from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError
from tqdm.auto import tqdm

from synthetic_data.clients import build_client, resolve_model
from synthetic_data.models import TitleVariants

DEFAULT_PARAMS = {
    "model": "gpt-5-mini",
    "api_key_env": None,
    "num_examples_per_title": 10,
    "temperature": None,
    "reasoning_effort": "minimal",
    "max_retries": 6,
    "retry_backoff": 5.0,
    "max_concurrent": 4,
    "seed": 42,
    "output_responses": "synthetic_data/data/jitter_responses.jsonl",
    "output_titles": "synthetic_data/data/jittered_titles.csv",
    "metrics_path": "synthetic_data/metrics/jitter_summary.json",
    "chunk_size": 1,
}

SYSTEM_PROMPT = (
    "You generate realistic, varied job-posting titles. Titles should resemble listings "
    "seen on job boards: occasional locations, teams, seniority, bonuses, or abbreviations. "
    "Avoid repeating a single pattern."
)


def load_params(params_path: Path) -> Dict:
    if not params_path.exists():
        return DEFAULT_PARAMS.copy()
    import yaml

    params = yaml.safe_load(params_path.read_text()) or {}
    sd_params = params.get("synthetic_data", {})
    merged = DEFAULT_PARAMS | sd_params
    return merged


def load_seed_titles(seed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(seed_path)
    expected_cols = {"seed_title", "onet_code", "onet_name"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"seed_titles.csv is missing columns: {', '.join(missing)}")
    return df


def load_existing_jsonl(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    existing: Dict[str, List[str]] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            existing[record["seed_title"]] = record["in_the_wild_titles"]
        except Exception:
            continue
    return existing


def persist_jsonl(path: Path, records: Iterable[TitleVariants]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json(ensure_ascii=False))
            f.write("\n")


def build_messages(seed_title: str, num_examples: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Seed title: {seed_title}\n"
                f"Return exactly {num_examples} distinct 'in-the-wild' job titles as JSON with "
                f"keys seed_title and in_the_wild_titles (list of strings). "
                "Do not include any prose outside the JSON."
            ),
        },
    ]


async def request_openai(
    client,
    model: str,
    seed_title: str,
    num_examples: int,
    temperature: Optional[float],
    reasoning_effort: str,
) -> TitleVariants:
    kwargs = {
        "model": model,
        "messages": build_messages(seed_title, num_examples),
        "response_format": {"type": "json_object"},
        "reasoning_effort": reasoning_effort,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    completion = await client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    return parse_response(seed_title, content)


async def request_gemini(
    client,
    model: str,
    seed_title: str,
    num_examples: int,
    temperature: Optional[float],
    reasoning_effort: str,
) -> TitleVariants:
    thinking_level = translate_reasoning_to_thinking(model, reasoning_effort)
    config = {
        "response_mime_type": "application/json",
        "thinking_level": thinking_level,
    }
    if temperature is not None:
        config["temperature"] = temperature

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents="\n".join(
            [
                SYSTEM_PROMPT,
                "",
                f"Seed title: {seed_title}",
                f"Return exactly {num_examples} titles in JSON with keys "
                f"seed_title and in_the_wild_titles (list of strings).",
            ]
        ),
        config=config,
    )
    payload = response.text or ""
    return parse_response(seed_title, payload)


def parse_response(seed_title: str, payload: str) -> TitleVariants:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return JSON: {payload[:120]}") from exc

    # Allow either full object or bare list for resilience.
    if isinstance(data, list):
        data = {"seed_title": seed_title, "in_the_wild_titles": data}
    if "seed_title" not in data:
        data["seed_title"] = seed_title
    try:
        return TitleVariants.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid payload for {seed_title}: {payload}") from exc


def translate_reasoning_to_thinking(model_name: str, reasoning_effort: str) -> str:
    """
    Map OpenAI-style reasoning levels to Gemini thinking levels.
    Gemini 3 Pro supports only low/high; Flash supports minimal/low/medium/high.
    """
    level = reasoning_effort.lower()
    if "gemini-3.0-pro" in model_name:
        if level in {"high", "medium"}:
            return "high"
        return "low"
    if level not in {"minimal", "low", "medium", "high"}:
        return "low"
    return level


async def generate_variations(
    client,
    provider: str,
    model_name: str,
    seed_title: str,
    num_examples: int,
    temperature: Optional[float],
    max_retries: int,
    backoff: float,
    reasoning_effort: str,
    concurrency_guard: asyncio.Semaphore,
) -> TitleVariants:
    for attempt in range(max_retries):
        try:
            async with concurrency_guard:
                if provider == "openai":
                    return await request_openai(
                        client, model_name, seed_title, num_examples, temperature, reasoning_effort
                    )
                return await request_gemini(
                    client, model_name, seed_title, num_examples, temperature, reasoning_effort
                )
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise RuntimeError(
                    f"Failed to generate titles for '{seed_title}' after {max_retries} attempts"
                ) from exc
            tqdm.write(
                f"[warn] attempt {attempt+1}/{max_retries} for '{seed_title}' failed: {exc}; retrying..."
            )
            await asyncio.sleep(backoff * (2 ** attempt))


async def run_pipeline(params: Dict, seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[TitleVariants], Dict]:
    model_info = resolve_model(params["model"])
    client = build_client(model_info, api_key_env=params.get("api_key_env"), async_mode=True)

    cache_path = Path(params["output_responses"])
    cached = load_existing_jsonl(cache_path)

    semaphore = asyncio.Semaphore(params["max_concurrent"])

    async def process_row(row) -> TitleVariants:
        seed_title = row["seed_title"]
        if seed_title in cached:
            return TitleVariants(seed_title=seed_title, in_the_wild_titles=cached[seed_title])

        return await generate_variations(
            client=client,
            provider=model_info.provider,
            model_name=model_info.model,
            seed_title=seed_title,
            num_examples=params["num_examples_per_title"],
            temperature=params["temperature"],
            max_retries=params["max_retries"],
            backoff=params["retry_backoff"],
            reasoning_effort=params["reasoning_effort"],
            concurrency_guard=semaphore,
        )

    tasks = [process_row(row) for _, row in seed_df.iterrows()]
    records: List[TitleVariants] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        records.append(await coro)

    persist_jsonl(cache_path, records)

    rows = []
    for record in records:
        meta = seed_df.loc[seed_df["seed_title"] == record.seed_title].iloc[0].to_dict()
        for jittered in record.in_the_wild_titles:
            rows.append(
                {
                    "seed_title": record.seed_title,
                    "jittered_title": jittered,
                    "onet_code": meta["onet_code"],
                    "onet_name": meta["onet_name"],
                }
            )

    jitter_df = pd.DataFrame(rows)
    metrics = {
        "num_seed_titles": len(seed_df),
        "generated_titles": len(jitter_df),
        "avg_titles_per_seed": len(jitter_df) / len(seed_df) if len(seed_df) else 0,
        "model_used": model_info.model,
        "provider": model_info.provider,
    }
    return jitter_df, records, metrics


def write_outputs(
    jitter_df: pd.DataFrame,
    metrics: Dict,
    params: Dict,
) -> None:
    titles_path = Path(params["output_titles"])
    metrics_path = Path(params["metrics_path"])

    titles_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    jitter_df.to_csv(titles_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic job titles via LLMs.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"), help="Path to params.yaml")
    parser.add_argument("--model", type=str, help="Override model name (e.g., gpt-5-mini)")
    parser.add_argument("--num", type=int, help="Override examples per seed title")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; omit for reasoning models")
    parser.add_argument("--reasoning-effort", type=str, help="Reasoning level (minimal/low/medium/high)")
    parser.add_argument("--max-concurrent", type=int, help="Maximum concurrent requests")
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params)
    if args.model:
        params["model"] = args.model
    if args.num:
        params["num_examples_per_title"] = args.num
    if args.seed:
        params["seed"] = args.seed
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.reasoning_effort:
        params["reasoning_effort"] = args.reasoning_effort
    if args.max_concurrent:
        params["max_concurrent"] = args.max_concurrent

    seed_df = load_seed_titles(Path("synthetic_data/data/seed_titles.csv"))
    jitter_df, records, metrics = asyncio.run(run_pipeline(params, seed_df))
    write_outputs(jitter_df, metrics, params)
    print(f"Saved {len(jitter_df)} rows to {params['output_titles']}")


if __name__ == "__main__":
    main()
