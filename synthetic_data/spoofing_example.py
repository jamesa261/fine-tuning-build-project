from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from synthetic_data.clients import build_client, resolve_model


class SentimentResult(BaseModel):
    sentiment: str = Field(description="Either Positive or Negative")


def build_prompt(sentence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Classify sentiment as 'Positive' or 'Negative'. Reply only with JSON.",
        },
        {"role": "user", "content": f"Sentence: {sentence}"},
    ]


def classify(sentence: str, model_name: str) -> SentimentResult:
    model = resolve_model(model_name)
    client = build_client(model)

    if model.provider == "openai":
        completion = client.chat.completions.create(
            model=model.model,
            messages=build_prompt(sentence),
            response_format={"type": "json_object"},
        )
        payload = completion.choices[0].message.content
    else:
        response = client.models.generate_content(
            model=model.model,
            contents="\n".join(
                [
                    "Classify sentiment as Positive or Negative. Respond with JSON object {\"sentiment\": \"Positive|Negative\"}.",
                    f"Sentence: {sentence}",
                ]
            ),
            config={"response_mime_type": "application/json"},
        )
        payload = response.text

    return SentimentResult.model_validate_json(payload)


def main():
    parser = argparse.ArgumentParser(description="Minimal sentiment spoofing demo.")
    parser.add_argument("sentence", help="Sentence to classify")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name to use")
    args = parser.parse_args()

    result = classify(args.sentence, args.model)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
