# Fine-Tuning Build Project

Generate synthetic job titles, train a sentence-transformer on them, and ship artifacts into a Streamlit search demo using DVC (https://doc.dvc.org/) to coordinate stages.

## Repo Structure
- `synthetic_data/` — LLM-based generation scripts and prompts.
- `fine_tuning/` — Triplet-training pipeline, metrics, plots.
- `streamlit_app/` — Demo app plus embedding prep.
- `dvc.yaml` — Pipelines wiring all stages together.
- `params.yaml` — Tunable defaults for generation and training.

## Pipelines (DVC)
1) **synthetic_data**: `python -m synthetic_data.generate --params params.yaml`  
   - Produces `synthetic_data/data/jittered_titles.csv` and `synthetic_data/metrics/jitter_summary.json`.
2) **fine_tuning**: `python -m fine_tuning.train --params params.yaml`  
   - Uses jittered titles, writes train/val/test splits, trained model, metrics, and t-SNE plot.
3) **publish_model**: copies the best checkpoint to `streamlit_app/data/fine_tuned_model`.
4) **prepare_embeddings**: `python streamlit_app/prepare_embeddings.py`  
   - Generates `default_embeddings.npy` and `fine_tuned_embeddings.npy` for the Streamlit app.

Run the whole chain:  
```bash
dvc repro prepare_embeddings
```  
*(Do **not** run generation/training if you want to avoid LLM calls—those stages are cached; rerun only when keys/quotas are set and you intend to bill.)*

## Setup
Using a uv (https://docs.astral.sh/uv/) managed virtual environment is recommended.
To set this up, run:
```bash
uv venv --python=3.13.11
source .venv/bin/activate
uv pip compile requirements.in -o requirements.txt --torch-backed=auto
uv pip sync requirements.txt
```

Environment:
- `OPENAI_API_KEY` for OpenAI models.
- `GEMINI_API_KEY` for Gemini/Gemma via Google GenAI.
- Authentication via Google Vertex AI (`gcloud auth login`) + environment variables for `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` & `GOOGLE_CLOUD_LOCATION`

## Key Scripts
- `synthetic_data/generate.py`: async LLM generation with JSON outputs, supports OpenAI & Gemini/Gemma. Configurable via `params.yaml`.
- `fine_tuning/train.py`: dynamic-negative triplet training (cosine margin), metrics JSON, t-SNE plot.
- `streamlit_app/prepare_embeddings.py`: builds baseline and fine-tuned embeddings for the demo.

## Useful Commands
```bash
# Reproduce only training (uses cached synthetic data)
dvc repro fine_tuning

# Publish trained model + embeddings (after training)
dvc repro publish_model prepare_embeddings

# Clean DVC outputs (careful)
dvc remove --outs
```

## Notes & Cautions
- LLM stages will call external APIs; ensure keys/quotas and be mindful of cost.
- Temperature defaults to `null` for reasoning models; `reasoning_effort` is configurable in `params.yaml`.
- Dynamic negatives are enabled by default; disable with `dynamic_negatives: false` in `params.yaml` if you prefer static triplets.
