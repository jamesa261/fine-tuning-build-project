# Fine-Tuning Build Project

Generate synthetic job titles, train a sentence-transformer on them, and ship artifacts into a Streamlit search demo using DVC (https://doc.dvc.org/) to coordinate stages.

## Repo Structure
- `synthetic_data/` — LLM-based generation scripts and prompts.
- `fine_tuning/` — Triplet-training pipeline, metrics, plots.
- `streamlit_app/` — Demo app plus embedding prep.
- `dvc.yaml` — Pipelines wiring all stages together.
- `params.yaml` — Tunable defaults for generation and training.

## Setup

### Virtual Environment
Using a uv (https://docs.astral.sh/uv/) managed virtual environment is recommended.
To set this up, run:
```bash
uv venv --python=3.13.11
source .venv/bin/activate
uv pip compile requirements.in -o requirements.txt --torch-backend=auto
uv pip sync requirements.txt --torch-backend=auto
```

### DVC Setup
To setup DVC, run `dvc init`. If you'd like to set up remote (or pseudo-remote in a folder on your machine), see DVC docs (https://doc.dvc.org/start) on `dvc remote`.

Environment:
- `OPENAI_API_KEY` for OpenAI models.
- `GEMINI_API_KEY` for Gemini/Gemma via Google GenAI.
- Authentication via Google Vertex AI (`gcloud auth login`) + environment variables for `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` & `GOOGLE_CLOUD_LOCATION`

## Pipelines (DVC)

### Testing with example dataset
To test the training and streamlit app on the included example dataset (i.e. without running the synthetic data generation):
```bash
cp synthetic_data/data/example_jittered_titles.csv synthetic_data/data/jittered_titles.csv
echo "{}" > synthetic_data/metrics/jitter_summary.json
dvc commit -df synthetic_data
```

This assigns the example dataset as the output of the synthetic data generation stage, allowing downstream stages in the pipeline to run with `dvc repro`.

### Run pipeline end-to-end
1) **synthetic_data**: `python -m synthetic_data.generate --params params.yaml`  
   - Produces `synthetic_data/data/jittered_titles.csv` and `synthetic_data/metrics/jitter_summary.json`.
2) **fine_tuning**: `python -m fine_tuning.train --params params.yaml`  
   - Uses jittered titles, writes train/val/test splits, trained model, metrics, and t-SNE plot.
3) **publish_model**: copies the best checkpoint to `streamlit_app/data/fine_tuned_model`.
4) **prepare_embeddings**: `python streamlit_app/prepare_embeddings.py`  
   - Generates `default_embeddings.npy` and `fine_tuned_embeddings.npy` for the Streamlit app.

Run the whole chain:  
```bash
dvc repro
```

Run a specific stage (+ preceding dependencies)
```bash
dvc repro fine_tuning
```

## Key Scripts
- `synthetic_data/generate.py`: async LLM generation with JSON outputs, supports OpenAI & Gemini/Gemma. Configurable via `params.yaml`.
- `fine_tuning/train.py`: dynamic-negative triplet training (cosine margin), metrics JSON, t-SNE plot.
- `streamlit_app/prepare_embeddings.py`: builds baseline and fine-tuned embeddings for the app.
