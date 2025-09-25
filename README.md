## RAG-based GenAI Evaluation Agent for Hackathon Ideas

### Overview

This repository provides an evaluation agent that ingests hackathon idea presentations in PowerPoint, extracts idea details into a structured schema, evaluates them against a rubric using multiple LLMs, ranks the ideas, and exports an Excel with key details and a concise ranking summary.

- **Input**: `.pptx` idea templates (problem, solution, impact, duration, cost, etc.)
- **Output**: `outputs/Ranked_Ideas.xlsx` ordered by rank (Rank 1 on top) with per-idea details and a concise rationale for the assigned rank. Optional per-model score breakdown and metadata sheets are included.
- **Constraint**: Retrieval uses only open-source models for embeddings (e.g., BGE/E5/GTE). Evaluation models may be open-source or a mix, configurable via `configs/models.yaml`.

### Key Features

- **End-to-end pipeline**: Ingestion → Extraction → RAG retrieval → Multi-model evaluation → Normalization/Ranking → Reporting
- **Open-source embeddings**: BAAI BGE/E5/GTE families with FAISS or `pgvector`
- **Multi-model evaluation**: Run identical prompts across multiple models; compare and ensemble results
- **Reproducible**: Config-driven rubric, prompts, and models; artifacts persisted with run metadata
- **Auditability**: Slide citations for each criterion; JSON schema validation
- **Final Excel**: Clean, column-wise, ranked view with a 4–6 sentence ranking summary

### Architecture (High Level)

1. Ingestion & OCR: Load `.pptx`, extract text/layout (and images via OCR if needed)
2. Extraction: Map slides to a structured idea schema and validate
3. RAG Indexing: Chunk per slide and embed using open-source models; index in FAISS/pgvector
4. Evaluation: Retrieve top-k slide chunks per criterion; run identical prompts across models
5. Aggregation & Ranking: Validate JSON, normalize per-model criteria, compute weighted scores and ranks
6. Reporting: Generate per-idea summaries and export Excel; persist artifacts and metadata

### Repository Structure

```text
repo/
  app/
    __init__.py
    config.py
    ingestion.py
    ocr.py
    pptx_parser.py
    extractor.py
    embeddings.py
    retriever.py
    prompts.py
    models/
      base.py
      openai_client.py         # optional
      anthropic_client.py      # optional
      google_client.py         # optional
      together_client.py       # optional / OSS hosting
      vllm_client.py           # optional local inference for OSS LLMs
    evaluator.py
    aggregator.py
    ranking.py
    reporting.py
    store.py
    api.py
  configs/
    rubric.yaml
    models.yaml
    export.yaml                # optional Excel column config
  scripts/
    run_ingest.py
    run_evaluate.py
    run_rank.py
    export_excel.py
  tests/
    test_parser.py
    test_evaluator.py
    test_ranking.py
  outputs/                     # generated artifacts
```

### Installation

Requirements: Python 3.10+, ffmpeg/tesseract if OCR is enabled, optional CUDA for acceleration.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install sentence-transformers flagembedding faiss-cpu numpy pandas openpyxl pydantic pyyaml python-pptx rapidfuzz langdetect tqdm
# Optional for GPU/quant:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install bitsandbytes faiss-gpu
# Optional for DB vector store:
# pip install psycopg2-binary pgvector
```

Set provider API keys if using hosted LLMs (optional):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

### Configuration

#### Embeddings (open-source only)

```yaml
# configs/models.yaml
embeddings:
  model_name: BAAI/bge-large-en-v1.5
  runner: sentence_transformers
  device: auto
  batch_size: 64
  normalize: true
  quantization: none
  multilingual_fallback:
    enabled: true
    model_name: BAAI/bge-m3
  index:
    backend: faiss
    type: hnsw
    hnsw_m: 32
    hnsw_ef_construction: 200
    ef_search: 64

evaluation_models:
  - name: llama31_70b_instruct
    provider: vllm
    model: meta-llama/Meta-Llama-3.1-70B-Instruct
  - name: qwen2_72b_instruct
    provider: vllm
    model: Qwen/Qwen2-72B-Instruct
params:
  temperature: 0.1
  top_p: 0.9
  max_tokens: 1200
```

#### Rubric

```yaml
# configs/rubric.yaml
criteria:
  clarity: { weight: 0.15 }
  impact: { weight: 0.25 }
  feasibility: { weight: 0.20 }
  novelty: { weight: 0.15 }
  cost_efficiency: { weight: 0.10 }
  time_to_value: { weight: 0.10 }
  org_fit: { weight: 0.05 }
normalization: zscore
aggregation: weighted_sum
tie_breakers: [impact, feasibility, novelty]
```

#### Excel Export (optional)

```yaml
# configs/export.yaml
rankings_sheet:
  include_columns:
    - rank
    - team_id
    - submission_id
    - title
    - problem_statement
    - proposed_solution
    - impact_summary
    - feasibility_summary
    - cost_estimate_usd
    - duration_estimate_weeks
    - ensemble_score
    - clarity
    - impact
    - feasibility
    - novelty
    - cost_efficiency
    - time_to_value
    - org_fit
    - evidence_slides
    - ranking_summary
text_wrap_columns:
  - problem_statement
  - proposed_solution
  - impact_summary
  - feasibility_summary
  - ranking_summary
```

### Running the Pipeline (CLI)

```bash
# 1) Ingest all PPTX files in ./submissions
python scripts/run_ingest.py --input_dir ./submissions

# 2) Evaluate all submissions using configured models & RAG
python scripts/run_evaluate.py --models_config configs/models.yaml --rubric configs/rubric.yaml

# 3) Aggregate, normalize, and compute rankings
python scripts/run_rank.py --rubric configs/rubric.yaml --out ./outputs/rankings.csv

# 4) Export the final Excel deliverable
python scripts/export_excel.py
```

Generated artifacts:

- `outputs/extractions/{submission_id}.json`
- `outputs/evaluations/{submission_id}/{model}.json`
- `outputs/ensemble_per_criterion.json`
- `outputs/consensus_summaries.json`
- `outputs/rankings.csv`
- `outputs/Ranked_Ideas.xlsx`

### Excel Deliverable

The main sheet, "Rankings", is sorted by `rank` ascending and includes:

- rank, team_id, submission_id, title
- problem_statement, proposed_solution (wrapped and truncated for readability)
- impact_summary, feasibility_summary
- cost_estimate_usd, duration_estimate_weeks
- ensemble_score and per-criterion scores (clarity..org_fit)
- evidence_slides (e.g., "3,4; 5")
- ranking_summary (4–6 sentences with slide references)

Additional sheets:

- "Per-Model Scores": one row per `(submission_id, model)` with per-criterion details
- "Metadata": run_id, timestamp, model list, prompt/rubric versions, normalization method

### Data Schema (Extraction)

```json
{
  "team_id": "string",
  "submission_id": "string",
  "title": "string",
  "problem_statement": "string",
  "proposed_solution": "string",
  "target_users": "string",
  "impact": {
    "business_value": "string",
    "kpis": ["string"],
    "org_alignment": "string"
  },
  "feasibility": {
    "tech_complexity": "string",
    "dependencies": ["string"],
    "risks": ["string"]
  },
  "cost_estimate_usd": 0,
  "duration_estimate_weeks": 0,
  "novelty": "string",
  "compliance_considerations": "string",
  "slide_citations": [
    { "field": "problem_statement", "slides": [2, 3] }
  ],
  "raw_slide_map": [
    { "slide_number": 1, "text": "string", "notes": "string" }
  ]
}
```

### Prompting & Evaluation

- A concise system prompt enforces strict JSON outputs and slide citations per criterion.
- Per-criterion retrieval provides top-k slide chunks (by cosine similarity) as evidence.
- Models return 1–5 scores for criteria: clarity, impact, feasibility, novelty, cost_efficiency, time_to_value, org_fit.
- A consensus step averages normalized scores (z-score by criterion) across models and produces a 4–6 sentence summary for Excel.

### RAG Details (Open-Source Embeddings)

- Default embedding: `BAAI/bge-large-en-v1.5` (1024-dim, L2-normalized)
- Multilingual fallback: `BAAI/bge-m3`
- Alternatives: `intfloat/e5-large-v2`, `Alibaba-NLP/gte-large-en-v1.5`, `bge-small` variants for CPU speed
- Vector store: FAISS (HNSW/IP). Optional `pgvector` for persistence and SQL joins
- Metadata per chunk: `submission_id`, `slide_number`, `slide_title`, `lang`, `chunk_id`

### Ranking & Aggregation

- Per model: weighted sum of normalized per-criterion scores
- Across models: average model totals (optionally weighted by reliability)
- Ties resolved by `impact → feasibility → novelty`
- Diagnostics: agreement metrics (Spearman/Kendall), per-criterion distributions

### Reproducibility & Observability

- Persist prompt strings, configs, model names, seeds/params, and checksums
- Structured logs for each model call (latency, tokens/cost where applicable)
- Store embedding metadata: repo, revision, vector_dim, normalize flag, index type

### Testing

```bash
pytest -q
```

- Unit tests for parser, schema validation, normalization, and ranking
- Golden tests: known PPTX → expected JSON, scores, and rank

### Optional: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -U pip \
    && pip install sentence-transformers flagembedding faiss-cpu numpy pandas openpyxl pydantic pyyaml python-pptx rapidfuzz langdetect tqdm
CMD ["bash", "-lc", "python scripts/run_ingest.py --input_dir ./submissions && python scripts/run_evaluate.py --models_config configs/models.yaml --rubric configs/rubric.yaml && python scripts/run_rank.py --rubric configs/rubric.yaml --out ./outputs/rankings.csv && python scripts/export_excel.py"]
```

### Troubleshooting

- Empty or malformed JSON from a model: enable JSON mode if supported or run a schema-repair pass
- Poor retrieval: verify embeddings are normalized and index uses IP; increase `top_k`
- OCR missing text: enable OCR and ensure Tesseract is installed; increase image DPI during extraction
- Memory issues: use `bge-small` or enable 8-bit quantization (`bitsandbytes`)

### License

Add your license of choice (e.g., Apache-2.0).

