## Specification: RAG-based GenAI Evaluation Agent for Hackathon Idea Presentations

### 1) Goals and Success Criteria

- **Objective**: Automatically ingest PowerPoint idea templates, extract idea details into a structured schema, evaluate each idea against a defined rubric using multiple LLMs, rank ideas, and produce rationale summaries with slide citations.
- **Success**:
  - ≥95% field extraction accuracy on mandatory template sections.
  - Deterministic, strictly valid JSON outputs across models.
  - Cross-model comparable scores with normalization and transparent ranking.
  - Reproducible runs with stored prompts, versions, seeds, inputs, and outputs.

### 2) High-Level Architecture

- **Ingestion & Preprocessing**: Collect `.pptx`, convert if needed, perform OCR for images.
- **Parser & Extractor**: Map slide content to structured fields; validate against schema.
- **RAG Indexing**: Chunk by slide; create embeddings with slide-number metadata for citations.
- **Evaluation Orchestrator**: Run identical prompts across multiple LLMs via a common interface.
- **Aggregation & Ranking**: Normalize/weight scores; compute final ranks; generate rationales.
- **Reporting**: Export CSV/JSON and a ranked Excel with summaries; optional dashboards.
- **Storage**: Object store for files, relational DB for structured data, vector store for retrieval.

### 3) Pipeline

1. Collect files → store raw files, compute checksums.
2. Convert `.ppt` → `.pptx`; extract text and layout using `python-pptx`; run OCR for images.
3. Extract fields (problem, solution, impact, feasibility, cost, duration, risks, etc.) with template-aware rules + fallback NLP/LLM.
4. Build slide-chunk RAG index with open-source embeddings; retain `slide_number` metadata for citations.
5. Evaluate per model:
   - Retrieve top-k slide chunks per criterion.
   - Use a single evaluation prompt (identical across models) with strict JSON schema.
   - Collect scores and rationales per criterion with slide citations.
6. Aggregate:
   - Validate JSON; normalize scores across models (z-score or min-max per criterion).
   - Weighted sum; handle ties; compute consensus rationale.
7. Rank:
   - Produce overall ranking and per-model rankings; compute agreement metrics.
8. Report:
   - Persist artifacts; export CSV/JSON; render the ranked Excel deliverable.

### 4) Data Model (Core Schema)

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

### 5) Evaluation Rubric (Config-Driven)

- **Criteria** (weight, definition, anchors):
  - **Clarity** (0.15): Problem and solution are specific, measurable, and coherent.
  - **Impact** (0.25): Business value, KPIs, scale of outcome.
  - **Feasibility** (0.20): Technical complexity, resources, dependencies, risks.
  - **Novelty** (0.15): Differentiation vs known solutions.
  - **Cost Efficiency** (0.10): Value per dollar; realistic cost.
  - **Time-to-Value** (0.10): Duration realism and phased delivery.
  - **Org Fit** (0.05): Strategic alignment, compliance-readiness.
- **Scoring Scale**: 1 (poor) to 5 (excellent), with anchor descriptions per criterion.
- **Normalization**:
  - Per model, per criterion: z-score across submissions, or min-max to [0,1].
  - Weighted sum on normalized scores; support model-averaged ensemble or adjudication.

Example rubric config (YAML):

```yaml
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

### 6) Multi-Model Strategy

- **Evaluation providers**: Mix of open-source via local runners (e.g., vLLM) and optionally hosted providers. An OSS-only profile is supported.
- **Common interface**: `ModelClient` with `generate(prompt, system, schema, params)`; supports JSON mode/function calling where available.
- **Consistency controls**: temperature=0.0–0.2, top_p=0.9, max_tokens tuned; fixed seed if supported.
- **Comparability**:
  - Use identical prompts and schema across models.
  - Normalize results per criterion across submissions per model.
  - Ensemble: average normalized criterion scores across models; optional model weighting.
  - Agreement metrics for diagnostics (Spearman, Kendall).

OSS-only evaluation profile (example):

```yaml
evaluation_models:
  - name: llama31_70b_instruct
    provider: vllm
    model: meta-llama/Meta-Llama-3.1-70B-Instruct
  - name: qwen2_72b_instruct
    provider: vllm
    model: Qwen/Qwen2-72B-Instruct
  - name: mixtral_8x22b_instruct
    provider: vllm
    model: mistralai/Mixtral-8x22B-Instruct-v0.1
params:
  temperature: 0.1
  top_p: 0.9
  max_tokens: 1200
```

### 7) Prompting (Shared Across Models)

System prompt:

```text
You are an impartial evaluator scoring hackathon ideas based ONLY on the provided slides content and retrieved excerpts. Output STRICT JSON matching the schema. Cite slide numbers for each criterion. If information is absent, score conservatively and explain the gap. Do not hallucinate. Do not include extra keys or commentary.
```

User prompt template:

```text
CONTEXT
- Team ID: {{team_id}}
- Title: {{title}}

EXTRACTED_FIELDS (may be partial)
{{extracted_fields_json}}

RETRIEVED_EXCERPTS
{{for each criterion: top-k chunks with slide numbers}}

TASK
Evaluate the idea on these criteria:
- clarity, impact, feasibility, novelty, cost_efficiency, time_to_value, org_fit

Return JSON only matching this schema:
{
  "scores": {
    "clarity": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "impact": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "feasibility": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "novelty": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "cost_efficiency": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "time_to_value": {"score": 1-5, "evidence_slides": [int], "justification": "string"},
    "org_fit": {"score": 1-5, "evidence_slides": [int], "justification": "string"}
  },
  "missing_info_notes": ["string"]
}
Output only JSON.
```

Consensus summary prompt (post-aggregation):

```text
You will produce a concise rationale for the final rank using the per-criterion averaged scores and slide citations.

INPUT
- Team ID: {{team_id}}, Title: {{title}}
- Averaged scores (normalized 0-1) per criterion with top evidence slides
- Tie-breaker notes (if applicable)

TASK
Write a 4-6 sentence summary explaining the ranking, explicitly referencing key slides (e.g., "slides 3–4") without adding new claims.
```

### 8) RAG Design (Open-Source Embeddings Only)

- **Chunking**: One chunk per slide; optionally split long slides by bullet groups while retaining `slide_number`.
- **Embeddings (OSS)**:
  - English: `BAAI/bge-large-en-v1.5`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-large-v2`, `Alibaba-NLP/gte-large-en-v1.5`.
  - Multilingual: `BAAI/bge-m3`, `distiluse-base-multilingual-cased-v2`.
  - Runners: `sentence-transformers` or `FlagEmbedding`; optional 8-bit with `bitsandbytes`.
  - Normalize embeddings (L2) before indexing; cosine/IP similarity.
- **Vector store**: FAISS (FlatIP or HNSW). Optional `pgvector` for persistence and joins.
- **Retrieval**: Per-criterion targeted queries; `top_k`=3–5; ensure evidence slides exist.
- **Citations**: Preserve `slide_number` and `slide_title` in metadata; require citations in outputs.

### 9) Validation and Guardrails

- Strict JSON validation against JSON Schema; auto-repair to schema if needed.
- Enforce score bounds [1..5].
- Evidence slides must exist in `raw_slide_map` and the vector store metadata.
- Missing field handling: score conservatively; record `missing_info_notes`.
- Rate limiting and retries per provider; exponential backoff; circuit breakers.
- Assert embedding model is in an OSS allowlist before indexing.

### 10) Ranking & Aggregation

- **Per model**: Weighted normalized score `S_model = Σ w_c * norm(score_c)`.
- **Across models**: `S_ensemble = average(S_model)`, optionally weighted by model reliability.
- **Tie-breaking**: Ordered by configured criteria (e.g., impact → feasibility → novelty).
- **Diagnostics**: Inter-model correlation (Spearman), agreement on top-10, bias checks.

Pseudocode:

```python
def rank_submissions(evaluations, rubric, method="zscore"):
    norm = normalize_per_model_criterion(evaluations, method)
    per_model_scores = {
        (sub, m): sum(rubric[w_c] * norm[(sub, m)][c] for c in rubric.criteria)
        for sub in submissions for m in models
    }
    ensemble = {
        sub: mean(per_model_scores[(sub, m)] for m in models)
        for sub in submissions
    }
    ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
    return ranked, per_model_scores
```

### 11) Storage & Artifacts

- **Object store**: `s3://.../submissions/{submission_id}/original.pptx` (or local path)
- **DB (Postgres/pgvector)**:
  - `submissions(id, team_id, title, created_at, checksum)`
  - `slides(submission_id, slide_number, text, notes, image_uri)`
  - `ideas(submission_id, jsonb_schema)`
  - `embeddings(submission_id, slide_number, vector, metadata)`
  - `evaluations(submission_id, model, raw_json, validated_json, created_at)`
  - `scores(submission_id, model, criterion, raw_score, normalized_score)`
  - `rankings(run_id, submission_id, ensemble_score, rank)`
  - `runs(id, config, rubric_version, code_version_hash)`

### 12) Base Code Skeleton (Python)

```text
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
    vllm_client.py
    openai_client.py
    anthropic_client.py
    google_client.py
    together_client.py
  evaluator.py
  aggregator.py
  ranking.py
  reporting.py
  store.py
  api.py
configs/
  rubric.yaml
  models.yaml
scripts/
  run_ingest.py
  run_evaluate.py
  run_rank.py
  export_excel.py
tests/
  test_parser.py
  test_evaluator.py
  test_ranking.py
```

Key interfaces:

```python
# app/models/base.py
from typing import Dict, Any

class ModelClient:
    def generate(self, system: str, prompt: str, schema: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# app/evaluator.py
def evaluate_submission(submission_id: str, model_clients: list, rubric, retriever, prompts):
    # 1) build inputs 2) call each model 3) validate/repair JSON 4) persist
    return evaluations

# app/aggregator.py
def aggregate_evaluations(evaluations, rubric, normalization="zscore"):
    # return per-model and ensemble scores, with normalized criterion scores
    return scores

# app/ranking.py
def compute_rankings(ensemble_scores, tie_breakers):
    # return ordered list with ties resolved
    return rankings
```

### 13) Configuration and Reproducibility

- **Versioning**: Pin rubric version and prompt version; store hash for every run.
- **Seeds**: Use deterministic parameters; log all generation params.
- **Run manifests**: Persist `configs/*.yaml`, prompt strings, and environment versions.

### 14) Observability & QA

- **Logging/Tracing**: Structured logs; trace each model call; latency and cost where applicable.
- **Prompt/Output Store**: Save all prompts and completions; attach run IDs to artifacts.
- **Dashboards**: Model agreement, per-criterion distributions, drift.
- **Tests**: Unit (parser, schema, normalizer), golden (PPTX → expected JSON, scores, rank), property (bounds, citation existence).

### 15) Deployment

- **Batch**: CLI jobs over a submissions folder; parallelized with a queue (RQ/Celery).
- **Service**: FastAPI endpoints:
  - `POST /ingest`
  - `POST /evaluate?submission_id=...`
  - `POST /rank?run_id=...`
  - `GET /report?submission_id=...`
- **Containerization**: Docker; GPU optional for OCR/LLM; secrets via env or vault.
- **Scaling**: Concurrency per provider limits; backpressure and retries.

### 16) Security & Compliance

- Secrets management (no keys in code), configurable data residency.
- Optional local-only mode (local LLM + local embeddings).
- PII redaction for slides/outputs if needed; access controls on artifacts.

### 17) Timeline (Indicative)

- Week 1: Parser + schema + tests + basic ingestion.
- Week 2: RAG indexing + single-model evaluator + JSON validation.
- Week 3: Multi-model adapters + normalization + ranking + reporting.
- Week 4: Observability + dashboards + golden tests + polish.

### 18) Minimal Example CLI Flow

```bash
python scripts/run_ingest.py --input_dir ./submissions
python scripts/run_evaluate.py --models_config configs/models.yaml --rubric configs/rubric.yaml
python scripts/run_rank.py --rubric configs/rubric.yaml --out ./outputs/rankings.csv
python scripts/export_excel.py
```

### 19) Output Artifacts

- `outputs/extractions/{submission_id}.json`
- `outputs/evaluations/{submission_id}/{model}.json`
- `outputs/ensemble_per_criterion.json`
- `outputs/consensus_summaries.json`
- `outputs/rankings.csv`
- `outputs/Ranked_Ideas.xlsx`

### 20) Risks and Mitigations

- **Template drift**: Add slide-heading synonym map and layout heuristics; fallback LLM extraction with schema validation.
- **Model JSON drift**: Use JSON mode/function calling; auto-repair validator; temperature near 0.
- **Cross-model scale bias**: Per-criterion normalization and model-averaged ensemble.
- **Hallucinations**: Strict instruction to cite slides; retrieval-limited context; penalize missing evidence.
- **OCR gaps**: Configure DPI upscaling; language packs; human-in-the-loop for unreadable slides.

### 21) Excel Output Specification (Final Deliverable)

- **File**: `outputs/Ranked_Ideas.xlsx`
- **Sheet 1 — Rankings** (default view):
  - Ordered by `rank` ascending (Rank 1 on top). Filters enabled; header row frozen.
  - Columns:
    - `rank` (int), `team_id`, `submission_id`, `title`
    - `problem_statement` (wrapped, ~400 chars)
    - `proposed_solution` (wrapped)
    - `impact_summary`, `feasibility_summary` (wrapped)
    - `cost_estimate_usd` (currency), `duration_estimate_weeks` (number)
    - `ensemble_score` (0–1, 3 decimals)
    - `clarity..org_fit` (either raw 1–5 or normalized consistently)
    - `evidence_slides` (e.g., "3,4; 5")
    - `ranking_summary` (4–6 sentence rationale with slide refs)
- **Sheet 2 — Per-Model Scores**: One row per `(submission_id, model_name)` with per-criterion details and totals.
- **Sheet 3 — Metadata**: `run_id`, timestamp, rubric weights, normalization method, model list/params, prompt/rubric versions.
- **Formatting**:
  - Currency format for `cost_estimate_usd`, numeric formats for durations/scores.
  - Text wrapping on long text columns; column widths set for readability.
  - Conditional formatting (3-color scale) on `ensemble_score`.

### 22) Open-Source Embeddings: Config and Code

Embedding config example:

```yaml
# configs/models.yaml (embeddings section)
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
```

Minimal embedding loader:

```python
# app/embeddings.py
from typing import List, Dict, Any
import numpy as np

def load_embedding_model(cfg: Dict[str, Any]):
    model_name = cfg["embeddings"]["model_name"]
    runner = cfg["embeddings"]["runner"]
    device = cfg["embeddings"].get("device", "auto")
    quantization = cfg["embeddings"].get("quantization", "none")

    if runner == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        extra_args = {}
        if quantization in {"bnb-8bit", "bnb-4bit"}:
            import torch
            extra_args["device"] = "cuda" if device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
            extra_args["trust_remote_code"] = True
        model = SentenceTransformer(model_name, **extra_args)
        return model
    elif runner == "flagembedding":
        from FlagEmbedding import FlagModel
        return FlagModel(model_name, query_instruction_for_retrieval="", use_fp16=True)
    else:
        raise ValueError(f"Unsupported runner: {runner}")

def embed_texts(model, texts: List[str], normalize: bool = True) -> np.ndarray:
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
    return emb
```

FAISS index snippet:

```python
# app/retriever.py (FAISS path)
import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int, use_hnsw: bool = True):
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.ids = []  # map to (submission_id, slide_number, chunk_id)

    def add(self, vectors: np.ndarray, meta_batch):
        self.index.add(vectors.astype(np.float32))
        self.ids.extend(meta_batch)

    def search(self, query_vecs: np.ndarray, top_k: int = 5):
        scores, idxs = self.index.search(query_vecs.astype(np.float32), top_k)
        results = []
        for row_scores, row_idxs in zip(scores, idxs):
            res = []
            for s, i in zip(row_scores, row_idxs):
                if i == -1:
                    continue
                res.append({"score": float(s), "meta": self.ids[i]})
            results.append(res)
        return results
```

### 23) Dependencies

- Core: `python-pptx`, `pandas`, `numpy`, `pyyaml`, `pydantic`, `rapidfuzz`, `tqdm`.
- Embeddings: `sentence-transformers`, `flagembedding`, `faiss-cpu` (or `faiss-gpu`).
- Optional OCR: `pytesseract`, `Pillow`, Tesseract binaries.
- Optional DB: `psycopg2-binary`, `pgvector`.
- Optional GPU/quant: `torch`, `bitsandbytes`.

### 24) Troubleshooting

- JSON not valid: enable JSON mode where possible; run schema-repair; set temperature to 0.1.
- Weak retrieval: confirm embeddings are L2-normalized; use IP similarity; increase `top_k`.
- OCR missing text: enable OCR; increase DPI; install correct language packs.
- Memory constraints: use `bge-small` or 8-bit quantization; reduce batch size.
- Cross-model drift: prefer z-score normalization per criterion; review rubric weights.
