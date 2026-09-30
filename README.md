# Databricks Custom LLM Serving — Qwen3.5-27B on RTX PRO 6000

End-to-end example of running **Qwen3.5-27B** (FP8 and BF16) on Databricks
[Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
using a single **NVIDIA RTX PRO 6000 (96 GB, `GPU_LARGE_RTX`)**.

The RTX PRO 6000's 96 GB GDDR7 fits the entire 27B model on one GPU (no tensor parallelism),
resulting in low-overhead, high-throughput inference backed by vLLM's continuous batching.

## Results at a glance

| Model | C=1 tok/s | C=32 tok/s | Peak tok/s | GPU |
|---|---|---|---|---|
| Qwen3.5-27B-FP8 | 37.9 | 1,016 | **2,282** (C=128) | RTX PRO 6000 |
| Qwen3.5-27B BF16 | 26.6 | 644 | **1,076** (C=64) | RTX PRO 6000 |

Full benchmark data: [BENCHMARK.md](BENCHMARK.md)

## Contents

```
├── notebooks/
│   ├── serve_qwen35_27b_fp8.py    # Registration notebook — FP8 (Databricks source format)
│   └── serve_qwen35_27b_bf16.py   # Registration notebook — BF16
├── scripts/
│   ├── create_endpoint.py         # Create the serving endpoint via REST API
│   └── benchmark.py               # Concurrency / throughput benchmark client
├── BENCHMARK.md                   # Full benchmark results
└── README.md
```

## Prerequisites

- Databricks workspace with **Custom LLM Serving** enabled  
  (Workspace Admin → Settings → Preview features → "Serve custom LLMs")
- Unity Catalog enabled; a catalog and schema you own
- Databricks CLI authenticated: `databricks auth login --host <workspace-url>`
- `GPU_LARGE_RTX` available in your workspace region  
  (available in: us-east-1, us-west-2, us-east-2, ap-northeast-1, ap-northeast-2 — see
  [region support](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms))
- HuggingFace access to `Qwen/Qwen3.5-27B-FP8` and `Qwen/Qwen3.5-27B`

> **Tokyo region (ap-northeast-1):** `GPU_LARGE_RTX` is available. Enable
> Custom LLM Serving in your workspace settings, then follow the same steps below.

## Step-by-step

### 1. Edit configuration

In `notebooks/serve_qwen35_27b_fp8.py`, set your catalog and schema:

```python
CATALOG = "your_catalog"
SCHEMA  = "your_schema"
```

### 2. Import the notebook

```bash
databricks workspace import \
  --file notebooks/serve_qwen35_27b_fp8.py \
  --format SOURCE --language PYTHON \
  /Users/<your-email>/serve_qwen35_27b_fp8
```

### 3. Run the notebook as a Serverless GPU job

The `register_model(env_pack="databricks_model_serving")` call **must** run on a
Serverless GPU job (not a classic cluster).

Create a job run via the UI or CLI with:
- **Compute:** Serverless GPU, `GPU_1xH100`
- **Environment:** AI Runtime client 4

The notebook downloads the model from HuggingFace, logs it to MLflow, and registers it
to Unity Catalog with the vLLM serving entrypoint baked in.

### 4. Create the serving endpoint

Edit `scripts/create_endpoint.py`:

```python
ENDPOINT_NAME = "qwen35-27b-fp8-rtx"
UC_MODEL_NAME = "your_catalog.your_schema.qwen35_27b_fp8"
MODEL_VERSION = "1"
```

Then run:

```bash
python scripts/create_endpoint.py
```

The endpoint takes 10–20 minutes to reach `READY` (model artifacts download +
vLLM startup + torch.compile).

### 5. Query the endpoint

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()
resp = w.serving_endpoints.query(
    name="qwen35-27b-fp8-rtx",
    messages=[ChatMessage(role=ChatMessageRole.USER, content="What is Databricks?")],
    max_tokens=256,
    extra_params={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

### 6. Benchmark

```bash
DATABRICKS_PROFILE=DEFAULT python scripts/benchmark.py \
  --endpoint qwen35-27b-fp8-rtx \
  --profile short \
  --max-tokens 256 \
  --levels 1,4,8,16,32,64
```

---

## Key differences from the official sample

Databricks provides two relevant official references:

- **[Qwen3.5-0.8B fine-tuning + serving tutorial](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-qwen3.5-0.8b)**
  — end-to-end SFT (H100) → serving on `GPU_MEDIUM` (A10) for a 0.8B model.
- **[Custom LLM Serving starter notebook](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)**
  — `vllm==0.11.2` / `transformers==4.57.6`, ~4B model on `GPU_MEDIUM` (A10).

Both official examples target A10 (`GPU_MEDIUM`), which is sufficient for sub-4B models.
At 27B scale, A10 has insufficient VRAM — hence `GPU_LARGE_RTX` (96 GB) — and two
dependency fixes are required:

### Fix 1 — Large model upload: `databricks-sdk>=0.102.0`

Without this, `mlflow.register_model(env_pack="databricks_model_serving")` times out
uploading `model_version.tar` for models larger than ~10 GB. The bundled
`databricks-sdk` in older AI Runtime images uses single-shot HTTP PUT;
`>=0.102.0` switches to multipart upload which eliminates the 5-minute per-file timeout.

Add to `%pip install`:
```python
"databricks-sdk>=0.102.0"
```

### Fix 2 — Health check failure: `fastapi<0.137.0`

FastAPI 0.137.0 introduced a routing change that breaks `prometheus_fastapi_instrumentator`.
This causes the `/v1/models` health check endpoint to return HTTP 500, so the Databricks
serving platform cannot confirm the model is healthy and the deployment never reaches `READY`.

Add to `extra_pip_requirements`:
```python
"fastapi<0.137.0"
```

---

## Dependency versions

All versions are pinned for reproducibility:

| Package | Version | Notes |
|---|---|---|
| vllm | 0.19.1 | Minimum for Qwen3.5 architecture |
| transformers | 5.5.4 | Required for `Qwen3_5ForConditionalGeneration` |
| mlflow | 3.12.0 | Minimum for `env_pack` support |
| databricks-sdk | ≥0.102.0 | Multipart upload for large models |
| fastapi | <0.137.0 | Health check fix |
| hf_transfer | 0.1.9 | Fast HuggingFace downloads |
| openai | 2.17.0 | Client for local smoke tests |

---

## Troubleshooting: serving base-image drift

The serving container ships a base image whose contents (e.g. vLLM, torch, flash_attn)
can change over time. If you rely on packages that happen to be in the base image, a
deploy that worked last week can fail today. Two symptoms and their fixes:

### `No module named 'vllm'` at container start

The base image stopped providing vLLM. **Fix:** pin the full serving stack in
`extra_pip_requirements` so the container installs it regardless of the base image:

```python
extra_pip_requirements=["vllm==0.19.1", "transformers==5.5.4",
                        "mlflow==3.12.0", "fastapi<0.137.0", "hf_transfer==0.1.9"]
```

(The notebooks in this repo already do this.)

### `flash_attn ... undefined symbol` / `EngineCore failed to start`

The base image's `flash_attn` was built against a different torch ABI, so importing it
crashes. For a multimodal model like Qwen3.5, vLLM's rotary-embedding init imports
`flash_attn` when it is discoverable. vLLM 0.19.1 guards this with
`if find_spec("flash_attn") is not None:` — so if `flash_attn` is **not** discoverable,
vLLM falls back to a native rotary implementation and starts cleanly. (vLLM's main
decoder attention uses the vendored `vllm_flash_attn`, which is unaffected — so text
throughput is not meaningfully impacted.)

**Fix:** make the broken `flash_attn` undiscoverable at the start of the serving
entrypoint (quote-free, so it is safe inside the `bash -lc '...'` wrapper):

```bash
mv <site-packages>/flash_attn <site-packages>/flash_attn_disabled 2>/dev/null || true
```

Find `<site-packages>` from the traceback in the endpoint's Service logs (the path in the
`ImportError` line). This is environment-specific, so it is documented here rather than
baked into the default entrypoint.

---

## References

- [Serve custom LLMs — Databricks documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
- [Serverless Optimized Deployments (SOD)](https://docs.databricks.com/aws/en/machine-learning/model-serving/serverless-optimized-deployments)
- [Feature and region support](https://docs.databricks.com/aws/en/resources/feature-region-support)
- [Qwen3.5 model card — HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B-FP8)
