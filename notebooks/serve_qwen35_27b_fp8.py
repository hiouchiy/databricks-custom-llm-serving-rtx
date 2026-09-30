# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM Serving — Qwen3.5-27B-FP8 on RTX PRO 6000 (GPU_LARGE_RTX)
# MAGIC
# MAGIC End-to-end Custom LLM Serving of Qwen3.5-27B-FP8 on a single RTX PRO 6000 (96 GB).
# MAGIC
# MAGIC | Stage | Compute |
# MAGIC |---|---|
# MAGIC | This notebook (download / log / register) | Serverless GPU 1×H100 (`GPU_1xH100`) |
# MAGIC | The serving endpoint | 1×RTX PRO 6000 96 GB (`GPU_LARGE_RTX`, TP=1) |
# MAGIC
# MAGIC ## Why a single RTX PRO 6000 is sufficient
# MAGIC The RTX PRO 6000 has 96 GB GDDR7.  Qwen3.5-27B-FP8 weighs ~27 GB, leaving ~69 GB
# MAGIC for KV cache and activations — all on one device with no inter-GPU communication overhead.
# MAGIC
# MAGIC ## Key differences from the official 4B sample
# MAGIC The official starter notebook ([docs.databricks.com](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms))
# MAGIC uses `vllm==0.11.2` / `transformers==4.57.6` for a 4B model.
# MAGIC Qwen3.5-27B requires newer dependencies and two additional fixes:
# MAGIC
# MAGIC | Item | Official sample | This notebook |
# MAGIC |---|---|---|
# MAGIC | vLLM | `0.11.2` | **`0.19.1`** |
# MAGIC | transformers | `4.57.6` | **`5.5.4`** (Qwen3.5 arch) |
# MAGIC | `databricks-sdk` | not pinned | **`>=0.102.0`** (multipart upload for large models) |
# MAGIC | `fastapi` | not pinned | **`<0.137.0`** (health-check fix) |

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi || echo "nvidia-smi not available"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dependencies
# MAGIC
# MAGIC All versions are pinned for reproducibility.
# MAGIC
# MAGIC **Why `databricks-sdk>=0.102.0`:** versions below 0.102 use single-shot PUT for artifact
# MAGIC uploads; large models (~28 GB tar) exceed the 5-minute per-file timeout.
# MAGIC `>=0.102.0` enables multipart upload and eliminates the timeout.
# MAGIC
# MAGIC **Why `fastapi<0.137.0`:** FastAPI 0.137.0 introduced a breaking change in
# MAGIC `prometheus_fastapi_instrumentator` routing that causes `/v1/models` to return 500,
# MAGIC making the serving health check fail and preventing the endpoint from becoming READY.

# COMMAND ----------

%pip install \
  "vllm==0.19.1" \
  "transformers==5.5.4" \
  "openai==2.17.0" \
  "hf_transfer==0.1.9" \
  "mlflow==3.12.0" \
  "databricks-sdk>=0.102.0"
%restart_python

# COMMAND ----------

%pip uninstall -y opencv-python-headless opencv-python

# COMMAND ----------

import os, tempfile, shutil

LOCAL_TMP = "/local_disk0/tmp"
os.makedirs(LOCAL_TMP, exist_ok=True)
os.environ["TMPDIR"] = LOCAL_TMP
tempfile.tempdir = LOCAL_TMP
workdir = tempfile.mkdtemp(dir=LOCAL_TMP)
os.chdir(workdir)
print("CWD:", workdir)
usage = shutil.disk_usage(LOCAL_TMP)
print(f"{LOCAL_TMP} free: {usage.free / 1e9:.1f} GB / total {usage.total / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md ## Configuration — edit these values before running

# COMMAND ----------

# ── Edit these three lines ────────────────────────────────────────────────────
CATALOG  = "your_catalog"   # Unity Catalog catalog name
SCHEMA   = "your_schema"    # Unity Catalog schema name
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REPO_ID     = "Qwen/Qwen3.5-27B-FP8"
ARTIFACTS_PATH    = "qwen35_27b_fp8"
SERVED_MODEL_NAME = "qwen"
DTYPE             = "bfloat16"
SERVING_PORT      = 8080

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qwen35_27b_fp8"

# COMMAND ----------

# MAGIC %md ## Download Qwen3.5-27B-FP8 (~27 GB)

# COMMAND ----------

import os, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
t0 = time.time()
snapshot_download(repo_id=MODEL_REPO_ID, local_dir=ARTIFACTS_PATH)
print(f"download {time.time()-t0:.0f}s; shards:",
      len([f for f in os.listdir(ARTIFACTS_PATH) if f.endswith(".safetensors")]))

# COMMAND ----------

# MAGIC %md ## Serving entrypoint

# COMMAND ----------

def entrypoint_serving(port: int) -> str:
    inner = (
        # opencv FIPS fix: FP8 path imports gguf which triggers a libcrypto abort.
        "python -m pip uninstall -y opencv-python-headless opencv-python >/dev/null 2>&1 || true; "
        "VLLM_WORKER_MULTIPROC_METHOD=fork "
        "exec python -u -m vllm.entrypoints.openai.api_server "
        f"--model {ARTIFACTS_PATH} --served-model-name {SERVED_MODEL_NAME} "
        f"--host 0.0.0.0 --port {port} --dtype {DTYPE} --max-model-len 8192 "
        "--gpu-memory-utilization 0.88 --max-num-seqs 128"
    )
    return f"bash -lc '{inner}'"

print("SERVING:", entrypoint_serving(SERVING_PORT))

# COMMAND ----------

# MAGIC %md ## Log and register (SOD)

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

model_info = mlflow.pyfunc.log_model(
    name=SERVED_MODEL_NAME,
    python_model=LLMModel(),
    artifacts={"model_dir": ARTIFACTS_PATH},
    metadata={"task": "llm/v1/chat", "entrypoint": entrypoint_serving(SERVING_PORT)},
    # fastapi<0.137.0: 0.137.0 breaks prometheus-fastapi-instrumentator → /v1/models 500
    # Pin the serving stack so the container does not depend on the serving base image,
    # which can drift over time (e.g. vLLM disappearing or a torch/flash_attn ABI mismatch).
    extra_pip_requirements=["vllm==0.19.1", "transformers==5.5.4",
                            "mlflow==3.12.0", "fastapi<0.137.0", "hf_transfer==0.1.9"],
)
print("model_uri:", model_info.model_uri)

# COMMAND ----------

# Must run on a GPU job (Serverless GPU) for env_pack to build a valid SOD artifact.
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME,
                                       env_pack="databricks_model_serving")
dbutils.notebook.exit(f"REGISTERED {UC_MODEL_NAME} v{model_version.version}")
