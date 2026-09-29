# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM Serving — Qwen3.5-27B BF16 on RTX PRO 6000 (GPU_LARGE_RTX)
# MAGIC
# MAGIC End-to-end Custom LLM Serving of the unquantized BF16 27B model on a single RTX PRO 6000 (96 GB).
# MAGIC
# MAGIC | Stage | Compute |
# MAGIC |---|---|
# MAGIC | This notebook (download / log / register) | Serverless GPU 1×H100 (`GPU_1xH100`) |
# MAGIC | The serving endpoint | 1×RTX PRO 6000 96 GB (`GPU_LARGE_RTX`, TP=1) |
# MAGIC
# MAGIC ## Memory budget
# MAGIC BF16 weights: ~54 GB. With `gpu_memory_utilization=0.88` → 84.5 GB usable.
# MAGIC KV cache budget: ~30 GB — supports `max_model_len=8192` with `max_num_seqs=64`.
# MAGIC
# MAGIC ## Δ vs FP8 sibling
# MAGIC | Property | FP8 | **BF16 (this notebook)** |
# MAGIC |---|---|---|
# MAGIC | Model | `Qwen/Qwen3.5-27B-FP8` | **`Qwen/Qwen3.5-27B`** |
# MAGIC | Weights on disk | ~27 GB | **~54 GB** |
# MAGIC | KV cache budget | ~69 GB | **~30 GB** |
# MAGIC | `--max-num-seqs` | 128 | **64** |
# MAGIC | opencv FIPS crash | Present (FP8 imports gguf) | Absent — removed defensively |

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi || echo "nvidia-smi not available"

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

# ── Edit these two lines ──────────────────────────────────────────────────────
CATALOG  = "your_catalog"   # Unity Catalog catalog name
SCHEMA   = "your_schema"    # Unity Catalog schema name
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REPO_ID     = "Qwen/Qwen3.5-27B"   # BF16 (no -FP8 suffix)
ARTIFACTS_PATH    = "qwen35_27b_bf16"
SERVED_MODEL_NAME = "qwen"
DTYPE             = "bfloat16"
SERVING_PORT      = 8080

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qwen35_27b_bf16"

# COMMAND ----------

# MAGIC %md ## Download Qwen3.5-27B BF16 (~54 GB)

# COMMAND ----------

import os, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
t0 = time.time()
snapshot_download(repo_id=MODEL_REPO_ID, local_dir=ARTIFACTS_PATH)
print(f"download {time.time()-t0:.0f}s; shards:",
      len([f for f in os.listdir(ARTIFACTS_PATH) if f.endswith(".safetensors")]))

# COMMAND ----------

def entrypoint_serving(port: int) -> str:
    inner = (
        "python -m pip uninstall -y opencv-python-headless opencv-python >/dev/null 2>&1 || true; "
        "VLLM_WORKER_MULTIPROC_METHOD=fork "
        "exec python -u -m vllm.entrypoints.openai.api_server "
        f"--model {ARTIFACTS_PATH} --served-model-name {SERVED_MODEL_NAME} "
        f"--host 0.0.0.0 --port {port} --dtype {DTYPE} --max-model-len 8192 "
        "--gpu-memory-utilization 0.88 --max-num-seqs 64"
    )
    return f"bash -lc '{inner}'"

print("SERVING:", entrypoint_serving(SERVING_PORT))

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
    extra_pip_requirements=["mlflow==3.12.0", "fastapi<0.137.0"],
)
print("model_uri:", model_info.model_uri)

# COMMAND ----------

model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME,
                                       env_pack="databricks_model_serving")
dbutils.notebook.exit(f"REGISTERED {UC_MODEL_NAME} v{model_version.version}")
