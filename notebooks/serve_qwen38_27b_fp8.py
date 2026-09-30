# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM Serving — Qwen3.8-27B-FP8 on RTX PRO 6000 (GPU_LARGE_RTX)
# MAGIC
# MAGIC | Stage | Compute |
# MAGIC |---|---|
# MAGIC | This notebook (download / log / register) | Serverless GPU 1×H100 (`GPU_1xH100`) |
# MAGIC | The serving endpoint | 1×RTX PRO 6000 96 GB (`GPU_LARGE_RTX`, TP=1) |
# MAGIC
# MAGIC Qwen3.8-27B-FP8: ~28 GiB weights, vLLM ≥ 0.17.0 required (using 0.19.1).
# MAGIC Architecture: hybrid attention (GatedDeltaNet + Gated Attention) + vision encoder.

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
usage = shutil.disk_usage(LOCAL_TMP)
print(f"CWD: {workdir}  free: {usage.free/1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

CATALOG  = "your_catalog"   # Unity Catalog catalog name
SCHEMA   = "your_schema"  # Unity Catalog schema name

MODEL_REPO_ID     = "Qwen/Qwen3.8-27B-FP8"
ARTIFACTS_PATH    = "qwen38_27b_fp8"
SERVED_MODEL_NAME = "qwen"
DTYPE             = "bfloat16"
SERVING_PORT      = 8080

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qwen38_27b_fp8"

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
        "--gpu-memory-utilization 0.88 --max-num-seqs 128"
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
