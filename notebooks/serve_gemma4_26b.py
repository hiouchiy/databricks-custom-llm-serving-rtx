# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM Serving — Gemma 4 26B-A4B-it on RTX PRO 6000 (GPU_LARGE_RTX)
# MAGIC
# MAGIC | Stage | Compute |
# MAGIC |---|---|
# MAGIC | This notebook (download / log / register) | Serverless GPU 1×H100 (`GPU_1xH100`) |
# MAGIC | The serving endpoint | 1×RTX PRO 6000 96 GB (`GPU_LARGE_RTX`, TP=1) |
# MAGIC
# MAGIC Model: `google/gemma-4-26B-A4B-it` (26B MoE, 4B active, ~52 GiB BF16)
# MAGIC vLLM: **0.24.0** (minimum for `Gemma4ForConditionalGeneration`)
# MAGIC
# MAGIC **HuggingFace token required.**
# MAGIC Gemma models require acceptance of Google's Community License Agreement.
# MAGIC Set `HF_TOKEN` as a Databricks secret or environment variable before running.

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi || echo "nvidia-smi not available"

# COMMAND ----------

# Gemma4 requires vLLM 0.24.0 (Gemma4ForConditionalGeneration support).
# This overrides the AI Runtime base vLLM; included in extra_pip_requirements too.
%pip install \
  "vllm==0.24.0" \
  "transformers>=5.5.3" \
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
# MAGIC
# MAGIC Set `HF_TOKEN` to your HuggingFace token (required for Gemma downloads).

# COMMAND ----------

import os
# Uncomment and set your HF token, or set via Databricks secrets:
# os.environ["HF_TOKEN"] = dbutils.secrets.get(scope="your_scope", key="hf_token")

CATALOG  = "serverless_stable_lxejgv_catalog"
SCHEMA   = "sgc"

MODEL_REPO_ID     = "google/gemma-4-26B-A4B-it"
ARTIFACTS_PATH    = "gemma4_26b_a4b_it"
SERVED_MODEL_NAME = "gemma4"
DTYPE             = "bfloat16"
SERVING_PORT      = 8080

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.gemma4_26b_a4b_it"

# COMMAND ----------

# MAGIC %md ## Download Gemma 4 26B-A4B-it (~52 GB BF16)

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
        # --model-impl vllm: use vLLM-native Gemma4 implementation
        "--model-impl vllm "
        "--gpu-memory-utilization 0.88 --max-num-seqs 32"
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
    # vllm==0.24.0 needed in serving container for Gemma4ForConditionalGeneration
    extra_pip_requirements=[
        "mlflow==3.12.0",
        "fastapi<0.137.0",
        "vllm==0.24.0",
        "transformers>=5.5.3",
    ],
)
print("model_uri:", model_info.model_uri)

# COMMAND ----------

model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME,
                                       env_pack="databricks_model_serving")
dbutils.notebook.exit(f"REGISTERED {UC_MODEL_NAME} v{model_version.version}")
