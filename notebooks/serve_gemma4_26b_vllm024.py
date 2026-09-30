# Databricks notebook source
# MAGIC %md
# MAGIC # Custom LLM Serving — Gemma 4 26B-A4B-it on RTX PRO 6000 (GPU_LARGE_RTX) [vLLM 0.24.0]
# MAGIC
# MAGIC | Stage | Compute |
# MAGIC |---|---|
# MAGIC | This notebook (download / log / register) | Serverless GPU 1×H100 (`GPU_1xH100`) |
# MAGIC | The serving endpoint | 1×RTX PRO 6000 96 GB (`GPU_LARGE_RTX`, TP=1) |
# MAGIC
# MAGIC Model: `google/gemma-4-26B-A4B-it` (26B MoE, 4B active, ~52 GiB BF16)
# MAGIC Serving container: **vLLM 0.24.0** via `extra_pip_requirements`
# MAGIC
# MAGIC ## vLLM 0.24.0 を使う理由
# MAGIC vLLM 0.19.1 (AI Runtime base) は Gemma4 をサポートしているが、
# MAGIC 0.24.0 はより安定したサポートが報告されている。
# MAGIC
# MAGIC ## 重要な依存関係
# MAGIC Serving-container pins (`extra_pip_requirements`):
# MAGIC - `vllm==0.24.0`
# MAGIC - `mlflow==3.14.0` — `mlflow==3.12.0` conflicts with `vllm==0.24.0` (starlette).
# MAGIC - `transformers==5.13.0` — must be pinned. Newer transformers treats Gemma4's `head_dim`
# MAGIC   as a per-layer attribute, and vLLM 0.24.0 fails at startup with
# MAGIC   `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`.
# MAGIC - `fastapi<0.137.0` — health-check fix (same as the Qwen notebooks).
# MAGIC
# MAGIC The notebook itself keeps the AI Runtime base vLLM (the pip constraints file pins it);
# MAGIC it only downloads, logs, and registers the model.

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi || echo "nvidia-smi not available"

# COMMAND ----------

# Do not install vllm here: the AI Runtime pip constraints file pins it and an upgrade fails.
# The serving container gets vllm==0.24.0 via extra_pip_requirements below.
%pip install \
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
# MAGIC
# MAGIC Set `HF_TOKEN` to your HuggingFace token (required for Gemma downloads).

# COMMAND ----------

import os
# Uncomment and set your HF token, or set via Databricks secrets:
# os.environ["HF_TOKEN"] = dbutils.secrets.get(scope="your_scope", key="hf_token")

CATALOG  = "your_catalog"   # Unity Catalog catalog name
SCHEMA   = "your_schema"  # Unity Catalog schema name

MODEL_REPO_ID     = "google/gemma-4-26B-A4B-it"
ARTIFACTS_PATH    = "gemma4_26b_a4b_it"
SERVED_MODEL_NAME = "gemma4"
DTYPE             = "bfloat16"
SERVING_PORT      = 8080

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.gemma4_26b_a4b_it_v024"  # separate from vLLM 0.19.1 version

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
    # See the header cell for why each pin is required.
    extra_pip_requirements=[
        "mlflow==3.14.0",
        "fastapi<0.137.0",
        "vllm==0.24.0",
        "transformers==5.13.0",
    ],
)
print("model_uri:", model_info.model_uri)

# COMMAND ----------

# Verify the serving requirements before registering: exactly one transformers pin, ==5.13.0.
reqs = open(mlflow.pyfunc.get_model_dependencies(model_info.model_uri)).read()
print(reqs)
tf_lines = [l for l in reqs.splitlines() if l.strip().lower().startswith("transformers")]
assert tf_lines == ["transformers==5.13.0"], f"unexpected transformers pins: {tf_lines}"

# COMMAND ----------

model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME,
                                       env_pack="databricks_model_serving")
dbutils.notebook.exit(f"REGISTERED {UC_MODEL_NAME} v{model_version.version}")
