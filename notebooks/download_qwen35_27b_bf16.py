# Databricks notebook source
# MAGIC %md
# MAGIC # Download Qwen3.5-27B BF16 → UC Volume
# MAGIC
# MAGIC Downloads `Qwen/Qwen3.5-27B` (~54 GB) from HuggingFace to a Unity Catalog Volume
# MAGIC so the weights persist across sessions and can be reused for BF16 serving validation.
# MAGIC
# MAGIC Edit the `VOLUME_PATH` variable below before running.

# COMMAND ----------

%pip install "hf_transfer==0.1.9" "huggingface_hub>=0.27"
%restart_python

# COMMAND ----------

import os, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ── Edit this path ────────────────────────────────────────────────────────────
VOLUME_PATH = "/Volumes/your_catalog/your_schema/your_volume/qwen35_27b_bf16"
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REPO_ID = "Qwen/Qwen3.5-27B"

from huggingface_hub import snapshot_download
print(f"Downloading {MODEL_REPO_ID} → {VOLUME_PATH}")
t0 = time.time()
snapshot_download(repo_id=MODEL_REPO_ID, local_dir=VOLUME_PATH)
elapsed = time.time() - t0
shards = len([f for f in os.listdir(VOLUME_PATH) if f.endswith(".safetensors")])
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min); shards: {shards}")
dbutils.notebook.exit(f"Downloaded {MODEL_REPO_ID} to {VOLUME_PATH} in {elapsed:.0f}s")
