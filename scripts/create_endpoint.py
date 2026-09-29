#!/usr/bin/env python3
"""Create a Custom LLM Serving (SOD) endpoint for Qwen3.5-27B on GPU_LARGE_RTX.

Usage:
    python create_endpoint.py

Configure the variables in the CONFIGURATION section below before running.
Auth: reads from the Databricks CLI profile set in DATABRICKS_PROFILE
      (defaults to "DEFAULT", i.e. ~/.databrickscfg [DEFAULT]).
"""
import json, os, subprocess, sys, urllib.request, urllib.error

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PROFILE        = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
ENDPOINT_NAME  = "qwen35-27b-fp8-rtx"             # name for the serving endpoint
SERVED_NAME    = "qwen35_27b_fp8_v1"               # served entity name
UC_MODEL_NAME  = "your_catalog.your_schema.qwen35_27b_fp8"  # UC model path
MODEL_VERSION  = "1"                               # model version to deploy
WORKLOAD_TYPE  = "GPU_LARGE_RTX"                   # 1× RTX PRO 6000 (96 GB)
WORKLOAD_SIZE  = "Small"
SCALE_TO_ZERO  = False
# ─────────────────────────────────────────────────────────────────────────────


def _host_token():
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    if PROFILE not in cfg:
        sys.exit(f"Profile '{PROFILE}' not found. Set DATABRICKS_PROFILE or edit ~/.databrickscfg.")
    host = cfg[PROFILE]["host"].rstrip("/")
    result = subprocess.run(["databricks", "auth", "token", "-p", PROFILE],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Token fetch failed. Re-auth: databricks auth login --host {host}\n{result.stderr}")
    tok = json.loads(result.stdout)["access_token"]
    return host, tok


def main():
    host, tok = _host_token()
    config = {
        "name": ENDPOINT_NAME,
        "config": {"served_entities": [{
            "name": SERVED_NAME,
            "entity_name": UC_MODEL_NAME,
            "entity_version": MODEL_VERSION,
            "workload_type": WORKLOAD_TYPE,
            "workload_size": WORKLOAD_SIZE,
            "scale_to_zero_enabled": SCALE_TO_ZERO,
        }]}
    }
    print(f"Creating endpoint '{ENDPOINT_NAME}' at {host}")
    print(f"  model:  {UC_MODEL_NAME} v{MODEL_VERSION}")
    print(f"  GPU:    {WORKLOAD_TYPE} (RTX PRO 6000, 96 GB, TP=1)")
    req = urllib.request.Request(
        f"{host}/api/2.0/serving-endpoints",
        data=json.dumps(config).encode(), method="POST",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        print("CREATE", r.status, r.read().decode()[:300])
        print(f"\nEndpoint URL: {host}/ml/endpoints/{ENDPOINT_NAME}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print("ERROR", e.code, body[:500])
        if "already exists" in body:
            print("Endpoint already exists. Delete it first or update via the UI.")


if __name__ == "__main__":
    main()
