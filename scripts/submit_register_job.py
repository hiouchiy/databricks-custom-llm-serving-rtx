#!/usr/bin/env python3
"""Upload a registration notebook to Databricks and submit it as a Serverless GPU job.

The registration step (mlflow.register_model with env_pack="databricks_model_serving")
MUST run on a Serverless GPU job — a classic cluster returns:
    ValueError: Serverless environment is required ... is_client_image=False

Usage:
    python scripts/submit_register_job.py --notebook notebooks/serve_qwen35_27b_fp8.py [--upload]

    --upload  Upload the notebook to the workspace before submitting.

Auth: uses the Databricks CLI profile set in DATABRICKS_PROFILE env var (default: DEFAULT).
"""
import argparse, base64, json, os, pathlib, subprocess, sys, urllib.request, urllib.error

PROFILE    = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
JOB_NAME   = "custom-llm-register"


def _host_token():
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    if PROFILE not in cfg:
        sys.exit(f"Profile '{PROFILE}' not found. "
                 f"Set DATABRICKS_PROFILE or run: databricks auth login --host <workspace-url>")
    host = cfg[PROFILE]["host"].rstrip("/")
    result = subprocess.run(["databricks", "auth", "token", "-p", PROFILE],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Token fetch failed.\nRe-auth: databricks auth login --host {host}\n{result.stderr}")
    tok = json.loads(result.stdout)["access_token"]
    return host, tok


def _current_user(host, tok):
    req = urllib.request.Request(f"{host}/api/2.0/preview/scim/v2/Me",
                                  headers={"Authorization": f"Bearer {tok}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["userName"]


def _upload_notebook(host, tok, local_path, user):
    ws_path = f"/Users/{user}/custom-llm-serving/{pathlib.Path(local_path).stem}"
    # Ensure parent directory exists
    mkdirs = urllib.request.Request(
        f"{host}/api/2.0/workspace/mkdirs",
        data=json.dumps({"path": f"/Users/{user}/custom-llm-serving"}).encode(),
        method="POST", headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    urllib.request.urlopen(mkdirs, timeout=30)
    # Upload
    content = pathlib.Path(local_path).read_bytes()
    payload = {"path": ws_path, "format": "SOURCE", "language": "PYTHON",
               "content": base64.b64encode(content).decode(), "overwrite": True}
    req = urllib.request.Request(
        f"{host}/api/2.0/workspace/import",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60)
    print(f"Uploaded → {ws_path}")
    return ws_path


def _submit(host, tok, notebook_path):
    job_spec = {
        "run_name": JOB_NAME,
        "environments": [{"environment_key": "ai-runtime", "spec": {"client": "4"}}],
        "tasks": [{
            "task_key": "register",
            "notebook_task": {"notebook_path": notebook_path},
            "environment_key": "ai-runtime",
            "compute": {"hardware_accelerator": "GPU_1xH100"},
        }],
    }
    req = urllib.request.Request(
        f"{host}/api/2.0/jobs/runs/submit",
        data=json.dumps(job_spec).encode(), method="POST",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    run_id = r["run_id"]
    print(f"Job submitted — run_id: {run_id}")
    print(f"Monitor: {host}/#job/runs/{run_id}")
    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True, help="Local notebook path to submit")
    ap.add_argument("--upload", action="store_true", help="Upload notebook before submitting")
    args = ap.parse_args()

    host, tok = _host_token()
    user = _current_user(host, tok)
    print(f"Workspace: {host}  (user: {user})")

    if args.upload:
        nb_path = _upload_notebook(host, tok, args.notebook, user)
    else:
        nb_path = f"/Users/{user}/custom-llm-serving/{pathlib.Path(args.notebook).stem}"
        print(f"Using workspace path: {nb_path}")

    _submit(host, tok, nb_path)


if __name__ == "__main__":
    main()
