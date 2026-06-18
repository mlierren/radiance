#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["runpod", "python-dotenv"]
# ///
"""RunPod Launcher for Radiance — Bible Worlds Unified 3DGS Asset PoC.

Spins up a GPU pod for the Relightable-3D-Gaussians + Gaussian-Grouping stack.
The pod bootstrap design:

  - A Network Volume is mounted at /workspace and holds EVERYTHING persistent:
    the uv venv (/workspace/envs/r3dg, with compiled CUDA extensions) AND the
    uv-managed Python interpreter it points at (/workspace/uv/python), the repo
    clones (/workspace/radiance + third_party), data and outputs.
    => Re-creating a pod on the same volume resumes in minutes (no recompile).
       NOTE: a venv only survives pod re-creation if its base interpreter also
       lives on the volume — hence the uv-managed Python is installed there too.
  - SSH is enabled so the local Emacs + emacs-jupyter workflow can tunnel to
    the remote Jupyter server:  ssh -L 8888:localhost:8888 <pod>
  - A JupyterLab server is started so org-babel (`:kernel r3dg`) can drive the
    remote kernel via emacs-jupyter. The r3dg kernel is registered from the
    uv venv during bootstrap.

GPU recommendation:
    NVIDIA GeForce RTX 3090 (Ampere sm_86, 24GB) — the repo's reference GPU.
    The stack pins PyTorch 1.12.1 + cu116; PyTorch 1.12 ships kernels for
    Ampere (sm_86) but NOT for Ada (sm_89: RTX 4090/L40) or Hopper (sm_90: H100),
    so stay on Ampere. Upgrade to RTX A6000 (same arch, 48GB) only if you hit
    VRAM limits during BVH ray tracing / compose.

The launcher itself runs via uv (PEP 723 inline deps above) — no manual install:
    uv run scripts/runpod_launcher.py --test-api

Usage:
    # Test API connection
    uv run scripts/runpod_launcher.py --test-api

    # One-time: create a network volume (sized for env + data + checkpoints)
    uv run scripts/runpod_launcher.py --create-volume \\
        --volume-name radiance-vol --volume-size 100 --datacenter EU-RO-1

    # List volumes to get the volume id
    uv run scripts/runpod_launcher.py --list-volumes

    # Create a pod on that volume (RTX 3090). First boot builds the env
    # (slow, one-time); later boots on the same volume are fast.
    uv run scripts/runpod_launcher.py --create --gpu "NVIDIA GeForce RTX 3090" \\
        --volume-id <vol_id> --repo-url git@github.com:you/radiance.git

    # Status (prints SSH tunnel + Jupyter URL) / stop / delete
    uv run scripts/runpod_launcher.py --status
    uv run scripts/runpod_launcher.py --stop
    uv run scripts/runpod_launcher.py --delete

Requires:
    uv (https://docs.astral.sh/uv/) on the local machine. Dependencies are
    declared inline (PEP 723) and fetched automatically by `uv run`.
    Create .env.runpod.local with:
        RUNPOD_API_KEY=your_runpod_api_key
        # optional:
        # RADIANCE_REPO_URL=git@github.com:you/radiance.git
"""

import argparse
import base64
import json
import os
import sys
import time
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' not found. pip install python-dotenv")
    sys.exit(1)

load_dotenv(".env.runpod.local")

try:
    import runpod
except ImportError:
    print("Error: 'runpod' not found. pip install runpod")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

BASE_POD_NAME = "radiance"

# CUDA 11.8 devel + Ubuntu 22.04. Relightable3DGaussian
# recommends compiling against CUDA 11.8 (the stack pins PyTorch 1.12.1+cu116).
# A *devel* image is required so nvcc is present for building
# r3dg-rasterization / bvh / simple-knn / nvdiffrast.
IMAGE_NAME_GPU = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# Ampere sm_86 — matches the stack's PyTorch 1.12.1+cu116 (see module docstring).
# This is the repo's reference GPU (24GB). A6000 (same arch, 48GB) also works.
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 3090"

# Network volume mounts at /workspace. All persistent state
# (venv, managed python, repos, data, outputs) lives there; container disk stays small.
VOLUME_MOUNT_PATH = "/workspace"
CONTAINER_DISK_SIZE = 40            # GB — ephemeral OS layer only
DEFAULT_FALLBACK_VOLUME_SIZE = 60   # GB — pod-local volume when NO network volume

DEFAULT_NETWORK_VOLUME_NAME = "radiance-vol"
# uv venv w/ torch + compiled CUDA ext (~10GB) + frames/COLMAP + checkpoints/outputs.
DEFAULT_NETWORK_VOLUME_SIZE = 100   # GB

# Where the canonical repo + external repos live on the volume.
WORKSPACE_REPO_DIR = "/workspace/radiance"
ENV_PREFIX = "/workspace/envs/r3dg"      # uv venv
PYTHON_VERSION = "3.10"                   # uv-managed; cu116 wheels exist for cp310
UV_HOME = "/workspace/uv"                 # uv-managed python + cache, on the volume

# External repos to clone on the pod.
THIRD_PARTY_REPOS = [
    # (dir_name, git_url, clone_recursive)
    ("Relightable3DGaussian", "https://github.com/NJU-3DV/Relightable3DGaussian.git", True),
    ("gaussian-grouping", "https://github.com/lkeab/gaussian-grouping.git", False),
    ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", False),
]

JUPYTER_TOKEN = "radiance_token"
JUPYTER_PORT = 8888

# `ssh <alias>` after --create/--start/--write-ssh-config (managed block in
# ~/.ssh/config). The pod's public IP+port change on every stop/start, so we
# rewrite the block each time. A dedicated known_hosts avoids host-key churn.
DEFAULT_SSH_ALIAS = "radiance-pod"
SSH_KNOWN_HOSTS = "~/.ssh/known_hosts.radiance"
SSH_CONFIG_PATH = "~/.ssh/config"

# RunPod datacenters with available storage clusters (as of 2026-05).
# RunPod returns the authoritative list in error messages if this drifts.
DATACENTERS = {
    dc: dc for dc in [
        "AP-JP-1",
        "CA-MTL-3", "CA-MTL-4",
        "EU-CZ-1", "EU-FR-1", "EU-NL-1", "EU-RO-1",
        "EUR-IS-3", "EUR-NO-1",
        "US-CA-2", "US-IL-1", "US-KS-2",
        "US-MO-1", "US-MO-2", "US-NC-2", "US-NE-1",
        "US-TX-3", "US-WA-1",
    ]
}


# ============================================================================
# Helpers
# ============================================================================

def get_ssh_public_key() -> Optional[str]:
    """Read a local SSH public key for injection into the pod via env var.

    Order: $SSH_PUBLIC_KEY env, then ~/.ssh/id_ed25519.pub, then ~/.ssh/id_rsa.pub.
    """
    explicit = os.environ.get("SSH_PUBLIC_KEY")
    if explicit:
        return explicit.strip()
    home = os.path.expanduser("~")
    for candidate in ("id_ed25519.pub", "id_rsa.pub"):
        path = os.path.join(home, ".ssh", candidate)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None


def get_pod_name(custom_name: Optional[str] = None) -> str:
    if custom_name:
        return custom_name
    # $USER is more reliable than os.getlogin() under sudo / non-tty shells
    username = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not username:
        try:
            username = os.getlogin()
        except OSError:
            username = "user"
    return f"{BASE_POD_NAME}-{username}"


def get_api_key(api_key_arg: Optional[str] = None) -> str:
    if api_key_arg:
        return api_key_arg
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY missing. Add it to .env.runpod.local")
        sys.exit(1)
    return api_key


def test_api_connection(api_key: str) -> bool:
    runpod.api_key = api_key
    try:
        pods = runpod.get_pods()
        print(f"RunPod API OK. {len(pods)} active pod(s).")
        return True
    except Exception as e:
        print(f"RunPod API failed: {e}")
        return False


def find_pod_by_name(name: str, api_key: str) -> Optional[Dict[str, Any]]:
    runpod.api_key = api_key
    for pod in runpod.get_pods():
        if pod.get("name") == name:
            return pod
    return None


# ============================================================================
# Network Volume Management
# ============================================================================

def list_network_volumes(api_key: str):
    runpod.api_key = api_key
    from runpod.api.graphql import run_graphql_query
    query = """
    query {
        myself {
            networkVolumes {
                id name size dataCenterId
            }
        }
    }
    """
    try:
        result = run_graphql_query(query)
        volumes = result.get("data", {}).get("myself", {}).get("networkVolumes", [])
        if not volumes:
            print("No network volumes found.")
            print(f"Create one with: --create-volume --volume-name {DEFAULT_NETWORK_VOLUME_NAME}")
            return []
        print(f"\nNetwork Volumes ({len(volumes)}):\n")
        print(f"{'ID':<22} {'Name':<26} {'Size':<8} {'DC':<12}")
        print("-" * 70)
        for v in volumes:
            print(f"{v['id']:<22} {v['name']:<26} {v['size']:<8} {v['dataCenterId']:<12}")
        return volumes
    except Exception as e:
        print(f"Failed to list volumes: {e}")
        return []


def create_network_volume(api_key: str, name: str, size: int, datacenter_id: str):
    runpod.api_key = api_key
    from runpod.api.graphql import run_graphql_query
    if datacenter_id not in DATACENTERS:
        print(f"Invalid datacenter. Choices: {', '.join(DATACENTERS.keys())}")
        return None
    print(f"Creating volume '{name}' ({size}GB) in {datacenter_id}...")
    # Inline literals (run_graphql_query doesn't accept variables in 1.9.0)
    name_lit = json.dumps(name)
    dc_lit = json.dumps(datacenter_id)
    mutation = f"""
    mutation {{
        createNetworkVolume(input: {{
            name: {name_lit}, size: {int(size)}, dataCenterId: {dc_lit}
        }}) {{
            id name size dataCenterId
        }}
    }}
    """
    try:
        result = run_graphql_query(mutation)
        vol = result.get("data", {}).get("createNetworkVolume")
        if vol:
            print(f"Created: {vol['id']} ({vol['name']}, {vol['size']}GB, {vol['dataCenterId']})")
            print(f"Use: --create --gpu '{DEFAULT_GPU_TYPE}' --volume-id {vol['id']}")
            print(f"Note: pod MUST be created in {datacenter_id}")
            return vol
        print(f"Unexpected response: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    return None


def delete_network_volume(api_key: str, volume_id: str) -> bool:
    runpod.api_key = api_key
    from runpod.api.graphql import run_graphql_query
    print(f"Deleting volume {volume_id}...")
    id_lit = json.dumps(volume_id)
    mutation = f"""
    mutation {{
        deleteNetworkVolume(input: {{ id: {id_lit} }})
    }}
    """
    try:
        run_graphql_query(mutation)
        print("Deleted.")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def get_volume_datacenter(api_key: str, volume_id: str) -> Optional[str]:
    runpod.api_key = api_key
    from runpod.api.graphql import run_graphql_query
    query = "query { myself { networkVolumes { id dataCenterId } } }"
    try:
        result = run_graphql_query(query)
        for v in result.get("data", {}).get("myself", {}).get("networkVolumes", []):
            if v["id"] == volume_id:
                return v["dataCenterId"]
    except Exception as e:
        print(f"Failed: {e}")
    return None


# ============================================================================
# Startup Script Builder
# ============================================================================

def _build_startup_script(
    no_jupyter: bool = False,
    run_command: Optional[str] = None,
    repo_url: Optional[str] = None,
    bootstrap_env: bool = True,
) -> str:
    """Build the pod startup script (Pod bootstrap).

    The script is idempotent and volume-aware: every expensive artifact (uv
    venv, compiled CUDA extensions, repo clones) lands on /workspace, so a pod
    re-created on the same network volume skips the slow steps.
    """

    # --- Clone the canonical repo (optional; needs a URL) ---
    repo_clone = ""
    if repo_url:
        repo_url_lit = json.dumps(repo_url)
        repo_clone = f"""
# --- Canonical repo ---
if [ -d "{WORKSPACE_REPO_DIR}/.git" ]; then
    echo "[REPO] {WORKSPACE_REPO_DIR} present; pulling..."
    git -C "{WORKSPACE_REPO_DIR}" pull --ff-only || echo "[REPO] pull skipped"
elif [ -d "{WORKSPACE_REPO_DIR}" ] && [ -n "$(ls -A "{WORKSPACE_REPO_DIR}" 2>/dev/null)" ]; then
    # dir exists & non-empty but not a git repo (e.g. only third_party/): adopt it
    # in place so the cloned src/ lands beside the already-built env/third_party.
    echo "[REPO] Adopting existing {WORKSPACE_REPO_DIR} as a git repo..."
    cd "{WORKSPACE_REPO_DIR}"
    git init -q
    git remote add origin {repo_url_lit} 2>/dev/null || git remote set-url origin {repo_url_lit}
    git fetch -q origin && git remote set-head origin -a 2>/dev/null
    DEF=$(git symbolic-ref -q --short refs/remotes/origin/HEAD | sed 's@^origin/@@')
    DEF=${{DEF:-main}}
    git reset -q --hard "origin/$DEF" && git checkout -q -B "$DEF" "origin/$DEF" \\
        || echo "[REPO] adopt failed (check repo is public / URL)"
else
    echo "[REPO] Cloning canonical repo..."
    git clone {repo_url_lit} "{WORKSPACE_REPO_DIR}" || echo "[REPO] clone failed (check URL is public)"
fi
"""
    else:
        repo_clone = f"""
# --- Canonical repo: no --repo-url given; create a bare working dir ---
mkdir -p "{WORKSPACE_REPO_DIR}"
echo "[REPO] No repo URL provided. Push your local radiance repo and re-launch"
echo "[REPO] with --repo-url, or 'git clone' into {WORKSPACE_REPO_DIR} manually."
"""

    # --- Clone external repos ---
    third_party_lines = [f'mkdir -p "{WORKSPACE_REPO_DIR}/third_party"']
    for dir_name, url, recursive in THIRD_PARTY_REPOS:
        dest = f"{WORKSPACE_REPO_DIR}/third_party/{dir_name}"
        rec = " --recursive" if recursive else ""
        third_party_lines.append(f"""
if [ -d "{dest}/.git" ]; then
    echo "[3P] {dir_name} present."
else
    echo "[3P] Cloning {dir_name}..."
    git clone{rec} {url} "{dest}" || echo "[3P] {dir_name} clone failed"
fi""")
    third_party_clone = "\n".join(third_party_lines)

    # --- uv venv + CUDA extensions + r3dg kernel ---
    env_bootstrap = ""
    if bootstrap_env:
        r3dg_dir = f"{WORKSPACE_REPO_DIR}/third_party/Relightable3DGaussian"
        env_bootstrap = f"""
# --- uv venv 'r3dg' on the volume (uv replaces conda) ---
# Both the venv AND its base interpreter must live on the volume to survive
# pod re-creation, so uv's managed Python + cache are pinned under {UV_HOME}.
export UV_PYTHON_INSTALL_DIR="{UV_HOME}/python"
export UV_CACHE_DIR="{UV_HOME}/cache"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:/workspace/bin:$PATH"
# Pin build targets to OUR GPUs (A100 sm_80, 3090/A6000/A5000/A4500 sm_86).
# Without this, torch_scatter + the CUDA submodules compile for EVERY arch torch
# supports (sm_35..sm_86) — minutes vs tens of minutes per extension.
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

if [ ! -x /workspace/bin/uv ]; then
    echo "[ENV] Installing uv to /workspace/bin (persists on volume)..."
    curl -LsSf https://astral.sh/uv/install.sh | \\
        env UV_INSTALL_DIR=/workspace/bin INSTALLER_NO_MODIFY_PATH=1 sh || echo "[ENV] uv install issue"
fi
UV=/workspace/bin/uv

if [ -x "{ENV_PREFIX}/bin/python" ]; then
    echo "[ENV] Reusing existing venv at {ENV_PREFIX} (compiled extensions preserved)."
else
    echo "[ENV] Building r3dg venv (SLOW, one-time; compiles CUDA extensions)..."
    "$UV" python install {PYTHON_VERSION} || echo "[ENV] uv python install issue"
    "$UV" venv --python {PYTHON_VERSION} --python-preference only-managed "{ENV_PREFIX}" \\
        || echo "[ENV] venv create issue"
    VPY="{ENV_PREFIX}/bin/python"

    # Build backend deps. PIN setuptools<80: torch 1.12's cpp_extension and the
    # CUDA submodules import pkg_resources, which setuptools>=81 removed — newer
    # setuptools breaks every extension build with "No module named pkg_resources".
    "$UV" pip install --python "$VPY" "setuptools==69.5.1" wheel ninja || echo "[ENV] build-deps issue"

    # Pin numpy<2 FIRST: torch 1.12.1 was built against the numpy 1.x C-ABI, so
    # numpy 2.x triggers "compiled against NumPy 1.x cannot be run in NumPy 2.x"
    # at runtime. Installing it before torch keeps uv from pulling numpy 2.x.
    "$UV" pip install --python "$VPY" "numpy<2" || echo "[ENV] numpy pin issue"

    # Pinned stack. Ampere-compatible PyTorch 1.12.1+cu116.
    "$UV" pip install --python "$VPY" \\
        torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \\
        --index-url https://download.pytorch.org/whl/cu116 || echo "[ENV] torch install issue"
    # torch_scatter matched to torch-1.12.1+cu116. --no-build-isolation so the
    # sdist build (when no matching wheel) can see the torch we just installed.
    "$UV" pip install --python "$VPY" --no-build-isolation torch_scatter==2.1.1 \\
        -f https://data.pyg.org/whl/torch-1.12.1+cu116.html || echo "[ENV] torch_scatter issue"
    "$UV" pip install --python "$VPY" kornia==0.6.12 || echo "[ENV] kornia issue"

    # nvdiffrast (cloned source) + the three CUDA submodules.
    # --no-build-isolation so the build sees the torch we just installed.
    cd "{r3dg_dir}"
    "$UV" pip install --python "$VPY" --no-build-isolation \\
        "{WORKSPACE_REPO_DIR}/third_party/nvdiffrast" || echo "[ENV] nvdiffrast build failed"
    "$UV" pip install --python "$VPY" --no-build-isolation ./submodules/simple-knn || echo "[ENV] simple-knn build failed"
    "$UV" pip install --python "$VPY" --no-build-isolation ./bvh                   || echo "[ENV] bvh build failed"
    "$UV" pip install --python "$VPY" --no-build-isolation ./r3dg-rasterization    || echo "[ENV] r3dg-rasterization build failed"

    # Runtime / dataset-prep / viz deps used by the FINAL pipeline (frame extract,
    # masks, analytic poses, training, Z-rescale, turntable render) and the
    # org-babel workflow. Pillow PINNED <10: torch 1.12's tensorboard uses
    # Image.ANTIALIAS, removed in Pillow 10 (--save_training_vis crashes otherwise).
    "$UV" pip install --python "$VPY" \\
        "pillow==9.5.0" tqdm plyfile scipy opencv-python-headless imageio \\
        matplotlib ipywidgets tensorboard pyexr dearpygui \\
        || echo "[ENV] runtime deps issue"
fi

# COLMAP + ffmpeg are native apps (not pip): install via apt.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq colmap ffmpeg || echo "[ENV] colmap/ffmpeg apt issue"

# --- Register the r3dg Jupyter kernel for emacs-jupyter ---
VPY="{ENV_PREFIX}/bin/python"
"$UV" pip install --python "$VPY" ipykernel jupyterlab || echo "[ENV] jupyter install issue"
"$VPY" -m ipykernel install --user --name r3dg --display-name "r3dg" || echo "[ENV] kernel register issue"

# Sanity: CUDA + the compiled extensions.
"$VPY" -c "
import torch
print('[ENV] CUDA:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
for mod in ('simple_knn','r3dg_rasterization','bvh','nvdiffrast'):
    try:
        __import__(mod); print('[ENV]', mod, 'OK')
    except Exception as e:
        print('[ENV]', mod, 'FAIL', type(e).__name__)
" || echo "[ENV] sanity check failed"
"""
    else:
        env_bootstrap = """
echo "[ENV] --skip-bootstrap set. Build the uv venv manually per the README."
"""

    # --- Jupyter launch ---
    jupyter_launch = ""
    if not no_jupyter:
        # If the r3dg env exists, serve from it (so the r3dg kernel is native);
        # otherwise fall back to the base python so connectivity still works.
        jupyter_launch = f"""
echo "[JL] Starting JupyterLab on :{JUPYTER_PORT} (emacs-jupyter target)..."
if [ -x "{ENV_PREFIX}/bin/jupyter" ]; then
    JUP="{ENV_PREFIX}/bin/jupyter"
    echo "[JL] Serving from r3dg env."
else
    pip install -q jupyterlab >/dev/null 2>&1 || true
    JUP="jupyter"
    echo "[JL] r3dg env not ready; serving from base python."
fi
cd /workspace
"$JUP" lab --ip=0.0.0.0 --port={JUPYTER_PORT} --allow-root --no-browser \\
    --ServerApp.token='{JUPYTER_TOKEN}' --ServerApp.password='' \\
    --ServerApp.allow_remote_access=True --ServerApp.disable_check_xsrf=True \\
    --ServerApp.allow_origin='*' --ServerApp.base_url='/'
"""
    else:
        jupyter_launch = "echo '[JL] Jupyter disabled. Tailing to keep container alive.'\ntail -f /dev/null\n"

    custom = ""
    if run_command:
        custom = f"""
# --- Custom command ---
cd "{WORKSPACE_REPO_DIR}"
echo "[RUN] {run_command}"
{run_command}
echo "[RUN] exit code: $?"
"""

    script = f"""#!/bin/bash
exec > >(tee /workspace/startup.log) 2>&1

echo "=== Radiance Pod Startup ==="
date

# --- Workspace layout ---
mkdir -p /workspace/data /workspace/outputs /workspace/envs

# --- System deps (build toolchain for CUDA extensions) ---
echo "[SYS] Installing apt packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \\
    git wget curl ffmpeg libgl1 libglib2.0-0 \\
    build-essential cmake ninja-build \\
    openssh-server

# --- SSH (for `ssh -L {JUPYTER_PORT}:localhost:{JUPYTER_PORT} <pod>` tunnel) ---
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ -n "$SSH_PUBLIC_KEY" ]; then
    echo "$SSH_PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi
ssh-keygen -A
service ssh start
{repo_clone}
{third_party_clone}
{env_bootstrap}
{custom}
echo "[OK] Radiance bootstrap done at $(date)"
{jupyter_launch}"""

    return script


# ============================================================================
# Repo URL resolution (local repo -> cloned on the pod)
# ============================================================================

def _normalize_github_https(url: str) -> str:
    """Convert an SSH GitHub remote to its HTTPS form so the pod can clone a
    PUBLIC repo without credentials (and `git pull` later, also no auth)."""
    url = url.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url[len("ssh://git@github.com/"):]
    return url


def resolve_repo_url(explicit: Optional[str] = None) -> Optional[str]:
    """Pick the repo to clone on the pod: explicit --repo-url, then
    RADIANCE_REPO_URL, then the local `git remote get-url origin` (auto).
    Always normalized to HTTPS for credential-free public clone/pull."""
    import subprocess
    candidate = explicit or os.environ.get("RADIANCE_REPO_URL")
    if not candidate:
        try:
            candidate = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, check=False,
            ).stdout.strip() or None
        except Exception:
            candidate = None
    return _normalize_github_https(candidate) if candidate else None


# ============================================================================
# Pod Management
# ============================================================================

def create_pod(
    api_key: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    pod_name: Optional[str] = None,
    no_jupyter: bool = False,
    run_command: Optional[str] = None,
    repo_url: Optional[str] = None,
    volume_id: Optional[str] = None,
    container_disk_gb: Optional[int] = None,
    pod_volume_gb: Optional[int] = None,
    skip_bootstrap: bool = False,
    cloud_type: str = "SECURE",
):
    runpod.api_key = api_key
    target_pod_name = get_pod_name(pod_name)

    if find_pod_by_name(target_pod_name, api_key):
        print(f"Pod '{target_pod_name}' already exists. Use --status or --delete.")
        return

    volume_datacenter = None
    if volume_id:
        volume_datacenter = get_volume_datacenter(api_key, volume_id)
        if not volume_datacenter:
            print(f"Volume {volume_id} not found.")
            return
        print(f"Volume {volume_id} in datacenter {volume_datacenter}")
    else:
        print("WARNING: no --volume-id. Without a network volume the uv venv "
              "and repos are LOST on pod deletion.")

    print(f"Creating pod '{target_pod_name}' with GPU {gpu_type}...")
    print(f"  Image: {IMAGE_NAME_GPU}")

    resolved_repo = resolve_repo_url(repo_url)
    if resolved_repo:
        print(f"  Repo (cloned to {WORKSPACE_REPO_DIR}): {resolved_repo}")
    else:
        print("  Repo: none detected (set --repo-url / RADIANCE_REPO_URL, or add a git "
              "remote). Pod gets an empty /workspace/radiance.")

    startup = _build_startup_script(
        no_jupyter=no_jupyter,
        run_command=run_command,
        repo_url=resolved_repo,
        bootstrap_env=not skip_bootstrap,
    )
    startup_b64 = base64.b64encode(startup.encode("utf-8")).decode("utf-8")
    docker_cmd = "bash -c 'echo $STARTUP_SCRIPT | base64 -d | bash'"

    env_vars = {"STARTUP_SCRIPT": startup_b64}
    ssh_key = get_ssh_public_key()
    if ssh_key:
        env_vars["SSH_PUBLIC_KEY"] = ssh_key
        fp = ssh_key.split()[1][-12:] if len(ssh_key.split()) > 1 else "?"
        print(f"  SSH key injected (...{fp})")
    else:
        print("  No local SSH public key found; SSH will require RunPod console key.")

    # 8888=Jupyter (org-babel), 22=SSH tunnel.
    ports = f"22/tcp" if no_jupyter else f"{JUPYTER_PORT}/http,22/tcp"

    pod_kwargs: Dict[str, Any] = {
        "name": target_pod_name,
        "image_name": IMAGE_NAME_GPU,
        "container_disk_in_gb": container_disk_gb or CONTAINER_DISK_SIZE,
        "env": env_vars,
        "docker_args": docker_cmd,
        "ports": ports,
        "gpu_type_id": gpu_type,
        "cloud_type": cloud_type,
    }

    if volume_id:
        # Network volume mounts at /workspace.
        pod_kwargs["network_volume_id"] = volume_id
        pod_kwargs["volume_mount_path"] = VOLUME_MOUNT_PATH
        pod_kwargs["volume_in_gb"] = 0
        if volume_datacenter:
            pod_kwargs["data_center_id"] = volume_datacenter
    else:
        # No network volume: give the pod a local volume at /workspace anyway,
        # so the bootstrap paths still resolve (state is ephemeral).
        pod_kwargs["volume_mount_path"] = VOLUME_MOUNT_PATH
        pod_kwargs["volume_in_gb"] = (
            pod_volume_gb if pod_volume_gb is not None else DEFAULT_FALLBACK_VOLUME_SIZE
        )

    try:
        pod = runpod.create_pod(**pod_kwargs)
        print(f"Pod created. ID: {pod['id']}. Waiting for RUNNING state...")
        for _ in range(60):
            p = runpod.get_pod(pod["id"])
            if p and p.get("desiredStatus") == "RUNNING" and (p.get("runtime") or {}).get("ports"):
                print("\nPod is RUNNING!")
                show_pod_status(api_key, pod["id"], pod_name=target_pod_name)
                write_ssh_config(api_key, pod_name=target_pod_name)
                return
            time.sleep(5)
            print(".", end="", flush=True)
        print("\nPod created but slow to start. Check the RunPod console.")
    except Exception as e:
        print(f"Failed to create pod: {e}")


def stop_pod(api_key: str, pod_name: Optional[str] = None):
    runpod.api_key = api_key
    target = get_pod_name(pod_name)
    pod = find_pod_by_name(target, api_key)
    if not pod:
        print(f"Pod '{target}' not found.")
        return
    print(f"Stopping {pod['id']} ({target})...")
    try:
        runpod.stop_pod(pod["id"])
        print("Stopped.")
    except Exception as e:
        print(f"Failed: {e}")


def delete_pod(api_key: str, pod_name: Optional[str] = None):
    runpod.api_key = api_key
    target = get_pod_name(pod_name)
    pod = find_pod_by_name(target, api_key)
    if not pod:
        print(f"Pod '{target}' not found.")
        return
    print(f"Terminating {pod['id']} ({target})...")
    try:
        runpod.terminate_pod(pod["id"])
        print("Terminated.")
    except Exception as e:
        print(f"Failed: {e}")


def _parse_runtime_endpoints(pod_id, runtime):
    """Return (ssh_ip, ssh_port, jupyter_url) from a pod's runtime.

    For exposed-TCP pods the public IP lives on the port object (port["ip"] +
    isIpPublic), NOT on runtime["publicIp"] (which is often null).
    """
    ssh_ip = ssh_port = jupyter_url = None
    for port in (runtime or {}).get("ports", []) or []:
        priv = port.get("privatePort")
        if priv == JUPYTER_PORT:
            if port.get("isIpPublic"):
                jupyter_url = f"http://{port['ip']}:{port['publicPort']}"
            else:
                jupyter_url = f"https://{pod_id}-{JUPYTER_PORT}.proxy.runpod.net"
        elif priv == 22 and port.get("type") == "tcp" and port.get("isIpPublic"):
            ssh_ip = port["ip"]
            ssh_port = port["publicPort"]
    return ssh_ip, ssh_port, jupyter_url


def show_pod_status(api_key: str, pod_id: Optional[str] = None, pod_name: Optional[str] = None):
    runpod.api_key = api_key
    if not pod_id:
        target = get_pod_name(pod_name)
        pod = find_pod_by_name(target, api_key)
        if not pod:
            print(f"Pod '{target}' not found.")
            return
        pod_id = pod["id"]
    else:
        pod = runpod.get_pod(pod_id)

    print(f"\nPod: {pod.get('name', pod_id)}")
    print(f"  ID: {pod['id']}")
    print(f"  Status: {pod['desiredStatus']}")
    print(f"  Image: {pod['imageName']}")
    print(f"  GPU: {pod.get('machine', {}).get('gpuDisplayName', 'Unknown')}")

    runtime = pod.get("runtime")
    if runtime:
        ssh_ip, ssh_port, jupyter_url = _parse_runtime_endpoints(pod_id, runtime)
        print("\nAccess:")
        if ssh_ip and ssh_port:
            print(f"  SSH:           ssh root@{ssh_ip} -p {ssh_port}")
            print("\n  emacs-jupyter workflow:")
            print(f"    1) Tunnel:   ssh -L {JUPYTER_PORT}:localhost:{JUPYTER_PORT} "
                  f"root@{ssh_ip} -p {ssh_port}")
            print(f"    2) Connect:  http://localhost:{JUPYTER_PORT}/lab?token={JUPYTER_TOKEN}")
            print(f"                 (emacs-jupyter -> kernel 'r3dg')")
            print(f"  Tip: `--write-ssh-config` writes this as `ssh {DEFAULT_SSH_ALIAS}` (+tunnel).")
        else:
            print("  SSH:           (no public TCP port yet — use the proxy URL below)")
        if jupyter_url:
            print(f"\n  Jupyter (proxy): {jupyter_url}/lab?token={JUPYTER_TOKEN}")
        print("\n  Bootstrap log:  ssh in, then `tail -f /workspace/startup.log`")
        print("  (first boot builds the uv venv + CUDA extensions — can take 15-30 min)")
    else:
        print("\nRuntime not ready yet.")
        print(f"  Potential Jupyter: https://{pod_id}-{JUPYTER_PORT}.proxy.runpod.net"
              f"/lab?token={JUPYTER_TOKEN}")


def write_ssh_config(api_key: str, pod_name: Optional[str] = None,
                     alias: str = DEFAULT_SSH_ALIAS, quiet: bool = False) -> bool:
    """Write/replace a managed `Host <alias>` block in ~/.ssh/config pointing at
    the pod's current SSH endpoint (which changes on every stop/start).
    Returns True on success."""
    import re
    import subprocess
    runpod.api_key = api_key
    target = get_pod_name(pod_name)
    pod = find_pod_by_name(target, api_key)
    if not pod:
        print(f"Pod '{target}' not found.")
        return False
    pod = runpod.get_pod(pod["id"])
    ssh_ip, ssh_port, _ = _parse_runtime_endpoints(pod["id"], pod.get("runtime"))
    if not (ssh_ip and ssh_port):
        print("No public TCP SSH endpoint yet (pod still starting?). "
              "Re-run --write-ssh-config once --status shows an SSH line.")
        return False

    begin = f"# >>> radiance-launcher {alias} (managed; do not edit) >>>"
    end = f"# <<< radiance-launcher {alias} <<<"
    block = "\n".join([
        begin,
        f"Host {alias}",
        f"    HostName {ssh_ip}",
        f"    Port {ssh_port}",
        "    User root",
        f"    LocalForward {JUPYTER_PORT} localhost:{JUPYTER_PORT}",
        "    IdentityFile ~/.ssh/id_ed25519",
        "    StrictHostKeyChecking accept-new",
        f"    UserKnownHostsFile {SSH_KNOWN_HOSTS}",
        end,
    ])

    path = os.path.expanduser(SSH_CONFIG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read()
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pat.search(existing):
        new = pat.sub(block, existing)
    else:
        prefix = existing.rstrip("\n")
        new = (prefix + "\n\n" if prefix else "") + block + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    os.chmod(path, 0o600)

    # Drop any stale host key for this exact target so a reused ip:port with a
    # fresh host key doesn't trip "REMOTE HOST IDENTIFICATION HAS CHANGED".
    kh = os.path.expanduser(SSH_KNOWN_HOSTS)
    try:
        subprocess.run(["ssh-keygen", "-R", f"[{ssh_ip}]:{ssh_port}", "-f", kh],
                       capture_output=True, check=False)
    except Exception:
        pass

    if not quiet:
        print(f"~/.ssh/config updated -> `ssh {alias}`  (root@{ssh_ip}:{ssh_port}, "
              f"auto-tunnels {JUPYTER_PORT}). emacs-jupyter: http://localhost:{JUPYTER_PORT}"
              f"/lab?token={JUPYTER_TOKEN} (kernel r3dg)")
    return True


def start_pod(api_key: str, pod_name: Optional[str] = None,
              alias: str = DEFAULT_SSH_ALIAS):
    """Resume a stopped pod, wait for RUNNING, then refresh the SSH config block
    (the public IP/port are reassigned on resume)."""
    runpod.api_key = api_key
    target = get_pod_name(pod_name)
    pod = find_pod_by_name(target, api_key)
    if not pod:
        print(f"Pod '{target}' not found.")
        return
    print(f"Resuming {pod['id']} ({target})...")
    try:
        runpod.resume_pod(pod["id"], gpu_count=1)
    except Exception as e:
        print(f"Failed to resume (GPU may be unavailable in the DC right now): {e}")
        return
    print("Waiting for RUNNING state...")
    for _ in range(60):
        p = runpod.get_pod(pod["id"])
        if p and p.get("desiredStatus") == "RUNNING" and (p.get("runtime") or {}).get("ports"):
            print("\nPod is RUNNING!")
            show_pod_status(api_key, pod["id"], pod_name=target)
            write_ssh_config(api_key, pod_name=target, alias=alias)
            print("\nNote: jupyter re-launches a few min after boot (apt + kernel "
                  "register). Watch `ssh " + alias + " 'tail -f /workspace/startup.log'`.")
            return
        time.sleep(5)
        print(".", end="", flush=True)
    print("\nResumed but slow to start. Re-run --status / --write-ssh-config shortly.")


def list_pods(api_key: str):
    runpod.api_key = api_key
    pods = runpod.get_pods()
    if not pods:
        print("No pods.")
        return
    print(f"\n{len(pods)} pod(s):")
    for pod in pods:
        status = pod.get("desiredStatus", "?")
        emoji = "[R]" if status == "RUNNING" else "[.]" if status == "EXITED" else "[?]"
        gpu = pod.get("machine", {}).get("gpuDisplayName", "CPU")
        print(f"  {emoji} {pod.get('name', '?')}  id={pod.get('id')}  gpu={gpu}  status={status}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Radiance RunPod Launcher (Bible Worlds 3DGS PoC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  1) Create volume:  --create-volume --volume-name radiance-vol --volume-size 100 --datacenter EU-RO-1
  2) List volumes:   --list-volumes
  3) Launch pod:     --create --gpu "NVIDIA GeForce RTX 3090" --volume-id <vol_id> --repo-url <git_url>
  4) Status:         --status   (prints SSH tunnel + Jupyter URL for emacs-jupyter)
""",
    )

    # Pod actions
    parser.add_argument("--test-api", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--start", action="store_true",
                        help="Resume a stopped pod and refresh ~/.ssh/config "
                             "(public IP/port are reassigned on resume).")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--write-ssh-config", action="store_true",
                        help=f"Write/replace a managed `Host {DEFAULT_SSH_ALIAS}` block in "
                             f"~/.ssh/config with the pod's current SSH endpoint + tunnel.")
    parser.add_argument("--ssh-alias", type=str, default=DEFAULT_SSH_ALIAS,
                        help=f"Host alias for --write-ssh-config/--start (default: {DEFAULT_SSH_ALIAS})")

    # Pod config
    parser.add_argument("--gpu", type=str, default=DEFAULT_GPU_TYPE,
                        help=f"GPU type (default: {DEFAULT_GPU_TYPE})")
    parser.add_argument("--no-jupyter", action="store_true")
    parser.add_argument("--run-command", type=str)
    parser.add_argument("--repo-url", type=str,
                        help="Canonical radiance repo git URL to clone on the pod "
                             "(or set RADIANCE_REPO_URL in .env.runpod.local)")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Don't build the uv venv at boot (do it manually per "
                             "the README). Repos are still cloned.")
    parser.add_argument("--cloud-type", type=str, default="SECURE",
                        choices=["ALL", "SECURE", "COMMUNITY"],
                        help="Cloud type filter (default: SECURE). The emacs-jupyter "
                             "SSH tunnel needs a real public IP that routes to the "
                             "container sshd — COMMUNITY exposed-TCP often does NOT, "
                             "so SSH fails there. Use ALL only if you don't need SSH.")

    # Identity & secrets
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--name", type=str,
                        help=f"Pod name (default: {BASE_POD_NAME}-{{username}})")

    # Network volume
    parser.add_argument("--list-volumes", action="store_true")
    parser.add_argument("--create-volume", action="store_true")
    parser.add_argument("--delete-volume", type=str, metavar="VOLUME_ID")
    parser.add_argument("--volume-name", type=str, default=DEFAULT_NETWORK_VOLUME_NAME)
    parser.add_argument("--volume-size", type=int, default=DEFAULT_NETWORK_VOLUME_SIZE)
    parser.add_argument("--datacenter", type=str, default="EU-RO-1",
                        help=f"Options: {', '.join(DATACENTERS.keys())}")
    parser.add_argument("--volume-id", type=str)
    parser.add_argument("--container-disk-gb", type=int, default=None)
    parser.add_argument("--pod-volume-gb", type=int, default=None)

    args = parser.parse_args()
    api_key = get_api_key(args.api_key)

    if args.list_volumes:
        list_network_volumes(api_key)
    elif args.create_volume:
        create_network_volume(api_key, args.volume_name, args.volume_size, args.datacenter)
    elif args.delete_volume:
        delete_network_volume(api_key, args.delete_volume)
    elif args.test_api:
        test_api_connection(api_key)
    elif args.create:
        create_pod(
            api_key=api_key,
            gpu_type=args.gpu,
            pod_name=args.name,
            no_jupyter=args.no_jupyter,
            run_command=args.run_command,
            repo_url=args.repo_url,
            volume_id=args.volume_id,
            container_disk_gb=args.container_disk_gb,
            pod_volume_gb=args.pod_volume_gb,
            skip_bootstrap=args.skip_bootstrap,
            cloud_type=args.cloud_type,
        )
    elif args.start:
        start_pod(api_key, args.name, alias=args.ssh_alias)
    elif args.list:
        list_pods(api_key)
    elif args.status:
        show_pod_status(api_key, pod_name=args.name)
    elif args.write_ssh_config:
        write_ssh_config(api_key, pod_name=args.name, alias=args.ssh_alias)
    elif args.stop:
        stop_pod(api_key, args.name)
    elif args.delete:
        delete_pod(api_key, args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
