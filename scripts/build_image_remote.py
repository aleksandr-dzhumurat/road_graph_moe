#!/usr/bin/env python3
"""
Build Docker image on a temporary Nebius VM and push to Nebius Container Registry.

create key
    ssh-keygen -t ed25519 -f ~/.ssh/nebius -C "nebius-build" -N ""


make build-remote SSH_KEY=~/.ssh/nebius

Creates VM → syncs project → builds image → pushes → destroys VM.
VM is always deleted in the finally block, even on failure.

Usage:
    python scripts/build_image.py
    python scripts/build_image.py --tag v2
    python scripts/build_image.py --ssh-key ~/.ssh/nebius_key
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

REGISTRY   = "cr.eu-north1.nebius.cloud/e00z8s1dsnskgapvef"
IMAGE_NAME = "trajectory-train"
VM_NAME    = "image-builder-tmp"
VM_USER    = "ubuntu"
PROJECT_ID = "project-e00k4rxqpr00w5gx8c3ent"
SUBNET_ID  = "vpcsubnet-e00kq5p36ty3gad3vf"


def _find_bin(name: str) -> str:
    """Resolve a CLI binary, checking common install dirs if not on PATH."""
    path = shutil.which(name)
    if path:
        return path
    candidates = [
        Path.home() / ".nebius" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"'{name}' not found. "
        f"Install: curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash"
    )


NEBIUS = _find_bin("nebius")


# ─── Logging ─────────────────────────────────────────────────────────────────


def log(msg: str, prefix: str = "[build]") -> None:
    print(f"{prefix} {msg}", flush=True)


# ─── Subprocess with threaded stdout/stderr streaming ────────────────────────


def _stream(stream, tag: str, sink: list) -> None:
    for line in iter(stream.readline, b""):
        text = line.decode(errors="replace").rstrip()
        print(f"  {tag} {text}", flush=True)
        sink.append(text)


def run(
    cmd: list,
    check: bool = True,
    stdin_data: "Optional[bytes]" = None,
) -> "tuple[int, str, str]":
    log(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if stdin_data is not None:
        proc.stdin.write(stdin_data)
        proc.stdin.close()

    out_lines: list[str] = []
    err_lines: list[str] = []
    t_out = threading.Thread(target=_stream, args=(proc.stdout, "out│", out_lines))
    t_err = threading.Thread(target=_stream, args=(proc.stderr, "err│", err_lines))
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    return proc.returncode, "\n".join(out_lines), "\n".join(err_lines)


def ssh_run(
    ip: str,
    cmd: str,
    key: Path,
    stdin_data: "Optional[bytes]" = None,
    check: bool = True,
) -> "tuple[int, str, str]":
    return run(
        [
            "ssh",
            "-i",
            str(key),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            f"{VM_USER}@{ip}",
            cmd,
        ],
        check=check,
        stdin_data=stdin_data,
    )


# ─── VM lifecycle ─────────────────────────────────────────────────────────────


def cleanup_stale_resources() -> None:
    """Delete any VM or disk left over from a previous failed build."""
    found = False

    rc, stdout, _ = run(
        [NEBIUS, "compute", "instance", "list", "--parent-id", PROJECT_ID, "--format", "json"],
        check=False,
    )
    if rc == 0:
        for item in json.loads(stdout).get("items", []):
            if item["metadata"].get("name") == VM_NAME:
                vm_id = item["metadata"]["id"]
                log(f"Removing stale VM {vm_id}...")
                run([NEBIUS, "compute", "instance", "delete", "--id", vm_id], check=False)
                found = True

    rc, stdout, _ = run(
        [NEBIUS, "compute", "disk", "list", "--parent-id", PROJECT_ID, "--format", "json"],
        check=False,
    )
    if rc == 0:
        for item in json.loads(stdout).get("items", []):
            if item["metadata"].get("name") == f"{VM_NAME}-disk":
                disk_id = item["metadata"]["id"]
                log(f"Removing stale disk {disk_id}...")
                run([NEBIUS, "compute", "disk", "delete", "--id", disk_id], check=False)
                found = True

    if found:
        log("Waiting 15s for Nebius API to settle after cleanup...")
        time.sleep(15)


def create_vm(ssh_key: Path) -> str:
    log("Creating build VM...")
    pub_key_path = ssh_key.with_suffix(".pub")
    if not pub_key_path.exists():
        raise FileNotFoundError(f"SSH public key not found: {pub_key_path}")
    pub_key = pub_key_path.read_text().strip()

    cloud_init = f"#cloud-config\nusers:\n  - name: ubuntu\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    ssh_authorized_keys:\n      - {pub_key}\n"
    network_interfaces = json.dumps([{
        "name": "eth0",
        "subnet_id": SUBNET_ID,
        "ip_address": {},
        "public_ip_address": {},
    }])

    _, stdout, _ = run([
        NEBIUS, "compute", "instance", "create",
        "--parent-id", PROJECT_ID,
        "--name", VM_NAME,
        "--resources-platform", "cpu-e2",
        "--resources-preset", "2vcpu-8gb",
        "--boot-disk-managed-disk-name", f"{VM_NAME}-disk",
        "--boot-disk-managed-disk-type", "network_ssd",
        "--boot-disk-managed-disk-size-gibibytes", "50",
        "--boot-disk-managed-disk-source-image-id", "computeimage-e00x8tej7rj2bpm8pk",
        "--boot-disk-attach-mode", "read_write",
        "--network-interfaces", network_interfaces,
        "--cloud-init-user-data", cloud_init,
        "--format", "json",
    ])
    vm_id = json.loads(stdout)["metadata"]["id"]
    log(f"VM created: {vm_id}")
    return vm_id


def get_vm_ip(vm_id: str, timeout: int = 300) -> str:
    log("Waiting for public IP...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, stdout, _ = run(
            [NEBIUS, "compute", "instance", "get", "--id", vm_id, "--format", "json"],
            check=False,
        )
        if rc == 0:
            try:
                ifaces = json.loads(stdout)["status"]["network_interfaces"]
                ip = ifaces[0]["public_ip_address"]["address"].split("/")[0]
                if ip:
                    log(f"Public IP: {ip}")
                    return ip
            except (KeyError, IndexError, json.JSONDecodeError):
                pass
        time.sleep(10)
    raise TimeoutError("Timed out waiting for VM public IP")


def wait_for_ssh(ip: str, key: Path, timeout: int = 360) -> None:
    log("Waiting for SSH...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, _, _ = ssh_run(ip, "echo ok", key, check=False)
        if rc == 0:
            log("SSH ready")
            return
        time.sleep(10)
    raise TimeoutError("SSH did not become available within timeout")


def delete_vm(vm_id: str) -> None:
    log(f"Deleting VM {vm_id}...")
    run([NEBIUS, "compute", "instance", "delete", "--id", vm_id], check=False)
    log("VM deleted")
    # Nebius may leave the boot disk behind; delete it explicitly
    rc, stdout, _ = run(
        [NEBIUS, "compute", "disk", "list", "--parent-id", PROJECT_ID, "--format", "json"],
        check=False,
    )
    if rc == 0:
        disks = json.loads(stdout).get("items", [])
        for disk in disks:
            if disk["metadata"].get("name") == f"{VM_NAME}-disk":
                disk_id = disk["metadata"]["id"]
                log(f"Deleting orphaned disk {disk_id}...")
                run([NEBIUS, "compute", "disk", "delete", "--id", disk_id], check=False)
                log("Disk deleted")


# ─── Build pipeline ───────────────────────────────────────────────────────────


def sync_project(ip: str, key: Path) -> None:
    log("Syncing project files...")
    project_root = Path(__file__).resolve().parent.parent
    run(
        [
            "rsync",
            "-av",
            "--progress",
            "-e",
            f"ssh -i {key} -o StrictHostKeyChecking=no",
            "--exclude=data/",
            "--exclude=.git/",
            "--exclude=.env",
            "--exclude=.claude/",
            "--exclude=.venv/",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            f"{project_root}/",
            f"{VM_USER}@{ip}:/home/{VM_USER}/project/",
        ]
    )


def build_and_push(ip: str, key: Path, full_image: str) -> None:
    log("Installing Docker...")
    ssh_run(ip, "curl -fsSL https://get.docker.com | sh", key)

    log("Authenticating with Nebius Container Registry...")
    _, token, _ = run([NEBIUS, "iam", "get-access-token"])
    ssh_run(
        ip,
        "sudo docker login cr.eu-north1.nebius.cloud --username iam --password-stdin",
        key,
        stdin_data=token.strip().encode(),
    )

    log("Building image...")
    ssh_run(
        ip, f"cd /home/{VM_USER}/project && sudo docker build -t {full_image} .", key
    )

    log("Pushing image...")
    ssh_run(ip, f"sudo docker push {full_image}", key)


# ─── Entrypoint ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Docker image on a Nebius VM.")
    parser.add_argument("--tag", default="latest", help="Image tag (default: latest)")
    parser.add_argument(
        "--ssh-key",
        type=lambda p: Path(p).expanduser(),
        default=Path("~/.ssh/id_rsa").expanduser(),
        help="Path to SSH private key (default: ~/.ssh/id_rsa)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_image = f"{REGISTRY}/{IMAGE_NAME}:{args.tag}"
    log(f"Target image: {full_image}")

    vm_id = None
    try:
        cleanup_stale_resources()
        for attempt in range(1, 4):
            try:
                vm_id = create_vm(args.ssh_key)
                break
            except RuntimeError as e:
                if attempt == 3:
                    raise
                log(f"VM creation failed (attempt {attempt}/3), cleaning up and retrying in 20s: {e}")
                cleanup_stale_resources()
                time.sleep(20)
        ip = get_vm_ip(vm_id)
        wait_for_ssh(ip, args.ssh_key)
        sync_project(ip, args.ssh_key)
        build_and_push(ip, args.ssh_key, full_image)
        log(f"Done — image available at: {full_image}")
    except Exception as e:
        log(f"ERROR: {e}", prefix="[build]")
        sys.exit(1)
    finally:
        if vm_id:
            delete_vm(vm_id)


if __name__ == "__main__":
    main()
