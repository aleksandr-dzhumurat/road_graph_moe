# Nebius cli

create key
```shell
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 \
  -out ~/nebius_private.pem
```

exract public key
```shell
openssl rsa -pubout -in ~/nebius_private.pem -out ~/nebius_public.pem
```

## Generating a new HMAC key

HMAC (S3) keys are scoped to a service account and grant access to all S3 buckets in the project. Generate a new one whenever you add a bucket that the existing key doesn't cover, or when the secret needs rotating.

1. Go to **Nebius console → IAM → Service accounts → `serviceaccount-e00w46nmwhmkseaak0` → Access keys**
2. Click **Create access key** — a new key pair is generated covering all project buckets
3. Copy both values immediately — the secret is shown **once only**
4. Update `~/.aws/credentials`:

```ini
[nebius]
aws_access_key_id = <new_key_id>
aws_secret_access_key = <new_secret>
```

5. Verify access to all required buckets:

```bash
for bucket in geo-trajectories geo-trajectories-checkpoints geo-trajectories-road-enhanced; do
  aws s3 ls s3://$bucket/ --profile nebius \
    --endpoint-url https://storage.eu-north1.nebius.cloud \
    && echo "$bucket OK" || echo "$bucket FAILED"
done
```

6. Recreate the Nebius job secret with the new key:

```bash
make delete-s3-secret && make create-s3-secret
```

```shell
export ACCESS_KEY_AWS_ID=***
export SECRET_ACCESS_KEY=***
```

Go to your service account 
Open the Public keys section
Add a new key by uploading ~/nebius_public.pem
Copy the new service account ID and public key ID that Nebius assigns

install cli

```shell
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
```

run

```shell
nebius profile create
```

Console output
```shell
Profile name: adzhumurat-nebius
✔ Set api endpoint: api.nebius.cloud█
Select authorization type: service account
Set service account ID: serviceaccount-e00w46nmwhmkseaak0
Set public key ID: publickey-e00tcek9msnmpmp54m
Set path to PEM encoded private key (file can be removed afterwards): ~/nebius_private.pem
Choose Tenant ID to use as a default: tenant-e00sdwxgt8km9vwtt3
Choose Project ID (parent-id) to use as a default: project-e00k4rxqpr00w5gx8c3ent
```

result:
```shell
profile "adzhumurat-nebius" configured and activated
```

Confirm service account name
```shell
nebius iam service-account list
```

Next go [step-by-step instruction](https://docs.nebius.com/object-storage/interfaces/aws-cli#aws-cli-setup-instructions)

Get system account ID (we have already created it on previous step)
```shell
export SA_ID=$(nebius iam service-account list --format json \
  | jq -r '.items[] | select(.metadata.name == "object-storage-sa") | .metadata.id')
```

Grant edit access to the service account:
```shell
export PROJECT_ID=$(nebius config get parent-id)
export TENANT_ID=$(nebius iam project get $PROJECT_ID --format jsonpath='{.metadata.parent_id}')
```

Get the ID of the default editors group:
```shell
export EDITORS_GROUP_ID=$(nebius iam group get-by-name \
  --name editors --parent-id $TENANT_ID \
  --format jsonpath='{.metadata.id}')
```

Add the service account to editors group:
```shell
nebius iam group-membership create \
  --parent-id $EDITORS_GROUP_ID \
  --member-id $SA_ID
```

Skip step with keys creating and configure aws
```shell
aws configure set aws_access_key_id $ACCESS_KEY_AWS_ID --profile nebius
aws configure set aws_secret_access_key $SECRET_ACCESS_KEY --profile nebius
```


# S3

Interactions with storage

list dirs in bucket
list buckets
```shell
aws s3 ls --profile nebius  --endpoint-url https://storage.eu-north1.nebius.cloud
```

create new bucket
```shell
aws s3 mb s3://geo-trajectories --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

copy dir to a bucket
```shell
aws s3 cp data/gold s3://geo-trajectories/ --recursive --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

Check bucket content
```shell
aws s3 ls s3://geo-trajectories/ --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

Or more detailed output
```shell
aws s3 ls s3://geo-trajectories/ --recursive --human-readable --summarize --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```


Before pushing to docker

```shell
/Users/adzhumurat/.nebius/bin/nebius iam get-access-token | docker login cr.eu-north1.nebius.cloud --username iam --password-stdin
```


# Remote build

Builds the Docker image on a temporary Nebius CPU VM (amd64) instead of locally.
Avoids QEMU emulation on Apple Silicon and produces a native Linux/amd64 image.

## One-time setup

Generate a dedicated SSH key:
```shell
ssh-keygen -t ed25519 -f ~/.ssh/nebius -C "nebius-build" -N ""
```

## Run

```shell
make build-remote SSH_KEY=~/.ssh/nebius
# or with a custom tag
make build-remote SSH_KEY=~/.ssh/nebius IMAGE_TAG=v2
```

Internally calls `python scripts/build_image.py --tag <tag> --ssh-key <key>`.

## What the script does

1. Creates a `cpu-e2 / 2vcpu-8gb` VM in project `project-e00k4rxqpr00w5gx8c3ent`
2. Injects the SSH public key via `--cloud-init-user-data`
3. Attaches a 50 GiB `network_ssd` boot disk from image `computeimage-e00x8tej7rj2bpm8pk` (ubuntu22.04-driverless)
4. Assigns a public IP via `--network-interfaces` with `"public_ip_address": {}`
5. Polls until the IP is reachable over SSH (strips `/32` CIDR suffix from the returned address)
6. `rsync`s the project (excludes `data/`, `.env`, `.git/`, `__pycache__/`)
7. Installs Docker, logs in to `cr.eu-north1.nebius.cloud` using an IAM token passed via stdin
8. Runs `docker build` + `docker push`
9. Deletes the VM in a `finally` block — always runs even on failure

## Key constants (scripts/build_image.py)

| Constant | Value |
|---|---|
| `PROJECT_ID` | `project-e00k4rxqpr00w5gx8c3ent` |
| `SUBNET_ID` | `vpcsubnet-e00kq5p36ty3gad3vf` |
| `REGISTRY` | `cr.eu-north1.nebius.cloud/registry-e00z8s1dsnskgapvef` |
| Boot image | `computeimage-e00x8tej7rj2bpm8pk` (ubuntu22.04-driverless) |

## Public images (eu-north1)

| ID | Name | Use for |
|---|---|---|
| `computeimage-e00x8tej7rj2bpm8pk` | ubuntu22.04-driverless | CPU build VMs |
| `computeimage-e00d7ctzs3waty7c1w` | ubuntu24.04-driverless | CPU build VMs |
| `computeimage-e00ckjm576pedy5rwv` | ubuntu22.04-cuda12 | GPU training VMs |
| `computeimage-e00vbaf7yyn7v8a6f3` | ubuntu24.04-cuda13 | GPU training VMs |

## Gotchas discovered

- `--platform` / `--preset` flags do not exist; correct flags are `--resources-platform` / `--resources-preset`
- `--ssh-key` flag does not exist; SSH key must be injected via `--cloud-init-user-data`
- `--image-family` flag does not exist; use `--boot-disk-managed-disk-source-image-id` with a concrete image ID
- Image families (e.g. `ubuntu22.04`) are not defined in this region; must use public image IDs from `nebius compute image list-public --region eu-north1`

### Boot disk survives VM deletion, blocking subsequent builds

**Symptom**: `make build-remote` fails immediately with:
```
AlreadyExists: disk with name "image-builder-tmp-disk" already exists
```

**Root cause**: Nebius does not automatically delete a VM's boot disk when the instance is deleted. If a previous build failed mid-run, the disk is left behind unattached (or still attached to a ghost instance).

**Fix**: Delete the orphaned VM and disk manually:

```bash
# Find and delete the orphaned VM
nebius compute instance list --parent-id project-e00k4rxqpr00w5gx8c3ent --format json | \
  python3 -c "
import json,sys,subprocess
for i in json.load(sys.stdin).get('items',[]):
    if 'image-builder' in i['metadata'].get('name',''):
        subprocess.run(['nebius','compute','instance','delete','--id',i['metadata']['id']])
"

# Find and delete the orphaned disk
nebius compute disk list --parent-id project-e00k4rxqpr00w5gx8c3ent --format json | \
  python3 -c "
import json,sys,subprocess
for i in json.load(sys.stdin).get('items',[]):
    if 'image-builder' in i['metadata'].get('name',''):
        subprocess.run(['nebius','compute','disk','delete','--id',i['metadata']['id']])
"
```

`build_image_remote.py` now explicitly deletes the disk in its `finally` block to prevent this on future failures.

### `python` not found — use `python3`

**Symptom**: `make build-remote` fails instantly with `/bin/sh: python: command not found`.

**Fix**: The Makefile target uses `python3`. If you see this on a fresh shell, ensure the venv is activated or that `python3` is on `PATH`.

### `bytes | None` union syntax requires Python 3.10+

**Symptom**: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` when `build_image_remote.py` is run with the system `python3` (e.g. 3.9 on macOS).

**Fix**: The script now uses `Optional[bytes]` from `typing` instead of `bytes | None`, which is compatible with Python 3.8+.
- The returned public IP includes a CIDR suffix (`/32`) that must be stripped before use with SSH
- `nebius` binary is installed to `~/.nebius/bin/` and may not be on PATH in non-login shells; `build_image.py` resolves it via `shutil.which` with a fallback to that path

### VM appears RUNNING but SSH never connects

**Symptom**: The VM is created and returns a public IP immediately, but all SSH attempts time out for the full 360s, after which `delete_vm` gets `NotFound` (Nebius already terminated the instance internally):

```
[build] Public IP: 89.169.102.61
[build] Waiting for SSH...
  err│ ssh: connect to host 89.169.102.61 port 22: Operation timed out
  err│ ssh: connect to host 89.169.102.61 port 22: Connection refused
  err│ Read from remote host 89.169.102.61: Connection reset by peer
  err│ ssh: connect to host 89.169.102.61 port 22: Operation timed out
  ...
[build] ERROR: SSH did not become available within timeout
  err│ Error: instance not found by id "computeinstance-e00ecmzkybsjqj7hc2"
```

**Root cause**: Nebius occasionally terminates a freshly created VM internally (likely a cloud-init or scheduling issue). The VM is returned as RUNNING in the create response but never becomes reachable.

**Fix**: Just re-run `make build-remote`. The script calls `cleanup_stale_resources()` first, which will find and delete any orphaned disk from the failed attempt. `delete_vm` uses `check=False` so the NotFound error on cleanup doesn't crash the script.

**Note**: SSH normally goes through several transient states before succeeding: `Operation timed out` → `Connection refused` → `Connection reset by peer` → `ok`. This is normal cloud-init startup. The failure mode above is when it never reaches `ok` and regresses back to timeouts.

### `.venv/` must be excluded from rsync

**Symptom**: rsync transfers hundreds of MB of `.venv/` files, then the build VM may fail if a macOS ARM64 venv is sent to a Linux AMD64 host.

**Fix**: The rsync call includes `--exclude=.venv/`. If you see large transfers (>50 MB) from `.venv`, the exclude is missing or `.venv` was renamed.


# Cloud run

Launches `backbone.py` as a Nebius AI serverless job on a GPU with S3-mounted training data.

## One-time setup

### 1. Create the checkpoints bucket (if it doesn't exist)

```shell
aws s3 mb s3://geo-trajectories-checkpoints --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

### 2. Create the MysteryBox secret for S3 credentials

```shell
make create-s3-secret
```

This reads `key_id` and `secret_key` from `.env` and creates a secret named `nebius-s3-creds`
with two entries: `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` — the exact key names Nebius requires.

To recreate after rotating credentials:
```shell
make delete-s3-secret
make create-s3-secret
```

## Run

```shell
make run-cloud
# debug run (50 trips, 300 steps, no GPU waste):
make run-cloud  # edit backbone.py --debug first
```

## Monitor

```shell
nebius ai job logs <job_id> --follow
nebius ai job get <job_id>
```

## Fetching job logs

The default `--since` window is **1 hour** and the output is silently capped at **100 lines**.
Use `--tail` to raise the cap and `--since` to reach back to job start:

```shell
# Full log from the start of a specific day
nebius ai job logs aijob-e00pc1astawdz8wmtq \
  --since 2026-06-07 \
  --tail 100000 \
  --profile nebius \
  > data/train_log_full.txt
```

Stream live (keeps running until Ctrl+C):

```shell
nebius ai job logs aijob-e00pc1astawdz8wmtq \
  --since 2026-06-07 \
  --follow \
  --profile nebius \
  > data/train_log_full.txt
```

Find job IDs for recent runs:

```shell
nebius ai job list --profile nebius
```

## Plotting training curves

`scripts/train_curve.py` auto-detects phase-1 (backbone) and phase-2 (expert) log formats.

Plot from a saved log file:

```shell
uv run python scripts/train_curve.py data/train_log_full.txt
# custom output path and smoothing window
uv run python scripts/train_curve.py data/train_log_full.txt \
  --out data/expert_curves.png \
  --smooth 20
```

Fetch from Nebius and plot in one shot:

```shell
nebius ai job logs aijob-e00pc1astawdz8wmtq \
  --since 2026-06-07 \
  --tail 100000 \
  --profile nebius \
  | uv run python scripts/train_curve.py /dev/stdin \
      --out data/expert_curves.png
```

## Debugging

### Step 1 — check recent jobs and their failure codes

```shell
nebius ai job list --format json | python3 -c "
import sys, json
for j in json.load(sys.stdin).get('items', [])[:5]:
    m = j['metadata']; s = j['status']
    print(m['id'], j['spec'].get('platform'), s['state'],
          s.get('state_details', {}).get('code', ''),
          s.get('state_details', {}).get('message', ''))
"
```

Two distinct failure modes:
- `NotEnoughResources` (code 8) → resource exhaustion, switch platform or add `--preemptible`
- `StartFailed` (code 9) → job got a slot but couldn't start; see steps below

### Step 2 — isolate image vs S3 volume

If `StartFailed`, bisect by removing components:

```shell
# Test A: public image, no volumes → rules out platform issues
nebius ai job create \
  --image ubuntu:22.04 --container-command "echo ok" \
  --platform gpu-l40s-d --preset 1gpu-16vcpu-96gb \
  --preemptible --subnet-id vpcsubnet-e00kq5p36ty3gad3vf

# Test B: private image, no volumes → rules out image pull issues
nebius ai job create \
  --image cr.eu-north1.nebius.cloud/e00z8s1dsnskgapvef/trajectory-train:latest \
  --container-command "python -c 'import torch; print(torch.__version__)'" \
  --platform gpu-l40s-d --preset 1gpu-16vcpu-96gb \
  --preemptible --subnet-id vpcsubnet-e00kq5p36ty3gad3vf
```

If A completes but B fails → image issue. If B completes but full job fails → S3 mount issue.

### Step 3 — verify S3 credentials have bucket access

```shell
# Using the key stored in the MysteryBox secret (read from ~/.aws/credentials [nebius])
python3 -c "
import configparser, os, subprocess
cfg = configparser.ConfigParser()
cfg.read(os.path.expanduser('~/.aws/credentials'))
k = cfg['nebius']['aws_access_key_id']
s = cfg['nebius']['aws_secret_access_key']
env = {**os.environ, 'AWS_ACCESS_KEY_ID': k, 'AWS_SECRET_ACCESS_KEY': s}
r = subprocess.run(['aws','s3','ls','s3://geo-trajectories/',
    '--endpoint-url','https://storage.eu-north1.nebius.cloud'],
    env=env, capture_output=True, text=True)
print('rc:', r.returncode, '— OK' if r.returncode == 0 else r.stderr[:200])
"
```

`AccessDenied` here means the wrong key is in the secret — recreate with `make delete-s3-secret && make create-s3-secret`.

### Step 4 — verify image exists in registry

```shell
TOKEN=$(nebius iam get-access-token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://cr.eu-north1.nebius.cloud/v2/e00z8s1dsnskgapvef/trajectory-train/tags/list"
# Expected: {"name":"e00z8s1dsnskgapvef/trajectory-train","tags":["latest"]}
```

### Step 5 — check the MysteryBox secret exists and has the right key names

```shell
nebius mysterybox secret list --parent-id project-e00k4rxqpr00w5gx8c3ent --format json | \
  python3 -c "
import sys, json
for i in json.load(sys.stdin).get('items', []):
    print(i['metadata']['id'], i['metadata']['name'])
"
# Expected: one secret named nebius-s3-creds
```

The secret payload cannot be inspected via CLI (sensitive). If in doubt, delete and recreate:
```shell
make delete-s3-secret
make create-s3-secret
```

## What `make run-cloud` does

```
nebius ai job create
  --image        cr.eu-north1.nebius.cloud/e00z8s1dsnskgapvef/trajectory-train:latest
  --platform     gpu-l40s-d
  --preset       1gpu-16vcpu-96gb
  --subnet-id    vpcsubnet-e00kq5p36ty3gad3vf
  --volume       s3://geo-trajectories:/app/data/gold:ro:nebius@nebius-s3-creds
  --volume       s3://geo-trajectories-checkpoints:/app/data/checkpoints:rw:nebius@nebius-s3-creds
  --env          S3_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
```

S3 buckets are FUSE-mounted into the container at infrastructure level before the process starts.
Training data is read-only; checkpoints are read-write.

## Available GPU platforms (eu-north1, project-e00k4rxqpr00w5gx8c3ent)

Check current presets:
```shell
nebius compute platform list --parent-id project-e00k4rxqpr00w5gx8c3ent
```

# Nebius GPU Presets

Single-GPU presets with host memory below 300 GB.

| platform | preset | GPU |
|----------|--------|-----|
| gpu-l40s-d | 1gpu-16vcpu-96gb | 1 × L40S (48GB) |
| gpu-l40s-d | 1gpu-32vcpu-192gb | 1 × L40S (48GB) |
| gpu-l40s-d | 1gpu-48vcpu-288gb | 1 × L40S (48GB) |
| gpu-h200-sxm | 1gpu-16vcpu-200gb | 1 × H200 (141GB) |
| gpu-h100-sxm | 1gpu-16vcpu-200gb | 1 × H100 (80GB) |
| gpu-l40s-a | 1gpu-8vcpu-32gb | 1 × L40S (48GB) |
| gpu-l40s-a | 1gpu-16vcpu-64gb | 1 × L40S (48GB) |
| gpu-l40s-a | 1gpu-24vcpu-96gb | 1 × L40S (48GB) |
| gpu-l40s-a | 1gpu-32vcpu-128gb | 1 × L40S (48GB) |
| gpu-l40s-a | 1gpu-40vcpu-160gb | 1 × L40S (48GB) |

## Price estimation

The billing calculator returns hourly and monthly estimates for any platform/preset combination.

### On-demand price

```shell
nebius billing v1alpha1 calculator estimate \
  --resource-spec-compute-instance-spec-resources-platform gpu-h100-sxm \
  --resource-spec-compute-instance-spec-resources-preset 1gpu-16vcpu-200gb \
  --resource-spec-compute-instance-spec-parent-id project-e00k4rxqpr00w5gx8c3ent \
  --format json
```

### Preemptible price

Add `--resource-spec-compute-instance-spec-preemptible-priority 1` (flag is deprecated but still returns the preemptible rate):

```shell
nebius billing v1alpha1 calculator estimate \
  --resource-spec-compute-instance-spec-resources-platform gpu-h100-sxm \
  --resource-spec-compute-instance-spec-resources-preset 1gpu-16vcpu-200gb \
  --resource-spec-compute-instance-spec-parent-id project-e00k4rxqpr00w5gx8c3ent \
  --resource-spec-compute-instance-spec-preemptible-priority 1 \
  --format json
```

### Actual rates (eu-north1, 2026-06-04)

| Platform | Preset | GPU | On-demand | Preemptible |
|---|---|---|---|---|
| gpu-h100-sxm | 1gpu-16vcpu-200gb | 1 × H100 80GB | $3.85/h | $2.15/h |

### Cost for a training run

```python
# backbone.py at 0.153 s/step on H100, 300k steps total
total_hours = 0.153 * 300_000 / 3600   # ≈ 12.75 h
on_demand   = total_hours * 3.85        # ≈ $49
preemptible = total_hours * 2.15        # ≈ $27
```

Use `--preemptible` in `make run-cloud` — `backbone.py` handles `SIGTERM` checkpointing so preemption only loses at most one 2k-step checkpoint interval (~5 min).

---

## Gotchas discovered

### MysteryBox secret format — the only working solution

The secret payload must be an **array** with exactly these two key names:
```json
[
  {"key": "S3_ACCESS_KEY_ID",     "string_value": "<access_key_id>"},
  {"key": "S3_SECRET_ACCESS_KEY", "string_value": "<secret_key>"}
]
```

Create it with `make create-s3-secret` (reads from `~/.aws/credentials [nebius]`).

All other formats fail:
| Format tried | Error |
|---|---|
| `{"entries":[...]}` wrapper | `proto: syntax error: unexpected token` |
| `aws_access_key_id` / `aws_secret_access_key` keys | `failed to get shared config profile` |
| `access_key` / `secret_key` keys | `failed to get shared config profile` |
| `[default]\n...` credentials file content | `failed to get shared config profile` |
| `[nebius]\n...` credentials file content | `s3 credentials secret is invalid, must have S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY` |

### The credentials in the secret must match the bucket owner

The S3 buckets were created with the `[nebius]` profile key in `~/.aws/credentials`.
The secret must use **the same key** — not any other key from a different service account access key pair.

Using the wrong key results in `Job failed to start` (no logs, FUSE mount fails silently).

Verify access before creating the secret:
```shell
AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> \
  aws s3 ls s3://geo-trajectories/ \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

`make create-s3-secret` reads from `~/.aws/credentials [nebius]` which is the correct key.

### AWS profile name in `--volume` is validated locally

The volume spec format is `s3://BUCKET:/path:MODE:PROFILE@SECRET_NAME`.

`PROFILE` is validated against the **local** `~/.aws/credentials` before the job is submitted.
It must match a profile that exists locally. `nebius` works; `default` fails with
`failed to get shared config profile, default` because there is no `[default]` locally.

At runtime, `PROFILE` is ignored — the Nebius runner uses `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`
directly from the secret regardless of the profile name.

### `gpu-l40s-a` is unreliable in eu-north1

`gpu-l40s-a` suffers from both resource exhaustion (code 8) and random `StartFailed` errors.
Use `gpu-l40s-d` (AMD EPYC, same L40S GPU, separate node pool) — it works reliably.

### `--preemptible` dramatically improves availability

On-demand `gpu-l40s-*` slots are frequently exhausted. Preemptible uses spare capacity
and is cheaper. `backbone.py` handles `SIGTERM` checkpointing so preemption is safe.

### S3 env vars do not reach the volume mount

`--env AWS_ACCESS_KEY_ID=...` and `--env-secret` only inject into the running container.
The FUSE S3 mount happens at infrastructure level before the container starts — credentials
must be supplied via `PROFILE@SECRET` in the volume spec, not via environment variables.

### `geo-trajectories-checkpoints` bucket must exist before job submission

The job creation step resolves both volumes upfront. Create the bucket first:
```shell
aws s3 mb s3://geo-trajectories-checkpoints --profile nebius \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

### `--subnet-id` is required

Multiple subnets exist in the project. Without it: `multiple subnets found`.
Use `vpcsubnet-e00kq5p36ty3gad3vf`.

### `create-s3-secret` was storing a credentials file instead of individual keys

**Symptom**: `s3 credentials secret is invalid, must have S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY keys in the payload`

**Root cause**: The original `create-s3-secret` Makefile target read `key_id`/`secret_key` env vars and wrapped them in an AWS credentials file string stored under a single `credentials` key — the format Nebius explicitly rejects (see the "MysteryBox secret format" gotcha above).

**Fix**: `create-s3-secret` now reads directly from `~/.aws/credentials [nebius]` and stores two separate keys:

```json
[
  {"key": "S3_ACCESS_KEY_ID",     "string_value": "<access_key_id>"},
  {"key": "S3_SECRET_ACCESS_KEY", "string_value": "<secret_key>"}
]
```

If you encounter this error, delete and recreate the secret:

```bash
make delete-s3-secret && make create-s3-secret
```

### Adding a new S3 bucket for a new job requires recreating the secret

**Symptom**: New job (`run-cloud-phase02`) gets `Job failed before startup completed` (code 9) after the bucket was created and data uploaded successfully.

**Root cause**: Nebius HMAC keys are bucket-scoped. The existing `nebius-s3-creds` secret was created with a key that didn't cover the new `geo-trajectories-road-enhanced` bucket. The FUSE mount fails silently at infrastructure level before the container starts.

**Diagnosis**:

```bash
python3 -c "
import configparser, os, subprocess
cfg = configparser.ConfigParser()
cfg.read(os.path.expanduser('~/.aws/credentials'))
k = cfg['nebius']['aws_access_key_id']
s = cfg['nebius']['aws_secret_access_key']
env = {**os.environ, 'AWS_ACCESS_KEY_ID': k, 'AWS_SECRET_ACCESS_KEY': s}
r = subprocess.run(['aws','s3','ls','s3://geo-trajectories-road-enhanced/',
    '--endpoint-url','https://storage.eu-north1.nebius.cloud'],
    env=env, capture_output=True, text=True)
print('rc:', r.returncode, '— OK' if r.returncode == 0 else r.stderr[:200])
"
```

**Fix**: Generate a new HMAC key in the Nebius console covering all required buckets, update `~/.aws/credentials [nebius]`, then recreate the secret:

```bash
make delete-s3-secret && make create-s3-secret
```

### `aws s3 sync` fails with `Unable to parse response, invalid XML received: b'0\r\n\r\n'`

**Symptom**: Some shards upload successfully, then others fail mid-sync with:
```
upload failed: ... Unable to parse response (syntax error: line 1, column 0), invalid XML received.
Further retries may succeed: b'0\r\n\r\n'
```

**Root cause**: The AWS CLI uses chunked transfer encoding by default. The Nebius S3 endpoint returns an empty chunked response (`0\r\n\r\n`) that the SDK cannot parse as XML.

**Fix**: Disable payload signing for the nebius profile in `~/.aws/config`:

```ini
[profile nebius]
region = eu-north1
endpoint_url = https://storage.eu-north1.nebius.cloud
s3 =
    payload_signing_enabled = false
```

This makes the CLI send a normal `Content-Length` request instead of a chunked one. Applies to all `aws s3` commands using the nebius profile.
