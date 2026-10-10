# Local Docker readiness

Verified on 2026-10-10. Docker Engine is running in a dedicated Colima VM on this Apple Silicon Mac. No model API calls were made.

## Installed

- Docker CLI 29.8.2
- Docker Engine 29.5.2 (Linux arm64, inside the VM)
- Colima 0.10.3 and Lima 2.2.0
- Docker Buildx 0.37.2

Installed from official Homebrew formulae. Docker context: `colima-contextlab`.

The VM uses Apple's Virtualization.Framework, 2 CPUs, 4 GiB RAM, a 10 GiB data disk and a 10 GiB root disk. Virtual disks are sparse. Its configured host mount is the local project directory; the task containers further restrict that to a read-only fixture directory. No login startup service was installed. This document records the development-machine setup; a fresh checkout does not create or start this VM automatically.

A broken Buildx symlink left by an earlier Docker Desktop installation was preserved as `~/.docker/cli-plugins/docker-buildx.before-contextlab`; the active link now points to the Homebrew plugin. Other Docker credentials were not modified.

## Built and executed

Image: `contextlab-task:local`, built from `Dockerfile.task` using Python 3.13.12 and pytest 9.1.1. `.dockerignore` restricts the build context to its Dockerfile and ignore file; project source, environments, recordings and run artifacts are excluded.

All **17 real-container checks passed**:

- Three fixture bugs are detected by public tests.
- Three intermediate fixes pass public tests but fail the independent requirements.
- Three final reference solutions pass public tests and independent checks.
- Isolation checks verify UID 65534, read-only root/workspace, loopback-only networking, no capabilities, no privilege escalation, no Docker socket or host credentials, and writable temporary storage.
- Cleanup leaves no new task containers behind.

These are reference-fixture infrastructure checks, not AI-agent performance results. The Docker daemon remains running; the short-lived verification containers exited and were removed.

Evidence: `runs/docker-smoke-20261010T083437Z-3a33dc69/verification.json`. Image metadata: `runs/docker-image-inspect.json`. The evidence pins the exact image ID used for every execution.

## Useful commands

```sh
docker version
docker context show
docker image ls contextlab-task

# Start after a machine restart, or after stopping the VM:
colima start contextlab
docker context use colima-contextlab

# Repeat the actual container checks from the project directory:
uv run python -m scripts.verify_docker --image contextlab-task:local

# Stop the VM when you no longer need Docker:
colima stop contextlab
```

After adding key-only `.env` loading for `check` and `run`, ContextLab's local preflight returned `{"ready": true, "issues": []}` on 2026-10-10. The credential file has owner-only permissions and is ignored by Git and Docker builds. No credential values were printed. The subsequent user-approved pilot authenticated successfully and completed all nine sessions, with 103 successful model calls and $0.53146455 in usage-derived cost. Cleanup left no task containers. See `docs/gates/M0.md` for results and the untested compaction gate. Further paid experiments require approval of their own printed plan and spend ceiling.

Installation references: [Colima Docker setup](https://github.com/abiosoft/colima#docker), [Docker CLI](https://formulae.brew.sh/formula/docker), [Buildx](https://formulae.brew.sh/formula/docker-buildx).
