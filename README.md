# Liqo + RHOAI 3.x GPU offloading spike

Proof of concept for offloading RHOAI 3.x GPU workloads from a tenant
OpenShift cluster to a bare-metal GPU worker using Liqo v1.2.0.

## Architecture

```
Tenant (OCP 4.19, RHOAI 3.x)          Worker (OCP 4.22, RTX 5090)
┌───────────────────────────┐          ┌───────────────────────────┐
│  RHOAI operators          │          │  GPU Operator             │
│  Kyverno (tolerations)    │          │  Kueue (admission/quota)  │
│  Liqo virtual node        │◄────────►│  Liqo agent               │
│  kube-scheduler           │  peering │  ShadowPod controller     │
└───────────┬───────────────┘          └───────────┬───────────────┘
            │                                      │
            └──────────── Submariner ──────────────┘
                     (Libreswan IPsec)
```

The tenant cluster runs RHOAI operators and sees the worker's GPU
capacity through Liqo's virtual node. The kube-scheduler places GPU
pods on the virtual node based on `nvidia.com/gpu` resource requests.
Liqo creates ShadowPods on the worker, where Kueue handles admission
and quota. Submariner provides cross-cluster networking (Service, DNS,
Route access to offloaded pods).

## What's tested

32 tests across 9 test files covering:

- Core dispatch (pod scheduling, status sync, resource reflection, cleanup)
- Storage (PVC reflection via `liqo` StorageClass, data persistence)
- RHOAI workloads (Notebook, PyTorchJob, TFJob, KServe, RayJob, RayCluster, XGBoost, DSPA, LMEval)
- Kueue integration (admission, queuing, preemption, ShadowPod label persistence)
- Networking (Submariner reachability, headless DNS, KServe/Notebook routes, tunnel recovery)
- RHOAI 3.x smoke (operator presence, no CRDs on worker, no OSSM)

## How to run

```bash
pip install -e .

RHOAI3_KUBECONFIG=~/.kube/tenant-rhoai3 \
WORKER_KUBECONFIG=~/.kube/worker \
  python -m pytest tests/liqo/ -v
```

Filter by category:

```bash
python -m pytest tests/liqo/ -v -m liqo           # all Liqo tests
python -m pytest tests/liqo/ -v -m rhoai           # RHOAI integration
python -m pytest tests/liqo/ -v -m networking      # Submariner
python -m pytest tests/liqo/ -v -m kueue           # Kueue admission/scheduling
```

## Lab environment

- Liqo v1.2.0 with networking disabled + Submariner v0.24
- Tenant: SNO (OCP 4.19), RHOAI 3.x
- Worker: SNO (OCP 4.22), NVIDIA RTX 5090 (Blackwell), GPU Operator, Kueue
- Kyverno on tenant for virtual-node toleration injection

## Documentation

- [Spike assessment](docs/liqo-spike-assessment.md) with all 32 test
  scenarios, deployment findings, and recommendation
- [Submariner architecture](docs/submariner-architecture.md)

## Setup

Ansible playbooks in `playbooks/` automate Liqo, Submariner, Kyverno,
Kueue, LVMS, and RHOAI installation. See the
[spike assessment](docs/liqo-spike-assessment.md) for configuration
details and deployment findings.
