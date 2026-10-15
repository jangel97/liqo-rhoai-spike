# Liqo + RHOAI 3.x GPU offloading spike

Proof of concept for offloading RHOAI 3.x GPU workloads from a tenant
OpenShift cluster to a bare-metal GPU worker using Liqo v1.2.0.

All 32 tests pass. The [spike assessment](docs/liqo-spike-assessment.md)
has the full scenario-by-scenario breakdown and recommendation.


## Architecture overview

```
Tenant (OCP 4.19, RHOAI 3.x)                 Worker (OCP 4.22, RTX 5090)
┌──────────────────────────────┐              ┌──────────────────────────────┐
│                              │              │                              │
│  RHOAI operators             │              │  GPU Operator                │
│    training-operator         │              │  Kueue (ClusterQueue, quota) │
│    notebook-controller       │              │  Kyverno (queue-name labels) │
│    kserve-controller         │              │                              │
│    kuberay-operator          │              │  Liqo provider agent         │
│                              │   Liqo       │    ShadowPod controller      │
│  Liqo consumer               │◄── peer ───►│    resource reflector        │
│    virtual node (10 GPU)     │              │    storage fabric            │
│  Kyverno (GPU tolerations)   │              │                              │
│  kube-scheduler              │              │                              │
│                              │              │                              │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               └──────────── Submariner v0.24 ───────────────┘
                          Libreswan IPsec (UDP 4500)
                     Pod CIDRs: 10.128/14 ←→ 10.132/14
```

RHOAI operators live only on the tenant. The worker has no RHOAI CRDs.
Users interact with RHOAI exactly as they normally would. The GPU
resource request on a pod is the only signal that triggers offloading.


## Pod lifecycle

Here is what happens when a user creates a GPU workload (Notebook, PyTorchJob, etc.) on the tenant:

1. The RHOAI operator (training-operator, notebook-controller, etc.)
   creates a pod spec with `nvidia.com/gpu: 1` in resource limits.

2. Liqo's mutating webhook (`pod.mutate.liqo.io`) adds a toleration for
   `virtual-node.liqo.io/not-allowed:NoExecute` to all pods in the
   offloaded namespace.

3. Kyverno's mutating policy matches the GPU resource request and injects
   a second toleration for `nvidia.com/gpu:NoSchedule`. This is the
   virtual node's GPU taint.

4. The kube-scheduler sees only one node with `nvidia.com/gpu` in
   allocatable: the Liqo virtual node. It places the pod there.
   CPU-only pods stay on the tenant because they lack the GPU toleration
   and the virtual node has a `NoSchedule` taint.

5. Liqo's consumer controller detects the pod on the virtual node and
   creates a ShadowPod CRD on the worker cluster, in the twin namespace
   (same name with `EnforceSameName` strategy).

6. The ShadowPod controller on the worker creates an actual pod from the
   ShadowPod spec. The spec passes through unmodified: security context,
   tolerations, resources, volume mounts all arrive intact.

7. On the worker, Kyverno injects Kueue scheduling labels (queue name,
   priority class). Kueue creates a Workload object and checks the
   ClusterQueue. If a GPU is available the pod runs; otherwise it queues
   until one frees up.

8. The pod runs on the worker's physical GPU. The GPU Operator attaches
   the device, the container starts.

9. Liqo syncs the worker pod's status back to the tenant: phase, PodIP,
   container statuses, exit codes, and log access. The tenant's standard
   endpoint controller picks up the PodIP and creates EndpointSlice
   entries pointing to the worker pod's real IP.

10. When the user deletes the workload on the tenant, Liqo deletes the
    ShadowPod, which deletes the worker pod. No orphaned resources.

```
User → RHOAI operator → Pod spec (nvidia.com/gpu: 1)
  → Liqo webhook (virtual-node toleration)
  → Kyverno (GPU taint toleration)
  → kube-scheduler → virtual node
  → Liqo consumer → ShadowPod on worker
  → Kyverno on worker (Kueue labels)
  → Kueue admission → GPU pod runs
  → Status sync back to tenant
```


## Kyverno toleration injection

The Kyverno ClusterPolicy in `deploy/tenant-gpu-toleration-policy.yml`
matches any pod with `nvidia.com/gpu` in container limits and injects
two tolerations via `patchStrategicMerge`:

- `nvidia.com/gpu: Exists:NoSchedule` (the virtual node's GPU taint)
- `virtual-node.liqo.io/not-allowed: Exists:NoExecute` (Liqo's default taint)

Both tolerations are needed because of webhook ordering. Liqo's webhook
runs first and adds the second toleration to all pods in the namespace.
Kyverno runs after (`reinvocationPolicy: IfNeeded`). If Kyverno only
injected the GPU toleration, a strategic merge could overwrite Liqo's
toleration array. By including both, the final pod has both regardless
of webhook execution order.

RHOAI operators (training-operator, KServe, notebook-controller) do not
strip these tolerations during reconciliation. The spike tests confirm
operator-created pods keep the Kyverno-injected tolerations through
the full create/reconcile cycle.


## Kueue integration

Kueue on the worker is the sole authority for GPU admission and quota.
Getting there requires two configuration tweaks because, by default,
the tenant scheduler and Liqo both try to enforce GPU capacity
independently.

### GPU over-provisioning

The virtual node's GPU count is set at peering time:

```bash
liqoctl peer --resource nvidia.com/gpu=10
```

The worker has 1 physical GPU, but the virtual node advertises 10.
This lets the tenant scheduler accept up to 10 GPU pods simultaneously.
All 10 get offloaded to the worker, where Kueue's ClusterQueue enforces
the real quota (1 GPU). Excess pods queue on the worker until a GPU
frees up.

Without this, the tenant scheduler rejects the second GPU pod before it
ever reaches Kueue, and queuing/preemption never work.

### Disabling Liqo resource enforcement

```bash
helm upgrade liqo liqo/liqo -n liqo --reuse-values \
  --set controllerManager.config.enableResourceEnforcement=false
```

Liqo's `shadowpod.validate.liqo.io` webhook normally checks whether the
ShadowPod's resource requests fit within the peering quota. With
enforcement disabled, the webhook passes all pods through. Kueue becomes
the single gatekeeper.

### ShadowPod label persistence

The ShadowPod controller preserves labels added by admission webhooks.
When Kyverno on the worker adds Kueue scheduling labels, the controller
does not overwrite them. Test L12 validates this explicitly: labels
injected by Kyverno survive from ShadowPod creation through to the
running pod. This is what makes the Kyverno-to-Kueue chain work.


## Submariner networking

### Why not Liqo's built-in networking

Liqo ships a WireGuard-based networking module, but it does not work on
OpenShift:

- Gateway pods need `privileged` SCC and `hostNetwork: true`
- Requires `NET_ADMIN`, `NET_RAW`, `SYS_ADMIN` capabilities
- The v1.2.0 SCC template has a rendering bug (`.` passed to the label
  helper without `.name`, produces a nil interpolation in the instance
  label)
- No Red Hat support or testing on OCP 4.19+

### What Submariner provides

Submariner creates Libreswan IPsec tunnels between clusters. In the
spike, traffic flows like this:

```
Tenant pod → Service on tenant
  → EndpointSlice points to worker PodIP (10.132.x.x)
  → OVN reroute policy on tenant
  → IPsec tunnel (UDP 4500) to worker gateway
  → OVN on worker routes to pod
  → GPU workload responds
  → Response returns through tunnel
```

This enables three workload classes that batch jobs do not need:

- **Notebooks**: Route on tenant reaches Jupyter on worker GPU via tunnel.
  WebSocket connections work through the tunnel.
- **KServe inference**: Inference endpoint on tenant routes to the model
  server running on the worker GPU.
- **Distributed training**: Headless Services reflected by Liqo allow
  inter-pod DNS resolution for PyTorch/TF workers.

### PodIP sync (critical detail)

The whole scheme depends on Liqo syncing the real worker PodIP back to
the tenant pod's status. When Liqo's networking is disabled, it does not
remap IPs. The tenant pod gets the actual worker pod IP (e.g.
10.132.0.148), which Submariner's route agent knows how to reach through
the IPsec tunnel.

This behavior is undocumented. It works in v1.2.0, but it is
implementation behavior, not an API contract. If a future Liqo upgrade
introduces IP remapping with `networking.enabled=false`, it would
silently break Submariner routing. Tests L3 (PodIP sync) and L13
(service reachability) should be the first tests to run after any Liqo
version change.

### CIDR requirements

The current lab uses non-overlapping CIDRs:

- Tenant pod CIDR: `10.128.0.0/14`, service CIDR: `172.30.0.0/16`
- Worker pod CIDR: `10.132.0.0/14`, service CIDR: `172.31.0.0/16`

If two clusters share the same default OpenShift CIDR (`10.128.0.0/14`),
Submariner's Globalnet feature can NAT around the overlap, but it adds
complexity. Better to enforce non-overlapping CIDRs at cluster
provisioning time.


## Storage model

Liqo creates a virtual StorageClass called `liqo` on the tenant
(provisioner: `liqo.io/storage`). PVCs using this class get reflected
to the worker, where Liqo remaps them to the worker's actual
StorageClass.

```bash
# At peering time, specify the worker's storage class
liqoctl peer --remote-real-storage-class-name=lvms-vg1
```

The tenant always uses `storageClassName: liqo`. Liqo handles the
mapping to whatever the worker runs:

| Worker backend | StorageClass |
|---------------|-------------|
| Bare metal (LVMS/TopoLVM) | `lvms-vg1` |
| AWS EBS | `gp3` |
| GCP Persistent Disk | `pd-ssd` |
| Azure Managed Disk | `managed-premium` |

If you need a specific SC on the worker instead of the default, set the
`liqo.io/remote-storage-class` annotation on the PVC.

### When to use which StorageClass

Do not set `liqo` as the default SC on the tenant. Liqo only reflects
PVCs for pods that actually get offloaded. Infrastructure components
that stay on the tenant (DSPA MariaDB/MinIO, dashboard, monitoring)
need a local SC. If those pods use `liqo`, the PV gets provisioned on
the worker while the pod runs on the tenant, so it cannot mount.

- **GPU workloads** (notebooks, training, inference): use `liqo` SC
- **Tenant infrastructure** (DSPA, dashboard, monitoring): use the
  cloud provider's SC or LVMS

The RHOAI Dashboard lets admins pick the SC when creating notebooks,
so this is a config choice, not a code change.


## RHOAI operator transparency

Every RHOAI operator works unchanged. No patches, no custom scheduling
logic, no nodeName overrides in pod specs. The GPU resource request is
the only signal, and the kube-scheduler handles placement automatically.

Operators tested in the spike:

| Operator | Workload | How it works |
|----------|----------|-------------|
| Training operator | PyTorchJob, TFJob, XGBoostJob | Creates pods with GPU request. Scheduler places on virtual node. Liqo offloads. Operator sees status via Liqo sync. |
| Notebook controller | Workbenches | Creates StatefulSet with GPU request. Jupyter runs on worker GPU. WebSocket tunneled through Submariner. |
| KServe | InferenceService | Creates Deployment with GPU request. Inference runs on worker. Endpoint accessible via Submariner. RawDeployment mode tested. |
| KubeRay | RayJob, RayCluster | Head pod requests GPU, gets offloaded. Worker pods without GPU stay on tenant. |
| TrustyAI | LMEvalJob | Evaluation pod has no GPU request, stays on tenant. Evaluates model locally. |
| DSPA | DataSciencePipelinesApplication | MariaDB, MinIO, API server have no GPU request, stay on tenant. Requires local SC and >= 5Gi MinIO PVC. |

The key property: any new RHOAI workload type that uses `nvidia.com/gpu`
in its pod spec will automatically get offloaded without touching the
dispatch layer.


## Known limitations

1. **PodIP sync is undocumented.** Submariner routing depends on Liqo
   syncing the real worker PodIP to the tenant pod. This works in v1.2.0
   but is not guaranteed in future versions. Validate with tests L3 and
   L13 after every Liqo upgrade.

2. **No Red Hat support for Liqo.** Liqo is an academic/community
   project from Politecnico di Torino. If it breaks, the only recourse
   is upstream GitHub issues or self-patching. Submariner, by contrast,
   is part of RHACM.

3. **Larger worker footprint than custom VK.** A custom Virtual Kubelet
   only needs a ServiceAccount and RBAC. Liqo installs a ShadowPod
   controller, auth module, storage fabric, resource reflector, and VK
   server. More components means more surface area for failure.

4. **Exec returns empty through Liqo VK.** `kubectl exec` through the
   Liqo virtual node returns empty output. Not relevant for end users
   (they use RHOAI Dashboard), but affects debugging.

5. **TrustyAI concurrent reconcile race.** LMEvalJob has a bug where
   two reconcile loops fire simultaneously, one creates the pod, the
   other gets "already exists" and stops reconciling. The job gets stuck
   at "Scheduled". Tests include a retry workaround (delete and recreate
   if stuck for 60s). This is a TrustyAI bug, not Liqo-related.

6. **TFJob needs CUDA base image for Blackwell.** TensorFlow 2.19.0
   lacks sm_120 kernels for Blackwell GPUs (RTX 5090). The spike tests
   use the CUDA base image (`nvcr.io/nvidia/cuda:12.8.1-base-ubi9`)
   with nvidia-smi instead, which validates the dispatch chain without
   depending on framework CUDA kernel compatibility.


## What's tested

32 tests across 9 test files:

| File | Tests | What it covers |
|------|-------|---------------|
| `test_basic_dispatch.py` | L1-L6 | Virtual node, GPU dispatch, PodIP sync, resource reflection, cleanup, log proxy |
| `test_storage.py` | L7-L8 | PVC reflection, SC remapping, persistence across pod restart |
| `test_rhoai.py` | L9-L10 | PyTorchJob and KServe via Liqo |
| `test_kueue.py` | L11-L12 | Kueue admission, ShadowPod label persistence |
| `test_networking.py` | L13-L14 | Submariner service reachability, headless DNS |
| `test_rhoai_kueue.py` | L15-L17 | RHOAI + Kueue end-to-end: admission, queuing, preemption |
| `test_rhoai_smoke.py` | L18-L20 | RHOAI 3.x components present, no CRDs on worker, no OSSM |
| `test_rhoai_workloads.py` | L21-L30 | Notebook, workspace PVC, checkpoint, RayJob, TFJob, XGBoost, RayCluster, DSPA, LMEval |
| `test_rhoai_networking.py` | L31-L33 | KServe/Notebook via Submariner, tunnel recovery |


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

- [Architecture](docs/architecture.md) with Mermaid dispatch diagram,
  Submariner design decision, PVC model, and setup sequence
- [Spike assessment](docs/liqo-spike-assessment.md) with all 32 test
  scenarios, deployment findings, and recommendation
- [Submariner architecture](docs/submariner-architecture.md)


## Setup

Ansible playbooks in `playbooks/` automate Liqo, Submariner, Kyverno,
Kueue, LVMS, and RHOAI installation. See the
[spike assessment](docs/liqo-spike-assessment.md) for configuration
details and deployment findings.
