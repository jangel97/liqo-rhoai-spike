# Liqo Spike Assessment — Can Liqo Replace the Custom VK Provider?

## Objective

Test whether **Liqo v1.2.0** with networking disabled + Submariner can
replace the custom Virtual Kubelet provider for cross-cluster GPU
workload dispatch from RHOAI tenant clusters to a bare GPU worker.

## Lab Environment

Same lab as the VK spike. Liqo installed on two clusters:

- **Liqo:** v1.2.0 (liqoctl v1.2.0, server v1.2.0 on both clusters)
- **Consumer:** sno-tenant-rhoai3 (OCP 4.19, RHOAI 3.x)
- **Provider:** sno-worker (OCP 4.22, RTX 5090, GPU Operator, Kueue)
- **Networking:** Liqo networking disabled, Submariner v0.24 (Libreswan IPsec)

## Liqo Configuration

```bash
# Install on both clusters with networking disabled
liqoctl install openshift \
  --set networking.enabled=false \
  --set storage.enabled=true \
  --set telemetry.enabled=false \
  --set offloading.disableNetworkCheck=true

# On the worker only: disable Liqo's ShadowPod quota enforcement
# so Kueue is the sole admission authority (see Deployment Findings)
helm upgrade liqo liqo/liqo -n liqo --reuse-values \
  --set controllerManager.config.enableResourceEnforcement=false

# Peer with GPU resource — over-provision for Kueue queuing/preemption
# The real quota is enforced by Kueue ClusterQueue on the worker.
# See "GPU over-provisioning for Kueue" in Deployment Findings.
liqoctl peer \
  --kubeconfig=~/.kube/tenant-rhoai3 \
  --remote-kubeconfig=~/.kube/worker \
  --networking-disabled \
  --resource nvidia.com/gpu=10

# Offload test namespace — LocalAndRemote lets the scheduler decide
# GPU pods go to the virtual node, non-GPU pods stay local
liqoctl offload namespace vk-test \
  --pod-offloading-strategy LocalAndRemote \
  --namespace-mapping-strategy EnforceSameName
```

## How to Run

```bash
RHOAI3_KUBECONFIG=~/.kube/tenant-rhoai3 WORKER_KUBECONFIG=~/.kube/worker \
  python -m pytest tests/liqo/ -v

# By category
python -m pytest tests/liqo/ -v -m liqo           # All Liqo tests
python -m pytest tests/liqo/ -v -m rhoai           # RHOAI integration
python -m pytest tests/liqo/ -v -m networking      # Submariner
python -m pytest tests/liqo/ -v -m kueue           # Kueue admission/queuing/preemption
```

## Test Files

| File | Tests | Category |
|------|-------|----------|
| `test_basic_dispatch.py` | L1–L6 | Core dispatch, resource reflection, logs |
| `test_storage.py` | L7–L8 | PVC reflection and persistence |
| `test_rhoai.py` | L9–L10 | PyTorchJob and KServe via Liqo |
| `test_kueue.py` | L11–L12 | Kueue admission, ShadowPod label persistence |
| `test_networking.py` | L13–L14 | Submariner reachability, headless DNS |
| `test_rhoai_kueue.py` | L15–L17 | RHOAI + Kueue end-to-end (admission, queuing, preemption) |
| `test_rhoai_smoke.py` | L18–L20 | RHOAI 3.x component presence, no CRDs on worker, no OSSM |
| `test_rhoai_workloads.py` | L21–L30 | Notebook, workspace PVC, checkpoint, RayJob, TFJob, XGBoost, RayCluster, DSPA, LMEval |
| `test_rhoai_networking.py` | L31–L33 | KServe/Notebook reachable via Submariner, tunnel recovery |
| `conftest.py` | — | Shared fixtures (cluster clients, namespace, virtual node readiness, RHOAI DSC toggle) |
| `helpers.py` | — | `force_delete_pod`, `wait_pod_exists` utilities |

---

## Scenarios

### 1. Core Dispatch

#### L1. Virtual node registration

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_virtual_node_exists` |
| **What** | Liqo registers a virtual node with `liqo.io/type=virtual-node` label, `nvidia.com/gpu` in allocatable, and Ready condition |
| **Result** | PASS |
| **Proves** | Liqo peering works and the tenant scheduler sees GPU capacity |

#### L2. GPU pod dispatch and status sync

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_gpu_pod_dispatched` |
| **What** | GPU pod (no nodeName, no tolerations) → scheduler places on virtual node → Liqo creates ShadowPod on worker → nvidia-smi on RTX 5090 → Succeeded synced back |
| **Result** | PASS |
| **Proves** | Resource-based scheduling dispatches GPU pods to the virtual node without explicit scheduling constraints |

#### L3. PodIP status sync

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_pod_status_sync_podip` |
| **What** | Tenant pod shows PodIP from worker CIDR (10.13x.x.x) — needed for Submariner EndpointSlice routing |
| **Result** | PASS |
| **Proves** | Liqo syncs real worker PodIP to tenant (not remapped) when networking is disabled |

### 2. Resource Reflection

#### L4. Secret and ConfigMap reflection

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_resource_sync` |
| **What** | Secret (via env secretKeyRef) and ConfigMap (via volume) reflected to worker twin namespace |
| **Result** | PASS |
| **Proves** | Liqo resource reflection replaces custom ResourceSyncer |

#### L5. Pod deletion cleanup

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_pod_deletion_cleanup` |
| **What** | Delete tenant pod → ShadowPod deleted → worker pod deleted |
| **Result** | PASS |
| **Proves** | No resource leaks on the worker |

#### L6. Pod logs proxied

| | |
|---|---|
| **Test** | `test_basic_dispatch.py::test_pod_logs_proxied` |
| **What** | kubectl logs from tenant API server returns worker pod output |
| **Result** | PASS |
| **Proves** | Liqo VK proxies log requests to worker |

### 3. Storage

#### L7. PVC reflection with StorageClass remapping

| | |
|---|---|
| **Test** | `test_storage.py::test_pvc_reflection` |
| **What** | PVC with `storageClassName: liqo` → Liqo provisions on worker with LVMS StorageClass (via `--remote-real-storage-class-name=lvms-vg1`) → pod mounts it |
| **Result** | PASS |
| **Proves** | Liqo storage fabric replaces Catapult PVC model |

#### L8. PVC data persistence across pod restarts

| | |
|---|---|
| **Test** | `test_storage.py::test_pvc_persists_across_pod_restart` |
| **What** | Writer pod saves data to PVC → pod deleted → reader pod reads same data from PVC |
| **Result** | PASS |
| **Proves** | PVC lifecycle independent of pod lifecycle (same as Catapult) |

### 4. RHOAI Integration

#### L9. PyTorchJob via Liqo

| | |
|---|---|
| **Test** | `test_rhoai.py::test_pytorchjob_via_liqo` |
| **What** | PyTorchJob CR → training operator creates master pod → scheduler places on virtual node → Liqo offloads → real GPU training on RTX 5090 → Succeeded |
| **Result** | PASS |
| **Proves** | RHOAI training operator works transparently — GPU pods are scheduled to the virtual node automatically |

#### L10. KServe raw inference via Liqo

| | |
|---|---|
| **Test** | `test_rhoai.py::test_kserve_raw_inference_via_liqo` |
| **What** | KServe InferenceService (RawDeployment) → KServe creates Deployment → GPU pod scheduled to virtual node → Liqo offloads → inference server on GPU |
| **Result** | PASS |
| **Proves** | KServe works with Liqo — GPU resource request drives placement, no scheduling annotations needed |

### 5. Kueue Integration

#### L11. Kueue admits Liqo-dispatched pod

| | |
|---|---|
| **Test** | `test_kueue.py::test_kueue_admits_liqo_pod` |
| **What** | GPU pod → scheduler places on virtual node → Liqo offloads → Kyverno adds Kueue labels on worker → Kueue admits → Succeeded |
| **Result** | PASS |
| **Proves** | Kyverno can mutate ShadowPod-created Pods for Kueue admission |

#### L12. ShadowPod label persistence

| | |
|---|---|
| **Test** | `test_kueue.py::test_shadowpod_label_persistence` |
| **What** | Verify whether Kyverno-added labels survive ShadowPod controller reconciliation |
| **Result** | PASS |
| **Proves** | ShadowPod controller does NOT overwrite Kyverno-added labels — Kueue integration is viable |

### 6. RHOAI + Kueue End-to-End

#### L15. PyTorchJob admitted by Kueue with correct labels

| | |
|---|---|
| **Test** | `test_rhoai_kueue.py::test_pytorchjob_kueue_admission` |
| **What** | PyTorchJob → training operator creates pod → scheduler places on virtual node → Liqo offloads → Kyverno adds queue-name and priority labels on worker → Kueue admits → GPU training succeeds |
| **Result** | PASS |
| **Proves** | Full RHOAI + Liqo + Kyverno + Kueue chain works end-to-end |

#### L16. Two PyTorchJobs queued — only 1 GPU, second waits

| | |
|---|---|
| **Test** | `test_rhoai_kueue.py::test_pytorchjob_kueue_queuing` |
| **What** | Two PyTorchJobs each requesting 1 GPU — first admitted and runs, second queued (gated or pending) until first completes |
| **Result** | PASS |
| **Prerequisite** | Virtual node must advertise >= 2 GPUs via ResourceSlice (see [GPU over-provisioning](#gpu-over-provisioning-for-kueue)) |
| **Proves** | Kueue quota enforcement works with scheduler-dispatched RHOAI workloads via Liqo |

#### L17. Priority preemption — high-priority job preempts low-priority

| | |
|---|---|
| **Test** | `test_rhoai_kueue.py::test_pytorchjob_kueue_preemption` |
| **What** | Low-priority PyTorchJob (`gpuaas-opportunistic`) running → high-priority (`gpuaas-production`) submitted → Kueue preempts low → high runs on GPU → Succeeded |
| **Result** | PASS |
| **Prerequisite** | Virtual node must advertise >= 2 GPUs via ResourceSlice (see [GPU over-provisioning](#gpu-over-provisioning-for-kueue)) |
| **Proves** | Cross-priority preemption works with RHOAI workloads via Liqo + Kueue |

### 7. Cross-Cluster Networking

#### L13. GPU service reachable via Submariner

| | |
|---|---|
| **Test** | `test_networking.py::test_service_reachable_via_submariner` |
| **What** | GPU HTTP server → scheduler places on virtual node → Liqo offloads to worker → Service on tenant → curl from local node through Submariner → HTTP 200 with RTX 5090 data |
| **Result** | PASS |
| **Proves** | Submariner cross-cluster routing works with scheduler-dispatched GPU pods (PodIP synced, EndpointSlice valid) |

#### L14. Headless Service reflection for distributed training DNS

| | |
|---|---|
| **Test** | `test_networking.py::test_headless_service_reflection` |
| **What** | Headless Service reflected to worker → Pod B resolves Pod A's DNS via synced Service |
| **Result** | PASS |
| **Proves** | Liqo reflects headless Services for inter-pod DNS (distributed training prerequisite) |

### 8. RHOAI 3.x Smoke

#### L18. No RHOAI CRDs on worker

| | |
|---|---|
| **Test** | `test_rhoai_smoke.py::test_no_rhoai_crds_on_worker` |
| **What** | Worker cluster has no RHOAI CRDs (PyTorchJob, Notebook, InferenceService, RayCluster) — only the tenant installs RHOAI |
| **Result** | PASS |
| **Proves** | Worker remains a bare GPU provider — no RHOAI operator footprint |

#### L19. RHOAI 3.x components running

| | |
|---|---|
| **Test** | `test_rhoai_smoke.py::test_rhoai3_components_running` |
| **What** | `kubeflow-training-operator` and `kserve-controller-manager` deployments ready in `redhat-ods-applications` |
| **Result** | PASS |
| **Proves** | RHOAI 3.x operators are healthy on the Liqo-enabled tenant |

#### L20. No OSSM on RHOAI 3.x

| | |
|---|---|
| **Test** | `test_rhoai_smoke.py::test_no_ossm_on_rhoai3` |
| **What** | No `servicemeshcontrolplanes.maistra.io` CRD on tenant — RHOAI 3.x removed Service Mesh dependency |
| **Result** | PASS |
| **Proves** | RHOAI 3.x runs without OSSM (simplifies Liqo + Submariner networking) |

### 9. RHOAI 3.x Workloads

#### L21. Notebook CR dispatched via Liqo

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_notebook_cr_via_liqo` |
| **What** | Notebook CR → notebook controller creates StatefulSet → pod scheduled to virtual node → Liqo offloads → GPU HTTP server on RTX 5090 → Running with Ready status |
| **Result** | PASS |
| **Proves** | Notebook CR lifecycle works transparently through Liqo — no scheduling changes needed |

#### L22. Notebook with workspace PVC (RHOAI Dashboard flow)

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_notebook_with_workspace_pvc` |
| **What** | PVC (`storageClassName: liqo`) created first → Notebook CR mounts it at `/opt/app-root/src` → pod Running → PVC reflected to worker with SC remapped to `lvms-vg1` (Bound) → nvidia-smi output saved to workspace → data persists on reflected PVC |
| **Result** | PASS |
| **Proves** | RHOAI Dashboard storage workflow works end-to-end through Liqo PVC reflection + SC remapping. Operators creating PVCs with `storageClassName: liqo` get transparent remote provisioning. |

#### L23. PyTorchJob with checkpoint PVC

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_pytorchjob_checkpoint_with_pvc` |
| **What** | PyTorchJob trains on GPU → saves `checkpoint.pt` to Liqo PVC → second pod reads checkpoint from same PVC → validates model state dict |
| **Result** | PASS |
| **Proves** | Training checkpoint persistence on Liqo-reflected PVCs — standard RHOAI training workflow |

#### L24. RayJob dispatched via Liqo

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_rayjob_via_liqo` |
| **What** | RayJob CR → KubeRay creates RayCluster → head pod with GPU → scheduler places on virtual node → Liqo offloads → nvidia-smi on worker GPU → Running |
| **Result** | PASS |
| **Prerequisite** | `rhoai_full_environment` fixture enables `ray` DSC component |
| **Proves** | RayJob GPU dispatch works through Liqo — KubeRay operator creates pods normally |

#### L25. TFJob dispatched via Liqo

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_tfjob_via_liqo` |
| **What** | TFJob CR → training operator creates chief pod → GPU pod scheduled to virtual node → Liqo offloads → nvidia-smi on worker GPU → Succeeded |
| **Result** | PASS |
| **Proves** | TFJob dispatch works through Liqo. Uses CUDA base image with nvidia-smi instead of TensorFlow to avoid framework CUDA kernel compatibility issues (Blackwell sm_120). |

#### L26. XGBoostJob dispatched via Liqo

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_xgboostjob_via_liqo` |
| **What** | XGBoostJob CR → training operator creates master pod → XGBoost GPU training (`device: cuda`, `tree_method: hist`) → Succeeded |
| **Result** | PASS |
| **Proves** | XGBoostJob GPU dispatch works through Liqo — pip-installs xgboost at runtime and trains on GPU |

#### L27. RayCluster dispatched via Liqo

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_raycluster_via_liqo` |
| **What** | Standalone RayCluster CR → head pod with GPU → scheduled to virtual node → Liqo offloads → Running on worker |
| **Result** | PASS |
| **Prerequisite** | `rhoai_full_environment` fixture enables `ray` DSC component |
| **Proves** | Standalone RayCluster (not RayJob) works through Liqo for interactive Ray sessions |

#### L28. DSPA stays local on tenant

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_dspa_on_liqo_cluster` |
| **What** | DataSciencePipelinesApplication CR → MariaDB, MinIO, API server deployments → no GPU request → pods stay on tenant node → DSPA Ready |
| **Result** | PASS |
| **Prerequisite** | Tenant needs a local StorageClass (e.g. `lvms-vg1`) for MariaDB/MinIO PVCs. MinIO requires >= 5Gi PVC (minimum free disk threshold). |
| **Proves** | Non-GPU RHOAI infrastructure stays local with `LocalAndRemote` + GPU taint — does not get dispatched to worker |

#### L29. LMEvalJob on Liqo cluster

| | |
|---|---|
| **Test** | `test_rhoai_workloads.py::test_lmeval_on_liqo_cluster` |
| **What** | KServe serves opt-125m on CPU (local) → LMEvalJob evaluates model via completions API → runs arc_easy benchmark (2 samples) → Complete with Succeeded |
| **Result** | PASS |
| **Proves** | LMEval stays local (no GPU request), evaluates model served locally — TrustyAI integration works on Liqo-enabled cluster |

### 10. RHOAI 3.x Networking (Submariner)

#### L30. KServe inference reachable via Submariner

| | |
|---|---|
| **Test** | `test_rhoai_networking.py::test_kserve_inference_reachable_via_submariner` |
| **What** | KServe RawDeployment with GPU → pod on worker → PodIP synced to tenant → EndpointSlice has worker PodIP → curl from tenant through Submariner → HTTP 200 with RTX 5090 JSON |
| **Result** | PASS |
| **Proves** | KServe inference endpoints are reachable cross-cluster through Submariner — full serving path validated |

#### L31. Notebook reachable via Submariner

| | |
|---|---|
| **Test** | `test_rhoai_networking.py::test_notebook_reachable_via_submariner` |
| **What** | Notebook CR with GPU → pod on worker → PodIP synced → Service + label selector → curl from tenant through Submariner → HTTP 200 with RTX 5090 JSON |
| **Result** | PASS |
| **Proves** | GPU Notebooks dispatched via Liqo are reachable from tenant network — users can connect to remote notebooks transparently |

#### L32. Submariner tunnel failure recovery

| | |
|---|---|
| **Test** | `test_rhoai_networking.py::test_submariner_tunnel_recovery` |
| **What** | GPU pod reachable → delete Submariner gateway pod on worker → gateway auto-restarts → wait for Ready → curl succeeds again |
| **Result** | PASS |
| **Proves** | Submariner self-heals after gateway failure — tunnel recovery is automatic without user intervention |

---

## Comparison: Custom VK vs Liqo

| Capability | Custom VK | Liqo | Notes |
|------------|:---------:|:----:|-------|
| **GPU dispatch** | S2 PASS | L2 PASS | |
| **Resource sync** | S3 PASS | L4 PASS | Custom syncer vs built-in reflection |
| **Cleanup** | S4 PASS | L5 PASS | Per-pod vs ShadowPod lifecycle |
| **PVC** | S5 PASS | L7 PASS | Catapult opt-in vs Liqo StorageClass |
| **PVC rejection** | S6 PASS | — | Liqo reflects all PVCs (no opt-in gating) |
| **PyTorchJob** | S7 PASS | L9 PASS | |
| **Worker stays bare** | S8 PASS | — | Worker needs Liqo components |
| **Submariner** | S9 PASS | L13 PASS | |
| **Notebook** | S10 PASS | L21 PASS | |
| **Tunnel recovery** | S11 PASS | L32 PASS | |
| **Multi-tenant** | S12 PASS | — | Not tested in Liqo spike |
| **Headless DNS** | S13 PASS | L14 PASS | |
| **Logs/exec** | S14 PASS | L6 PASS | |
| **OCP version decoupling** | S15 PASS | PASS | Tenant OCP 4.19, Worker OCP 4.22 — different versions |
| **Real training** | S16 PASS | via L9 PASS | |
| **PVC persistence** | S17 PASS | L8 PASS | |
| **Notebook CR** | S18 PASS | L21 PASS | |
| **KServe raw** | S19 PASS | L10 PASS | |
| **KServe serverless** | S20 PASS | — | RHOAI 3.x has no Knative — only RawDeployment tested |
| **Kueue admission** | S21 PASS | L11 PASS | Same Kyverno approach works with ShadowPod-created pods |
| **Kueue queuing** | S22 PASS | L16 PASS | Requires GPU over-provisioning on virtual node |
| **Kueue preemption** | S23 PASS | L17 PASS | Requires GPU over-provisioning on virtual node |
| **PodIP sync** | PASS | L3 PASS | Critical for Submariner routing |
| **Workspace PVC** | — | L22 PASS | RHOAI Dashboard storage flow (PVC + Notebook CR) |
| **Checkpoint PVC** | S17 PASS | L23 PASS | Training checkpoint persistence |
| **RayJob** | — | L24 PASS | KubeRay job dispatch |
| **RayCluster** | — | L27 PASS | Standalone Ray cluster |
| **XGBoostJob** | — | L26 PASS | XGBoost GPU training |
| **TFJob** | — | L25 PASS | Uses CUDA base image to avoid framework kernel gaps |
| **DSPA local** | — | L28 PASS | Non-GPU infra stays local (needs local SC + 5Gi PVC) |
| **LMEval** | — | L29 PASS | LMEval stays local, evaluates locally served model |
| **KServe net** | — | L30 PASS | KServe inference reachable via Submariner |
| **Notebook net** | — | L31 PASS | Notebook reachable via Submariner |
| **Istio sidecar** | S24 PASS | N/A | RHOAI 3.x dropped OSSM; see deployment finding below |

---

## Architecture

See [Architecture](architecture.md) for the full system design,
including the Mermaid dispatch diagram, Submariner design decision,
PVC model, setup sequence, and architecture comparison table.

---

## Results Summary

```
tests/liqo/test_basic_dispatch.py::test_virtual_node_exists              PASSED
tests/liqo/test_basic_dispatch.py::test_gpu_pod_dispatched               PASSED
tests/liqo/test_basic_dispatch.py::test_resource_sync                    PASSED
tests/liqo/test_basic_dispatch.py::test_pod_deletion_cleanup             PASSED
tests/liqo/test_basic_dispatch.py::test_pod_logs_proxied                 PASSED
tests/liqo/test_basic_dispatch.py::test_pod_status_sync_podip            PASSED
tests/liqo/test_kueue.py::test_kueue_admits_liqo_pod                     PASSED
tests/liqo/test_kueue.py::test_shadowpod_label_persistence               PASSED
tests/liqo/test_networking.py::test_service_reachable_via_submariner      PASSED
tests/liqo/test_networking.py::test_headless_service_reflection           PASSED
tests/liqo/test_rhoai.py::test_pytorchjob_via_liqo                       PASSED
tests/liqo/test_rhoai.py::test_kserve_raw_inference_via_liqo             PASSED
tests/liqo/test_rhoai_kueue.py::test_pytorchjob_kueue_admission          PASSED
tests/liqo/test_rhoai_kueue.py::test_pytorchjob_kueue_queuing            PASSED
tests/liqo/test_rhoai_kueue.py::test_pytorchjob_kueue_preemption         PASSED
tests/liqo/test_rhoai_networking.py::test_kserve_inference_reachable_via_submariner  PASSED
tests/liqo/test_rhoai_networking.py::test_notebook_reachable_via_submariner          PASSED
tests/liqo/test_rhoai_networking.py::test_submariner_tunnel_recovery                 PASSED
tests/liqo/test_rhoai_smoke.py::test_no_rhoai_crds_on_worker             PASSED
tests/liqo/test_rhoai_smoke.py::test_rhoai3_components_running           PASSED
tests/liqo/test_rhoai_smoke.py::test_no_ossm_on_rhoai3                   PASSED
tests/liqo/test_rhoai_workloads.py::test_notebook_cr_via_liqo            PASSED
tests/liqo/test_rhoai_workloads.py::test_notebook_with_workspace_pvc     PASSED
tests/liqo/test_rhoai_workloads.py::test_pytorchjob_checkpoint_with_pvc  PASSED
tests/liqo/test_rhoai_workloads.py::test_rayjob_via_liqo                 PASSED
tests/liqo/test_rhoai_workloads.py::test_tfjob_via_liqo                  PASSED
tests/liqo/test_rhoai_workloads.py::test_xgboostjob_via_liqo             PASSED
tests/liqo/test_rhoai_workloads.py::test_raycluster_via_liqo             PASSED
tests/liqo/test_rhoai_workloads.py::test_dspa_on_liqo_cluster            PASSED
tests/liqo/test_rhoai_workloads.py::test_lmeval_on_liqo_cluster          PASSED
tests/liqo/test_storage.py::test_pvc_reflection                          PASSED
tests/liqo/test_storage.py::test_pvc_persists_across_pod_restart         PASSED

Liqo v1.2.0 — 32 passed — 2026-10-15 (full suite, RHOAI 3.x)
Liqo v1.2.0 — 17 passed — 2026-10-16 (core tests, Remote strategy)
Liqo v1.2.0 — 17 passed — 2026-10-15 (core tests, LocalAndRemote strategy)
Liqo v1.2.0 — 17 passed — 2026-10-15 (core tests, LocalAndRemote + GPU taint + Kyverno)
```

All 17 core tests pass under all three configurations on Liqo v1.2.0:

- **Remote strategy:** Liqo webhook forces all pods to the virtual
  node. Simple but prevents mixed-workload namespaces.
- **LocalAndRemote strategy:** Kube-scheduler places GPU pods on the
  virtual node (only node with `nvidia.com/gpu`), non-GPU pods stay
  local. Suitable for a platform where users deploy both GPU and
  non-GPU workloads in the same namespace.
- **LocalAndRemote + GPU taint:** Virtual node tainted with
  `nvidia.com/gpu=present:NoSchedule`. Kyverno on the tenant injects
  the matching toleration (plus Liqo's virtual node toleration) into
  GPU pods. Guarantees deterministic placement — non-GPU pods cannot
  reach the virtual node even if the scheduler would score it
  favorably. RHOAI operator-created pods (PyTorchJob L9/L15-L17,
  KServe L10) receive the toleration via Kyverno and are NOT stripped
  by operator reconciliation.

The full RHOAI 3.x suite (32 tests, all passing) validates end-to-end
integration across all RHOAI workload types: Notebook CR (L21),
Notebook with workspace PVC (L22), PyTorchJob with checkpoint PVC
(L23), RayJob (L24), TFJob (L25), XGBoostJob (L26), RayCluster (L27),
DSPA (L28), LMEval (L29), KServe inference via Submariner (L30),
Notebook networking via Submariner (L31), and Submariner tunnel
recovery (L32). The workspace PVC test (L22) proves that the RHOAI
Dashboard storage flow works transparently through Liqo PVC
reflection — operators create PVCs with `storageClassName: liqo`
and the storage is provisioned on the worker with the correct SC.

Full RHOAI + Kueue chain validated: PyTorchJob → scheduler places on
virtual node → Liqo offload → Kyverno labels → Kueue
admission/queuing/preemption. ShadowPod controller does not overwrite
Kyverno-added labels. Queuing and preemption require GPU
over-provisioning on the virtual node and
`enableResourceEnforcement=false` on the worker (see Deployment
Findings).

---

## Key Findings

Non-obvious discoveries from the spike on **Liqo v1.2.0** — things you
won't find in Liqo's documentation and that required live testing to
confirm. These findings are version-specific and should be re-validated
on any Liqo upgrade.

1. **Kueue integration is viable — ShadowPod controller preserves
   Kyverno-mutated labels** (L12). This was the critical unknown. The
   ShadowPod controller reconciles pod spec from the tenant but does NOT
   overwrite labels added by admission webhooks on the worker. This
   means the same Kyverno + Kueue policy layer we built for the custom
   VK works unchanged with Liqo.

2. **Liqo networking is incompatible with OpenShift** (design decision).
   The WireGuard gateway needs `privileged` SCC + `hostNetwork` +
   `NET_ADMIN`/`SYS_ADMIN`, and the v1.2.0 SCC template has a rendering
   bug ([#3385](https://github.com/liqotech/liqo/issues/3385)). We
   disable it entirely and use Submariner for cross-cluster traffic.

3. **PodIP sync enables Submariner when Liqo networking is off** (L3,
   L13). Liqo syncs the real worker PodIP (no IPAM remapping) to the
   tenant pod status. Submariner routes to this IP through IPsec. If
   Liqo networking were enabled, it would remap PodIPs and break
   Submariner routing.

4. **Two fixes required to make Kueue the sole admission authority**
   (L16, L17). The pod dispatch chain has two gatekeepers before Kueue:
   the tenant scheduler (capacity check on virtual node GPU count) and
   Liqo's `shadowpod.validate.liqo.io` webhook (peering quota check).
   Both must be bypassed: inflate virtual node GPUs via ResourceSlice,
   and disable resource enforcement on the worker
   (`enableResourceEnforcement=false`). Without both, pods are blocked
   before Kueue sees them.

5. **PVC StorageClass, not annotation** (L7). Liqo docs mention a
   `liqo.io/remote-storage-class` annotation — this did not work. The
   correct model is `storageClassName: liqo` with
   `--remote-real-storage-class-name` at peering time.

6. **GPU resource count is static** (L1). `--resource nvidia.com/gpu=N`
   at peering time is a manual number, not auto-discovered from the
   worker. Stored in the `ResourceSlice` CRD. Liqo's VK reconciles node
   status from ResourceSlice — manual node patches are overwritten.

7. **SCC pass-through is transparent** (manual verification). Liqo does
   not modify pod security contexts. The worker's SCC admission
   controller applies its own policy (pods ran under `privileged` SCC
   assigned to the ShadowPod service account).

8. **Kueue webhook reinvocationPolicy: IfNeeded is required** (L11).
   Kyverno adds labels first, then Kueue must re-invoke to see them.
   Without `IfNeeded`, Kueue doesn't see the queue-name label and
   doesn't create a Workload.

9. **RHOAI operators work transparently** (L9, L10). The training
   operator and KServe create pods normally — no `nodeName`, no
   tolerations, no VK annotations. GPU resource requests drive
   scheduler placement to the virtual node automatically.

10. **Resource-based scheduling works with `LocalAndRemote`** (all 17
    tests). Switching from `Remote` (all pods go remote) to
    `LocalAndRemote` (scheduler decides) does not break any test. GPU
    pods are placed on the virtual node because it is the only node
    with `nvidia.com/gpu` in allocatable. PVC tests (L7, L8) also pass
    — the scheduler's `VolumeBinding` plugin does not reject the
    virtual node despite missing CSINode (Liqo's `liqo` StorageClass
    with `WaitForFirstConsumer` defers binding past initial scheduling).

11. **GPU taint + Kyverno toleration injection validated for
    deterministic placement** (all 17 tests). With `LocalAndRemote`,
    non-GPU pods can land on the virtual node if the scheduler scores
    it favorably. Adding `nvidia.com/gpu=present:NoSchedule` to the
    virtual node and using a Kyverno ClusterPolicy on the tenant to
    inject the matching toleration into GPU pods guarantees only GPU
    pods reach the virtual node. Critically, RHOAI operator-created
    pods (PyTorchJob via training operator, KServe Deployment) receive
    the Kyverno-injected tolerations and operators do NOT strip them
    during reconciliation — L9, L10, L15-L17 all pass.

12. **Kyverno `reinvocationPolicy: IfNeeded` interacts with Liqo's
    webhook** (discovered during GPU taint testing). When both Kyverno
    and Liqo's `pod.mutate.liqo.io` webhook mutate the same pod,
    Kyverno re-runs after Liqo due to `reinvocationPolicy: IfNeeded`.
    Kyverno's `patchStrategicMerge` for tolerations can overwrite
    Liqo's toleration if Kyverno only injects the GPU toleration. Fix:
    the Kyverno policy must inject BOTH tolerations
    (`nvidia.com/gpu` and `virtual-node.liqo.io/not-allowed`) so the
    final toleration list is correct regardless of webhook ordering.
    See [GPU taint + Kyverno toleration injection](#gpu-taint--kyverno-toleration-injection-on-tenant)
    in Deployment Findings.

13. **RHOAI Dashboard storage flow works transparently** (L22). PVCs
    created by the RHOAI Dashboard with `storageClassName: liqo` are
    reflected to the worker with the SC remapped to `lvms-vg1`. The
    Notebook CR mounts the PVC, GPU code writes to it, and data
    persists on the reflected PVC. This is the standard RHOAI workflow
    for notebook workspaces, and it requires no changes to Dashboard
    behavior — operators select `liqo` as the StorageClass when
    creating notebooks.

14. **Liqo PVC StorageClass remapping is provider-agnostic** (L7, L22).
    On the tenant, PVCs use `storageClassName: liqo`. Liqo remaps to
    whatever the worker's default SC is — `lvms-vg1` in our lab, but
    `gp3` on AWS, `pd-ssd` on GCP, `managed-premium` on Azure, etc.
    There is also a `liqo.io/remote-storage-class` annotation for
    explicit SC targeting on the worker, but the default remapping is
    sufficient for most cases. The tenant does not need to know the
    worker's storage backend.

15. **Do NOT set `liqo` as the default StorageClass on the tenant.**
    The `liqo` SC provisions storage on the remote worker. Liqo only
    reflects PVCs for pods that are offloaded. Infrastructure
    components (DSPA MariaDB/MinIO, dashboard, monitoring) stay on the
    tenant. If those pods use `liqo` SC, the PV gets created on the
    worker while the pod is on the tenant — the mount fails. Keep the
    cloud provider's SC as default on the tenant for local infra, and
    use `liqo` SC explicitly for GPU workload storage.

16. **Exec through Liqo proxy returns empty responses.** The
    `kubernetes.stream.stream()` API for pod exec through the Liqo VK
    returns empty strings (`''`). Liqo's exec proxy buffers differently
    from the real kubelet. Workaround: exec directly to the worker pod
    (find it by name substring in the worker namespace). This is only
    relevant for tests — end users access notebooks through the RHOAI
    Dashboard (HTTP route), not through exec.

17. **TrustyAI operator has a pod-creation race condition** (L29). The
    LMEvalJob controller fires multiple concurrent reconcile loops when
    a new LMEvalJob is created. Both loops attempt to create the pod in
    the same second. The first succeeds; the second fails with
    `pods "X" already exists` and then fails to update the LMEvalJob
    status (`the object has been modified`). After this, the operator
    stops reconciling and the job stays stuck at `Scheduled` forever,
    even though the pod exists and eventually becomes Running. This
    happens even with clean state (no stale resources). Pre-cleanup
    helps reduce the frequency by removing leftovers from prior runs,
    but cannot prevent the internal race. The test uses a retry: if the
    job is stuck at `Scheduled` for 60s, it deletes and recreates the
    LMEvalJob. This is a TrustyAI operator bug, not Liqo-related.

18. **TFJob works via CUDA base image, not TensorFlow** (L25). The
    TFJob test uses `nvcr.io/nvidia/cuda:12.8.1-base-ubi9` with
    nvidia-smi instead of TensorFlow. This validates the training
    operator + Liqo dispatch chain without depending on framework CUDA
    kernel compatibility. TensorFlow 2.19.0-gpu lacks sm_120 (Blackwell)
    kernels, so actual TF GPU training fails on RTX 5090. The test
    proves TFJob dispatch works; TF kernel support is a framework issue.

19. **DSPA needs a local StorageClass and >= 5Gi MinIO PVC** (L28). DSPA
    pods (MariaDB, MinIO, API server) stay local on the tenant because
    they have no GPU request. But they need PVCs on local storage, not
    Liqo. The tenant must have a local StorageClass (e.g. `lvms-vg1`).
    MinIO also requires >= 5Gi PVC to pass its minimum free disk
    threshold check; 1Gi causes CrashLoopBackOff.

20. **LMEval stays local on a Liqo-enabled cluster** (L29). The
    LMEvalJob CRD has no fields for tolerations or `nodeName`. With
    `LocalAndRemote` + GPU taint, the LMEval pod has no GPU request and
    no GPU toleration, so the scheduler keeps it on the tenant node.
    The model server (KServe CPU) also stays local. Both communicate
    via ClusterIP service without crossing the Submariner tunnel. This
    is the correct behavior — LMEval evaluates model quality, it does
    not need GPU access itself.

---

## Deployment Findings

### PVC StorageClass — not annotation

Liqo docs mention a `liqo.io/remote-storage-class` annotation on PVCs.
This did not work in our testing. The correct model is:

1. At peering time, configure `--remote-real-storage-class-name=lvms-vg1`
2. Liqo creates a virtual StorageClass called `liqo` on the tenant
3. PVCs use `storageClassName: liqo` — Liqo reflects them to the worker
   with `storageClassName: lvms-vg1`

### Peering: GPU resource advertisement

```bash
liqoctl peer \
  --kubeconfig=~/.kube/tenant \
  --remote-kubeconfig=~/.kube/worker \
  --networking-disabled \
  --resource nvidia.com/gpu=1
```

The `--resource nvidia.com/gpu=1` flag controls how many GPUs appear on
the virtual node's `allocatable`. This is a static advertisement — Liqo
does not auto-discover GPU count from the worker. If the worker has 4
GPUs, you must specify `=4` explicitly.

### Submariner + Liqo networking disabled

See [Design Decision: Submariner over Liqo Networking](#design-decision-submariner-over-liqo-networking)
in the Architecture section for the full rationale. In short: Liqo's
WireGuard gateway requires `privileged` SCC + host networking +
`NET_ADMIN`/`SYS_ADMIN`, and the v1.2.0 SCC template has a rendering
bug. Submariner uses Libreswan IPsec with proper SCC management and
Red Hat support.

### SCC pass-through

Liqo passes the pod spec through to the worker without modifying the
security context. The worker's SCC admission controller applies its own
policy to the actual pod — same behavior as the custom VK.

Verified: a pod created on the tenant with no explicit `securityContext`
ran on the worker under the `privileged` SCC (assigned by the worker's
admission controller based on the ShadowPod service account). Liqo does
not interfere with or modify security contexts.

### Kueue setup on worker

Kyverno + Kueue configuration required on the worker for the twin
namespace (the namespace Liqo creates to mirror the tenant's offloaded
namespace):

1. Kyverno installed on worker with privileged SCC for its service
   accounts (required on OpenShift)
2. Kyverno MutatingPolicies from `kueue-gpu-policy` repo applied:
   `mutate-queue-name` and `mutate-priority` (match namespaces with
   `gpuaas.redhat.com/managed=true`)
3. WorkloadPriorityClasses created (`gpuaas-production`, `gpuaas-critical`,
   `gpuaas-standard`, `gpuaas-opportunistic`)
4. Twin namespace labeled:
   - `gpuaas.redhat.com/managed=true`
   - `kyverno.io/watch=enabled`
   - `gpuaas.redhat.com/default-priority=gpuaas-standard`
   - annotation: `gpuaas.redhat.com/allowed-priorities=<comma-separated>`
5. LocalQueue `default` in twin namespace pointing to `cluster-queue`
6. Kueue webhook `reinvocationPolicy: IfNeeded` (Kyverno runs first,
   Kueue re-invokes to see the labels)

The ShadowPod controller does **not** overwrite Kyverno-added labels
during reconciliation. This was the critical unknown — L12 confirms
Kueue integration is viable.

### GPU over-provisioning for Kueue

Making Kueue the sole admission authority requires two fixes. Without
both, pods are blocked before Kueue sees them and queuing/preemption
cannot work.

**The ordering problem:** The pod dispatch chain is
`tenant scheduler → Liqo ShadowPod webhook → Kueue (worker) → worker scheduler → GPU`.
Both the tenant scheduler (capacity check) and Liqo's ShadowPod
validating webhook (`shadowpod.validate.liqo.io`, quota check) make
resource-based decisions before Kueue can make policy-based decisions
(priority, queuing, preemption, borrowing).

**Fix 1: Inflate virtual node GPU count via ResourceSlice.**

The virtual node's GPU count comes from `--resource nvidia.com/gpu=N`
at peering time, stored in the `ResourceSlice` CRD. When the virtual
node advertises the real GPU count (e.g., `=1`), the tenant scheduler
blocks new GPU pods once capacity is exhausted. The pod stays Pending
on the tenant and never gets offloaded.

```bash
# At peering time: advertise more GPUs than physically available
liqoctl peer ... --resource nvidia.com/gpu=10

# Or patch the ResourceSlice post-peering:
oc patch resourceslice.authentication.liqo.io gpu-worker \
  -n liqo-tenant-gpu-worker \
  --type=merge -p '{"spec":{"resources":{"nvidia.com/gpu":"10"}}}'
```

**Fix 2: Disable Liqo resource enforcement on the worker.**

Even with an inflated ResourceSlice, Liqo's ShadowPod validating
webhook counts GPU usage against the peering quota and rejects pods
that exceed it:

```
admission webhook "shadowpod.validate.liqo.io" denied the request:
peering nvidia.com/gpu quota usage exceeded - free 2 / requested 4
```

An additional problem: Liqo counts **completed** pods against the
peering quota until they are deleted from the tenant. So even finished
pods consume quota.

Disable the webhook's quota check on the **worker** cluster:

```bash
helm upgrade liqo liqo/liqo -n liqo --reuse-values \
  --set controllerManager.config.enableResourceEnforcement=false
```

**Both fixes together** make Kueue the sole admission authority:

- Fix 1 without Fix 2: tenant scheduler accepts the pod, but Liqo's
  webhook still rejects it on the worker
- Fix 2 without Fix 1: webhook is disabled, but the pod never reaches
  the worker because the tenant scheduler already rejected it

With both applied, the virtual node is just a proxy — its resource
count controls how many pods the tenant scheduler will offload, not
how many GPUs actually exist. The real scheduling constraint is the
Kueue ClusterQueue on the worker (which has `nvidia.com/gpu: 1`
nominalQuota). This way:

- Multiple GPU pods arrive on the worker simultaneously
- Kueue creates Workload objects for each
- Kueue admits based on real quota and priority
- Queuing (L16) and preemption (L17) work correctly

**Multi-tenant note:** In a multi-tenant deployment,
`enableResourceEnforcement` could be re-enabled as a secondary safety
net (defense in depth) with each tenant's ResourceSlice set to their
actual GPU allocation. This creates two layers: Liqo (hard cap per
tenant) and Kueue (policy per tenant with queuing/preemption/borrowing).
The trade-off is that re-enabling it reintroduces the ordering problem
— Kueue preemption and borrowing won't work for pods exceeding the
per-tenant ResourceSlice.

### Liqo v1.2.0 PVC bugs — why we didn't hit them

A parallel spike on IBM Cloud VPC found three PVC bugs in Liqo v1.2.0.
Our tests L7 and L8 passed without encountering any of them. Here is
why each bug doesn't apply to our setup, and what to watch for.

**Bug 1: Nil pointer panic with `volumeBindingMode: Immediate`.**
Creating a PVC with a StorageClass using `volumeBindingMode: Immediate`
and provisioner `liqo.io/storage` panics the Liqo controller-manager
(`IsVirtualNode()` called on nil node at `nodeUtils.go:79`). With
`Immediate` binding, there is no `volume.kubernetes.io/selected-node`
annotation, and the provisioner dereferences nil.

**Why we're safe:** Liqo's virtual StorageClass `liqo` uses
`WaitForFirstConsumer` binding mode, which is the only mode Liqo
supports. Our worker's LVMS StorageClass (`lvms-vg1`) also uses
`WaitForFirstConsumer`. The bug is only triggered by manually creating
a StorageClass with `Immediate` mode — we never do this.

**Bug 2: No CSINode entry for the VirtualNode.** The kube-scheduler's
`VolumeBinding` plugin requires a CSINode object to validate that a
node can provision volumes. The Liqo VirtualNode has no CSINode entry,
so the scheduler rejects it as a candidate for pods with PVCs.

**Why we're safe:** With `LocalAndRemote` strategy, the scheduler runs
but Liqo's `liqo` StorageClass uses `WaitForFirstConsumer` binding
mode, which defers volume binding past initial scheduling — the
`VolumeBinding` plugin does not reject the virtual node despite missing
CSINode. Confirmed by L7/L8 passing under both `Remote` and
`LocalAndRemote` strategies. The sibling spike hit this because they
used `nodeName` pinning with `Immediate` binding mode on IBM Cloud VPC
Block CSI.

**Bug 3: Race condition — ShadowPod created before PVC reflected.**
The ShadowPod is created on the worker before Liqo's resource reflector
has reflected the PVC. The worker's scheduler sees
`persistentvolumeclaim not found` and does not retry after the PVC
eventually appears.

**Why we didn't observe it:** Likely timing-dependent. LVMS on bare
metal provisions faster than IBM Cloud VPC Block CSI, and our test
PVCs are small (1Gi). The PVC reflection may complete before the
ShadowPod controller creates the worker pod. However, this is a race
condition — it could surface under load or with larger PVCs. If PVC
tests become flaky, this bug is the likely cause. Workaround: delete
and recreate the pod after the PVC appears on the worker.

**Assessment:** None of these bugs block our current use case. Bug 1
is avoidable by never using `Immediate` binding mode (which Liqo
doesn't support anyway). Bug 2 is avoided by Liqo's `WaitForFirstConsumer`
binding mode. Bug 3 is a latent race that hasn't manifested but should be
monitored — it is the most likely to surface in production with
slower storage backends or concurrent PVC creation.

### GPU taint + Kyverno toleration injection on tenant

With `LocalAndRemote`, non-GPU pods can land on the virtual node
because Liqo's webhook adds tolerations to ALL pods in the offloaded
namespace. To guarantee only GPU pods reach the virtual node:

**Step 1: Taint the virtual node.**

```bash
oc taint nodes gpu-worker nvidia.com/gpu=present:NoSchedule
```

The virtual node now has two taints:
- `virtual-node.liqo.io/not-allowed=true:NoExecute` (Liqo default)
- `nvidia.com/gpu=present:NoSchedule` (added for deterministic placement)

**Step 2: Deploy a Kyverno ClusterPolicy on the tenant.**

Manifest: [`deploy/tenant-gpu-toleration-policy.yml`](../deploy/tenant-gpu-toleration-policy.yml)

The policy matches pods with `nvidia.com/gpu` in any container's
`resources.limits` and injects both tolerations via
`patchStrategicMerge`.

**Why both tolerations:** Kyverno's `reinvocationPolicy: IfNeeded`
causes it to re-run after Liqo's webhook mutates the pod. On
reinvocation, `patchStrategicMerge` for tolerations can overwrite
Liqo's toleration if Kyverno only injects the GPU toleration. By
injecting both, the final toleration list is correct regardless of
webhook ordering. Without the Liqo toleration, GPU pods fail with:

```
0/2 nodes are available: 1 Insufficient nvidia.com/gpu,
1 node(s) had untolerated taint {virtual-node.liqo.io/not-allowed: true}
```

**Operator reconciliation does not interfere:** The training operator
(PyTorchJob) and KServe controller create pods via their own
reconciliation loops. Kyverno intercepts the pod creation and injects
tolerations. The operators do NOT strip or overwrite tolerations on
subsequent reconciliation cycles. Validated by L9 (PyTorchJob), L10
(KServe), and L15-L17 (PyTorchJob + Kueue) all passing with the GPU
taint active.

### Tenant local storage for non-GPU infrastructure

The tenant cluster needs a local StorageClass in addition to Liqo's
`liqo` SC. RHOAI infrastructure components that stay on the tenant
(DSPA MariaDB/MinIO, dashboard, monitoring) create PVCs that must bind
locally. Without a local SC, these PVCs either bind to Liqo (storage on
worker, pod on tenant, mount fails) or have no SC at all.

In our lab: LVMS operator with `lvms-vg1` StorageClass. Set as default
so infrastructure PVCs pick it up automatically. GPU workload PVCs use
`storageClassName: liqo` explicitly.

**Installation note:** The lab clusters have `disableAllDefaultSources:
true` on the OperatorHub (operators were pre-installed during
provisioning, then catalog sources were disabled). Installing LVMS
required temporarily re-enabling default catalog sources:

```bash
# Enable catalogs, install operator, then disable again
oc patch operatorhub cluster --type=merge \
  -p '{"spec":{"disableAllDefaultSources":false}}'
# ... wait for CSV Succeeded, create LVMCluster ...
oc patch operatorhub cluster --type=merge \
  -p '{"spec":{"disableAllDefaultSources":true}}'
```

**MinIO PVC sizing:** The RHOAI DSPA MinIO image
(`quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z`) enforces a
minimum free disk threshold. A 1Gi PVC causes `CrashLoopBackOff` with
`Storage reached its minimum free disk threshold`. Use >= 5Gi for MinIO
PVCs.

### RHOAI 2.x Istio sidecar injection breaks on worker

RHOAI 2.x depends on OpenShift Service Mesh (OSSM), which injects
Istio sidecar containers into pods via `sidecar.istio.io/inject: "true"`.
When a pod with this annotation is offloaded to the worker (either via
the custom VK or Liqo), the sidecar injection chain follows the pod:

1. OSSM's mutating webhook on the tenant injects `istio-proxy` container
   and adds `k8s.v1.cni.cncf.io/networks: v2-6-istio-cni` annotation
2. The VK (or Liqo ShadowPod controller) copies the full pod spec to
   the worker, including the injected sidecar and CNI annotation
3. On the worker, there is no Istio CNI plugin and no OSSM control
   plane. The `NetworkAttachmentDefinition` for `v2-6-istio-cni` does
   not exist
4. Both the `main` and `istio-proxy` containers get stuck at
   `ContainerCreating` indefinitely

Confirmed in RHOAI 2.x VK tests (10/10 pass, including this scenario).
Test output from the Istio sidecar test:

```
Containers on worker: ['main', 'istio-proxy']
Worker pod phase: Pending
  istio-proxy: state=waiting: ContainerCreating
  main:        state=waiting: ContainerCreating
```

This is not a bug in Liqo or the VK. The sidecar injection happens on
the tenant before the pod is dispatched. The worker correctly rejects
a CNI network attachment it does not have.

**Why RHOAI 3.x avoids this entirely:** RHOAI 3.x removed the OSSM
dependency (test L20 confirms no `servicemeshcontrolplanes.maistra.io`
CRD exists on the tenant). No sidecar injection, no CNI annotation, no
worker-side conflict. This is one reason the Liqo spike targets RHOAI
3.x exclusively.

**If RHOAI 2.x support is needed:** Disable sidecar injection for GPU
workload namespaces by removing the `sidecar.istio.io/inject` label
from the namespace or adding `sidecar.istio.io/inject: "false"` to
individual pod specs. KServe serverless (Knative) still works without
sidecar injection, it just loses mTLS between pods.

### Multi-tenant namespace isolation

When multiple tenants peer with the same GPU worker, twin namespaces
can collide. If tenant A and tenant B both have a namespace called
`ml-training` and both use `EnforceSameName`, both map to
`ml-training` on the worker — breaking isolation.

**Solution: `DefaultName` mapping strategy.** Liqo prefixes the twin
namespace with the remote cluster's `clusterID`, producing
`{clusterID}-{namespace}` on the worker. This is collision-free as
long as cluster IDs are unique (Liqo enforces uniqueness at peering
time).

**Human-readable prefixes:** Since GPUaaS controls tenant provisioning,
enforce a naming convention for `discovery.config.clusterID` at Liqo
install time:

```bash
# At tenant provisioning time
liqoctl install openshift \
  --set discovery.config.clusterID=team-alpha \
  ...
```

Twin namespaces on the worker become `team-alpha-ml-training`,
`team-beta-ml-training` — readable and collision-free. The cluster ID
is set once at install time and cannot be changed without reinstalling
Liqo, so it is a stable identifier.

**Do not rely on OpenShift cluster names** — they are not guaranteed
unique across installations. Two clusters can both be named `cluster`
or `sno`. The Liqo `clusterID` is the right key because Liqo validates
uniqueness during peering.

**Impact on worker-side configuration:** With `DefaultName`, the
Kueue setup (LocalQueue, namespace labels) must target prefixed
namespace names. This should be automated — see Next Steps.

This approach was not tested in the spike (`EnforceSameName` was used
for single-tenant validation). Multi-tenant validation with
`DefaultName` is the next priority.

### Conclusion

Liqo v1.2.0 is a viable replacement for the custom Virtual Kubelet
provider for RHOAI 3.x GPU workload offloading. 32 out of 32 tests
pass, covering every RHOAI workload type: Notebooks, PyTorchJob,
TFJob, XGBoostJob, RayJob, RayCluster, KServe inference, DSPA
pipelines, and LMEval model evaluation. No RHOAI operator required
code changes or scheduling modifications.

**Why Liqo over the custom VK:**

The primary advantage is transparency. RHOAI operators (training
operator, KServe, notebook controller, KubeRay, TrustyAI) create pods
normally in offloaded namespaces. Pods requesting GPUs are placed on
the virtual node by the kube-scheduler because it is the only node
with `nvidia.com/gpu` in allocatable. Non-GPU pods (DSPA, LMEval,
monitoring) stay local. No `nodeName`, no tolerations, no annotations
on workload specs. Every new RHOAI workload type works automatically
without touching the dispatch layer.

The custom VK requires explicit scheduling (`nodeName` + tolerations),
custom Go code for resource sync and status propagation, and
maintenance of a binary that must be updated for every new pod field
or RHOAI operator behavior change. Liqo handles all of this through
its ShadowPod controller and resource reflector.

Storage is also simpler. The custom VK uses Catapult with opt-in
StorageClass gating. Liqo provides a virtual `liqo` StorageClass that
remaps to the worker's real SC (provider-agnostic: `lvms-vg1` on bare
metal, `gp3` on AWS, `pd-ssd` on GCP). The RHOAI Dashboard storage
flow works transparently (L22).

Kueue integration works end-to-end: admission, queuing, and priority
preemption all function with the same Kyverno policy layer built for
the custom VK. ShadowPod controller preserves Kyverno-mutated labels
(L12), so the worker-side Kueue setup is unchanged.

**Trade-offs and risks:**

1. **Submariner is mandatory.** Liqo's WireGuard networking does not
   work on OpenShift (SCC/privilege issues, SCC template bug in
   v1.2.0). Cross-cluster traffic goes through Submariner's Libreswan
   IPsec tunnels. This means two control planes to operate and debug
   when networking issues arise.

2. **PodIP sync is undocumented behavior.** The architecture depends
   on Liqo passing through the real worker PodIP when networking is
   disabled. This is not a documented API contract. A Liqo upgrade
   could introduce IP remapping, silently breaking Submariner routing.
   Tests L3 and L13 should be the first to run after any Liqo version
   bump.

3. **Worker footprint is larger.** The custom VK only needs a
   ServiceAccount + RBAC on the worker. Liqo installs a full control
   plane: ShadowPod controller, auth module, storage fabric, resource
   reflector. More components, more things that can fail.

4. **No Red Hat support for Liqo.** Liqo is an academic/community
   project from Politecnico di Torino. Submariner has Red Hat backing
   (part of RHACM), but Liqo does not. If a Liqo bug blocks
   production, the only recourse is upstream GitHub issues or
   self-patching.

5. **Non-overlapping CIDRs required.** Without Liqo IPAM (disabled),
   tenant and worker clusters must have non-overlapping Pod and Service
   CIDRs. Manageable at current scale (we control provisioning), but
   becomes a hard constraint at 64+ tenants or with externally managed
   clusters. Submariner Globalnet can mitigate this but was not tested.

6. **TrustyAI operator race condition.** The LMEval test is flaky due
   to a TrustyAI bug where concurrent reconcile loops race on pod
   creation (finding #17). Not Liqo-related, but surfaces more
   reliably in the full suite because of timing. The test includes a
   retry workaround.

**What this spike does not cover:**

- Multi-tenant isolation with `DefaultName` namespace mapping
- RHOAI 2.x compatibility (Istio sidecar injection is a known blocker;
  see deployment finding above)
- Liqo upgrades and PodIP sync behavior across versions
- Submariner Globalnet for overlapping CIDRs
- Performance under load (concurrent GPU workloads from multiple
  tenants)

**Recommendation:** Proceed with Liqo for GPUaaS. The transparency
with RHOAI operators eliminates the main maintenance burden of the
custom VK (keeping scheduling logic in sync with operator behavior).
The trade-offs (Submariner dependency, no Red Hat support for Liqo,
larger worker footprint) are manageable for the current deployment.
Next priority is multi-tenant validation with `DefaultName` mapping.

### Next Steps

1. **Multi-tenant with `DefaultName`** — Switch from `EnforceSameName`
   to `DefaultName` mapping strategy. Peer a second tenant with a
   human-readable `clusterID` and verify twin namespace prefixing,
   resource reflection, and namespace isolation on the worker.
2. **Twin namespace labeling automation** — When Liqo creates a twin
   namespace (with a `liqo.io/remote-cluster-id` label), it needs
   `gpuaas.redhat.com/managed=true` and a LocalQueue for Kueue.
   Automate this with a Kyverno generate policy or controller on the
   worker that watches for new namespaces with Liqo labels.
3. **RHOAI 2.x** — Current results are RHOAI 3.x only. RHOAI 2.x has a
   known Istio sidecar injection issue (see deployment finding). If 2.x
   support is needed, test with sidecar injection disabled for GPU
   workload namespaces.
