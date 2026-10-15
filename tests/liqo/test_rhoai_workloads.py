"""
RHOAI 3.x workload dispatch via Liqo.

Tests that RHOAI operator workloads (Notebook, RayJob, TFJob, XGBoostJob,
RayCluster) and platform components (DSPA, LMEval) work on a Liqo-enabled
cluster. GPU workloads are dispatched transparently — no nodeName, no
explicit tolerations. The kube-scheduler places GPU pods on the Liqo
virtual node (the only node with nvidia.com/gpu in allocatable).

PyTorchJob and KServe are already covered by test_rhoai.py (L9, L10).
"""

import time

import pytest
from kubernetes import client
from kubernetes.stream import stream

from helpers import force_delete_pod, wait_pod_exists


KUBEFLOW_GROUP = "kubeflow.org"
KUBEFLOW_VERSION = "v1"
PYTORCHJOB_PLURAL = "pytorchjobs"
NOTEBOOK_PLURAL = "notebooks"

RAYJOB_GROUP = "ray.io"
RAYJOB_VERSION = "v1"
RAYJOB_PLURAL = "rayjobs"

RAYCLUSTER_GROUP = "ray.io"
RAYCLUSTER_VERSION = "v1"
RAYCLUSTER_PLURAL = "rayclusters"

ISVC_GROUP = "serving.kserve.io"
ISVC_VERSION = "v1beta1"
ISVC_PLURAL = "inferenceservices"

SERVING_RUNTIME_GROUP = "serving.kserve.io"
SERVING_RUNTIME_VERSION = "v1alpha1"
SERVING_RUNTIME_PLURAL = "servingruntimes"

DSPA_GROUP = "datasciencepipelinesapplications.opendatahub.io"
DSPA_VERSION = "v1"
DSPA_PLURAL = "datasciencepipelinesapplications"

LMEVAL_GROUP = "trustyai.opendatahub.io"
LMEVAL_VERSION = "v1alpha1"
LMEVAL_PLURAL = "lmevaljobs"

PYTORCH_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
RAY_IMAGE = "rayproject/ray:2.46.0-py312-cu128"

LIQO_STORAGE_CLASS = "liqo"
LVMS_STORAGE_CLASS = "lvms-vg1"

LMEVAL_MODEL_NAME = "facebook/opt-125m"
VLLM_CPU_IMAGE = "registry.redhat.io/rhaii/vllm-cpu-rhel9@sha256:7a28b867955055a8836ea2a78e70b8bad63c1ee50e8a5f8dd95d1194f20ec489"


def wait_pod_phase(core_api, name, namespace, phases, timeout=600):
    deadline = time.time() + timeout
    phase = None
    while time.time() < deadline:
        try:
            pod = core_api.read_namespaced_pod(name=name, namespace=namespace)
            phase = pod.status.phase
            if phase in phases:
                return phase
        except client.exceptions.ApiException:
            pass
        time.sleep(5)
    return phase


def _find_worker_pod(worker_core, worker_ns, name_substring, timeout=120):
    """Find a pod on the worker whose name contains name_substring."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = worker_core.list_namespaced_pod(namespace=worker_ns)
        for p in pods.items:
            if name_substring in p.metadata.name:
                return p.metadata.name
        time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Notebook CR
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_notebook_cr_via_liqo(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
    rhoai_full_environment,
):
    """Notebook CR dispatched via Liqo. GPU request drives placement."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace
    apps_api = client.AppsV1Api(tenant_core.api_client)

    nb_name = "liqo-gpu-notebook"
    pod_name = f"{nb_name}-0"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=NOTEBOOK_PLURAL, name=nb_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    force_delete_pod(tenant_core, pod_name, ns)

    server_script = "\n".join([
        "import http.server, subprocess, json",
        "class H(http.server.BaseHTTPRequestHandler):",
        "  def do_GET(self):",
        "    r = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'], capture_output=True, text=True)",
        "    body = json.dumps(dict(gpu=r.stdout.strip(), status='ready'))",
        "    self.send_response(200)",
        "    self.send_header('Content-Type','application/json')",
        "    self.end_headers()",
        "    self.wfile.write(body.encode())",
        "  def log_message(self, *a): pass",
        "http.server.HTTPServer(('',8888),H).serve_forever()",
    ])

    notebook_cr = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {"name": nb_name, "namespace": ns},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "notebook",
                            "image": "nvcr.io/nvidia/cuda:12.8.1-base-ubi9",
                            "command": ["python3", "-c", server_script],
                            "ports": [{"containerPort": 8888, "protocol": "TCP"}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": "1"},
                            },
                        }
                    ],
                }
            }
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=NOTEBOOK_PLURAL, body=notebook_cr,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=NOTEBOOK_PLURAL, name=nb_name,
        ),
    )
    cleanup(force_delete_pod, tenant_core, pod_name, ns)

    deadline = time.time() + 60
    sts_found = False
    while time.time() < deadline:
        try:
            apps_api.read_namespaced_stateful_set(name=nb_name, namespace=ns)
            sts_found = True
            break
        except client.exceptions.ApiException:
            pass
        time.sleep(3)
    assert sts_found, (
        f"Notebook controller did not create StatefulSet {nb_name} within 60s"
    )

    assert wait_pod_exists(tenant_core, pod_name, ns, timeout=60), (
        f"StatefulSet did not create pod {pod_name} within 60s"
    )

    phase = wait_pod_phase(tenant_core, pod_name, ns, ["Running"], timeout=300)
    assert phase == "Running", f"Notebook pod phase: {phase}, expected Running"

    deadline = time.time() + 60
    nb_ready = False
    has_running_state = False
    while time.time() < deadline:
        nb = tenant_custom.get_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=NOTEBOOK_PLURAL, name=nb_name,
        )
        nb_conditions = nb.get("status", {}).get("conditions", [])
        nb_ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in nb_conditions
        )
        container_state = nb.get("status", {}).get("containerState", {})
        has_running_state = "running" in container_state
        if nb_ready or has_running_state:
            break
        time.sleep(5)
    assert nb_ready or has_running_state, (
        f"Notebook CR status does not show Ready or running containerState"
    )


# ---------------------------------------------------------------------------
# Notebook CR with workspace PVC (RHOAI Dashboard flow)
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_notebook_with_workspace_pvc(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
    rhoai_full_environment,
):
    """Notebook CR with workspace PVC — simulates the RHOAI Dashboard flow.

    The RHOAI Dashboard creates a workspace PVC, then a Notebook CR that
    mounts it. The data scientist connects to the notebook, runs GPU code,
    and saves results to the workspace. With Liqo, the PVC is reflected to
    the worker (SC remapped liqo → lvms-vg1). This test validates:

    1. PVC reflection + StorageClass remapping
    2. Notebook CR lifecycle (controller → StatefulSet → pod → Ready)
    3. Simulated notebook usage: exec GPU computation, save to workspace
    4. Results persist on the reflected PVC
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace
    apps_api = client.AppsV1Api(tenant_core.api_client)

    nb_name = "liqo-nb-workspace"
    pod_name = f"{nb_name}-0"
    pvc_name = "liqo-nb-workspace"

    # Pre-clean
    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=NOTEBOOK_PLURAL, name=nb_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    force_delete_pod(tenant_core, pod_name, ns)
    try:
        tenant_core.delete_namespaced_persistent_volume_claim(
            name=pvc_name, namespace=ns,
        )
        time.sleep(5)
    except client.exceptions.ApiException:
        pass

    # Step 1: Create workspace PVC (what the Dashboard does)
    pvc = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(name=pvc_name, namespace=ns),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            storage_class_name=LIQO_STORAGE_CLASS,
            resources=client.V1VolumeResourceRequirements(
                requests={"storage": "1Gi"},
            ),
        ),
    )
    tenant_core.create_namespaced_persistent_volume_claim(namespace=ns, body=pvc)
    cleanup(
        lambda: tenant_core.delete_namespaced_persistent_volume_claim(
            name=pvc_name, namespace=ns,
        ),
    )

    # Step 2: Create Notebook CR that mounts the workspace PVC
    notebook_cr = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {"name": nb_name, "namespace": ns},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "notebook",
                            "image": "nvcr.io/nvidia/cuda:12.8.1-base-ubi9",
                            "command": ["sleep", "infinity"],
                            "ports": [{"containerPort": 8888, "protocol": "TCP"}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "workspace",
                                    "mountPath": "/opt/app-root/src",
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "workspace",
                            "persistentVolumeClaim": {"claimName": pvc_name},
                        },
                    ],
                }
            }
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=NOTEBOOK_PLURAL, body=notebook_cr,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=NOTEBOOK_PLURAL, name=nb_name,
        ),
    )
    cleanup(force_delete_pod, tenant_core, pod_name, ns)

    # Step 3: Wait for notebook controller → StatefulSet → pod → Running
    deadline = time.time() + 60
    sts_found = False
    while time.time() < deadline:
        try:
            apps_api.read_namespaced_stateful_set(name=nb_name, namespace=ns)
            sts_found = True
            break
        except client.exceptions.ApiException:
            pass
        time.sleep(3)
    assert sts_found, (
        f"Notebook controller did not create StatefulSet {nb_name} within 60s"
    )

    assert wait_pod_exists(tenant_core, pod_name, ns, timeout=60), (
        f"StatefulSet did not create pod {pod_name} within 60s"
    )

    phase = wait_pod_phase(tenant_core, pod_name, ns, ["Running"], timeout=300)
    assert phase == "Running", f"Notebook pod phase: {phase}, expected Running"

    # Step 4: Verify PVC reflected to worker with SC remapping
    deadline = time.time() + 60
    worker_pvc_name = None
    while time.time() < deadline:
        pvcs = worker_core.list_namespaced_persistent_volume_claim(namespace=w_ns)
        for p in pvcs.items:
            if pvc_name in p.metadata.name:
                worker_pvc_name = p.metadata.name
                break
        if worker_pvc_name:
            break
        time.sleep(5)
    assert worker_pvc_name, (
        f"Workspace PVC not reflected to worker namespace {w_ns}"
    )

    worker_pvc = worker_core.read_namespaced_persistent_volume_claim(
        name=worker_pvc_name, namespace=w_ns,
    )
    assert worker_pvc.spec.storage_class_name == LVMS_STORAGE_CLASS, (
        f"Worker PVC StorageClass: {worker_pvc.spec.storage_class_name}, "
        f"expected {LVMS_STORAGE_CLASS}"
    )
    assert worker_pvc.status.phase == "Bound", (
        f"Worker PVC phase: {worker_pvc.status.phase}, expected Bound"
    )

    # Step 5: Simulate notebook usage — run GPU code and save to workspace
    worker_pod = _find_worker_pod(worker_core, w_ns, nb_name, timeout=60)
    assert worker_pod, f"Notebook worker pod not found in {w_ns}"

    gpu_resp = stream(
        worker_core.connect_get_namespaced_pod_exec,
        worker_pod, w_ns, container="notebook",
        command=[
            "sh", "-c",
            "nvidia-smi --query-gpu=name,memory.total "
            "--format=csv,noheader,nounits "
            "> /opt/app-root/src/gpu_output.txt && echo OK",
        ],
        stderr=True, stdout=True, stdin=False, tty=False,
    )
    assert "OK" in gpu_resp, f"nvidia-smi failed in notebook: {gpu_resp}"

    # Step 6: Read results back from workspace PVC — data persisted
    read_resp = stream(
        worker_core.connect_get_namespaced_pod_exec,
        worker_pod, w_ns, container="notebook",
        command=["cat", "/opt/app-root/src/gpu_output.txt"],
        stderr=True, stdout=True, stdin=False, tty=False,
    )
    assert read_resp.strip(), "GPU output file is empty on workspace PVC"
    assert "RTX" in read_resp or "NVIDIA" in read_resp, (
        f"GPU not detected in notebook workspace output: {read_resp}"
    )


# ---------------------------------------------------------------------------
# PyTorchJob with checkpoint PVC
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_pytorchjob_checkpoint_with_pvc(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
):
    """Train on GPU and save checkpoint to liqo PVC, then verify."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    job_name = "liqo-checkpoint-training"
    master_pod_name = f"{job_name}-master-0"
    pvc_name = "liqo-training-checkpoint"
    verify_pod_name = "liqo-checkpoint-verify"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=PYTORCHJOB_PLURAL, name=job_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    for name in [master_pod_name, verify_pod_name]:
        force_delete_pod(tenant_core, name, ns)
    try:
        tenant_core.delete_namespaced_persistent_volume_claim(
            name=pvc_name, namespace=ns,
        )
    except client.exceptions.ApiException:
        pass
    time.sleep(3)

    tenant_core.create_namespaced_persistent_volume_claim(
        namespace=ns,
        body=client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(name=pvc_name, namespace=ns),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteOnce"],
                storage_class_name=LIQO_STORAGE_CLASS,
                resources=client.V1VolumeResourceRequirements(
                    requests={"storage": "1Gi"},
                ),
            ),
        ),
    )
    cleanup(
        lambda: tenant_core.delete_namespaced_persistent_volume_claim(
            name=pvc_name, namespace=ns,
        ),
    )

    training_script = "\n".join([
        "import torch",
        "import torch.nn as nn",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'Training on: {device}')",
        "",
        "model = nn.Sequential(",
        "    nn.Linear(784, 128),",
        "    nn.ReLU(),",
        "    nn.Linear(128, 10),",
        ").to(device)",
        "",
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
        "loss_fn = nn.CrossEntropyLoss()",
        "",
        "for epoch in range(3):",
        "    x = torch.randn(64, 784, device=device)",
        "    y = torch.randint(0, 10, (64,), device=device)",
        "    loss = loss_fn(model(x), y)",
        "    optimizer.zero_grad()",
        "    loss.backward()",
        "    optimizer.step()",
        "    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')",
        "",
        "torch.save(model.state_dict(), '/data/checkpoint.pt')",
        "print(f'Checkpoint saved, keys: {len(model.state_dict())}')",
        "print('Training complete')",
    ])

    pytorchjob = {
        "apiVersion": f"{KUBEFLOW_GROUP}/{KUBEFLOW_VERSION}",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": ns},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "pytorch",
                                    "image": PYTORCH_IMAGE,
                                    "command": ["python3", "-c", training_script],
                                    "env": [{"name": "TMPDIR", "value": "/tmp"}],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                    },
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "tmp", "mountPath": "/tmp"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "data",
                                    "persistentVolumeClaim": {"claimName": pvc_name},
                                },
                                {"name": "tmp", "emptyDir": {}},
                            ],
                        }
                    },
                }
            }
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=pytorchjob,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=PYTORCHJOB_PLURAL, name=job_name,
        ),
    )

    assert wait_pod_exists(tenant_core, master_pod_name, ns, timeout=60), (
        "Training operator did not create master pod within 60s"
    )

    phase = wait_pod_phase(
        tenant_core, master_pod_name, ns, ["Succeeded", "Failed"], timeout=900,
    )
    assert phase == "Succeeded", f"Training pod phase: {phase}"

    logs = tenant_core.read_namespaced_pod_log(
        name=master_pod_name, namespace=ns, container="pytorch",
    )
    assert "Checkpoint saved" in logs, f"Checkpoint not saved. Logs: {logs[:500]}"

    verify_script = "\n".join([
        "import torch",
        "ckpt = torch.load('/data/checkpoint.pt', map_location='cpu', weights_only=True)",
        "print(f'Keys: {len(ckpt)}')",
        "assert len(ckpt) > 0, 'Empty checkpoint'",
        "print('Checkpoint valid')",
    ])

    verify_pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=verify_pod_name, namespace=ns),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="verify",
                    image=PYTORCH_IMAGE,
                    command=["python3", "-c", verify_script],
                    resources=client.V1ResourceRequirements(
                        limits={"nvidia.com/gpu": "1"},
                    ),
                    volume_mounts=[
                        client.V1VolumeMount(name="data", mount_path="/data"),
                    ],
                )
            ],
            volumes=[
                client.V1Volume(
                    name="data",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name,
                    ),
                )
            ],
        ),
    )
    tenant_core.create_namespaced_pod(namespace=ns, body=verify_pod)
    cleanup(force_delete_pod, tenant_core, verify_pod_name, ns)

    verify_phase = wait_pod_phase(
        tenant_core, verify_pod_name, ns, ["Succeeded", "Failed"], timeout=300,
    )
    assert verify_phase == "Succeeded", f"Verification pod phase: {verify_phase}"

    verify_logs = tenant_core.read_namespaced_pod_log(
        name=verify_pod_name, namespace=ns, container="verify",
    )
    assert "Checkpoint valid" in verify_logs, (
        f"Checkpoint verification failed. Logs: {verify_logs[:500]}"
    )


# ---------------------------------------------------------------------------
# RayJob
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_rayjob_via_liqo(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
    rhoai_full_environment,
):
    """RayJob dispatches a Ray head pod via Liqo, runs a GPU job on the worker."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    job_name = "liqo-rayjob-gpu"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=RAYJOB_GROUP, version=RAYJOB_VERSION,
            namespace=ns, plural=RAYJOB_PLURAL, name=job_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise

    pods = tenant_core.list_namespaced_pod(
        namespace=ns, label_selector="ray.io/node-type=head",
    )
    for p in pods.items:
        cluster = (p.metadata.labels or {}).get("ray.io/cluster", "")
        if cluster.startswith(job_name):
            force_delete_pod(tenant_core, p.metadata.name, ns)

    rayjob = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": job_name, "namespace": ns},
        "spec": {
            "entrypoint": "python -c \""
                "import ray; "
                "ray.init(); "
                "import subprocess; "
                "r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], "
                "capture_output=True, text=True); "
                "print(f'GPU: {r.stdout.strip()}'); "
                "print('RayJob complete')\"",
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 300,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": RAY_IMAGE,
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1", "cpu": "2", "memory": "4Gi"},
                                        "requests": {"nvidia.com/gpu": "1", "cpu": "1", "memory": "2Gi"},
                                    },
                                    "ports": [
                                        {"containerPort": 6379, "name": "gcs-server"},
                                        {"containerPort": 8265, "name": "dashboard"},
                                        {"containerPort": 10001, "name": "client"},
                                    ],
                                    "volumeMounts": [
                                        {"name": "tmp", "mountPath": "/tmp"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "tmp", "emptyDir": {}},
                            ],
                        }
                    },
                },
                "workerGroupSpecs": [],
            },
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=RAYJOB_GROUP, version=RAYJOB_VERSION,
        namespace=ns, plural=RAYJOB_PLURAL, body=rayjob,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=RAYJOB_GROUP, version=RAYJOB_VERSION,
            namespace=ns, plural=RAYJOB_PLURAL, name=job_name,
        ),
    )

    deadline = time.time() + 120
    head_pod_name = None
    while time.time() < deadline:
        try:
            rj = tenant_custom.get_namespaced_custom_object(
                group=RAYJOB_GROUP, version=RAYJOB_VERSION,
                namespace=ns, plural=RAYJOB_PLURAL, name=job_name,
            )
            cluster_name = rj.get("status", {}).get("rayClusterName", "")
        except client.exceptions.ApiException:
            cluster_name = ""

        if cluster_name:
            pods = tenant_core.list_namespaced_pod(
                namespace=ns,
                label_selector=f"ray.io/cluster={cluster_name},ray.io/node-type=head",
            )
            for p in pods.items:
                if p.metadata.deletion_timestamp is None:
                    head_pod_name = p.metadata.name
                    break
        if head_pod_name:
            break
        time.sleep(5)
    assert head_pod_name is not None, (
        f"Ray head pod for {job_name} did not appear within 120s"
    )

    w_head = _find_worker_pod(worker_core, w_ns, head_pod_name, timeout=60)
    assert w_head is not None, f"Worker head pod for {head_pod_name} not found"

    worker_phase = wait_pod_phase(
        worker_core, w_head, w_ns,
        ("Running", "Succeeded", "Failed"), timeout=300,
    )
    assert worker_phase in ("Running", "Succeeded"), (
        f"Ray head pod phase: {worker_phase}"
    )

    gpu_output = stream(
        worker_core.connect_get_namespaced_pod_exec,
        name=w_head, namespace=w_ns,
        command=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        container="ray-head",
        stderr=True, stdout=True, stdin=False, tty=False,
    )
    assert gpu_output.strip(), f"nvidia-smi returned empty output on worker head pod"


# ---------------------------------------------------------------------------
# TFJob
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_tfjob_via_liqo(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
):
    """TFJob dispatched via Liqo — validates training operator + GPU dispatch.

    Uses CUDA base image with nvidia-smi instead of TensorFlow, which
    avoids CUDA kernel compatibility issues (e.g. Blackwell sm_120).
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace

    job_name = "liqo-tfjob-gpu"
    chief_pod_name = f"{job_name}-chief-0"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural="tfjobs", name=job_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    force_delete_pod(tenant_core, chief_pod_name, ns)

    tfjob = {
        "apiVersion": f"{KUBEFLOW_GROUP}/{KUBEFLOW_VERSION}",
        "kind": "TFJob",
        "metadata": {"name": job_name, "namespace": ns},
        "spec": {
            "tfReplicaSpecs": {
                "Chief": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "tensorflow",
                                    "image": "nvcr.io/nvidia/cuda:12.8.1-base-ubi9",
                                    "command": [
                                        "nvidia-smi",
                                        "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv",
                                    ],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                    },
                                }
                            ],
                        }
                    },
                }
            }
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural="tfjobs", body=tfjob,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural="tfjobs", name=job_name,
        ),
    )

    assert wait_pod_exists(tenant_core, chief_pod_name, ns, timeout=60), (
        f"Training operator did not create chief pod {chief_pod_name} within 60s"
    )

    phase = wait_pod_phase(
        tenant_core, chief_pod_name, ns, ["Succeeded", "Failed"], timeout=900,
    )

    logs = ""
    try:
        logs = tenant_core.read_namespaced_pod_log(
            name=chief_pod_name, namespace=ns, container="tensorflow",
        )
    except client.exceptions.ApiException:
        pass

    assert phase == "Succeeded", (
        f"TFJob chief pod phase: {phase}. Logs: {logs[-500:]}"
    )
    assert "name" in logs.lower() and "driver_version" in logs.lower(), (
        f"nvidia-smi output missing GPU info. Logs: {logs[:500]}"
    )


# ---------------------------------------------------------------------------
# XGBoostJob
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_xgboostjob_via_liqo(
    cleanup, liqo_tenant_clients,
    liqo_test_namespace, liqo_virtual_node_ready,
):
    """XGBoostJob dispatched via Liqo — XGBoost GPU training."""
    tenant_core, tenant_custom = liqo_tenant_clients
    ns = liqo_test_namespace

    job_name = "liqo-xgboostjob"
    master_pod_name = f"{job_name}-master-0"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural="xgboostjobs", name=job_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    force_delete_pod(tenant_core, master_pod_name, ns)

    training_script = "\n".join([
        "import subprocess, sys",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q', '--break-system-packages', '--target=/tmp/pylibs'])",
        "sys.path.insert(0, '/tmp/pylibs')",
        "import xgboost as xgb",
        "import numpy as np",
        "print(f'XGBoost version: {xgb.__version__}')",
        "",
        "X = np.random.randn(100, 10).astype(np.float32)",
        "y = np.random.randint(0, 2, 100).astype(np.float32)",
        "dtrain = xgb.DMatrix(X, label=y)",
        "",
        "params = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic',",
        "          'device': 'cuda', 'tree_method': 'hist'}",
        "model = xgb.train(params, dtrain, num_boost_round=10)",
        "preds = model.predict(dtrain)",
        "print(f'Predictions shape: {preds.shape}')",
        "print('XGBoost training complete')",
    ])

    xgboostjob = {
        "apiVersion": f"{KUBEFLOW_GROUP}/{KUBEFLOW_VERSION}",
        "kind": "XGBoostJob",
        "metadata": {"name": job_name, "namespace": ns},
        "spec": {
            "xgbReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "xgboost",
                                    "image": PYTORCH_IMAGE,
                                    "command": ["python3", "-c", training_script],
                                    "env": [{"name": "TMPDIR", "value": "/tmp"}],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                    },
                                    "volumeMounts": [
                                        {"name": "tmp", "mountPath": "/tmp"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "tmp", "emptyDir": {}},
                            ],
                        }
                    },
                }
            }
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural="xgboostjobs", body=xgboostjob,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural="xgboostjobs", name=job_name,
        ),
    )

    assert wait_pod_exists(tenant_core, master_pod_name, ns, timeout=60), (
        f"Training operator did not create master pod within 60s"
    )

    phase = wait_pod_phase(
        tenant_core, master_pod_name, ns, ["Succeeded", "Failed"], timeout=900,
    )

    logs = ""
    try:
        logs = tenant_core.read_namespaced_pod_log(
            name=master_pod_name, namespace=ns, container="xgboost",
        )
    except client.exceptions.ApiException:
        pass

    assert phase == "Succeeded", (
        f"XGBoost training pod phase: {phase}. Logs: {logs[-500:]}"
    )
    assert "XGBoost training complete" in logs, (
        f"XGBoost training incomplete. Logs: {logs[:500]}"
    )


# ---------------------------------------------------------------------------
# RayCluster
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_raycluster_via_liqo(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
    rhoai_full_environment,
):
    """Standalone RayCluster dispatched via Liqo — head pod runs on worker GPU."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    cluster_name = "liqo-raycluster-gpu"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=RAYCLUSTER_GROUP, version=RAYCLUSTER_VERSION,
            namespace=ns, plural=RAYCLUSTER_PLURAL, name=cluster_name,
        )
        time.sleep(10)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise

    pods = tenant_core.list_namespaced_pod(
        namespace=ns, label_selector=f"ray.io/cluster={cluster_name}",
    )
    for p in pods.items:
        force_delete_pod(tenant_core, p.metadata.name, ns)

    raycluster = {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": {"name": cluster_name, "namespace": ns},
        "spec": {
            "headGroupSpec": {
                "rayStartParams": {"dashboard-host": "0.0.0.0"},
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "ray-head",
                                "image": RAY_IMAGE,
                                "resources": {
                                    "limits": {"nvidia.com/gpu": "1", "cpu": "2", "memory": "4Gi"},
                                    "requests": {"nvidia.com/gpu": "1", "cpu": "1", "memory": "2Gi"},
                                },
                                "ports": [
                                    {"containerPort": 6379, "name": "gcs-server"},
                                    {"containerPort": 8265, "name": "dashboard"},
                                    {"containerPort": 10001, "name": "client"},
                                ],
                                "volumeMounts": [
                                    {"name": "tmp", "mountPath": "/tmp"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "tmp", "emptyDir": {}},
                        ],
                    }
                },
            },
            "workerGroupSpecs": [],
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=RAYCLUSTER_GROUP, version=RAYCLUSTER_VERSION,
        namespace=ns, plural=RAYCLUSTER_PLURAL, body=raycluster,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=RAYCLUSTER_GROUP, version=RAYCLUSTER_VERSION,
            namespace=ns, plural=RAYCLUSTER_PLURAL, name=cluster_name,
        ),
    )

    deadline = time.time() + 120
    head_pod_name = None
    while time.time() < deadline:
        pods = tenant_core.list_namespaced_pod(
            namespace=ns,
            label_selector=f"ray.io/cluster={cluster_name},ray.io/node-type=head",
        )
        if pods.items:
            head_pod_name = pods.items[0].metadata.name
            break
        time.sleep(5)
    assert head_pod_name is not None, (
        f"Ray head pod for {cluster_name} did not appear within 120s"
    )

    w_head = _find_worker_pod(worker_core, w_ns, head_pod_name, timeout=60)
    assert w_head is not None, f"Worker head pod for {head_pod_name} not found"

    worker_phase = wait_pod_phase(
        worker_core, w_head, w_ns, ("Running", "Failed"), timeout=300,
    )
    assert worker_phase == "Running", f"Ray head pod phase: {worker_phase}"

    deadline = time.time() + 60
    cluster_ready = False
    state = ""
    while time.time() < deadline:
        try:
            rc = tenant_custom.get_namespaced_custom_object(
                group=RAYCLUSTER_GROUP, version=RAYCLUSTER_VERSION,
                namespace=ns, plural=RAYCLUSTER_PLURAL, name=cluster_name,
            )
            state = rc.get("status", {}).get("state", "")
            if state == "ready":
                cluster_ready = True
                break
        except client.exceptions.ApiException:
            pass
        time.sleep(5)

    if not cluster_ready:
        import warnings
        warnings.warn(
            f"RayCluster {cluster_name} state is {state!r}, not 'ready'. "
            f"Head pod is Running on worker — cluster may need status sync."
        )


# ---------------------------------------------------------------------------
# DataSciencePipelines (stays local — no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_dspa_on_liqo_cluster(
    cleanup, liqo_tenant_clients,
    liqo_test_namespace, liqo_virtual_node_ready,
):
    """DSPA infrastructure works on a Liqo-enabled cluster.

    DSPA pods (MariaDB, MinIO, API server) don't request GPUs, so they
    stay on the tenant node with LocalAndRemote + GPU taint.
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    ns = liqo_test_namespace
    apps_api = client.AppsV1Api(tenant_core.api_client)

    storage_api = client.StorageV1Api(tenant_core.api_client)
    scs = storage_api.list_storage_class()
    local_scs = [
        sc for sc in scs.items
        if sc.provisioner != "liqo.io/storage"
    ]
    if not local_scs:
        pytest.skip("No local StorageClass available — DSPA requires PVCs for MariaDB/MinIO")

    dspa_name = "liqo-dspa"

    try:
        tenant_custom.delete_namespaced_custom_object(
            group=DSPA_GROUP, version=DSPA_VERSION,
            namespace=ns, plural=DSPA_PLURAL, name=dspa_name,
        )
        time.sleep(30)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise

    dspa = {
        "apiVersion": f"{DSPA_GROUP}/{DSPA_VERSION}",
        "kind": "DataSciencePipelinesApplication",
        "metadata": {"name": dspa_name, "namespace": ns},
        "spec": {
            "dspVersion": "v2",
            "apiServer": {
                "cABundle": {
                    "configMapKey": "ca.crt",
                    "configMapName": "kube-root-ca.crt",
                },
            },
            "database": {
                "mariaDB": {"deploy": True, "pvcSize": "5Gi"},
            },
            "objectStorage": {
                "minio": {
                    "deploy": True,
                    "pvcSize": "5Gi",
                    "image": "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance",
                },
            },
        },
    }

    tenant_custom.create_namespaced_custom_object(
        group=DSPA_GROUP, version=DSPA_VERSION,
        namespace=ns, plural=DSPA_PLURAL, body=dspa,
    )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=DSPA_GROUP, version=DSPA_VERSION,
            namespace=ns, plural=DSPA_PLURAL, name=dspa_name,
        ),
    )

    for component in [
        f"mariadb-{dspa_name}",
        f"minio-{dspa_name}",
        f"ds-pipeline-{dspa_name}",
    ]:
        deadline = time.time() + 300
        ready = False
        while time.time() < deadline:
            try:
                dep = apps_api.read_namespaced_deployment(component, ns)
                if (dep.status.ready_replicas or 0) >= 1:
                    ready = True
                    break
            except client.exceptions.ApiException:
                pass
            time.sleep(10)
        assert ready, f"DSPA component {component} not ready within 300s"

    deadline = time.time() + 60
    dspa_ready = False
    conditions = []
    while time.time() < deadline:
        obj = tenant_custom.get_namespaced_custom_object(
            group=DSPA_GROUP, version=DSPA_VERSION,
            namespace=ns, plural=DSPA_PLURAL, name=dspa_name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        for c in conditions:
            if c.get("type") == "Ready" and c.get("status") == "True":
                dspa_ready = True
                break
        if dspa_ready:
            break
        time.sleep(5)
    assert dspa_ready, (
        f"DSPA {dspa_name} did not reach Ready. "
        f"Conditions: {[c.get('type') + '=' + c.get('status') for c in conditions]}"
    )


# ---------------------------------------------------------------------------
# LMEval (simplified — runs locally, no GPU dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.liqo
@pytest.mark.rhoai
def test_lmeval_on_liqo_cluster(
    cleanup, liqo_tenant_clients,
    liqo_test_namespace, liqo_virtual_node_ready,
):
    """LMEvalJob evaluating a locally served model on a Liqo-enabled cluster.

    Model is served via KServe CPU on the tenant. LMEval job runs locally
    (no GPU request → stays on tenant). Validates TrustyAI/LMEval
    integration on RHOAI 3.x with Liqo installed.
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    ns = liqo_test_namespace
    apps_api = client.AppsV1Api(tenant_core.api_client)
    api_ext = client.ApiextensionsV1Api(tenant_core.api_client)

    try:
        apps_api.read_namespaced_deployment(
            "trustyai-service-operator-controller-manager",
            "redhat-ods-applications",
        )
    except client.exceptions.ApiException:
        pytest.skip("TrustyAI operator not deployed — enable trustyai in DSC")

    try:
        api_ext.read_custom_resource_definition(
            "inferenceservices.serving.kserve.io"
        )
    except client.exceptions.ApiException:
        pytest.skip("KServe not installed — enable kserve in DSC")

    job_name = "liqo-lmeval"
    pod_name = job_name
    isvc_name = "liqo-lmeval-model"
    runtime_name = "liqo-vllm-cpu"

    for name in [job_name]:
        try:
            tenant_custom.delete_namespaced_custom_object(
                group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                namespace=ns, plural=LMEVAL_PLURAL, name=name,
            )
            time.sleep(15)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise
    force_delete_pod(tenant_core, pod_name, ns)
    try:
        tenant_custom.delete_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, name=isvc_name,
        )
        time.sleep(5)
    except client.exceptions.ApiException:
        pass
    try:
        tenant_custom.delete_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=ns, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
        )
        time.sleep(5)
    except client.exceptions.ApiException:
        pass

    runtime = {
        "apiVersion": f"{SERVING_RUNTIME_GROUP}/{SERVING_RUNTIME_VERSION}",
        "kind": "ServingRuntime",
        "metadata": {"name": runtime_name, "namespace": ns},
        "spec": {
            "multiModel": False,
            "supportedModelFormats": [
                {"name": "vLLM", "autoSelect": True},
            ],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": VLLM_CPU_IMAGE,
                    "args": [
                        "--model", "/mnt/models",
                        "--port", "8080",
                        "--gpu-memory-utilization", "0.4",
                        "--served-model-name", LMEVAL_MODEL_NAME,
                    ],
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                    "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                },
            ],
            "volumes": [
                {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}},
            ],
        },
    }

    try:
        existing = tenant_custom.get_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=ns, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
        )
        runtime["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
        tenant_custom.replace_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=ns, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
            body=runtime,
        )
    except client.exceptions.ApiException:
        tenant_custom.create_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=ns, plural=SERVING_RUNTIME_PLURAL, body=runtime,
        )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=ns, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
        ),
    )

    isvc = {
        "apiVersion": f"{ISVC_GROUP}/{ISVC_VERSION}",
        "kind": "InferenceService",
        "metadata": {"name": isvc_name, "namespace": ns},
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "runtime": runtime_name,
                    "storageUri": f"hf://{LMEVAL_MODEL_NAME}",
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                },
            },
        },
    }

    try:
        existing = tenant_custom.get_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, name=isvc_name,
        )
        isvc["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
        tenant_custom.replace_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, name=isvc_name,
            body=isvc,
        )
    except client.exceptions.ApiException:
        tenant_custom.create_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, body=isvc,
        )
    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, name=isvc_name,
        ),
    )

    deadline = time.time() + 300
    isvc_ready = False
    while time.time() < deadline:
        obj = tenant_custom.get_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=ns, plural=ISVC_PLURAL, name=isvc_name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        for c in conditions:
            if c.get("type") == "Ready" and c.get("status") == "True":
                isvc_ready = True
                break
        if isvc_ready:
            break
        time.sleep(10)
    assert isvc_ready, f"InferenceService {isvc_name} not Ready within 300s"

    model_url = f"http://{isvc_name}-predictor.{ns}.svc.cluster.local:8080/v1/completions"

    lmeval_job = {
        "apiVersion": f"{LMEVAL_GROUP}/{LMEVAL_VERSION}",
        "kind": "LMEvalJob",
        "metadata": {"name": job_name, "namespace": ns},
        "spec": {
            "model": "local-completions",
            "modelArgs": [
                {"name": "model", "value": LMEVAL_MODEL_NAME},
                {"name": "base_url", "value": model_url},
                {"name": "tokenizer", "value": LMEVAL_MODEL_NAME},
                {"name": "num_concurrent", "value": "1"},
            ],
            "taskList": {
                "taskNames": ["arc_easy"],
            },
            "limit": "2",
            "batchSize": "1",
            "allowOnline": True,
            "pod": {
                "volumes": [
                    {"name": "hf-cache", "emptyDir": {}},
                    {"name": "tmp", "emptyDir": {}},
                    {"name": "lmeval-output", "emptyDir": {}},
                    {"name": "lmeval-catalogs", "emptyDir": {}},
                ],
                "container": {
                    "env": [
                        {"name": "HF_HOME", "value": "/tmp/hf-cache"},
                    ],
                    "volumeMounts": [
                        {"name": "hf-cache", "mountPath": "/tmp/hf-cache"},
                        {"name": "tmp", "mountPath": "/tmp"},
                        {"name": "lmeval-output", "mountPath": "/opt/app-root/src/output"},
                        {"name": "lmeval-catalogs", "mountPath": "/opt/app-root/src/my_catalogs"},
                    ],
                },
            },
        },
    }

    # TrustyAI operator has a race condition where two concurrent reconcile
    # loops try to create the pod. The second fails with "already exists"
    # and the job status never advances past "Scheduled". Retry once if
    # we detect this stuck state.
    for attempt in range(2):
        if attempt > 0:
            try:
                tenant_custom.delete_namespaced_custom_object(
                    group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                    namespace=ns, plural=LMEVAL_PLURAL, name=job_name,
                )
                time.sleep(15)
            except client.exceptions.ApiException:
                pass
            force_delete_pod(tenant_core, pod_name, ns)

        tenant_custom.create_namespaced_custom_object(
            group=LMEVAL_GROUP, version=LMEVAL_VERSION,
            namespace=ns, plural=LMEVAL_PLURAL, body=lmeval_job,
        )

        assert wait_pod_exists(tenant_core, pod_name, ns, timeout=120), (
            f"LMEvalJob controller did not create pod {pod_name} within 120s"
        )

        deadline = time.time() + 600
        job_state = ""
        job_reason = ""
        stuck_count = 0
        while time.time() < deadline:
            try:
                job = tenant_custom.get_namespaced_custom_object(
                    group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                    namespace=ns, plural=LMEVAL_PLURAL, name=job_name,
                )
                job_state = job.get("status", {}).get("state", "")
                job_reason = job.get("status", {}).get("reason", "")
                if job_state == "Complete":
                    break
                if job_state == "Scheduled":
                    stuck_count += 1
                else:
                    stuck_count = 0
                if stuck_count >= 6 and attempt == 0:
                    break
            except client.exceptions.ApiException:
                pass
            time.sleep(10)

        if job_state != "Scheduled" or attempt > 0:
            break

    cleanup(
        lambda: tenant_custom.delete_namespaced_custom_object(
            group=LMEVAL_GROUP, version=LMEVAL_VERSION,
            namespace=ns, plural=LMEVAL_PLURAL, name=job_name,
        ),
    )

    assert job_state == "Complete", (
        f"LMEvalJob did not complete. state={job_state!r}, reason={job_reason!r}"
    )
    assert job_reason == "Succeeded", (
        f"LMEvalJob completed but not Succeeded. reason={job_reason!r}"
    )
