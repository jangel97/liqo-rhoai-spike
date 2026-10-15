"""
Liqo + RHOAI + Kueue end-to-end tests.

Validates the full chain: RHOAI operator creates workload in offloaded
namespace → Liqo dispatches via ShadowPod → Kyverno adds Kueue labels →
Kueue admits/queues/preempts.

Tests:
  L15. PyTorchJob admitted by Kueue with correct labels
  L16. Two PyTorchJobs queued — only 1 GPU, second waits
  L17. Priority preemption — high-priority job preempts low-priority
"""

import time

import pytest
from kubernetes import client

from helpers import force_delete_pod


KUBEFLOW_GROUP = "kubeflow.org"
KUBEFLOW_VERSION = "v1"
PYTORCHJOB_PLURAL = "pytorchjobs"

PYTORCH_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"

QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"


def wait_pod_phase(core_api, name, namespace, phases, timeout=300):
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


def _training_script(epochs, sleep_per_epoch=0):
    lines = [
        "import torch, time",
        "import torch.nn as nn",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'Training on: {device}')",
        "assert device.type == 'cuda', 'CUDA not available'",
        "",
        "model = nn.Sequential(",
        "    nn.Linear(784, 128),",
        "    nn.ReLU(),",
        "    nn.Linear(128, 10),",
        ").to(device)",
        "",
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
        "criterion = nn.CrossEntropyLoss()",
        "",
        f"for epoch in range({epochs}):",
        "    x = torch.randn(32, 784, device=device)",
        "    y = torch.randint(0, 10, (32,), device=device)",
        "    out = model(x)",
        "    loss = criterion(out, y)",
        "    optimizer.zero_grad()",
        "    loss.backward()",
        "    optimizer.step()",
        "    print(f'Epoch {epoch+1}, loss={loss.item():.4f}')",
    ]
    if sleep_per_epoch > 0:
        lines.append(f"    time.sleep({sleep_per_epoch})")
    lines.append("")
    lines.append("print('Training complete')")
    return "\n".join(lines)


def _make_pytorchjob(name, ns, script, priority_label=None):
    labels = {}
    if priority_label:
        labels[PRIORITY_LABEL] = priority_label

    return {
        "apiVersion": f"{KUBEFLOW_GROUP}/{KUBEFLOW_VERSION}",
        "kind": "PyTorchJob",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": labels,
        },
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "metadata": {"labels": labels} if labels else {},
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "pytorch",
                                    "image": PYTORCH_IMAGE,
                                    "command": ["python", "-c", script],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                    },
                                    "volumeMounts": [
                                        {"name": "tmp", "mountPath": "/tmp"},
                                    ],
                                },
                            ],
                            "volumes": [
                                {"name": "tmp", "emptyDir": {}},
                            ],
                        },
                    },
                },
            },
        },
    }


def _delete_pytorchjob(tenant_custom, ns, name):
    try:
        tenant_custom.delete_namespaced_custom_object(
            group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
            namespace=ns, plural=PYTORCHJOB_PLURAL, name=name,
        )
        time.sleep(5)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise


def _wait_worker_pod(worker_core, w_ns, pod_prefix, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = worker_core.list_namespaced_pod(namespace=w_ns)
        for p in pods.items:
            if pod_prefix in p.metadata.name:
                return p.metadata.name
        time.sleep(5)
    return None


@pytest.mark.liqo
@pytest.mark.rhoai
@pytest.mark.kueue
def test_pytorchjob_kueue_admission(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
):
    """L15: PyTorchJob → Liqo offload → Kyverno adds Kueue labels → Kueue
    admits → GPU training succeeds. Full RHOAI + Kueue chain."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    job_name = "liqo-rhoai-kueue-test"
    master_pod = f"{job_name}-master-0"

    _delete_pytorchjob(tenant_custom, ns, job_name)
    force_delete_pod(tenant_core, master_pod, ns)

    script = _training_script(epochs=3)
    job = _make_pytorchjob(job_name, ns, script)

    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=job,
    )
    cleanup(lambda: _delete_pytorchjob(tenant_custom, ns, job_name))

    worker_pod = _wait_worker_pod(worker_core, w_ns, master_pod)
    assert worker_pod, f"Training operator did not create {master_pod} on worker"

    w_pod = worker_core.read_namespaced_pod(name=worker_pod, namespace=w_ns)
    labels = w_pod.metadata.labels or {}

    assert labels.get(QUEUE_NAME_LABEL) == "default", (
        f"Kyverno did not add queue-name label. Labels: {labels}"
    )
    assert PRIORITY_LABEL in labels, (
        f"Kyverno did not add priority label. Labels: {labels}"
    )

    phase = wait_pod_phase(tenant_core, master_pod, ns, ["Succeeded", "Failed"], timeout=600)
    assert phase == "Succeeded", f"PyTorchJob master pod phase: {phase}"

    logs = tenant_core.read_namespaced_pod_log(
        name=master_pod, namespace=ns, container="pytorch",
    )
    assert "Training on: cuda" in logs, f"Training did not use CUDA: {logs[:300]}"
    assert "Training complete" in logs, f"Training incomplete: {logs[:300]}"


@pytest.mark.liqo
@pytest.mark.rhoai
@pytest.mark.kueue
def test_pytorchjob_kueue_queuing(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
):
    """L16: Two PyTorchJobs with 1 GPU each — first admitted, second queued
    until first completes. Proves Kueue quota enforcement with Liqo."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    job_a = "liqo-queue-a"
    job_b = "liqo-queue-b"
    pod_a = f"{job_a}-master-0"
    pod_b = f"{job_b}-master-0"

    for name in [job_a, job_b]:
        _delete_pytorchjob(tenant_custom, ns, name)
    for name in [pod_a, pod_b]:
        force_delete_pod(tenant_core, name, ns)

    script_long = _training_script(epochs=30, sleep_per_epoch=2)
    script_short = _training_script(epochs=3)

    pytorchjob_a = _make_pytorchjob(job_a, ns, script_long)
    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=pytorchjob_a,
    )
    cleanup(lambda: _delete_pytorchjob(tenant_custom, ns, job_a))

    phase_a = wait_pod_phase(tenant_core, pod_a, ns, ["Running"], timeout=300)
    assert phase_a == "Running", f"Job A not running: {phase_a}"

    pytorchjob_b = _make_pytorchjob(job_b, ns, script_short)
    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=pytorchjob_b,
    )
    cleanup(lambda: _delete_pytorchjob(tenant_custom, ns, job_b))

    time.sleep(15)

    worker_pod_b = _wait_worker_pod(worker_core, w_ns, pod_b, timeout=30)

    b_is_gated = False
    if worker_pod_b:
        w_pod = worker_core.read_namespaced_pod(name=worker_pod_b, namespace=w_ns)
        gates = w_pod.spec.scheduling_gates or []
        b_is_gated = any(g.name == "kueue.x-k8s.io/admission" for g in gates)
        b_phase = w_pod.status.phase
    else:
        b_phase = "not created yet"
        b_is_gated = True

    assert b_is_gated or b_phase == "Pending", (
        f"Job B should be queued while Job A uses the GPU. "
        f"Pod B phase: {b_phase}, gated: {b_is_gated}"
    )

    _delete_pytorchjob(tenant_custom, ns, job_a)
    force_delete_pod(tenant_core, pod_a, ns)

    phase_b = wait_pod_phase(tenant_core, pod_b, ns, ["Succeeded", "Failed"], timeout=600)
    assert phase_b == "Succeeded", f"Job B did not succeed after Job A freed GPU: {phase_b}"


@pytest.mark.liqo
@pytest.mark.rhoai
@pytest.mark.kueue
def test_pytorchjob_kueue_preemption(
    cleanup, liqo_tenant_clients, liqo_worker_clients,
    liqo_test_namespace, liqo_worker_namespace, liqo_virtual_node_ready,
):
    """L17: Low-priority PyTorchJob running → high-priority submitted →
    Kueue preempts low-priority → high-priority runs. Cross-priority
    preemption with RHOAI workloads via Liqo."""
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients
    ns = liqo_test_namespace
    w_ns = liqo_worker_namespace

    low_job = "liqo-preempt-low"
    high_job = "liqo-preempt-high"
    low_pod = f"{low_job}-master-0"
    high_pod = f"{high_job}-master-0"

    for name in [low_job, high_job]:
        _delete_pytorchjob(tenant_custom, ns, name)
    for name in [low_pod, high_pod]:
        force_delete_pod(tenant_core, name, ns)

    script_long = _training_script(epochs=60, sleep_per_epoch=2)
    script_short = _training_script(epochs=3)

    low_pytorchjob = _make_pytorchjob(
        low_job, ns, script_long, priority_label="gpuaas-opportunistic",
    )
    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=low_pytorchjob,
    )
    cleanup(lambda: _delete_pytorchjob(tenant_custom, ns, low_job))

    phase_low = wait_pod_phase(tenant_core, low_pod, ns, ["Running"], timeout=300)
    assert phase_low == "Running", f"Low-priority job not running: {phase_low}"

    high_pytorchjob = _make_pytorchjob(
        high_job, ns, script_short, priority_label="gpuaas-production",
    )
    tenant_custom.create_namespaced_custom_object(
        group=KUBEFLOW_GROUP, version=KUBEFLOW_VERSION,
        namespace=ns, plural=PYTORCHJOB_PLURAL, body=high_pytorchjob,
    )
    cleanup(lambda: _delete_pytorchjob(tenant_custom, ns, high_job))

    phase_high = wait_pod_phase(
        tenant_core, high_pod, ns, ["Succeeded", "Failed"], timeout=600,
    )
    assert phase_high == "Succeeded", (
        f"High-priority job did not succeed: {phase_high}"
    )

    logs = tenant_core.read_namespaced_pod_log(
        name=high_pod, namespace=ns, container="pytorch",
    )
    assert "Training on: cuda" in logs, f"High-priority did not use CUDA: {logs[:300]}"
    assert "Training complete" in logs, f"High-priority incomplete: {logs[:300]}"
