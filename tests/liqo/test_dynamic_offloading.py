"""Dynamic namespace offloading via Kyverno generate policy.

Proves that a GPU pod in a fresh (never-offloaded) namespace triggers
automatic namespace offloading through a Kyverno generate rule, without
any manual `liqoctl offload namespace` step.

Prerequisites:
  - Kyverno ClusterPolicy `auto-offload-gpu-namespace` applied on tenant
  - Kyverno ClusterRole `kyverno-generate-namespace-offloading` applied
  - sno-tenant-rhoai3 and sno-worker running

Run:
  RHOAI3_KUBECONFIG=~/.kube/tenant-rhoai3 \\
  WORKER_KUBECONFIG=~/.kube/worker \\
    python -m pytest tests/liqo/test_dynamic_offloading.py -v -s
"""

import time

import pytest
from kubernetes import client


OFFLOADING_GROUP = "offloading.liqo.io"
OFFLOADING_VERSION = "v1beta1"
OFFLOADING_PLURAL = "namespaceoffloadings"

DYNAMIC_NS = "gpu-dynamic-test"
IDEMPOTENT_NS = "gpu-idempotent-test"
DELETION_NS = "gpu-deletion-test"


def _wait_for_offloading_cr(custom_api, namespace, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            custom_api.get_namespaced_custom_object(
                group=OFFLOADING_GROUP, version=OFFLOADING_VERSION,
                namespace=namespace, plural=OFFLOADING_PLURAL,
                name="offloading",
            )
            return True
        except client.exceptions.ApiException:
            pass
        time.sleep(5)
    return False


def _wait_for_twin_namespace(worker_core, namespace, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            worker_core.read_namespace(name=namespace)
            return True
        except client.exceptions.ApiException:
            pass
        time.sleep(5)
    return False


def _wait_pod_phase(core_api, name, namespace, phases, timeout=300):
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


@pytest.mark.liqo
def test_dynamic_namespace_offloading(
    liqo_tenant_clients, liqo_worker_clients, liqo_virtual_node_ready,
):
    """GPU pod in a fresh namespace triggers auto-offloading via Kyverno.

    1. Create a new namespace (no offloading configured)
    2. Create a GPU pod
    3. Kyverno generates the NamespaceOffloading CR
    4. Liqo processes it, creates twin namespace on worker
    5. Pod runs on worker GPU and completes
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients

    try:
        tenant_core.delete_namespace(name=DYNAMIC_NS)
        time.sleep(10)
    except client.exceptions.ApiException:
        pass

    print(f"\n--- Creating fresh namespace '{DYNAMIC_NS}' ---")
    tenant_core.create_namespace(
        body=client.V1Namespace(
            metadata=client.V1ObjectMeta(name=DYNAMIC_NS),
        ),
    )

    try:
        pod_name = "gpu-auto-offload-test"
        print(f"--- Creating GPU pod '{pod_name}' (no offloading configured yet) ---")
        tenant_core.create_namespaced_pod(
            namespace=DYNAMIC_NS,
            body=client.V1Pod(
                metadata=client.V1ObjectMeta(name=pod_name, namespace=DYNAMIC_NS),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="gpu",
                            image="nvcr.io/nvidia/cuda:12.8.0-base-ubi9",
                            command=["nvidia-smi"],
                            resources=client.V1ResourceRequirements(
                                limits={"nvidia.com/gpu": "1"},
                            ),
                        ),
                    ],
                ),
            ),
        )

        print("--- Waiting for Kyverno to generate NamespaceOffloading CR ---")
        offloading_created = _wait_for_offloading_cr(tenant_custom, DYNAMIC_NS)
        print(f"  NamespaceOffloading CR created: {offloading_created}")
        assert offloading_created, (
            f"Kyverno did not generate NamespaceOffloading CR in '{DYNAMIC_NS}' within 60s"
        )

        cr = tenant_custom.get_namespaced_custom_object(
            group=OFFLOADING_GROUP, version=OFFLOADING_VERSION,
            namespace=DYNAMIC_NS, plural=OFFLOADING_PLURAL,
            name="offloading",
        )
        spec = cr.get("spec", {})
        print(f"  Strategy: pod={spec.get('podOffloadingStrategy')}, "
              f"ns={spec.get('namespaceMappingStrategy')}")

        print("--- Waiting for twin namespace on worker ---")
        twin_exists = _wait_for_twin_namespace(worker_core, DYNAMIC_NS)
        print(f"  Twin namespace '{DYNAMIC_NS}' on worker: {twin_exists}")
        assert twin_exists, (
            f"Liqo did not create twin namespace '{DYNAMIC_NS}' on worker within 120s"
        )

        print("--- Checking if pod completes on first attempt ---")
        phase = _wait_pod_phase(
            tenant_core, pod_name, DYNAMIC_NS,
            phases=("Succeeded", "Failed"), timeout=300,
        )
        print(f"  Pod phase (first attempt): {phase}")

        if phase not in ("Succeeded", "Failed"):
            print("--- First pod stuck, deleting and recreating ---")
            try:
                tenant_core.delete_namespaced_pod(
                    name=pod_name, namespace=DYNAMIC_NS,
                    grace_period_seconds=0,
                )
                time.sleep(10)
            except client.exceptions.ApiException:
                pass

            tenant_core.create_namespaced_pod(
                namespace=DYNAMIC_NS,
                body=client.V1Pod(
                    metadata=client.V1ObjectMeta(
                        name=pod_name, namespace=DYNAMIC_NS,
                    ),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[
                            client.V1Container(
                                name="gpu",
                                image="nvcr.io/nvidia/cuda:12.8.0-base-ubi9",
                                command=["nvidia-smi"],
                                resources=client.V1ResourceRequirements(
                                    limits={"nvidia.com/gpu": "1"},
                                ),
                            ),
                        ],
                    ),
                ),
            )

            phase = _wait_pod_phase(
                tenant_core, pod_name, DYNAMIC_NS,
                phases=("Succeeded", "Failed"), timeout=300,
            )
            print(f"  Pod phase (second attempt): {phase}")

        assert phase == "Succeeded", (
            f"GPU pod did not succeed in dynamic namespace. phase={phase!r}"
        )

        worker_pod = worker_core.read_namespaced_pod(
            name=pod_name, namespace=DYNAMIC_NS,
        )
        print(f"  Worker pod phase: {worker_pod.status.phase}")
        print(f"  Worker pod node: {worker_pod.spec.node_name}")

        print("\n--- Result: dynamic namespace offloading works ---")

    finally:
        print(f"\n--- Cleanup: deleting namespace '{DYNAMIC_NS}' ---")
        try:
            tenant_core.delete_namespace(name=DYNAMIC_NS)
        except client.exceptions.ApiException:
            pass


def _wait_ns_gone(core_api, namespace, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            core_api.read_namespace(name=namespace)
        except client.exceptions.ApiException:
            return True
        time.sleep(5)
    return False


@pytest.mark.liqo
def test_offloading_idempotent(
    liqo_tenant_clients, liqo_worker_clients, liqo_virtual_node_ready,
):
    """Second GPU pod in an already-offloaded namespace does not conflict.

    Kyverno's generate rule fires for every GPU pod, but the
    NamespaceOffloading CR already exists. Verifies the second pod
    works without errors.
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients

    try:
        tenant_core.delete_namespace(name=IDEMPOTENT_NS)
        _wait_ns_gone(tenant_core, IDEMPOTENT_NS)
    except client.exceptions.ApiException:
        pass

    print(f"\n--- Creating namespace '{IDEMPOTENT_NS}' ---")
    tenant_core.create_namespace(
        body=client.V1Namespace(metadata=client.V1ObjectMeta(name=IDEMPOTENT_NS)),
    )

    try:
        pod1 = "gpu-first"
        print(f"--- Creating first GPU pod '{pod1}' ---")
        tenant_core.create_namespaced_pod(
            namespace=IDEMPOTENT_NS,
            body=client.V1Pod(
                metadata=client.V1ObjectMeta(name=pod1, namespace=IDEMPOTENT_NS),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="gpu",
                            image="nvcr.io/nvidia/cuda:12.8.0-base-ubi9",
                            command=["nvidia-smi"],
                            resources=client.V1ResourceRequirements(
                                limits={"nvidia.com/gpu": "1"},
                            ),
                        ),
                    ],
                ),
            ),
        )

        assert _wait_for_offloading_cr(tenant_custom, IDEMPOTENT_NS), (
            "NamespaceOffloading CR not created for first pod"
        )
        assert _wait_for_twin_namespace(worker_core, IDEMPOTENT_NS), (
            "Twin namespace not created on worker"
        )

        phase1 = _wait_pod_phase(
            tenant_core, pod1, IDEMPOTENT_NS,
            phases=("Succeeded", "Failed"), timeout=300,
        )
        print(f"  First pod phase: {phase1}")
        assert phase1 == "Succeeded", f"First pod failed: {phase1}"

        pod2 = "gpu-second"
        print(f"--- Creating second GPU pod '{pod2}' (offloading already exists) ---")
        tenant_core.create_namespaced_pod(
            namespace=IDEMPOTENT_NS,
            body=client.V1Pod(
                metadata=client.V1ObjectMeta(name=pod2, namespace=IDEMPOTENT_NS),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="gpu",
                            image="nvcr.io/nvidia/cuda:12.8.0-base-ubi9",
                            command=["nvidia-smi"],
                            resources=client.V1ResourceRequirements(
                                limits={"nvidia.com/gpu": "1"},
                            ),
                        ),
                    ],
                ),
            ),
        )

        phase2 = _wait_pod_phase(
            tenant_core, pod2, IDEMPOTENT_NS,
            phases=("Succeeded", "Failed"), timeout=300,
        )
        print(f"  Second pod phase: {phase2}")
        assert phase2 == "Succeeded", f"Second pod failed: {phase2}"

        cr = tenant_custom.get_namespaced_custom_object(
            group=OFFLOADING_GROUP, version=OFFLOADING_VERSION,
            namespace=IDEMPOTENT_NS, plural=OFFLOADING_PLURAL,
            name="offloading",
        )
        print(f"  NamespaceOffloading CR still exists: True")
        print(f"  CR resourceVersion: {cr['metadata']['resourceVersion']}")

        print("\n--- Result: idempotent, second pod works without conflict ---")

    finally:
        print(f"\n--- Cleanup: deleting namespace '{IDEMPOTENT_NS}' ---")
        try:
            tenant_core.delete_namespace(name=IDEMPOTENT_NS)
        except client.exceptions.ApiException:
            pass


@pytest.mark.liqo
def test_namespace_deletion_propagates_to_worker(
    liqo_tenant_clients, liqo_worker_clients, liqo_virtual_node_ready,
):
    """Deleting the tenant namespace cleans up the twin namespace on the worker.

    When the namespace is deleted on the tenant, the NamespaceOffloading
    CR is deleted with it (cascade). Liqo should detect this and remove
    the twin namespace on the worker.
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    worker_core, _ = liqo_worker_clients

    try:
        tenant_core.delete_namespace(name=DELETION_NS)
        _wait_ns_gone(tenant_core, DELETION_NS)
    except client.exceptions.ApiException:
        pass
    _wait_ns_gone(worker_core, DELETION_NS, timeout=30)

    print(f"\n--- Creating namespace '{DELETION_NS}' ---")
    tenant_core.create_namespace(
        body=client.V1Namespace(metadata=client.V1ObjectMeta(name=DELETION_NS)),
    )

    pod_name = "gpu-deletion-test"
    print(f"--- Creating GPU pod to trigger offloading ---")
    tenant_core.create_namespaced_pod(
        namespace=DELETION_NS,
        body=client.V1Pod(
            metadata=client.V1ObjectMeta(name=pod_name, namespace=DELETION_NS),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[
                    client.V1Container(
                        name="gpu",
                        image="nvcr.io/nvidia/cuda:12.8.0-base-ubi9",
                        command=["nvidia-smi"],
                        resources=client.V1ResourceRequirements(
                            limits={"nvidia.com/gpu": "1"},
                        ),
                    ),
                ],
            ),
        ),
    )

    assert _wait_for_twin_namespace(worker_core, DELETION_NS), (
        "Twin namespace not created on worker"
    )
    print(f"  Twin namespace '{DELETION_NS}' exists on worker")

    phase = _wait_pod_phase(
        tenant_core, pod_name, DELETION_NS,
        phases=("Succeeded", "Failed"), timeout=300,
    )
    print(f"  Pod phase: {phase}")

    print(f"--- Deleting namespace '{DELETION_NS}' on tenant ---")
    tenant_core.delete_namespace(name=DELETION_NS)
    assert _wait_ns_gone(tenant_core, DELETION_NS), (
        f"Namespace '{DELETION_NS}' not deleted on tenant within 120s"
    )
    print(f"  Tenant namespace deleted")

    print(f"--- Checking if twin namespace is cleaned up on worker ---")
    twin_gone = _wait_ns_gone(worker_core, DELETION_NS, timeout=120)
    print(f"  Twin namespace gone from worker: {twin_gone}")
    assert twin_gone, (
        f"Twin namespace '{DELETION_NS}' still exists on worker after "
        f"tenant namespace deletion. Liqo did not propagate the deletion."
    )
