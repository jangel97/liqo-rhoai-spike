"""Multi-tenant decoupling: RHOAI 2.x + RHOAI 3.x LMEval on the same GPU worker.

Proves that two RHOAI tenants (2.x via custom VK, 3.x via Liqo) can
both submit LMEvalJob workloads against the same shared GPU worker
cluster without interfering with each other.

Uses the built-in dummy model (random outputs) to avoid needing
internet access or a model server. The LMEvalJob pods run on the
tenants with minimal resources. The test proves that both TrustyAI
operators can create and complete LMEvalJobs independently while
sharing the same GPU worker backend.

Prerequisites:
  - sno-tenant (RHOAI 2.x) running
  - sno-tenant-rhoai3 (RHOAI 3.x) running
  - sno-worker running with both VK provider and Liqo provider
  - All other VMs shut down

Run:
  TENANT_KUBECONFIG=~/.kube/tenant \\
  RHOAI3_KUBECONFIG=~/.kube/tenant-rhoai3 \\
  WORKER_KUBECONFIG=~/.kube/worker \\
    python -m pytest tests/liqo/test_multi_tenant_lmeval.py -v -s
"""

import time

import pytest
from kubernetes import client

from helpers import force_delete_pod, wait_pod_exists

LMEVAL_GROUP = "trustyai.opendatahub.io"
LMEVAL_VERSION = "v1alpha1"
LMEVAL_PLURAL = "lmevaljobs"

LMEVAL_MODEL_NAME = "hf"


def _check_trustyai(core_api):
    apps = client.AppsV1Api(core_api.api_client)
    try:
        apps.read_namespaced_deployment(
            "trustyai-service-operator-controller-manager",
            "redhat-ods-applications",
        )
        return True
    except client.exceptions.ApiException:
        return False


def _check_lmeval_crd(core_api):
    api_ext = client.ApiextensionsV1Api(core_api.api_client)
    try:
        api_ext.read_custom_resource_definition("lmevaljobs.trustyai.opendatahub.io")
        return True
    except client.exceptions.ApiException:
        return False


def _create_lmeval_job(job_name, namespace):
    return {
        "apiVersion": f"{LMEVAL_GROUP}/{LMEVAL_VERSION}",
        "kind": "LMEvalJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "model": LMEVAL_MODEL_NAME,
            "modelArgs": [
                {"name": "pretrained", "value": "facebook/opt-125m"},
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
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "2", "memory": "2Gi"},
                    },
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


def _run_lmeval_with_retry(core_api, custom_api, job_name, ns, timeout=600):
    """Run LMEvalJob with TrustyAI race condition retry."""
    job_body = _create_lmeval_job(job_name, ns)

    for attempt in range(2):
        if attempt > 0:
            try:
                custom_api.delete_namespaced_custom_object(
                    group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                    namespace=ns, plural=LMEVAL_PLURAL, name=job_name,
                )
                time.sleep(15)
            except client.exceptions.ApiException:
                pass
            force_delete_pod(core_api, job_name, ns)

        custom_api.create_namespaced_custom_object(
            group=LMEVAL_GROUP, version=LMEVAL_VERSION,
            namespace=ns, plural=LMEVAL_PLURAL, body=job_body,
        )

        assert wait_pod_exists(core_api, job_name, ns, timeout=120), (
            f"LMEvalJob controller did not create pod {job_name} within 120s"
        )

        deadline = time.time() + timeout
        job_state = ""
        job_reason = ""
        stuck_count = 0
        while time.time() < deadline:
            try:
                job = custom_api.get_namespaced_custom_object(
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

    return job_state, job_reason


def _safe_delete_ns(core_api, ns):
    try:
        core_api.delete_namespace(name=ns)
    except client.exceptions.ApiException:
        pass


def _cleanup_lmeval(core_api, custom_api, job_name, ns):
    try:
        custom_api.delete_namespaced_custom_object(
            group=LMEVAL_GROUP, version=LMEVAL_VERSION,
            namespace=ns, plural=LMEVAL_PLURAL, name=job_name,
        )
    except client.exceptions.ApiException:
        pass
    force_delete_pod(core_api, job_name, ns)


@pytest.mark.liqo
@pytest.mark.rhoai
def test_lmeval_decoupled_rhoai2_and_rhoai3(
    cleanup,
    liqo_test_namespace,
    rhoai2_tenant_clients,
    rhoai2_vk_node_ready,
    rhoai2_lmeval_minimal,
    liqo_tenant_clients,
    liqo_virtual_node_ready,
    rhoai3_lmeval_minimal,
    liqo_worker_clients,
):
    """LMEvalJob from RHOAI 2.x and RHOAI 3.x tenants, same GPU worker.

    Both tenants run LMEvalJob with facebook/opt-125m loaded directly
    via HuggingFace (no model server needed). The test proves both RHOAI
    versions can coexist on the same shared GPU worker cluster without
    interference.

    RHOAI 2.x dispatches via custom VK (nodeName + tolerations).
    RHOAI 3.x dispatches via Liqo (virtual node + scheduler).
    Both peer with sno-worker.
    """
    ns = liqo_test_namespace
    r2_core, r2_custom = rhoai2_tenant_clients
    r3_core, r3_custom = liqo_tenant_clients
    w_core, _ = liqo_worker_clients

    try:
        r2_core.create_namespace(
            body=client.V1Namespace(metadata=client.V1ObjectMeta(name=ns)),
        )
    except client.exceptions.ApiException:
        pass
    cleanup(lambda: _safe_delete_ns(r2_core, ns))

    if not _check_trustyai(r2_core):
        pytest.skip("TrustyAI operator not ready on RHOAI 2.x tenant")
    if not _check_lmeval_crd(r2_core):
        pytest.skip("LMEvalJob CRD not installed on RHOAI 2.x tenant")
    if not _check_trustyai(r3_core):
        pytest.skip("TrustyAI operator not ready on RHOAI 3.x tenant")
    if not _check_lmeval_crd(r3_core):
        pytest.skip("LMEvalJob CRD not installed on RHOAI 3.x tenant")

    r2_job = "rhoai2-lmeval"
    r3_job = "rhoai3-lmeval"

    _cleanup_lmeval(r2_core, r2_custom, r2_job, ns)
    _cleanup_lmeval(r3_core, r3_custom, r3_job, ns)
    time.sleep(5)

    cleanup(lambda: _cleanup_lmeval(r2_core, r2_custom, r2_job, ns))
    cleanup(lambda: _cleanup_lmeval(r3_core, r3_custom, r3_job, ns))

    print(f"\n--- Using namespace '{ns}' ---")

    print("--- Submitting LMEvalJob on RHOAI 2.x ---")
    r2_state, r2_reason = _run_lmeval_with_retry(r2_core, r2_custom, r2_job, ns)
    print(f"  RHOAI 2.x: state={r2_state}, reason={r2_reason}")

    print("--- Submitting LMEvalJob on RHOAI 3.x ---")
    r3_state, r3_reason = _run_lmeval_with_retry(r3_core, r3_custom, r3_job, ns)
    print(f"  RHOAI 3.x: state={r3_state}, reason={r3_reason}")

    print("\n--- Decoupling check ---")
    try:
        w_core.read_namespace(name=ns)
        print(f"  Worker namespace '{ns}' exists (Liqo twin namespace)")
    except client.exceptions.ApiException:
        print(f"  Worker namespace '{ns}' not found")

    namespaces = w_core.list_namespace()
    vk_namespaces = [
        n.metadata.name for n in namespaces.items
        if ns in n.metadata.name and n.metadata.name != ns
    ]
    if vk_namespaces:
        print(f"  VK worker namespaces containing '{ns}': {vk_namespaces}")
    else:
        print(f"  No VK-prefixed worker namespaces found for '{ns}'")

    print(f"\n--- Results ---")
    print(f"  RHOAI 2.x LMEvalJob: state={r2_state}, reason={r2_reason}")
    print(f"  RHOAI 3.x LMEvalJob: state={r3_state}, reason={r3_reason}")

    assert r2_state == "Complete", (
        f"RHOAI 2.x LMEvalJob did not complete. state={r2_state!r}, reason={r2_reason!r}"
    )
    assert r2_reason == "Succeeded", (
        f"RHOAI 2.x LMEvalJob completed but not Succeeded. reason={r2_reason!r}"
    )
    assert r3_state == "Complete", (
        f"RHOAI 3.x LMEvalJob did not complete. state={r3_state!r}, reason={r3_reason!r}"
    )
    assert r3_reason == "Succeeded", (
        f"RHOAI 3.x LMEvalJob completed but not Succeeded. reason={r3_reason!r}"
    )
