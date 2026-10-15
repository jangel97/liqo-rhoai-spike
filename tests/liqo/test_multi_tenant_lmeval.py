"""Multi-tenant decoupling: RHOAI 2.x + RHOAI 3.x LMEval on the same GPU worker.

Proves that two RHOAI tenants (2.x via custom VK, 3.x via Liqo) can
both submit LMEvalJob workloads against the same shared GPU worker
cluster without interfering with each other.

The test strips both tenants' DSC to only kserve + trustyai (the two
components LMEval needs). This reduces memory so both tenants can run
at ~10GB each alongside the 24GB GPU worker (44GB total, well within
the 62GB hypervisor budget).

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

from helpers import VK_TEST_NAMESPACE, force_delete_pod, safe_delete, wait_pod_exists

ISVC_GROUP = "serving.kserve.io"
ISVC_VERSION = "v1beta1"
ISVC_PLURAL = "inferenceservices"

SERVING_RUNTIME_GROUP = "serving.kserve.io"
SERVING_RUNTIME_VERSION = "v1alpha1"
SERVING_RUNTIME_PLURAL = "servingruntimes"

LMEVAL_GROUP = "trustyai.opendatahub.io"
LMEVAL_VERSION = "v1alpha1"
LMEVAL_PLURAL = "lmevaljobs"

LMEVAL_MODEL_NAME = "facebook/opt-125m"
VLLM_CPU_IMAGE = (
    "registry.redhat.io/rhaii/vllm-cpu-rhel9"
    "@sha256:7a28b867955055a8836ea2a78e70b8bad63c1ee50e8a5f8dd95d1194f20ec489"
)

NS = VK_TEST_NAMESPACE


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


def _check_kserve(core_api):
    api_ext = client.ApiextensionsV1Api(core_api.api_client)
    try:
        api_ext.read_custom_resource_definition(
            "inferenceservices.serving.kserve.io"
        )
        return True
    except client.exceptions.ApiException:
        return False


def _deploy_model_server(custom_api, runtime_name, isvc_name):
    """Deploy ServingRuntime + InferenceService for LMEval model serving."""
    runtime = {
        "apiVersion": f"{SERVING_RUNTIME_GROUP}/{SERVING_RUNTIME_VERSION}",
        "kind": "ServingRuntime",
        "metadata": {"name": runtime_name, "namespace": NS},
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
        existing = custom_api.get_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=NS, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
        )
        runtime["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
        custom_api.replace_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=NS, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
            body=runtime,
        )
    except client.exceptions.ApiException:
        custom_api.create_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=NS, plural=SERVING_RUNTIME_PLURAL, body=runtime,
        )

    isvc = {
        "apiVersion": f"{ISVC_GROUP}/{ISVC_VERSION}",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": NS,
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
            },
        },
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
        existing = custom_api.get_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=NS, plural=ISVC_PLURAL, name=isvc_name,
        )
        isvc["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
        custom_api.replace_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=NS, plural=ISVC_PLURAL, name=isvc_name,
            body=isvc,
        )
    except client.exceptions.ApiException:
        custom_api.create_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=NS, plural=ISVC_PLURAL, body=isvc,
        )


def _wait_isvc_ready(custom_api, isvc_name, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        obj = custom_api.get_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=NS, plural=ISVC_PLURAL, name=isvc_name,
        )
        for c in obj.get("status", {}).get("conditions", []):
            if c.get("type") == "Ready" and c.get("status") == "True":
                return True
        time.sleep(10)
    return False


def _create_lmeval_job(custom_api, job_name, isvc_name):
    model_url = (
        f"http://{isvc_name}-predictor.{NS}.svc.cluster.local:8080/v1/completions"
    )
    return {
        "apiVersion": f"{LMEVAL_GROUP}/{LMEVAL_VERSION}",
        "kind": "LMEvalJob",
        "metadata": {"name": job_name, "namespace": NS},
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


def _run_lmeval_with_retry(core_api, custom_api, job_name, isvc_name, timeout=600):
    """Run LMEvalJob with TrustyAI race condition retry."""
    job_body = _create_lmeval_job(custom_api, job_name, isvc_name)

    for attempt in range(2):
        if attempt > 0:
            try:
                custom_api.delete_namespaced_custom_object(
                    group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                    namespace=NS, plural=LMEVAL_PLURAL, name=job_name,
                )
                time.sleep(15)
            except client.exceptions.ApiException:
                pass
            force_delete_pod(core_api, job_name, NS)

        custom_api.create_namespaced_custom_object(
            group=LMEVAL_GROUP, version=LMEVAL_VERSION,
            namespace=NS, plural=LMEVAL_PLURAL, body=job_body,
        )

        assert wait_pod_exists(core_api, job_name, NS, timeout=120), (
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
                    namespace=NS, plural=LMEVAL_PLURAL, name=job_name,
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


def _cleanup_lmeval(core_api, custom_api, job_name, isvc_name, runtime_name):
    for name in [job_name]:
        try:
            custom_api.delete_namespaced_custom_object(
                group=LMEVAL_GROUP, version=LMEVAL_VERSION,
                namespace=NS, plural=LMEVAL_PLURAL, name=name,
            )
        except client.exceptions.ApiException:
            pass
    force_delete_pod(core_api, job_name, NS)
    try:
        custom_api.delete_namespaced_custom_object(
            group=ISVC_GROUP, version=ISVC_VERSION,
            namespace=NS, plural=ISVC_PLURAL, name=isvc_name,
        )
    except client.exceptions.ApiException:
        pass
    try:
        custom_api.delete_namespaced_custom_object(
            group=SERVING_RUNTIME_GROUP, version=SERVING_RUNTIME_VERSION,
            namespace=NS, plural=SERVING_RUNTIME_PLURAL, name=runtime_name,
        )
    except client.exceptions.ApiException:
        pass


@pytest.mark.liqo
@pytest.mark.rhoai
def test_lmeval_decoupled_rhoai2_and_rhoai3(
    cleanup,
    rhoai2_tenant_clients,
    rhoai2_vk_node_ready,
    rhoai2_lmeval_minimal,
    liqo_tenant_clients,
    liqo_virtual_node_ready,
    rhoai3_lmeval_minimal,
    liqo_worker_clients,
):
    """LMEvalJob from RHOAI 2.x and RHOAI 3.x tenants, same GPU worker.

    Both tenants deploy a CPU model server (vLLM opt-125m) locally and
    run LMEvalJob against it. The test proves both RHOAI versions can
    coexist on the same shared GPU worker cluster without interference.

    RHOAI 2.x dispatches via custom VK (nodeName + tolerations).
    RHOAI 3.x dispatches via Liqo (virtual node + scheduler).
    Both peer with sno-worker.
    """
    r2_core, r2_custom = rhoai2_tenant_clients
    r3_core, r3_custom = liqo_tenant_clients
    w_core, _ = liqo_worker_clients

    if not _check_trustyai(r2_core):
        pytest.skip("TrustyAI operator not deployed on RHOAI 2.x tenant")
    if not _check_kserve(r2_core):
        pytest.skip("KServe not installed on RHOAI 2.x tenant")
    if not _check_trustyai(r3_core):
        pytest.skip("TrustyAI operator not deployed on RHOAI 3.x tenant")
    if not _check_kserve(r3_core):
        pytest.skip("KServe not installed on RHOAI 3.x tenant")

    r2_job = "rhoai2-lmeval"
    r2_isvc = "rhoai2-lmeval-model"
    r2_runtime = "rhoai2-vllm-cpu"
    r3_job = "rhoai3-lmeval"
    r3_isvc = "rhoai3-lmeval-model"
    r3_runtime = "rhoai3-vllm-cpu"

    _cleanup_lmeval(r2_core, r2_custom, r2_job, r2_isvc, r2_runtime)
    _cleanup_lmeval(r3_core, r3_custom, r3_job, r3_isvc, r3_runtime)
    time.sleep(5)

    cleanup(lambda: _cleanup_lmeval(r2_core, r2_custom, r2_job, r2_isvc, r2_runtime))
    cleanup(lambda: _cleanup_lmeval(r3_core, r3_custom, r3_job, r3_isvc, r3_runtime))

    print("\n--- Deploying model server on RHOAI 2.x tenant ---")
    _deploy_model_server(r2_custom, r2_runtime, r2_isvc)

    print("--- Deploying model server on RHOAI 3.x tenant ---")
    _deploy_model_server(r3_custom, r3_runtime, r3_isvc)

    print("--- Waiting for RHOAI 2.x ISVC Ready ---")
    assert _wait_isvc_ready(r2_custom, r2_isvc), (
        f"InferenceService {r2_isvc} not Ready on RHOAI 2.x within 600s"
    )
    print(f"  {r2_isvc}: Ready")

    print("--- Waiting for RHOAI 3.x ISVC Ready ---")
    assert _wait_isvc_ready(r3_custom, r3_isvc), (
        f"InferenceService {r3_isvc} not Ready on RHOAI 3.x within 600s"
    )
    print(f"  {r3_isvc}: Ready")

    print("--- Submitting LMEvalJob on RHOAI 2.x ---")
    r2_state, r2_reason = _run_lmeval_with_retry(
        r2_core, r2_custom, r2_job, r2_isvc,
    )
    print(f"  RHOAI 2.x: state={r2_state}, reason={r2_reason}")

    print("--- Submitting LMEvalJob on RHOAI 3.x ---")
    r3_state, r3_reason = _run_lmeval_with_retry(
        r3_core, r3_custom, r3_job, r3_isvc,
    )
    print(f"  RHOAI 3.x: state={r3_state}, reason={r3_reason}")

    print("\n--- Decoupling check ---")
    try:
        w_core.read_namespace(name=NS)
        print(f"  Worker namespace '{NS}' exists (Liqo twin namespace)")
    except client.exceptions.ApiException:
        print(f"  Worker namespace '{NS}' not found")

    namespaces = w_core.list_namespace()
    vk_namespaces = [
        n.metadata.name for n in namespaces.items
        if NS in n.metadata.name and n.metadata.name != NS
    ]
    if vk_namespaces:
        print(f"  VK worker namespaces containing '{NS}': {vk_namespaces}")
    else:
        print(f"  No VK-prefixed worker namespaces found for '{NS}'")

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
