"""
Liqo spike test fixtures.

Namespace offloading uses LocalAndRemote strategy — the kube-scheduler
places GPU pods on the Liqo virtual node (the only node with
nvidia.com/gpu in allocatable). Liqo's webhook adds tolerations for the
virtual node taint but does NOT set nodeName. No explicit scheduling
constraints needed on the pod spec — resource requests drive placement.

The worker (twin) namespace has the same name as the tenant namespace
when using EnforceSameName mapping strategy.
"""

import logging
import os
import time

import pytest
import urllib3
from kubernetes import client, config

from helpers import VK_TEST_NAMESPACE, safe_delete

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger(__name__)

LIQO_TEST_NAMESPACE = VK_TEST_NAMESPACE

DSC_GROUP = "datasciencecluster.opendatahub.io"
DSC_VERSION = "v1"
DSC_PLURAL = "datascienceclusters"
DSC_NAME = "default-dsc"
RHOAI_APPS_NS = "redhat-ods-applications"


def _load_clients(kubeconfig_path):
    api_client = config.new_client_from_config(config_file=kubeconfig_path)
    return client.CoreV1Api(api_client), client.CustomObjectsApi(api_client)


def _enable_dsc_component(tenant_custom, component):
    """Enable a DSC component, return original state for teardown."""
    dsc = tenant_custom.get_cluster_custom_object(
        group=DSC_GROUP, version=DSC_VERSION, plural=DSC_PLURAL, name=DSC_NAME,
    )
    original = (
        dsc.get("spec", {}).get("components", {})
        .get(component, {}).get("managementState", "Removed")
    )
    if original != "Managed":
        tenant_custom.patch_cluster_custom_object(
            group=DSC_GROUP, version=DSC_VERSION,
            plural=DSC_PLURAL, name=DSC_NAME,
            body={"spec": {"components": {component: {"managementState": "Managed"}}}},
        )
        log.info("DSC %s → Managed (was %s)", component, original)
    return original


def _revert_dsc_component(tenant_custom, component, original_state):
    if original_state != "Managed":
        try:
            tenant_custom.patch_cluster_custom_object(
                group=DSC_GROUP, version=DSC_VERSION,
                plural=DSC_PLURAL, name=DSC_NAME,
                body={"spec": {"components": {component: {"managementState": original_state}}}},
            )
            log.info("DSC %s → %s (reverted)", component, original_state)
        except Exception as exc:
            log.warning("Failed to revert DSC %s: %s", component, exc)


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def cleanup():
    _callbacks = []

    def register(fn, *args, **kwargs):
        _callbacks.append((fn, args, kwargs))

    yield register

    for fn, args, kwargs in reversed(_callbacks):
        safe_delete(fn, *args, **kwargs)


@pytest.fixture(scope="session")
def liqo_tenant_kubeconfig():
    return os.environ.get(
        "RHOAI3_KUBECONFIG", os.path.expanduser("~/.kube/tenant-rhoai3"),
    )


@pytest.fixture(scope="session")
def liqo_tenant_clients(liqo_tenant_kubeconfig):
    return _load_clients(liqo_tenant_kubeconfig)


@pytest.fixture(scope="session")
def liqo_worker_clients():
    path = os.environ.get(
        "WORKER_KUBECONFIG", os.path.expanduser("~/.kube/worker"),
    )
    return _load_clients(path)


@pytest.fixture(scope="session")
def liqo_test_namespace():
    return LIQO_TEST_NAMESPACE


@pytest.fixture(scope="session")
def liqo_worker_namespace():
    """Twin namespace on worker — same name with EnforceSameName strategy."""
    return LIQO_TEST_NAMESPACE


@pytest.fixture(scope="session")
def liqo_virtual_node(liqo_tenant_clients):
    """Discover the Liqo virtual node name on the tenant."""
    tenant_core, _ = liqo_tenant_clients
    deadline = time.time() + 120
    while time.time() < deadline:
        nodes = tenant_core.list_node(label_selector="liqo.io/type=virtual-node")
        if nodes.items:
            return nodes.items[0].metadata.name
        time.sleep(10)
    pytest.fail("No Liqo virtual node found within 120s")


@pytest.fixture(scope="session")
def liqo_virtual_node_ready(liqo_tenant_clients, liqo_virtual_node):
    """Wait for the Liqo virtual node to be Ready."""
    tenant_core, _ = liqo_tenant_clients
    deadline = time.time() + 120
    while time.time() < deadline:
        node = tenant_core.read_node(name=liqo_virtual_node)
        for cond in (node.status.conditions or []):
            if cond.type == "Ready" and cond.status == "True":
                return True
        time.sleep(10)
    pytest.fail(f"Liqo virtual node {liqo_virtual_node} not Ready within 120s")


@pytest.fixture(scope="session")
def rhoai2_tenant_clients():
    path = os.environ.get(
        "TENANT_KUBECONFIG", os.path.expanduser("~/.kube/tenant"),
    )
    return _load_clients(path)


LMEVAL_REQUIRED_COMPONENTS = {"kserve", "trustyai"}
LMEVAL_DISABLED_COMPONENTS = [
    "dashboard", "workbenches", "ray", "training",
    "datasciencepipelines", "modelmeshserving", "codeflare",
    "modelregistry", "kueue",
]


def _wait_for_rhods_operator(core_api, timeout=300):
    """Wait for rhods-operator to be ready (webhook must be reachable)."""
    apps = client.AppsV1Api(core_api.api_client)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            dep = apps.read_namespaced_deployment(
                "rhods-operator", "redhat-ods-operator",
            )
            if (dep.status.ready_replicas or 0) >= 1:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def _patch_dsc_with_retry(custom_api, comp, state, retries=6, delay=10):
    for attempt in range(retries):
        try:
            custom_api.patch_cluster_custom_object(
                group=DSC_GROUP, version=DSC_VERSION,
                plural=DSC_PLURAL, name=DSC_NAME,
                body={"spec": {"components": {comp: {"managementState": state}}}},
            )
            return True
        except Exception as exc:
            if attempt < retries - 1:
                log.info("DSC patch %s → %s failed (attempt %d), retrying: %s",
                         comp, state, attempt + 1, exc)
                time.sleep(delay)
            else:
                log.warning("Failed to set DSC %s → %s after %d attempts: %s",
                            comp, state, retries, exc)
    return False


def _minimize_dsc_for_lmeval(core_api, custom_api):
    """Disable all DSC components except kserve and trustyai.

    Waits for the rhods-operator webhook to be ready before patching.
    Returns a dict of {component: original_state} for teardown.
    """
    log.info("Waiting for rhods-operator to be ready before DSC changes...")
    if not _wait_for_rhods_operator(core_api):
        log.warning("rhods-operator not ready after 300s, proceeding anyway")

    dsc = None
    for attempt in range(6):
        try:
            dsc = custom_api.get_cluster_custom_object(
                group=DSC_GROUP, version=DSC_VERSION, plural=DSC_PLURAL, name=DSC_NAME,
            )
            break
        except Exception as exc:
            if attempt < 5:
                log.info("DSC GET failed (attempt %d), retrying in 10s: %s", attempt + 1, exc)
                time.sleep(10)
            else:
                raise
    components = dsc.get("spec", {}).get("components", {})
    originals = {}

    for comp in LMEVAL_DISABLED_COMPONENTS:
        state = components.get(comp, {}).get("managementState", "Removed")
        originals[comp] = state
        if state == "Managed":
            if _patch_dsc_with_retry(custom_api, comp, "Removed"):
                log.info("DSC %s → Removed (was %s)", comp, state)

    for comp in LMEVAL_REQUIRED_COMPONENTS:
        state = components.get(comp, {}).get("managementState", "Removed")
        originals[comp] = state
        if state != "Managed":
            if _patch_dsc_with_retry(custom_api, comp, "Managed"):
                log.info("DSC %s → Managed (was %s)", comp, state)

    return originals


def _restore_dsc(custom_api, originals):
    for comp, state in originals.items():
        current_dsc = custom_api.get_cluster_custom_object(
            group=DSC_GROUP, version=DSC_VERSION, plural=DSC_PLURAL, name=DSC_NAME,
        )
        current = (
            current_dsc.get("spec", {}).get("components", {})
            .get(comp, {}).get("managementState", "Removed")
        )
        if current != state:
            try:
                custom_api.patch_cluster_custom_object(
                    group=DSC_GROUP, version=DSC_VERSION,
                    plural=DSC_PLURAL, name=DSC_NAME,
                    body={"spec": {"components": {comp: {"managementState": state}}}},
                )
                log.info("DSC %s → %s (restored)", comp, state)
            except Exception as exc:
                log.warning("Failed to restore DSC %s: %s", comp, exc)


def _wait_trustyai_ready(core_api, timeout=180):
    """Wait for TrustyAI controller deployment to exist and have ready pods."""
    apps = client.AppsV1Api(core_api.api_client)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            dep = apps.read_namespaced_deployment(
                "trustyai-service-operator-controller-manager",
                RHOAI_APPS_NS,
            )
            if (dep.status.ready_replicas or 0) >= 1:
                log.info("TrustyAI controller ready")
                return True
        except client.exceptions.ApiException:
            pass
        time.sleep(10)
    log.warning("TrustyAI controller not ready after %ds", timeout)
    return False


@pytest.fixture(scope="session")
def rhoai2_lmeval_minimal(request, rhoai2_tenant_clients):
    """Strip RHOAI 2.x DSC to kserve + trustyai only."""
    core, custom = rhoai2_tenant_clients
    originals = _minimize_dsc_for_lmeval(core, custom)
    request.addfinalizer(lambda: _restore_dsc(custom, originals))
    _wait_trustyai_ready(core)


@pytest.fixture(scope="session")
def rhoai3_lmeval_minimal(request, liqo_tenant_clients):
    """Strip RHOAI 3.x DSC to kserve + trustyai only."""
    core, custom = liqo_tenant_clients
    originals = _minimize_dsc_for_lmeval(core, custom)
    request.addfinalizer(lambda: _restore_dsc(custom, originals))
    _wait_trustyai_ready(core)


@pytest.fixture(scope="session")
def rhoai2_vk_node_ready(rhoai2_tenant_clients):
    """Wait for the custom VK node gpu-worker on the RHOAI 2.x tenant."""
    tenant_core, _ = rhoai2_tenant_clients
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            node = tenant_core.read_node(name="gpu-worker")
            for cond in (node.status.conditions or []):
                if cond.type == "Ready" and cond.status == "True":
                    return True
        except client.exceptions.ApiException:
            pass
        time.sleep(10)
    pytest.fail("VK node gpu-worker not Ready on RHOAI 2.x tenant within 120s")


@pytest.fixture(scope="session")
def rhoai_full_environment(request, liqo_tenant_clients):
    """Enable RHOAI DSC components for full e2e coverage.

    Enables workbenches and ray in the DataScienceCluster.
    Cleanup always runs via addfinalizer — even if setup fails.
    """
    tenant_core, tenant_custom = liqo_tenant_clients
    apps = client.AppsV1Api(tenant_core.api_client)

    original_workbenches = _enable_dsc_component(tenant_custom, "workbenches")
    original_ray = _enable_dsc_component(tenant_custom, "ray")

    def _teardown():
        _revert_dsc_component(tenant_custom, "workbenches", original_workbenches)
        _revert_dsc_component(tenant_custom, "ray", original_ray)

    request.addfinalizer(_teardown)

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            dep = apps.read_namespaced_deployment(
                "notebook-controller-deployment", RHOAI_APPS_NS,
            )
            if (dep.status.ready_replicas or 0) >= 1:
                break
        except client.exceptions.ApiException:
            pass
        time.sleep(10)
    else:
        pytest.fail("notebook-controller-deployment not ready within 300s")

    deadline = time.time() + 300
    while time.time() < deadline:
        pods = tenant_core.list_namespaced_pod(
            namespace=RHOAI_APPS_NS,
            label_selector="app.kubernetes.io/name=kuberay",
        )
        running = [p for p in pods.items if p.status.phase == "Running"]
        if running:
            break
        time.sleep(10)
    else:
        pytest.fail("KubeRay operator pod not Running within 300s")
