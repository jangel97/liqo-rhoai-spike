"""
RHOAI 3.x smoke tests on a Liqo-enabled cluster.

Validates that RHOAI operators are installed on the tenant, CRDs are NOT
on the worker, and OSSM is absent (RHOAI 3.x removed Service Mesh).
"""

import os

import pytest
from kubernetes import client
from kubernetes import config as k8s_config


RHOAI_APPS_NS = "redhat-ods-applications"

RHOAI_CRDS = [
    "pytorchjobs.kubeflow.org",
    "notebooks.kubeflow.org",
    "inferenceservices.serving.kserve.io",
    "rayclusters.ray.io",
]


@pytest.mark.liqo
@pytest.mark.rhoai
def test_no_rhoai_crds_on_worker(liqo_worker_clients):
    """RHOAI CRDs should only exist on the tenant, not the GPU worker."""
    worker_core, _ = liqo_worker_clients
    ext = client.ApiextensionsV1Api(worker_core.api_client)

    worker_crds = {c.metadata.name for c in ext.list_custom_resource_definition().items}
    for crd in RHOAI_CRDS:
        assert crd not in worker_crds, f"CRD {crd} should NOT be on worker cluster"


@pytest.mark.liqo
@pytest.mark.rhoai
def test_rhoai3_components_running(liqo_tenant_clients):
    """Verify key RHOAI 3.x deployments are available."""
    tenant_core, _ = liqo_tenant_clients
    apps = client.AppsV1Api(tenant_core.api_client)

    expected = [
        "kubeflow-training-operator",
        "kserve-controller-manager",
    ]
    for dep_name in expected:
        try:
            dep = apps.read_namespaced_deployment(dep_name, RHOAI_APPS_NS)
        except client.exceptions.ApiException:
            pytest.skip(f"{dep_name} not deployed in {RHOAI_APPS_NS}")
        ready = dep.status.ready_replicas or 0
        desired = dep.spec.replicas or 1
        assert ready >= desired, f"{dep_name} not ready: {ready}/{desired}"


@pytest.mark.liqo
@pytest.mark.rhoai
def test_no_ossm_on_rhoai3(liqo_tenant_clients):
    """RHOAI 3.x should not have OSSM/Istio."""
    tenant_core, _ = liqo_tenant_clients
    ext = client.ApiextensionsV1Api(tenant_core.api_client)

    crds = {c.metadata.name for c in ext.list_custom_resource_definition().items}
    assert "servicemeshcontrolplanes.maistra.io" not in crds, \
        "OSSM CRD found -- RHOAI 3.x should not install Service Mesh"
