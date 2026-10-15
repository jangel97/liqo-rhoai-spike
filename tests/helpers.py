"""Shared test utilities and constants."""

import time

from kubernetes import client


VK_TEST_NAMESPACE = "vk-test"


def safe_delete(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def wait_pod_exists(core_api, name, namespace, timeout=120):
    """Wait for a pod to exist. Returns True if found."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            core_api.read_namespaced_pod(name=name, namespace=namespace)
            return True
        except client.exceptions.ApiException:
            pass
        time.sleep(3)
    return False


def force_delete_pod(core_api, name, namespace, timeout=60):
    try:
        core_api.delete_namespaced_pod(
            name=name, namespace=namespace, grace_period_seconds=0,
        )
    except client.exceptions.ApiException:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            core_api.read_namespaced_pod(name=name, namespace=namespace)
        except client.exceptions.ApiException:
            return
        time.sleep(2)
