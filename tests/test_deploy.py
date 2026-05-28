"""Unit tests for DeployPipeline.generate_k8s_manifests().

Pure Python — no database or HTTP calls needed.
"""

import pytest
import yaml

from src.utils.deploy_pipeline import DeployPipeline

EXPECTED_KEYS = {
    "k8s/backend-deployment.yaml",
    "k8s/backend-service.yaml",
    "k8s/frontend-deployment.yaml",
    "k8s/frontend-service.yaml",
    "k8s/ingress.yaml",
    "k8s/secrets.yaml",
    "k8s/kustomization.yaml",
}


@pytest.fixture
def pipeline():
    return DeployPipeline()


# ── manifest set ──────────────────────────────────────────────────────────────


def test_returns_seven_manifests(pipeline):
    result = pipeline.generate_k8s_manifests("todo-app")
    assert set(result.keys()) == EXPECTED_KEYS


def test_all_manifests_are_valid_yaml(pipeline):
    for path, content in pipeline.generate_k8s_manifests("todo-app").items():
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"{path} is not valid YAML: {e}")


# ── image defaults ────────────────────────────────────────────────────────────


def test_default_backend_image(pipeline):
    manifests = pipeline.generate_k8s_manifests("my-project")
    deployment = manifests["k8s/backend-deployment.yaml"]
    assert "my-project-backend:latest" in deployment


def test_default_frontend_image(pipeline):
    manifests = pipeline.generate_k8s_manifests("my-project")
    deployment = manifests["k8s/frontend-deployment.yaml"]
    assert "my-project-frontend:latest" in deployment


def test_custom_backend_image(pipeline):
    manifests = pipeline.generate_k8s_manifests(
        "my-project", backend_image="registry.example.com/backend:v2"
    )
    assert "registry.example.com/backend:v2" in manifests["k8s/backend-deployment.yaml"]


def test_custom_frontend_image(pipeline):
    manifests = pipeline.generate_k8s_manifests(
        "my-project", frontend_image="registry.example.com/frontend:v2"
    )
    assert "registry.example.com/frontend:v2" in manifests["k8s/frontend-deployment.yaml"]


# ── underscore → hyphen slug ──────────────────────────────────────────────────


def test_slug_converts_underscores(pipeline):
    manifests = pipeline.generate_k8s_manifests("my_app")
    deployment = manifests["k8s/backend-deployment.yaml"]
    assert "my-app-backend" in deployment
    # underscore form should NOT appear in image or resource names
    assert "my_app-backend" not in deployment


# ── domain / ingress ──────────────────────────────────────────────────────────


def test_custom_domain_in_ingress(pipeline):
    manifests = pipeline.generate_k8s_manifests("todo-app", domain="todo.myapp.com")
    assert "todo.myapp.com" in manifests["k8s/ingress.yaml"]


def test_default_domain_fallback(pipeline):
    manifests = pipeline.generate_k8s_manifests("todo-app")
    assert "todo-app.example.com" in manifests["k8s/ingress.yaml"]


def test_ingress_routes_api_to_backend(pipeline):
    ingress = pipeline.generate_k8s_manifests("todo-app")["k8s/ingress.yaml"]
    assert "/api" in ingress
    assert "todo-app-backend" in ingress


def test_ingress_routes_root_to_frontend(pipeline):
    ingress = pipeline.generate_k8s_manifests("todo-app")["k8s/ingress.yaml"]
    assert "todo-app-frontend" in ingress


# ── namespace ─────────────────────────────────────────────────────────────────


def test_default_namespace_is_default(pipeline):
    for path, content in pipeline.generate_k8s_manifests("todo-app").items():
        if path == "k8s/kustomization.yaml":
            continue
        assert "namespace: default" in content


def test_custom_namespace(pipeline):
    manifests = pipeline.generate_k8s_manifests("todo-app", namespace="production")
    for path, content in manifests.items():
        if path == "k8s/kustomization.yaml":
            assert "namespace: production" in content
        else:
            assert "namespace: production" in content


# ── secrets template ──────────────────────────────────────────────────────────


def test_secrets_contains_change_me_placeholder(pipeline):
    secrets = pipeline.generate_k8s_manifests("todo-app")["k8s/secrets.yaml"]
    assert "CHANGE_ME" in secrets


def test_secrets_db_name_derived_from_project(pipeline):
    secrets = pipeline.generate_k8s_manifests("my-project")["k8s/secrets.yaml"]
    # slug "my-project" → db_name "my_project"
    assert "my_project" in secrets


# ── kustomization ─────────────────────────────────────────────────────────────


def test_kustomization_lists_all_resources(pipeline):
    kustomization = pipeline.generate_k8s_manifests("todo-app")["k8s/kustomization.yaml"]
    for resource in (
        "backend-deployment.yaml",
        "backend-service.yaml",
        "frontend-deployment.yaml",
        "frontend-service.yaml",
        "ingress.yaml",
        "secrets.yaml",
    ):
        assert resource in kustomization


# ── readiness probes ──────────────────────────────────────────────────────────


def test_backend_readiness_probe_on_health(pipeline):
    deployment = pipeline.generate_k8s_manifests("todo-app")["k8s/backend-deployment.yaml"]
    assert "/health" in deployment
    assert "readinessProbe" in deployment


def test_frontend_readiness_probe_on_root(pipeline):
    deployment = pipeline.generate_k8s_manifests("todo-app")["k8s/frontend-deployment.yaml"]
    assert "readinessProbe" in deployment
