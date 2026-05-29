"""
NexusFlow Deployment Pipeline
Pushes generated projects to GitHub automatically.
"""

import base64
import json
import httpx
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REGISTER_STUB = """import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export default function Register() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await axios.post(`${API_URL}/api/auth/register`, { name, email, password });
      navigate('/login');
    } catch (err: unknown) {
      setError('Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Create account</h1>
        <p className="text-gray-500 mb-8">Sign up to get started</p>
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500"
              placeholder="Your name" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500"
              placeholder="you@example.com" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500"
              placeholder="••••••••" />
          </div>
          <button type="submit" disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors">
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>
        <p className="mt-6 text-center text-gray-500">
          Already have an account? <Link to="/login" className="text-blue-600 font-semibold hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
"""


class DeployPipeline:
    def __init__(self):
        self.github_token = settings.GITHUB_TOKEN
        self.github_username = settings.GITHUB_USERNAME
        logger.info("DeployPipeline initialised (GitHub-only mode)")

    async def deploy_project(self, project_name: str, files: list[dict]) -> dict:
        """Push project files to a new GitHub repository."""
        results: dict = {
            "status": "deploying",
            "github_repo": None,
            "errors": [],
        }

        files = self._ensure_register_stub(files)

        logger.info("DeployPipeline: pushing to GitHub...")
        github_result = await self._push_to_github(project_name, files)
        if github_result["status"] == "success":
            results["github_repo"] = github_result["repo_url"]
            results["status"] = "success"
            logger.info("DeployPipeline: GitHub push successful: %s", github_result["repo_url"])
        else:
            results["errors"].append(f"GitHub: {github_result['error']}")
            results["status"] = "failed"

        return results

    def _ensure_register_stub(self, files: list[dict]) -> list[dict]:
        """Inject a Register.tsx stub if auth is detected but the page is missing."""
        paths = {f.get("path", "") for f in files}
        has_auth = any("auth.py" in p or "AuthContext.tsx" in p for p in paths)
        has_login = any("pages/Login.tsx" in p for p in paths)
        has_register = any("pages/Register.tsx" in p for p in paths)
        if has_auth and has_login and not has_register:
            files = list(files)
            files.append({"path": "frontend/src/pages/Register.tsx", "content": _REGISTER_STUB})
            logger.info("DeployPipeline: injected Register.tsx stub")
        return files

    async def _push_to_github(self, project_name: str, files: list[dict]) -> dict:
        """Create a GitHub repo and push all project files."""
        if not self.github_token or not self.github_username:
            return {"status": "error", "error": "GITHUB_TOKEN or GITHUB_USERNAME not configured in .env"}

        repo_name = f"nexusflow-{project_name.lower().replace(' ', '-')}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            # Create repository
            create_resp = await client.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json={
                    "name": repo_name,
                    "description": f"Generated by NexusFlow — {project_name}",
                    "private": False,
                    "auto_init": True,
                },
            )
            if create_resp.status_code not in (200, 201, 422):
                return {"status": "error", "error": f"Failed to create repo: {create_resp.text}"}

            repo_url = f"https://github.com/{self.github_username}/{repo_name}"

            # Push each file
            for f in files:
                file_path = f.get("path", "")
                content = f.get("content", "")
                if not file_path or not content:
                    continue

                encoded = base64.b64encode(content.encode()).decode()

                # Fetch existing SHA (for updates)
                get_resp = await client.get(
                    f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{file_path}",
                    headers=headers,
                )
                payload: dict = {"message": f"Add {file_path}", "content": encoded}
                if get_resp.status_code == 200:
                    payload["sha"] = get_resp.json().get("sha", "")

                await client.put(
                    f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{file_path}",
                    headers=headers,
                    json=payload,
                )

            # Push Tailwind config if project uses Tailwind
            index_css = next(
                (f.get("content", "") for f in files if f.get("path", "").endswith("frontend/src/index.css")),
                "",
            )
            if "@tailwind" in index_css:
                tailwind_config = """/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
"""
                postcss_config = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""
                for tw_path, tw_content in [
                    ("frontend/tailwind.config.js", tailwind_config),
                    ("frontend/postcss.config.js", postcss_config),
                ]:
                    get_tw = await client.get(
                        f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{tw_path}",
                        headers=headers,
                    )
                    tw_payload: dict = {
                        "message": f"Add {tw_path}",
                        "content": base64.b64encode(tw_content.encode()).decode(),
                    }
                    if get_tw.status_code == 200:
                        tw_payload["sha"] = get_tw.json().get("sha", "")
                    await client.put(
                        f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{tw_path}",
                        headers=headers,
                        json=tw_payload,
                    )
                    logger.info("DeployPipeline: pushed %s", tw_path)

            return {"status": "success", "repo_url": repo_url, "repo_name": repo_name}

    async def get_deploy_status(self, project_id: str) -> dict:
        return {
            "status": "github_only",
            "message": "Project is deployed to GitHub. Check the repository for the latest status.",
        }

    def generate_k8s_manifests(
        self,
        project_name: str,
        backend_image: str = "",
        frontend_image: str = "",
        namespace: str = "default",
        backend_port: int = 8001,
        frontend_port: int = 80,
        domain: str = "",
    ) -> dict[str, str]:
        """Generate Kubernetes manifests for a generated full-stack project."""
        slug = project_name.lower().replace("_", "-")
        be_image = backend_image or f"{slug}-backend:latest"
        fe_image = frontend_image or f"{slug}-frontend:latest"
        host = domain or f"{slug}.example.com"
        db_name = slug.replace("-", "_")

        backend_deployment = f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {slug}-backend
  namespace: {namespace}
  labels:
    app: {slug}-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {slug}-backend
  template:
    metadata:
      labels:
        app: {slug}-backend
    spec:
      containers:
        - name: backend
          image: {be_image}
          ports:
            - containerPort: {backend_port}
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {slug}-secrets
                  key: database-url
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /health
              port: {backend_port}
            initialDelaySeconds: 10
            periodSeconds: 5
"""

        backend_service = f"""\
apiVersion: v1
kind: Service
metadata:
  name: {slug}-backend
  namespace: {namespace}
spec:
  selector:
    app: {slug}-backend
  ports:
    - port: {backend_port}
      targetPort: {backend_port}
  type: ClusterIP
"""

        frontend_deployment = f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {slug}-frontend
  namespace: {namespace}
  labels:
    app: {slug}-frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {slug}-frontend
  template:
    metadata:
      labels:
        app: {slug}-frontend
    spec:
      containers:
        - name: frontend
          image: {fe_image}
          ports:
            - containerPort: {frontend_port}
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
          readinessProbe:
            httpGet:
              path: /
              port: {frontend_port}
            initialDelaySeconds: 5
            periodSeconds: 5
"""

        frontend_service = f"""\
apiVersion: v1
kind: Service
metadata:
  name: {slug}-frontend
  namespace: {namespace}
spec:
  selector:
    app: {slug}-frontend
  ports:
    - port: {frontend_port}
      targetPort: {frontend_port}
  type: ClusterIP
"""

        ingress = f"""\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {slug}-ingress
  namespace: {namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
spec:
  ingressClassName: nginx
  rules:
    - host: {host}
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: {slug}-backend
                port:
                  number: {backend_port}
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {slug}-frontend
                port:
                  number: {frontend_port}
"""

        secrets_template = f"""\
apiVersion: v1
kind: Secret
metadata:
  name: {slug}-secrets
  namespace: {namespace}
type: Opaque
stringData:
  database-url: "postgresql+asyncpg://postgres:CHANGE_ME@postgres-service:5432/{db_name}"
"""

        kustomization = f"""\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - backend-deployment.yaml
  - backend-service.yaml
  - frontend-deployment.yaml
  - frontend-service.yaml
  - ingress.yaml
  - secrets.yaml
namespace: {namespace}
"""

        logger.info("generate_k8s_manifests: generated 7 manifests for %r (namespace=%s)", project_name, namespace)
        return {
            "k8s/backend-deployment.yaml": backend_deployment,
            "k8s/backend-service.yaml": backend_service,
            "k8s/frontend-deployment.yaml": frontend_deployment,
            "k8s/frontend-service.yaml": frontend_service,
            "k8s/ingress.yaml": ingress,
            "k8s/secrets.yaml": secrets_template,
            "k8s/kustomization.yaml": kustomization,
        }


deploy_pipeline = DeployPipeline()
