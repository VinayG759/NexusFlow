"""
NexusFlow One-Click Deployment Pipeline
Deploys generated projects to GitHub + Render + Vercel automatically.
"""

import httpx
import base64
import json
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
        self.render_api_key = settings.RENDER_API_KEY
        self.vercel_token = settings.VERCEL_TOKEN
        self.github_username = settings.GITHUB_USERNAME
        logger.info("DeployPipeline initialised")

    async def deploy_project(self, project_name: str, files: list[dict]) -> dict:
        """Full deployment pipeline: GitHub → Render → Vercel"""
        results = {
            "status": "deploying",
            "github_repo": None,
            "render_url": None,
            "vercel_url": None,
            "errors": []
        }

        # Ensure Register.tsx is present before pushing (prevents Vercel build failures)
        files = self._ensure_register_stub(files)

        # Step 1: Push to GitHub
        logger.info("DeployPipeline: pushing to GitHub...")
        github_result = await self._push_to_github(project_name, files)
        if github_result["status"] == "success":
            results["github_repo"] = github_result["repo_url"]
            logger.info("DeployPipeline: GitHub push successful: %s", github_result["repo_url"])
        else:
            results["errors"].append(f"GitHub: {github_result['error']}")
            return {**results, "status": "failed"}

        # Step 2: Deploy backend to Render
        logger.info("DeployPipeline: deploying backend to Render...")
        render_result = await self._deploy_to_render(
            project_name,
            github_result["repo_url"],
            github_result["repo_name"]
        )
        if render_result["status"] == "success":
            results["render_url"] = render_result["url"]
        else:
            results["errors"].append(f"Render: {render_result['error']}")

        # Step 3: Deploy frontend to Vercel
        logger.info("DeployPipeline: deploying frontend to Vercel...")
        vercel_result = await self._deploy_to_vercel(
            project_name,
            github_result["repo_url"],
            github_result["repo_name"],
            results["render_url"] or "http://localhost:8001"
        )
        if vercel_result["status"] == "success":
            results["vercel_url"] = vercel_result["url"]
        else:
            results["errors"].append(f"Vercel: {vercel_result['error']}")

        results["status"] = "success" if not results["errors"] else "partial"
        return results

    def _ensure_register_stub(self, files: list[dict]) -> list[dict]:
        """Inject Register.tsx stub if auth is detected but the page is missing."""
        paths = {f.get("path", "") for f in files}
        has_auth = any("auth.py" in p or "AuthContext.tsx" in p for p in paths)
        has_login = any("pages/Login.tsx" in p for p in paths)
        has_register = any("pages/Register.tsx" in p for p in paths)
        if has_auth and has_login and not has_register:
            files = list(files)
            files.append({"path": "frontend/src/pages/Register.tsx", "content": _REGISTER_STUB})
            logger.info("DeployPipeline: injected Register.tsx stub before GitHub push")
        return files

    async def _push_to_github(self, project_name: str, files: list[dict]) -> dict:
        """Create GitHub repo and push all project files."""
        if not self.github_token or not self.github_username:
            return {"status": "error", "error": "GITHUB_TOKEN or GITHUB_USERNAME not set"}

        repo_name = f"nexusflow-{project_name.lower().replace(' ', '-')}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        async with httpx.AsyncClient(timeout=30) as client:
            # Create repo
            create_response = await client.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json={
                    "name": repo_name,
                    "description": f"Generated by NexusFlow - {project_name}",
                    "private": False,
                    "auto_init": True
                }
            )

            if create_response.status_code not in (200, 201, 422):
                return {"status": "error", "error": f"Failed to create repo: {create_response.text}"}

            repo_url = f"https://github.com/{self.github_username}/{repo_name}"

            # Push each file
            for f in files:
                file_path = f.get("path", "")
                content = f.get("content", "")
                if not file_path or not content:
                    continue

                encoded = base64.b64encode(content.encode()).decode()

                # Check if file exists (for update)
                get_response = await client.get(
                    f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{file_path}",
                    headers=headers
                )

                payload = {
                    "message": f"Add {file_path}",
                    "content": encoded
                }

                if get_response.status_code == 200:
                    payload["sha"] = get_response.json().get("sha", "")

                await client.put(
                    f"https://api.github.com/repos/{self.github_username}/{repo_name}/contents/{file_path}",
                    headers=headers,
                    json=payload
                )

            return {
                "status": "success",
                "repo_url": repo_url,
                "repo_name": repo_name
            }

    async def _deploy_to_render(self, project_name: str, repo_url: str, repo_name: str) -> dict:
        """Deploy backend to Render via API."""
        if not self.render_api_key:
            return {"status": "error", "error": "RENDER_API_KEY not set"}

        headers = {
            "Authorization": f"Bearer {self.render_api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch owner ID (required by Render API v1)
            owners_resp = await client.get("https://api.render.com/v1/owners", headers=headers)
            if owners_resp.status_code != 200:
                return {"status": "error", "error": f"Failed to get Render owner ID: {owners_resp.text}"}
            owners = owners_resp.json()
            owner_id = owners[0]["owner"]["id"] if owners else None
            if not owner_id:
                return {"status": "error", "error": "No Render owner account found"}

            response = await client.post(
                "https://api.render.com/v1/services",
                headers=headers,
                json={
                    "type": "web_service",
                    "name": f"{project_name}-backend",
                    "ownerId": owner_id,
                    "repo": repo_url,
                    "branch": "main",
                    "serviceDetails": {
                        "env": "python",
                        "plan": "free",
                        "pullRequestPreviewsEnabled": "no",
                        "rootDir": "backend",
                        "envSpecificDetails": {
                            "buildCommand": "pip install -r requirements.txt",
                            "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
                        },
                        "envVars": [
                            {"key": "PYTHON_VERSION", "value": "3.11.0"},
                            {"key": "DATABASE_URL", "value": settings.DATABASE_URL}
                        ]
                    }
                }
            )

            if response.status_code in (200, 201):
                data = response.json()
                service_url = data.get("service", {}).get("serviceDetails", {}).get("url", "")
                if service_url and not service_url.startswith("http"):
                    service_url = f"https://{service_url}"
                return {"status": "success", "url": service_url}
            else:
                return {"status": "error", "error": f"Render API error: {response.text}"}

    async def _deploy_to_vercel(self, project_name: str, repo_url: str, repo_name: str, backend_url: str) -> dict:
        """Deploy frontend to Vercel via API (create project → trigger deployment)."""
        if not self.vercel_token:
            return {"status": "error", "error": "VERCEL_TOKEN not set"}

        headers = {
            "Authorization": f"Bearer {self.vercel_token}",
            "Content-Type": "application/json"
        }

        project_slug = f"{repo_name}-frontend"

        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Create Vercel project linked to GitHub repo
            create_resp = await client.post(
                "https://api.vercel.com/v9/projects",
                headers=headers,
                json={
                    "name": project_slug,
                    "gitRepository": {
                        "type": "github",
                        "repo": f"{self.github_username}/{repo_name}"
                    },
                    "rootDirectory": "frontend",
                    "buildCommand": "npm run build",
                    "outputDirectory": "dist",
                    "environmentVariables": [
                        {
                            "key": "VITE_API_URL",
                            "value": backend_url,
                            "type": "plain",
                            "target": ["production"]
                        }
                    ]
                }
            )

            if create_resp.status_code == 409:
                # Project already exists — fetch its ID
                get_resp = await client.get(
                    f"https://api.vercel.com/v9/projects/{project_slug}",
                    headers=headers
                )
                project_id = get_resp.json().get("id", "") if get_resp.status_code == 200 else ""
            elif create_resp.status_code in (200, 201):
                project_id = create_resp.json().get("id", "")
            else:
                return {"status": "error", "error": f"Vercel project creation error: {create_resp.text}"}

            if not project_id:
                return {"status": "error", "error": "Could not obtain Vercel project ID"}

            # Step 2: Trigger deployment from GitHub main branch
            deploy_resp = await client.post(
                "https://api.vercel.com/v13/deployments",
                headers=headers,
                json={
                    "name": project_slug,
                    "project": project_id,
                    "gitSource": {
                        "type": "github",
                        "org": self.github_username,
                        "repo": repo_name,
                        "ref": "main"
                    },
                    "target": "production"
                }
            )

            if deploy_resp.status_code in (200, 201):
                data = deploy_resp.json()
                url = data.get("url", "")
                return {"status": "success", "url": f"https://{url}" if url and not url.startswith("http") else url}
            else:
                return {"status": "error", "error": f"Vercel deploy error: {deploy_resp.text}"}

    async def get_deploy_status(self, service_id: str) -> dict:
        """Check deployment status on Render."""
        if not self.render_api_key:
            return {"status": "error"}

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"https://api.render.com/v1/services/{service_id}",
                headers={"Authorization": f"Bearer {self.render_api_key}"}
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "error"}


deploy_pipeline = DeployPipeline()
