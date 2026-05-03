"""Curated training examples for NexusFlow debugging system.

235 examples across 11 error categories covering common failures during
project generation. Used for:
- Instant-fix lookup in TrainingCollector.get_instant_fix()
- Database seeding via seed_training_data()
- LLM fine-tuning export via /training/export
"""

TRAINING_DATA: list[dict] = [

    # ═══ CATEGORY 1: Missing Files (20 examples) ═══

    {
        "error": "Can't find file: ./index.css",
        "error_type": "missing_css",
        "fix": "Create src/index.css with base reset styles",
        "file_to_fix": "frontend/src/index.css",
        "fix_content": (
            "body {\n  margin: 0;\n  padding: 0;\n"
            "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;\n"
            "  -webkit-font-smoothing: antialiased;\n}\n\n"
            "* { box-sizing: border-box; }\n\n"
            "a { text-decoration: none; }\n\n"
            "button { cursor: pointer; border: none; outline: none; }"
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './App.css'",
        "error_type": "missing_css",
        "fix": "Create src/App.css with minimal app styles",
        "file_to_fix": "frontend/src/App.css",
        "fix_content": "/* App styles */\n.App {\n  min-height: 100vh;\n}",
        "instant_fix": True,
    },
    {
        "error": "public/index.html not found",
        "error_type": "missing_index_html",
        "fix": "Create public/index.html with React root div",
        "file_to_fix": "frontend/public/index.html",
        "fix_content": (
            '<!DOCTYPE html>\n<html lang="en">\n  <head>\n'
            '    <meta charset="utf-8" />\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
            "    <title>App</title>\n  </head>\n  <body>\n"
            '    <noscript>You need to enable JavaScript to run this app.</noscript>\n'
            '    <div id="root"></div>\n  </body>\n</html>'
        ),
        "instant_fix": True,
    },
    {
        "error": "ENOENT: no such file or directory, open 'frontend/.env'",
        "error_type": "missing_env",
        "fix": "Create frontend/.env with API URL",
        "file_to_fix": "frontend/.env",
        "fix_content": "REACT_APP_API_URL=http://localhost:8000",
        "instant_fix": True,
    },
    {
        "error": "ENOENT: no such file or directory, open 'frontend/public/favicon.ico'",
        "error_type": "missing_file",
        "fix": "Add a placeholder favicon or remove the favicon reference from index.html",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module './setupTests' from 'src/App.test.tsx'",
        "error_type": "missing_file",
        "fix": "Create src/setupTests.ts with import '@testing-library/jest-dom'",
        "file_to_fix": "frontend/src/setupTests.ts",
        "fix_content": "import '@testing-library/jest-dom';",
        "instant_fix": True,
    },
    {
        "error": "Module not found: Can't resolve './serviceWorker'",
        "error_type": "missing_file",
        "fix": "Remove the serviceWorker import from index.tsx or create a stub",
        "instant_fix": False,
    },
    {
        "error": "ENOENT: no such file or directory, open 'frontend/src/reportWebVitals.ts'",
        "error_type": "missing_file",
        "fix": "Create reportWebVitals.ts or remove its import from index.tsx",
        "file_to_fix": "frontend/src/reportWebVitals.ts",
        "fix_content": (
            "import { ReportHandler } from 'web-vitals';\n\n"
            "const reportWebVitals = (onPerfEntry?: ReportHandler) => {\n"
            "  if (onPerfEntry && onPerfEntry instanceof Function) {\n"
            "    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {\n"
            "      getCLS(onPerfEntry); getFID(onPerfEntry); getFCP(onPerfEntry);\n"
            "      getLCP(onPerfEntry); getTTFB(onPerfEntry);\n"
            "    });\n  }\n};\n\nexport default reportWebVitals;"
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module '../assets/logo.png'",
        "error_type": "missing_file",
        "fix": "Add a logo.png to frontend/src/assets/ or remove the import",
        "instant_fix": False,
    },
    {
        "error": "ENOENT: no such file or directory 'backend/requirements.txt'",
        "error_type": "missing_file",
        "fix": "Create backend/requirements.txt with required Python packages",
        "file_to_fix": "backend/requirements.txt",
        "fix_content": (
            "fastapi\nuvicorn[standard]\nsqlalchemy[asyncio]\nasyncpg\n"
            "python-dotenv\npydantic\nhttpx\nalembic\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "ENOENT: no such file or directory 'backend/.env'",
        "error_type": "missing_env",
        "fix": "Create backend/.env with default configuration",
        "file_to_fix": "backend/.env",
        "fix_content": (
            "DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/app\n"
            "SECRET_KEY=changeme\nDEBUG=True\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "ENOENT: no such file or directory 'frontend/tsconfig.json'",
        "error_type": "missing_file",
        "fix": "Create frontend/tsconfig.json with standard React TypeScript settings",
        "file_to_fix": "frontend/tsconfig.json",
        "fix_content": (
            '{\n  "compilerOptions": {\n    "target": "es5",\n'
            '    "lib": ["dom", "dom.iterable", "esnext"],\n'
            '    "allowJs": true,\n    "skipLibCheck": true,\n'
            '    "esModuleInterop": true,\n    "allowSyntheticDefaultImports": true,\n'
            '    "strict": true,\n    "module": "esnext",\n'
            '    "moduleResolution": "node",\n    "resolveJsonModule": true,\n'
            '    "isolatedModules": true,\n    "noEmit": true,\n'
            '    "jsx": "react-jsx"\n  },\n'
            '  "include": ["src"]\n}'
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './store' from 'src/App.tsx'",
        "error_type": "wrong_import_path",
        "fix": "Check if store file exists at src/store/index.ts or create a stub",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve './config'",
        "error_type": "wrong_import_path",
        "fix": "Create src/config.ts with application configuration constants",
        "file_to_fix": "frontend/src/config.ts",
        "fix_content": (
            "export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';\n"
            "export const APP_NAME = process.env.REACT_APP_NAME || 'App';\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './utils/helpers'",
        "error_type": "wrong_import_path",
        "fix": "Create src/utils/helpers.ts with common utility functions",
        "file_to_fix": "frontend/src/utils/helpers.ts",
        "fix_content": (
            "export const formatDate = (date: string | Date): string =>\n"
            "  new Date(date).toLocaleDateString();\n\n"
            "export const truncate = (str: string, n: number): string =>\n"
            "  str.length > n ? str.slice(0, n) + '...' : str;\n\n"
            "export const classNames = (...classes: string[]): string =>\n"
            "  classes.filter(Boolean).join(' ');\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './api/client'",
        "error_type": "wrong_import_path",
        "fix": "Create src/api/client.ts with axios instance",
        "file_to_fix": "frontend/src/api/client.ts",
        "fix_content": (
            "import axios from 'axios';\n\n"
            "const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';\n\n"
            "const client = axios.create({ baseURL: API_URL });\n\n"
            "export default client;\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './hooks/useAuth'",
        "error_type": "wrong_import_path",
        "fix": "Create src/hooks/useAuth.ts or fix the import path",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module './contexts/AuthContext'",
        "error_type": "wrong_import_path",
        "fix": "Create src/contexts/AuthContext.tsx with auth provider",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module './constants'",
        "error_type": "wrong_import_path",
        "fix": "Create src/constants.ts with application constants",
        "file_to_fix": "frontend/src/constants.ts",
        "fix_content": (
            "export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';\n"
            "export const MAX_RETRIES = 3;\n"
            "export const TIMEOUT_MS = 30000;\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "Module not found: Can't resolve './types'",
        "error_type": "wrong_import_path",
        "fix": "Create src/types.ts with TypeScript interfaces for the project",
        "instant_fix": False,
    },

    # ═══ CATEGORY 2: Wrong Import Paths (20 examples) ═══

    {
        "error": "Module not found: Error: Can't resolve './styles/globals.css'",
        "error_type": "wrong_import_path",
        "fix": "Replace './styles/globals.css' with './index.css' in the import",
        "pattern": "./styles/globals.css",
        "replacement": "./index.css",
        "instant_fix": True,
    },
    {
        "error": "Module not found: Error: Can't resolve '../styles/globals.css'",
        "error_type": "wrong_import_path",
        "fix": "Replace '../styles/globals.css' with '../index.css'",
        "pattern": "../styles/globals.css",
        "replacement": "../index.css",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module './components/Button' from 'src/App.tsx'",
        "error_type": "wrong_import_path",
        "fix": "Check if file exists at src/components/Button.tsx or Button/index.tsx",
        "instant_fix": False,
        "llm_prompt": "Fix the import './components/Button' — check if the file exists or fix the path.",
    },
    {
        "error": "Module not found: Can't resolve './pages/Home'",
        "error_type": "wrong_import_path",
        "fix": "Create the missing page component or fix the import path",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module '../utils/api' from 'src/components/Dashboard.tsx'",
        "error_type": "wrong_import_path",
        "fix": "Ensure src/utils/api.ts exists or correct the relative import path",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Error: Can't resolve '@/components/ui/Button'",
        "error_type": "wrong_import_path",
        "fix": "Add paths alias to tsconfig.json: '@/*': ['src/*'] or fix the import to './components/ui/Button'",
        "instant_fix": False,
        "llm_prompt": "Configure TypeScript path alias @/ in tsconfig.json compilerOptions.paths or rewrite the import.",
    },
    {
        "error": "Module not found: Error: Can't resolve '@/utils/cn'",
        "error_type": "wrong_import_path",
        "fix": "Add paths alias to tsconfig and create src/utils/cn.ts, or rewrite import to relative path",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module '../hooks/useLocalStorage'",
        "error_type": "wrong_import_path",
        "fix": "Create src/hooks/useLocalStorage.ts with the custom hook",
        "file_to_fix": "frontend/src/hooks/useLocalStorage.ts",
        "fix_content": (
            "import { useState, useEffect } from 'react';\n\n"
            "function useLocalStorage<T>(key: string, initialValue: T) {\n"
            "  const [storedValue, setStoredValue] = useState<T>(() => {\n"
            "    try {\n      const item = window.localStorage.getItem(key);\n"
            "      return item ? JSON.parse(item) : initialValue;\n"
            "    } catch { return initialValue; }\n  });\n\n"
            "  const setValue = (value: T) => {\n"
            "    try {\n      setStoredValue(value);\n"
            "      window.localStorage.setItem(key, JSON.stringify(value));\n"
            "    } catch (error) { console.error(error); }\n  };\n\n"
            "  return [storedValue, setValue] as const;\n}\n\n"
            "export default useLocalStorage;\n"
        ),
        "instant_fix": True,
    },
    {
        "error": "Module not found: Can't resolve './styles/index.css'",
        "error_type": "wrong_import_path",
        "fix": "Replace './styles/index.css' with './index.css'",
        "pattern": "./styles/index.css",
        "replacement": "./index.css",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module '../../shared/types'",
        "error_type": "wrong_import_path",
        "fix": "Create the shared types file or fix the relative import path",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve '@/pages/Dashboard'",
        "error_type": "wrong_import_path",
        "fix": "Configure TypeScript path alias or rewrite import to relative path",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module '../store/useStore'",
        "error_type": "wrong_import_path",
        "fix": "Create src/store/useStore.ts with zustand store or fix the import path",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve './assets/images/hero.jpg'",
        "error_type": "wrong_import_path",
        "fix": "Remove the image import and use a placeholder URL instead",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module '../components/Layout'",
        "error_type": "wrong_import_path",
        "fix": "Create src/components/Layout.tsx or fix the relative import path",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve '@/lib/utils'",
        "error_type": "wrong_import_path",
        "fix": "Create src/lib/utils.ts with utility functions and configure @ alias",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module './components/Navbar'",
        "error_type": "wrong_import_path",
        "fix": "Create src/components/Navbar.tsx or fix the import path",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve '../services/api'",
        "error_type": "wrong_import_path",
        "fix": "Create src/services/api.ts with API service functions",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module './routes'",
        "error_type": "wrong_import_path",
        "fix": "Create src/routes.tsx with React Router route definitions",
        "instant_fix": False,
    },
    {
        "error": "Module not found: Can't resolve '@/contexts/ThemeContext'",
        "error_type": "wrong_import_path",
        "fix": "Create src/contexts/ThemeContext.tsx or configure @ path alias",
        "instant_fix": False,
    },
    {
        "error": "Cannot find module '../../components/common/Modal'",
        "error_type": "wrong_import_path",
        "fix": "Create src/components/common/Modal.tsx or fix the deep relative import",
        "instant_fix": False,
    },

    # ═══ CATEGORY 3: Fake/Invalid npm Packages (20 examples) ═══

    {
        "error": "404 Not Found - GET https://registry.npmjs.org/@react-bits%2freact",
        "error_type": "fake_package",
        "fix": "Remove @react-bits/react from package.json dependencies",
        "package_to_remove": "@react-bits/react",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 Not Found - GET https://registry.npmjs.org/@react-bits%2fui",
        "error_type": "fake_package",
        "fix": "Remove @react-bits/ui from package.json",
        "package_to_remove": "@react-bits/ui",
        "instant_fix": True,
    },
    {
        "error": "The requested resource '@shadcn/ui@^1.0.0' could not be found",
        "error_type": "fake_package",
        "fix": "Remove @shadcn/ui. Use lucide-react for icons or @radix-ui/* for primitives instead.",
        "package_to_remove": "@shadcn/ui",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - '@flowbite/react' is not in the npm registry",
        "error_type": "fake_package",
        "fix": "Remove @flowbite/react. The correct package is 'flowbite-react'.",
        "package_to_remove": "@flowbite/react",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - GET https://registry.npmjs.org/@tabler%2ficons",
        "error_type": "fake_package",
        "fix": "Replace @tabler/icons with @tabler/icons-react (the correct npm package name)",
        "package_to_remove": "@tabler/icons",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - 'react-bits' is not in the npm registry",
        "error_type": "fake_package",
        "fix": "Remove react-bits from package.json",
        "package_to_remove": "react-bits",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - GET https://registry.npmjs.org/react-awesome-components",
        "error_type": "fake_package",
        "fix": "Remove react-awesome-components — it doesn't exist on npm",
        "package_to_remove": "react-awesome-components",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - '@ui/components' is not in this registry",
        "error_type": "fake_package",
        "fix": "Remove @ui/components and use shadcn-ui CLI or @radix-ui/* instead",
        "package_to_remove": "@ui/components",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - GET https://registry.npmjs.org/react-styled-kit",
        "error_type": "fake_package",
        "fix": "Remove react-styled-kit. Use styled-components or @emotion/styled instead.",
        "package_to_remove": "react-styled-kit",
        "instant_fix": True,
    },
    {
        "error": "not found @react-bits/core",
        "error_type": "fake_package",
        "fix": "Remove @react-bits/core from package.json",
        "package_to_remove": "@react-bits/core",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - 'tailwindui' is not in the npm registry",
        "error_type": "fake_package",
        "fix": "Remove tailwindui. Tailwind UI components are at @headlessui/react or @tailwindcss/ui doesn't exist.",
        "package_to_remove": "tailwindui",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - GET https://registry.npmjs.org/react-pro-kit",
        "error_type": "fake_package",
        "fix": "Remove react-pro-kit — it doesn't exist on npm",
        "package_to_remove": "react-pro-kit",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - '@antfu/components' is not in this registry",
        "error_type": "fake_package",
        "fix": "Remove @antfu/components — it doesn't exist on npm",
        "package_to_remove": "@antfu/components",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - react-animation-kit not found",
        "error_type": "fake_package",
        "fix": "Remove react-animation-kit. Use framer-motion for animations instead.",
        "package_to_remove": "react-animation-kit",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - 'next-auth-react' is not in the npm registry",
        "error_type": "fake_package",
        "fix": "Remove next-auth-react. The correct package for auth is 'next-auth' (for Next.js) or use a custom auth hook.",
        "package_to_remove": "next-auth-react",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - '@react-bits/forms' not found",
        "error_type": "fake_package",
        "fix": "Remove @react-bits/forms. Use react-hook-form instead.",
        "package_to_remove": "@react-bits/forms",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - 'react-ui-components' is not in this registry",
        "error_type": "fake_package",
        "fix": "Remove react-ui-components — it doesn't exist on npm",
        "package_to_remove": "react-ui-components",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - 'react-design-system' not found",
        "error_type": "fake_package",
        "fix": "Remove react-design-system and use a real component library like @mui/material",
        "package_to_remove": "react-design-system",
        "instant_fix": True,
    },
    {
        "error": "npm error 404 - '@components/core' is not in this registry",
        "error_type": "fake_package",
        "fix": "Remove @components/core — it doesn't exist on npm",
        "package_to_remove": "@components/core",
        "instant_fix": True,
    },
    {
        "error": "404 Not Found - 'pretty-react-hooks' not found",
        "error_type": "fake_package",
        "fix": "Remove pretty-react-hooks. Use ahooks or react-use for custom hooks.",
        "package_to_remove": "pretty-react-hooks",
        "instant_fix": True,
    },

    # ═══ CATEGORY 4: Missing npm Packages (30 examples) ═══

    {
        "error": "Cannot find module 'react-router-dom'",
        "error_type": "missing_package",
        "fix": "Add react-router-dom to package.json",
        "package_to_add": "react-router-dom",
        "version": "^6.8.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'axios'",
        "error_type": "missing_package",
        "fix": "Add axios to package.json",
        "package_to_add": "axios",
        "version": "^1.3.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'framer-motion'",
        "error_type": "missing_package",
        "fix": "Add framer-motion to package.json",
        "package_to_add": "framer-motion",
        "version": "^10.0.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'lucide-react'",
        "error_type": "missing_package",
        "fix": "Add lucide-react to package.json",
        "package_to_add": "lucide-react",
        "version": "^0.263.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'zustand'",
        "error_type": "missing_package",
        "fix": "Add zustand to package.json",
        "package_to_add": "zustand",
        "version": "^4.3.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'recharts'",
        "error_type": "missing_package",
        "fix": "Add recharts to package.json",
        "package_to_add": "recharts",
        "version": "^2.5.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module '@tanstack/react-query'",
        "error_type": "missing_package",
        "fix": "Add @tanstack/react-query to package.json",
        "package_to_add": "@tanstack/react-query",
        "version": "^4.0.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'date-fns'",
        "error_type": "missing_package",
        "fix": "Add date-fns to package.json",
        "package_to_add": "date-fns",
        "version": "^2.29.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'uuid'",
        "error_type": "missing_package",
        "fix": "Add uuid and @types/uuid to package.json",
        "package_to_add": "uuid",
        "version": "^9.0.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-hook-form'",
        "error_type": "missing_package",
        "fix": "Add react-hook-form to package.json",
        "package_to_add": "react-hook-form",
        "version": "^7.43.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'zod'",
        "error_type": "missing_package",
        "fix": "Add zod to package.json",
        "package_to_add": "zod",
        "version": "^3.20.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-toastify'",
        "error_type": "missing_package",
        "fix": "Add react-toastify to package.json",
        "package_to_add": "react-toastify",
        "version": "^9.1.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-hot-toast'",
        "error_type": "missing_package",
        "fix": "Add react-hot-toast to package.json",
        "package_to_add": "react-hot-toast",
        "version": "^2.4.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'socket.io-client'",
        "error_type": "missing_package",
        "fix": "Add socket.io-client to package.json",
        "package_to_add": "socket.io-client",
        "version": "^4.6.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-icons'",
        "error_type": "missing_package",
        "fix": "Add react-icons to package.json",
        "package_to_add": "react-icons",
        "version": "^4.7.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'clsx'",
        "error_type": "missing_package",
        "fix": "Add clsx to package.json",
        "package_to_add": "clsx",
        "version": "^1.2.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'dayjs'",
        "error_type": "missing_package",
        "fix": "Add dayjs to package.json",
        "package_to_add": "dayjs",
        "version": "^1.11.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'lodash'",
        "error_type": "missing_package",
        "fix": "Add lodash and @types/lodash to package.json",
        "package_to_add": "lodash",
        "version": "^4.17.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-markdown'",
        "error_type": "missing_package",
        "fix": "Add react-markdown to package.json",
        "package_to_add": "react-markdown",
        "version": "^8.0.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module '@emotion/react'",
        "error_type": "missing_package",
        "fix": "Add @emotion/react to package.json",
        "package_to_add": "@emotion/react",
        "version": "^11.10.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'styled-components'",
        "error_type": "missing_package",
        "fix": "Add styled-components to package.json",
        "package_to_add": "styled-components",
        "version": "^5.3.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'chart.js'",
        "error_type": "missing_package",
        "fix": "Add chart.js to package.json",
        "package_to_add": "chart.js",
        "version": "^4.2.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-chartjs-2'",
        "error_type": "missing_package",
        "fix": "Add react-chartjs-2 and chart.js to package.json",
        "package_to_add": "react-chartjs-2",
        "version": "^5.2.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module '@headlessui/react'",
        "error_type": "missing_package",
        "fix": "Add @headlessui/react to package.json",
        "package_to_add": "@headlessui/react",
        "version": "^1.7.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'immer'",
        "error_type": "missing_package",
        "fix": "Add immer to package.json",
        "package_to_add": "immer",
        "version": "^9.0.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'swr'",
        "error_type": "missing_package",
        "fix": "Add swr to package.json",
        "package_to_add": "swr",
        "version": "^2.1.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-select'",
        "error_type": "missing_package",
        "fix": "Add react-select to package.json",
        "package_to_add": "react-select",
        "version": "^5.7.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-datepicker'",
        "error_type": "missing_package",
        "fix": "Add react-datepicker to package.json",
        "package_to_add": "react-datepicker",
        "version": "^4.10.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'classnames'",
        "error_type": "missing_package",
        "fix": "Add classnames to package.json (or use clsx as a lighter alternative)",
        "package_to_add": "classnames",
        "version": "^2.3.0",
        "instant_fix": True,
    },
    {
        "error": "Cannot find module 'react-dropzone'",
        "error_type": "missing_package",
        "fix": "Add react-dropzone to package.json",
        "package_to_add": "react-dropzone",
        "version": "^14.2.0",
        "instant_fix": True,
    },

    # ═══ CATEGORY 5: Python Syntax and Runtime Errors (30 examples) ═══

    {
        "error": "SyntaxError: invalid syntax in backend/main.py",
        "error_type": "python_syntax",
        "fix": "Fix Python syntax error using LLM",
        "instant_fix": False,
        "llm_prompt": "Fix the syntax error in this FastAPI Python file. Common issues: missing colon, unclosed brackets, wrong indentation.",
    },
    {
        "error": "ImportError: cannot import name 'AsyncSession' from 'sqlalchemy.ext.asyncio'",
        "error_type": "python_import",
        "fix": "Add: from sqlalchemy.ext.asyncio import AsyncSession",
        "instant_fix": False,
        "llm_prompt": "Fix the SQLAlchemy async import error.",
    },
    {
        "error": "ModuleNotFoundError: No module named 'fastapi'",
        "error_type": "missing_python_package",
        "fix": "Add fastapi to requirements.txt",
        "package_to_add": "fastapi",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'uvicorn'",
        "error_type": "missing_python_package",
        "fix": "Add uvicorn[standard] to requirements.txt",
        "package_to_add": "uvicorn",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'sqlalchemy'",
        "error_type": "missing_python_package",
        "fix": "Add sqlalchemy[asyncio] to requirements.txt",
        "package_to_add": "sqlalchemy",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'asyncpg'",
        "error_type": "missing_python_package",
        "fix": "Add asyncpg to requirements.txt",
        "package_to_add": "asyncpg",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'dotenv'",
        "error_type": "missing_python_package",
        "fix": "Add python-dotenv to requirements.txt",
        "package_to_add": "dotenv",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'pydantic'",
        "error_type": "missing_python_package",
        "fix": "Add pydantic to requirements.txt",
        "package_to_add": "pydantic",
        "instant_fix": True,
    },
    {
        "error": "IndentationError: unexpected indent",
        "error_type": "python_syntax",
        "fix": "Fix indentation — Python requires consistent 4-space indentation",
        "instant_fix": False,
    },
    {
        "error": "NameError: name 'db' is not defined",
        "error_type": "python_runtime",
        "fix": "Add db: AsyncSession = Depends(get_db) to the route function parameters",
        "instant_fix": False,
    },
    {
        "error": "ModuleNotFoundError: No module named 'httpx'",
        "error_type": "missing_python_package",
        "fix": "Add httpx to requirements.txt",
        "package_to_add": "httpx",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'alembic'",
        "error_type": "missing_python_package",
        "fix": "Add alembic to requirements.txt",
        "package_to_add": "alembic",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'aiofiles'",
        "error_type": "missing_python_package",
        "fix": "Add aiofiles to requirements.txt",
        "package_to_add": "aiofiles",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'passlib'",
        "error_type": "missing_python_package",
        "fix": "Add passlib[bcrypt] to requirements.txt",
        "package_to_add": "passlib",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'jose'",
        "error_type": "missing_python_package",
        "fix": "Add python-jose[cryptography] to requirements.txt",
        "package_to_add": "jose",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'anthropic'",
        "error_type": "missing_python_package",
        "fix": "Add anthropic to requirements.txt",
        "package_to_add": "anthropic",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'openai'",
        "error_type": "missing_python_package",
        "fix": "Add openai to requirements.txt",
        "package_to_add": "openai",
        "instant_fix": True,
    },
    {
        "error": "SyntaxError: EOL while scanning string literal",
        "error_type": "python_syntax",
        "fix": "Fix unterminated string literal — a quote is missing",
        "instant_fix": False,
    },
    {
        "error": "SyntaxError: unexpected EOF while parsing",
        "error_type": "python_syntax",
        "fix": "Fix unclosed bracket, parenthesis, or block — something is missing at end of file",
        "instant_fix": False,
    },
    {
        "error": "ImportError: cannot import name 'CORSMiddleware' from 'fastapi.middleware.cors'",
        "error_type": "python_import",
        "fix": "Fix import: from fastapi.middleware.cors import CORSMiddleware",
        "instant_fix": False,
    },
    {
        "error": "TypeError: object is not subscriptable",
        "error_type": "python_runtime",
        "fix": "You are indexing a non-subscriptable object. Check the type before indexing.",
        "instant_fix": False,
    },
    {
        "error": "AttributeError: 'NoneType' object has no attribute 'id'",
        "error_type": "python_runtime",
        "fix": "Add a null check before accessing .id. The object may not be found in the database.",
        "instant_fix": False,
    },
    {
        "error": "ValueError: too many values to unpack",
        "error_type": "python_runtime",
        "fix": "Check the number of variables on the left side of the assignment matches the iterable",
        "instant_fix": False,
    },
    {
        "error": "ModuleNotFoundError: No module named 'redis'",
        "error_type": "missing_python_package",
        "fix": "Add redis to requirements.txt",
        "package_to_add": "redis",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'celery'",
        "error_type": "missing_python_package",
        "fix": "Add celery to requirements.txt",
        "package_to_add": "celery",
        "instant_fix": True,
    },
    {
        "error": "ModuleNotFoundError: No module named 'PIL'",
        "error_type": "missing_python_package",
        "fix": "Add Pillow to requirements.txt (PIL is the old name)",
        "package_to_add": "PIL",
        "instant_fix": True,
    },
    {
        "error": "IndentationError: expected an indented block after function definition",
        "error_type": "python_syntax",
        "fix": "Add a function body — use 'pass' as placeholder if needed",
        "instant_fix": False,
    },
    {
        "error": "NameError: name 'settings' is not defined",
        "error_type": "python_runtime",
        "fix": "Import settings: from src.config.settings import settings",
        "instant_fix": False,
    },
    {
        "error": "ImportError: cannot import name 'Mapped' from 'sqlalchemy.orm'",
        "error_type": "python_import",
        "fix": "Upgrade SQLAlchemy to 2.0+. Mapped is only available in SQLAlchemy 2.x.",
        "instant_fix": False,
    },
    {
        "error": "ModuleNotFoundError: No module named 'stripe'",
        "error_type": "missing_python_package",
        "fix": "Add stripe to requirements.txt",
        "package_to_add": "stripe",
        "instant_fix": True,
    },
    {
        "error": "SyntaxError: invalid character in identifier",
        "error_type": "python_syntax",
        "fix": "Remove non-ASCII characters (like curly quotes or Unicode dashes) from code",
        "instant_fix": False,
    },

    # ═══ CATEGORY 6: TypeScript Errors (30 examples) ═══

    {
        "error": "TS7016: Could not find a declaration file for module 'three'",
        "error_type": "typescript_missing_types",
        "fix": "Add @ts-ignore before the import OR add @types/three to devDependencies",
        "instant_fix": True,
        "fix_action": "add_ts_ignore",
    },
    {
        "error": "TS2304: Cannot find name 'ShieldCheck'",
        "error_type": "typescript_missing_import",
        "fix": "Add ShieldCheck to the lucide-react import statement",
        "instant_fix": False,
    },
    {
        "error": "TS2339: Property 'data' does not exist on type 'AxiosResponse'",
        "error_type": "typescript_type",
        "fix": "Add proper TypeScript interface for the API response or use AxiosResponse<YourType>",
        "instant_fix": False,
    },
    {
        "error": "TS1139: Type parameter declaration expected",
        "error_type": "typescript_version",
        "fix": "Add skipLibCheck: true to tsconfig.json compilerOptions",
        "instant_fix": True,
        "fix_action": "add_skip_lib_check",
    },
    {
        "error": "TS2561: Object literal may only specify known properties, but 'insetX' does not exist",
        "error_type": "typescript_css",
        "fix": "Replace insetX with left: 0, right: 0 in the style object",
        "pattern": "insetX: 0",
        "replacement": "left: 0, right: 0",
        "instant_fix": True,
    },
    {
        "error": "TS2307: Cannot find module 'react' or its corresponding type declarations",
        "error_type": "missing_package",
        "fix": "Add @types/react and @types/react-dom to devDependencies",
        "package_to_add": "@types/react",
        "version": "^18.0.0",
        "instant_fix": True,
    },
    {
        "error": "TS2345: Argument of type 'string | undefined' is not assignable to parameter of type 'string'",
        "error_type": "typescript_type",
        "fix": "Add a null check: if (!value) return; or use non-null assertion: value!",
        "instant_fix": False,
    },
    {
        "error": "TS7006: Parameter 'event' implicitly has an 'any' type",
        "error_type": "typescript_type",
        "fix": "Add explicit type: (event: React.ChangeEvent<HTMLInputElement>) => void",
        "instant_fix": False,
    },
    {
        "error": "TS2322: Type 'string' is not assignable to type 'number'",
        "error_type": "typescript_type",
        "fix": "Convert string to number: Number(value) or parseInt(value, 10)",
        "instant_fix": False,
    },
    {
        "error": "TS2304: Cannot find name 'React'",
        "error_type": "typescript_missing_import",
        "fix": "Add import React from 'react'; at the top of the file",
        "instant_fix": False,
    },
    {
        "error": "TS7016: Could not find a declaration file for module 'd3'",
        "error_type": "typescript_missing_types",
        "fix": "Add // @ts-ignore before d3 import, or add @types/d3 to devDependencies",
        "instant_fix": True,
        "fix_action": "add_ts_ignore",
    },
    {
        "error": "TS2739: Type '{}' is missing the following properties from type 'User'",
        "error_type": "typescript_type",
        "fix": "Provide all required properties of the User type or use Partial<User>",
        "instant_fix": False,
    },
    {
        "error": "TS1005: ',' expected",
        "error_type": "typescript_type",
        "fix": "Fix missing comma in function parameters, destructuring, or object literal",
        "instant_fix": False,
    },
    {
        "error": "TS2531: Object is possibly 'null'",
        "error_type": "typescript_type",
        "fix": "Add optional chaining: obj?.property or a null check before access",
        "instant_fix": False,
    },
    {
        "error": "TS2532: Object is possibly 'undefined'",
        "error_type": "typescript_type",
        "fix": "Add optional chaining: arr?.[0] or initialize with a default value",
        "instant_fix": False,
    },
    {
        "error": "TS2551: Property 'classname' does not exist on type — did you mean 'className'?",
        "error_type": "typescript_type",
        "fix": "Fix the prop name: use className instead of classname in JSX",
        "instant_fix": False,
    },
    {
        "error": "TS7016: Could not find a declaration file for module 'react-beautiful-dnd'",
        "error_type": "typescript_missing_types",
        "fix": "Add @types/react-beautiful-dnd to devDependencies",
        "instant_fix": True,
        "fix_action": "add_ts_ignore",
    },
    {
        "error": "TS2367: This condition will always return 'false' since types have no overlap",
        "error_type": "typescript_type",
        "fix": "Fix the type comparison — one side is always a different type than expected",
        "instant_fix": False,
    },
    {
        "error": "TS1128: Declaration or statement expected",
        "error_type": "typescript_type",
        "fix": "Fix syntax error — an unexpected token at the top level of the file",
        "instant_fix": False,
    },
    {
        "error": "TS2614: Module has no exported member 'default'",
        "error_type": "typescript_missing_import",
        "fix": "Change 'import X from' to 'import { X } from', or check if the module has a default export",
        "instant_fix": False,
    },
    {
        "error": "TS2497: This module can only be referenced with ECMAScript imports/exports",
        "error_type": "typescript_version",
        "fix": "Add esModuleInterop: true and allowSyntheticDefaultImports: true in tsconfig.json",
        "instant_fix": True,
        "fix_action": "add_skip_lib_check",
    },
    {
        "error": "TS7016: Could not find a declaration file for module 'lodash'",
        "error_type": "typescript_missing_types",
        "fix": "Add @types/lodash to devDependencies in package.json",
        "instant_fix": True,
        "fix_action": "add_ts_ignore",
    },
    {
        "error": "TS2695: Left side of comma operator is unused and has no side effects",
        "error_type": "typescript_type",
        "fix": "Remove the dead expression before the comma",
        "instant_fix": False,
    },
    {
        "error": "TS2430: Interface 'Props' incorrectly extends interface 'HTMLAttributes'",
        "error_type": "typescript_type",
        "fix": "Check the Props interface — one of the property types conflicts with HTMLAttributes",
        "instant_fix": False,
    },
    {
        "error": "TS18048: 'items' is possibly 'undefined'",
        "error_type": "typescript_type",
        "fix": "Initialize state as empty array: useState<Item[]>([]) instead of useState()",
        "instant_fix": False,
    },
    {
        "error": "TS2786: 'Component' cannot be used as a JSX component. Its return type 'Element | undefined' is not a valid JSX element.",
        "error_type": "typescript_type",
        "fix": "Change return type to ReactElement | null — never return undefined from a component",
        "instant_fix": False,
    },
    {
        "error": "TS7031: Binding element 'children' implicitly has an 'any' type",
        "error_type": "typescript_type",
        "fix": "Add type to props: ({ children }: { children: React.ReactNode })",
        "instant_fix": False,
    },
    {
        "error": "TS2416: Property 'render' in type 'MyComponent' is not assignable to the same property in base type 'Component'",
        "error_type": "typescript_type",
        "fix": "Fix the render method return type to match React.ReactNode",
        "instant_fix": False,
    },
    {
        "error": "TS2561: Object literal may only specify known properties, but 'insetY' does not exist in type 'CSSProperties'",
        "error_type": "typescript_css",
        "fix": "Replace insetY with top: 0, bottom: 0 in the style object",
        "pattern": "insetY: 0",
        "replacement": "top: 0, bottom: 0",
        "instant_fix": True,
    },
    {
        "error": "TS2454: Variable 'data' is used before being assigned",
        "error_type": "typescript_type",
        "fix": "Initialize the variable before use or add definite assignment assertion: data!",
        "instant_fix": False,
    },
    {
        "error": "TS2304: Cannot find name 'process'",
        "error_type": "typescript_missing_import",
        "fix": "Add @types/node to devDependencies so process.env is typed",
        "instant_fix": False,
    },

    # ═══ CATEGORY 7: Database Errors (20 examples) ═══

    {
        "error": "asyncpg.exceptions.ConnectionRefusedError: connection refused",
        "error_type": "db_connection",
        "fix": "Check DATABASE_URL in .env file. Make sure PostgreSQL is running on port 5432.",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.ProgrammingError: relation 'todos' does not exist",
        "error_type": "db_table_missing",
        "fix": "Run database migrations: alembic upgrade head, or add create_all() to startup lifespan",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.InvalidPasswordError: password authentication failed",
        "error_type": "db_auth",
        "fix": "Check database credentials in DATABASE_URL. Verify the password is correct.",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.OperationalError: could not translate host name 'db' to address",
        "error_type": "db_host",
        "fix": "Change host in DATABASE_URL from 'db' (Docker service name) to 'localhost' for local dev",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.ProgrammingError: column 'users.created_at' does not exist",
        "error_type": "db_table_missing",
        "fix": "Run migrations to add the missing column: alembic revision --autogenerate && alembic upgrade head",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions._base.InterfaceError: cannot perform operation: another operation is in progress",
        "error_type": "db_connection",
        "fix": "Use a new DB session per request. Do not share AsyncSession across concurrent tasks.",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.IntegrityError: duplicate key value violates unique constraint",
        "error_type": "db_table_missing",
        "fix": "Handle duplicate key with try/except IntegrityError and return 409 Conflict response",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.UndefinedTableError: relation 'users' does not exist",
        "error_type": "db_table_missing",
        "fix": "Run init_db() or create_all() on startup to create all tables",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.OperationalError: (asyncpg.exceptions.TooManyConnectionsError)",
        "error_type": "db_connection",
        "fix": "Add pool_size and max_overflow limits to create_async_engine(): pool_size=5, max_overflow=10",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.ConnectionDoesNotExistError: connection is closed",
        "error_type": "db_connection",
        "fix": "Recreate the session — the connection was closed. Use a fresh AsyncSession from the factory.",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.ProgrammingError: column 'id' is of type integer but expression is of type text",
        "error_type": "db_table_missing",
        "fix": "Cast the value to the correct type or check the column type in the ORM model",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.ForeignKeyViolationError: insert or update on table violates foreign key constraint",
        "error_type": "db_table_missing",
        "fix": "Ensure the referenced record exists before inserting the dependent record",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.orm.exc.DetachedInstanceError: Instance is not bound to a Session",
        "error_type": "db_connection",
        "fix": "Access lazy-loaded attributes before closing the session, or use selectinload() in the query",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.DataError: invalid input syntax for type integer",
        "error_type": "db_table_missing",
        "fix": "Validate and convert the input to int before the database query",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.PendingRollbackError: This Session's transaction has been rolled back",
        "error_type": "db_connection",
        "fix": "Add await db.rollback() in the exception handler before continuing to use the session",
        "instant_fix": False,
    },
    {
        "error": "could not connect to server: No such file or directory — is the server running locally?",
        "error_type": "db_connection",
        "fix": "Start PostgreSQL: brew services start postgresql OR pg_ctl start -D /usr/local/var/postgres",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.exceptions.InvalidSchemaNameError: schema 'public' does not exist",
        "error_type": "db_auth",
        "fix": "Create the public schema: CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO postgres;",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.NoResultFound: No row was found when one was required",
        "error_type": "db_table_missing",
        "fix": "Use scalar_one_or_none() instead of scalar_one() and handle the None case",
        "instant_fix": False,
    },
    {
        "error": "asyncpg.PostgresError: SSL connection has been closed unexpectedly",
        "error_type": "db_connection",
        "fix": "Add ssl=True or ssl='require' to the asyncpg connection string for cloud databases",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.ArgumentError: Could not parse SQLAlchemy URL from string 'DATABASE_URL'",
        "error_type": "db_connection",
        "fix": "Set DATABASE_URL in .env — it should be a full connection string not the literal text 'DATABASE_URL'",
        "instant_fix": False,
    },

    # ═══ CATEGORY 8: CORS Errors (10 examples) ═══

    {
        "error": "Access to fetch blocked by CORS policy: No 'Access-Control-Allow-Origin' header",
        "error_type": "cors_error",
        "fix": "Add CORSMiddleware to FastAPI app with allow_origins=['*']",
        "instant_fix": False,
        "llm_prompt": "Add CORS middleware to the FastAPI backend to allow all origins.",
    },
    {
        "error": "CORS error: Response to preflight request doesn't pass access control check",
        "error_type": "cors_preflight",
        "fix": "Set allow_methods=['*'] and allow_headers=['*'] in CORSMiddleware",
        "instant_fix": False,
    },
    {
        "error": "Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource",
        "error_type": "cors_error",
        "fix": "Configure CORS on the backend to allow the frontend's origin",
        "instant_fix": False,
    },
    {
        "error": "CORS policy blocked: The request client is not a secure context and the resource is in more-private address space",
        "error_type": "cors_preflight",
        "fix": "Add allow_origin_regex or list the specific origin in allow_origins in CORSMiddleware",
        "instant_fix": False,
    },
    {
        "error": "Access-Control-Allow-Origin header missing from server response",
        "error_type": "cors_error",
        "fix": "Ensure CORSMiddleware is added before route definitions in FastAPI app",
        "instant_fix": False,
    },
    {
        "error": "XMLHttpRequest cannot load. No 'Access-Control-Allow-Origin' header is present",
        "error_type": "cors_error",
        "fix": "Add CORSMiddleware: app.add_middleware(CORSMiddleware, allow_origins=['*'])",
        "instant_fix": False,
    },
    {
        "error": "CORS blocked: preflight response has invalid HTTP status code 404",
        "error_type": "cors_preflight",
        "fix": "The preflight OPTIONS request returns 404 — check that FastAPI handles OPTIONS methods",
        "instant_fix": False,
    },
    {
        "error": "CORS: Request header 'content-type' is not allowed",
        "error_type": "cors_preflight",
        "fix": "Add 'Content-Type' to allow_headers in CORSMiddleware or use allow_headers=['*']",
        "instant_fix": False,
    },
    {
        "error": "CORS error on credentials: The value of the 'Access-Control-Allow-Origin' header must not be the wildcard '*' when the request's credentials mode is 'include'",
        "error_type": "cors_error",
        "fix": "Set a specific origin in allow_origins and set allow_credentials=True in CORSMiddleware",
        "instant_fix": False,
    },
    {
        "error": "CORS blocked: Access-Control-Allow-Methods does not include POST",
        "error_type": "cors_preflight",
        "fix": "Set allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'] in CORSMiddleware",
        "instant_fix": False,
    },

    # ═══ CATEGORY 9: React Runtime Errors (20 examples) ═══

    {
        "error": "Objects are not valid as a React child (found: object with keys {id, name})",
        "error_type": "react_invalid_child",
        "fix": "Convert object to string before rendering: {JSON.stringify(obj)} or access specific property like {obj.name}",
        "instant_fix": False,
    },
    {
        "error": "Cannot read properties of undefined (reading 'map')",
        "error_type": "react_undefined",
        "fix": "Add optional chaining: data?.map() or initialize state as empty array: useState<Item[]>([])",
        "instant_fix": False,
    },
    {
        "error": "Each child in a list should have a unique 'key' prop",
        "error_type": "react_key",
        "fix": "Add key prop to each list item: items.map((item, i) => <div key={item.id ?? i}>)",
        "instant_fix": False,
    },
    {
        "error": "Warning: Can't perform a React state update on an unmounted component",
        "error_type": "react_cleanup",
        "fix": "Add cleanup in useEffect: let mounted = true; return () => { mounted = false; }; check mounted before setState",
        "instant_fix": False,
    },
    {
        "error": "Too many re-renders. React limits the number of renders to prevent an infinite loop",
        "error_type": "react_infinite_loop",
        "fix": "Move state update inside useEffect or event handler — never call setState directly in the render body",
        "instant_fix": False,
    },
    {
        "error": "Cannot read properties of null (reading 'useState')",
        "error_type": "react_undefined",
        "fix": "React hooks are called out of order — check that all hooks are called unconditionally at the top level",
        "instant_fix": False,
    },
    {
        "error": "Uncaught Error: Rendered more hooks than during the previous render",
        "error_type": "react_invalid_child",
        "fix": "Hooks cannot be called conditionally. Move all hook calls before any early returns.",
        "instant_fix": False,
    },
    {
        "error": "Warning: Each child in a list should have a unique key prop. Check the render method of 'App'",
        "error_type": "react_key",
        "fix": "Add a stable unique key: use item.id if available, not array index, to avoid re-render bugs",
        "instant_fix": False,
    },
    {
        "error": "Error: Hydration failed because the initial UI does not match what was rendered on the server",
        "error_type": "react_invalid_child",
        "fix": "Ensure server-rendered HTML matches client render. Avoid browser-only APIs during SSR.",
        "instant_fix": False,
    },
    {
        "error": "Cannot read properties of undefined (reading 'filter')",
        "error_type": "react_undefined",
        "fix": "Initialize array state: useState<Item[]>([]) and guard: (data ?? []).filter(...)",
        "instant_fix": False,
    },
    {
        "error": "Warning: ReactDOM.render is no longer supported in React 18",
        "error_type": "react_invalid_child",
        "fix": "Replace ReactDOM.render with createRoot: const root = createRoot(document.getElementById('root')!); root.render(<App />);",
        "instant_fix": False,
    },
    {
        "error": "Uncaught ReferenceError: process is not defined",
        "error_type": "react_invalid_child",
        "fix": "Use import.meta.env.VITE_* for Vite or process.env.REACT_APP_* for CRA — don't use raw process in browser",
        "instant_fix": False,
    },
    {
        "error": "Error: Invalid hook call. Hooks can only be called inside of a function component",
        "error_type": "react_invalid_child",
        "fix": "Move the hook call inside a React function component, not a class component or utility function",
        "instant_fix": False,
    },
    {
        "error": "Warning: Failed prop type: The prop 'onClick' is marked as required in 'Button', but its value is 'undefined'",
        "error_type": "react_undefined",
        "fix": "Pass the onClick prop to the Button component, or make it optional in PropTypes",
        "instant_fix": False,
    },
    {
        "error": "Uncaught TypeError: dispatch is not a function",
        "error_type": "react_undefined",
        "fix": "Ensure the component is wrapped in the Redux Provider or Zustand store — check context setup",
        "instant_fix": False,
    },
    {
        "error": "Warning: Update to component from inside the function body of a different component",
        "error_type": "react_infinite_loop",
        "fix": "Wrap the state update in useEffect or move it to an event handler",
        "instant_fix": False,
    },
    {
        "error": "Error: Objects are not valid as a React child (found: [object Promise])",
        "error_type": "react_invalid_child",
        "fix": "Await the Promise and store in state before rendering: const [data] = useState(); useEffect(() => { fetchData().then(setData); }, [])",
        "instant_fix": False,
    },
    {
        "error": "Uncaught SyntaxError: Unexpected token '<'",
        "error_type": "react_invalid_child",
        "fix": "The browser received HTML instead of JavaScript — check API URL and ensure the dev server is running",
        "instant_fix": False,
    },
    {
        "error": "Warning: Cannot update a component from inside the function body of a different component",
        "error_type": "react_cleanup",
        "fix": "Wrap side effects in useEffect with correct dependencies",
        "instant_fix": False,
    },
    {
        "error": "Error: Minified React error #31",
        "error_type": "react_invalid_child",
        "fix": "A non-element value is being rendered directly. Wrap it or convert to string.",
        "instant_fix": False,
    },

    # ═══ CATEGORY 10: FastAPI Errors (20 examples) ═══

    {
        "error": "422 Unprocessable Entity",
        "error_type": "fastapi_validation",
        "fix": "Check request body matches the Pydantic model schema — all required fields must be present with correct types",
        "instant_fix": False,
    },
    {
        "error": "AttributeError: 'coroutine' object has no attribute 'id'",
        "error_type": "fastapi_async",
        "fix": "Add 'await' before the async function call",
        "instant_fix": False,
    },
    {
        "error": "RuntimeError: no running event loop",
        "error_type": "fastapi_eventloop",
        "fix": "Use asyncio.run() or call from inside an async function, not synchronously",
        "instant_fix": False,
    },
    {
        "error": "fastapi.exceptions.ResponseValidationError",
        "error_type": "fastapi_response",
        "fix": "Fix the return type of the route to match the response_model annotation",
        "instant_fix": False,
    },
    {
        "error": "HTTPException: 404 Not Found",
        "error_type": "fastapi_validation",
        "fix": "The resource was not found. Add proper 404 handling: raise HTTPException(status_code=404, detail='Not found')",
        "instant_fix": False,
    },
    {
        "error": "RuntimeError: Task attached to a different loop",
        "error_type": "fastapi_eventloop",
        "fix": "Do not create asyncio tasks in one loop and await in another. Use FastAPI's built-in async routes.",
        "instant_fix": False,
    },
    {
        "error": "pydantic.error_wrappers.ValidationError: value is not a valid integer",
        "error_type": "fastapi_validation",
        "fix": "Pass an integer value in the request body, not a string",
        "instant_fix": False,
    },
    {
        "error": "sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called",
        "error_type": "fastapi_async",
        "fix": "Add await before all SQLAlchemy async operations, or use run_sync() for sync operations",
        "instant_fix": False,
    },
    {
        "error": "starlette.exceptions.HTTPException: Method Not Allowed",
        "error_type": "fastapi_validation",
        "fix": "Check the HTTP method used in the request matches the route decorator (@app.get vs @app.post)",
        "instant_fix": False,
    },
    {
        "error": "AttributeError: 'async_generator' object has no attribute 'close'",
        "error_type": "fastapi_async",
        "fix": "Use 'async for' with async generators and handle cleanup in a finally block",
        "instant_fix": False,
    },
    {
        "error": "ValueError: [TypeError(\"'coroutine' object is not iterable\")] while executing",
        "error_type": "fastapi_async",
        "fix": "Await the coroutine before iterating: result = await coroutine()",
        "instant_fix": False,
    },
    {
        "error": "Exception: FastAPI application startup failed",
        "error_type": "fastapi_validation",
        "fix": "Check the lifespan function for errors during startup — examine logs for the root cause",
        "instant_fix": False,
    },
    {
        "error": "pydantic.ValidationError: field required (type=value_error.missing)",
        "error_type": "fastapi_validation",
        "fix": "The required field is missing from the request. Make the field optional with Optional[str] = None",
        "instant_fix": False,
    },
    {
        "error": "TypeError: Object of type UUID is not JSON serializable",
        "error_type": "fastapi_response",
        "fix": "Convert UUID to string in the response: str(obj.id), or use response_model with a Pydantic schema",
        "instant_fix": False,
    },
    {
        "error": "TypeError: Object of type datetime is not JSON serializable",
        "error_type": "fastapi_response",
        "fix": "Add json_encoders to Pydantic Config: {datetime: lambda v: v.isoformat()} or use response_model",
        "instant_fix": False,
    },
    {
        "error": "RuntimeError: Cannot use async context manager in non-async code",
        "error_type": "fastapi_eventloop",
        "fix": "Ensure you are using 'async with' inside an 'async def' function",
        "instant_fix": False,
    },
    {
        "error": "fastapi.exceptions.RequestValidationError: ensure this value has at most 100 characters",
        "error_type": "fastapi_validation",
        "fix": "Truncate input before sending, or increase the max_length constraint in the Pydantic model",
        "instant_fix": False,
    },
    {
        "error": "HTTPException: 401 Unauthorized",
        "error_type": "fastapi_validation",
        "fix": "Include a valid Authorization header: Authorization: Bearer <token>",
        "instant_fix": False,
    },
    {
        "error": "HTTPException: 500 Internal Server Error",
        "error_type": "fastapi_response",
        "fix": "Check server logs for the exception traceback and fix the underlying error",
        "instant_fix": False,
    },
    {
        "error": "ImportError: cannot import name 'get_db' from 'src.database.connection'",
        "error_type": "python_import",
        "fix": "Ensure get_db is defined and exported in src/database/connection.py",
        "instant_fix": False,
    },

    # ═══ CATEGORY 11: StackBlitz Preview Errors (15 examples) ═══

    {
        "error": "Import error, can't find file: ./index.css",
        "error_type": "stackblitz_missing_css",
        "fix": "Create src/index.css file with basic reset styles",
        "instant_fix": True,
        "fix_action": "create_index_css",
    },
    {
        "error": "Can't find packages: react-router-dom",
        "error_type": "stackblitz_missing_package",
        "fix": "Add react-router-dom: ^6.8.0 to package.json dependencies",
        "package_to_add": "react-router-dom",
        "version": "^6.8.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: axios",
        "error_type": "stackblitz_missing_package",
        "fix": "Add axios: ^1.3.0 to package.json dependencies",
        "package_to_add": "axios",
        "version": "^1.3.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: framer-motion",
        "error_type": "stackblitz_missing_package",
        "fix": "Add framer-motion: ^10.0.0 to package.json dependencies",
        "package_to_add": "framer-motion",
        "version": "^10.0.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: lucide-react",
        "error_type": "stackblitz_missing_package",
        "fix": "Add lucide-react: ^0.263.0 to package.json dependencies",
        "package_to_add": "lucide-react",
        "version": "^0.263.0",
        "instant_fix": True,
    },
    {
        "error": "WebContainer failed to boot: out of memory",
        "error_type": "stackblitz_oom",
        "fix": "Reduce number of dependencies in package.json — remove unused packages to decrease install size",
        "instant_fix": False,
    },
    {
        "error": "Can't find packages: zustand",
        "error_type": "stackblitz_missing_package",
        "fix": "Add zustand: ^4.3.0 to package.json dependencies",
        "package_to_add": "zustand",
        "version": "^4.3.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: recharts",
        "error_type": "stackblitz_missing_package",
        "fix": "Add recharts: ^2.5.0 to package.json dependencies",
        "package_to_add": "recharts",
        "version": "^2.5.0",
        "instant_fix": True,
    },
    {
        "error": "Import error, can't find file: ./App.css",
        "error_type": "stackblitz_missing_css",
        "fix": "Create src/App.css with minimal styles",
        "instant_fix": True,
        "fix_action": "create_index_css",
    },
    {
        "error": "StackBlitz preview: Killed (memory limit exceeded)",
        "error_type": "stackblitz_oom",
        "fix": "Remove heavy packages (d3, three.js) and replace with lighter alternatives",
        "instant_fix": False,
    },
    {
        "error": "Can't find packages: react-hook-form",
        "error_type": "stackblitz_missing_package",
        "fix": "Add react-hook-form: ^7.43.0 to package.json dependencies",
        "package_to_add": "react-hook-form",
        "version": "^7.43.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: zod",
        "error_type": "stackblitz_missing_package",
        "fix": "Add zod: ^3.20.0 to package.json dependencies",
        "package_to_add": "zod",
        "version": "^3.20.0",
        "instant_fix": True,
    },
    {
        "error": "StackBlitz error: react-scripts not found",
        "error_type": "stackblitz_missing_package",
        "fix": "Ensure react-scripts: 5.0.1 is in package.json dependencies and scripts uses 'react-scripts start'",
        "package_to_add": "react-scripts",
        "version": "5.0.1",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: date-fns",
        "error_type": "stackblitz_missing_package",
        "fix": "Add date-fns: ^2.29.0 to package.json dependencies",
        "package_to_add": "date-fns",
        "version": "^2.29.0",
        "instant_fix": True,
    },
    {
        "error": "Can't find packages: clsx",
        "error_type": "stackblitz_missing_package",
        "fix": "Add clsx: ^1.2.0 to package.json dependencies",
        "package_to_add": "clsx",
        "version": "^1.2.0",
        "instant_fix": True,
    },

    # ═══ CATEGORY 12: Advanced SQLAlchemy Errors (40 examples) ═══

    {"error": "sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called; can't call await_only() here.", "error_type": "sqlalchemy_greenlet", "fix": "Add await before every SQLAlchemy async call. Use AsyncSession, not the sync Session.", "instant_fix": False, "llm_prompt": "Fix MissingGreenlet: every SQLAlchemy operation inside an async function must be awaited (execute, get, merge, refresh, etc.)."},
    {"error": "MissingGreenlet: greenlet_spawn has not been called; can't call await_only() here. Was IO attempted in an unexpected place?", "error_type": "sqlalchemy_greenlet", "fix": "Never call sync SQLAlchemy inside async context. Use 'await session.execute(select(Model))'.", "instant_fix": False},
    {"error": "sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called. If using asyncio, use AsyncSession and selectinload() for relationships.", "error_type": "sqlalchemy_greenlet", "fix": "Replace sync session with AsyncSession and eager-load relationships with selectinload() or joinedload().", "instant_fix": False},
    {"error": "MissingGreenlet error when accessing user.posts outside session — lazy load cannot proceed in async context", "error_type": "sqlalchemy_greenlet", "fix": "Use selectinload(User.posts) in the query: select(User).options(selectinload(User.posts))", "instant_fix": False},
    {"error": "sqlalchemy.exc.MissingGreenlet: relationship 'User.orders' triggered a lazy load inside an async context", "error_type": "sqlalchemy_greenlet", "fix": "Add .options(selectinload(User.orders)) to the query that fetches the User.", "instant_fix": False},

    {"error": "sqlalchemy.orm.exc.DetachedInstanceError: Instance <User at 0x7f...> is not bound to a Session; attribute refresh operation cannot proceed", "error_type": "sqlalchemy_detached", "fix": "Access all needed attributes before the session closes. Use selectinload() for relationships accessed after session scope.", "instant_fix": False},
    {"error": "DetachedInstanceError: Instance <Post at 0x...> is not bound to a Session; lazy load operation of attribute 'author' cannot proceed", "error_type": "sqlalchemy_detached", "fix": "Eager-load 'author' with joinedload(Post.author) in the query, or access it within the session context manager.", "instant_fix": False},
    {"error": "sqlalchemy.orm.exc.DetachedInstanceError: Parent instance <Comment> is not bound to a Session; lazy load cannot proceed", "error_type": "sqlalchemy_detached", "fix": "Use AsyncSession as a context manager (async with) and access all relationships before the 'with' block exits.", "instant_fix": False},
    {"error": "DetachedInstanceError: Instance is not bound to a Session; this operation requires a persistent instance", "error_type": "sqlalchemy_detached", "fix": "Re-attach the instance: instance = await db.merge(instance), or re-query it in the current session.", "instant_fix": False},

    {"error": "sqlalchemy.exc.IntegrityError: (asyncpg.exceptions.ForeignKeyViolationError) insert or update on table 'comments' violates foreign key constraint 'comments_user_id_fkey'", "error_type": "sqlalchemy_integrity", "fix": "Ensure the referenced user_id exists before inserting the comment. Raise 404 if the parent row is not found.", "instant_fix": False},
    {"error": "sqlalchemy.exc.IntegrityError: (asyncpg.exceptions.UniqueViolationError) duplicate key value violates unique constraint 'users_email_key'", "error_type": "sqlalchemy_integrity", "fix": "Catch IntegrityError and return HTTP 409 Conflict. Check for existing email before insert with a SELECT.", "instant_fix": False},
    {"error": "sqlalchemy.exc.IntegrityError: (asyncpg.exceptions.NotNullViolationError) null value in column 'name' of relation 'users' violates not-null constraint", "error_type": "sqlalchemy_integrity", "fix": "Ensure 'name' is provided in the request body and not None before the database insert.", "instant_fix": False},
    {"error": "sqlalchemy.exc.IntegrityError: (asyncpg.exceptions.CheckViolationError) new row for relation 'products' violates check constraint 'products_price_check'", "error_type": "sqlalchemy_integrity", "fix": "Validate the value meets the check constraint (e.g. price > 0) in the Pydantic model before inserting.", "instant_fix": False},
    {"error": "DETAIL: Key (user_id)=(999) is not present in table 'users'. (sqlalchemy.exc.IntegrityError)", "error_type": "sqlalchemy_integrity", "fix": "Query for the user before referencing it. Raise HTTPException(404) if not found.", "instant_fix": False},

    {"error": "asyncpg.exceptions.UniqueViolationError: duplicate key value violates unique constraint 'users_email_key'", "error_type": "sqlalchemy_unique", "fix": "Wrap the insert in try/except IntegrityError and return 409. Or SELECT-then-INSERT with a check.", "instant_fix": False},
    {"error": "sqlalchemy.exc.IntegrityError: (asyncpg.exceptions.UniqueViolationError) duplicate key value violates unique constraint 'users_username_key' DETAIL: Key (username)=(john) already exists.", "error_type": "sqlalchemy_unique", "fix": "Check uniqueness before insert: result = await db.execute(select(User).where(User.username == username)). Return 409 if found.", "instant_fix": False},
    {"error": "UniqueViolationError: duplicate key value violates unique constraint on table 'api_keys'. DETAIL: Key (key)=(sk-abc123) already exists.", "error_type": "sqlalchemy_unique", "fix": "Generate a new UUID/key until unique, or use ON CONFLICT DO NOTHING with a returning clause.", "instant_fix": False},

    {"error": "asyncpg.exceptions.DataError: value too long for type character varying(255)", "error_type": "sqlalchemy_data", "fix": "Truncate the value to 255 characters before saving, or increase the column size in the model: String(1000).", "instant_fix": False},
    {"error": "sqlalchemy.exc.DataError: (asyncpg.exceptions.DataError) invalid input syntax for type integer: 'abc'", "error_type": "sqlalchemy_data", "fix": "Validate input is an integer in the Pydantic model. Add: id: int (not id: str) in the schema.", "instant_fix": False},
    {"error": "asyncpg.exceptions.DataError: invalid input syntax for type uuid: 'not-a-uuid'", "error_type": "sqlalchemy_data", "fix": "Validate UUID format before querying: try uuid.UUID(value) except ValueError: raise HTTPException(400).", "instant_fix": False},

    {"error": "asyncpg.exceptions.NotNullViolationError: null value in column 'email' of relation 'users' violates not-null constraint", "error_type": "sqlalchemy_notnull", "fix": "Make 'email' required in the Pydantic schema (remove Optional). Ensure it is passed in the request.", "instant_fix": False},
    {"error": "sqlalchemy.exc.IntegrityError: null value in column 'created_at' violates not-null constraint", "error_type": "sqlalchemy_notnull", "fix": "Add server_default=func.now() to the column: created_at: Mapped[datetime] = mapped_column(server_default=func.now())", "instant_fix": False},
    {"error": "NOT NULL constraint failed: users.password_hash", "error_type": "sqlalchemy_notnull", "fix": "Hash the password before saving: user.password_hash = bcrypt.hash(password)", "instant_fix": False},

    {"error": "sqlalchemy.exc.MissingGreenlet: lazy load for attribute 'User.posts' triggered in async context; use selectinload()", "error_type": "sqlalchemy_lazy", "fix": "Add selectinload to query: select(User).options(selectinload(User.posts)).where(User.id == user_id)", "instant_fix": False},
    {"error": "greenlet_spawn lazy load error: accessing relationship 'Order.items' after session closed", "error_type": "sqlalchemy_lazy", "fix": "Query with options(selectinload(Order.items)) or access items within the active session scope.", "instant_fix": False},
    {"error": "sqlalchemy.exc.InvalidRequestError: 'User.posts' is not available due to lazy='dynamic' in async mode", "error_type": "sqlalchemy_lazy", "fix": "Remove lazy='dynamic'. In async SQLAlchemy use lazy='select' with selectinload() or lazy='joined' with joinedload().", "instant_fix": False},
    {"error": "MissingGreenlet: 'comment.post' relationship accessed outside async session — add joinedload or selectinload to the query", "error_type": "sqlalchemy_lazy", "fix": "Eager-load: select(Comment).options(joinedload(Comment.post)).where(...)", "instant_fix": False},

    {"error": "sqlalchemy.exc.InvalidRequestError: This Session's transaction has been rolled back due to a previous exception. Issue Session.rollback() before starting a new transaction.", "error_type": "sqlalchemy_session", "fix": "Add await db.rollback() in the except block before re-using the session.", "instant_fix": False},
    {"error": "sqlalchemy.exc.InvalidRequestError: Can't operate on a closed Session.", "error_type": "sqlalchemy_session", "fix": "Don't reuse the session after it's been closed. Create a fresh session per request via Depends(get_db).", "instant_fix": False},
    {"error": "sqlalchemy.exc.InvalidRequestError: A transaction is already begun on this Session.", "error_type": "sqlalchemy_session", "fix": "Don't call session.begin() manually; FastAPI's get_db dependency manages the transaction lifecycle.", "instant_fix": False},

    {"error": "sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30", "error_type": "sqlalchemy_pool", "fix": "Increase pool_size and max_overflow in create_async_engine(): create_async_engine(url, pool_size=20, max_overflow=0)", "instant_fix": False},
    {"error": "asyncpg: could not connect to server within timeout — connection pool is exhausted", "error_type": "sqlalchemy_pool", "fix": "Close sessions properly with async context managers. Ensure every Depends(get_db) session is yielded inside try/finally.", "instant_fix": False},
    {"error": "sqlalchemy pool timeout: all connections in use. Consider reducing concurrent requests or increasing pool_size.", "error_type": "sqlalchemy_pool", "fix": "Add pool_pre_ping=True and pool_recycle=3600 to create_async_engine() to recycle stale connections.", "instant_fix": False},

    {"error": "sqlalchemy.exc.PendingRollbackError: This Session's transaction has been rolled back due to a previous exception. Please rollback() or close() this Session.", "error_type": "sqlalchemy_transaction", "fix": "Add await db.rollback() in the except handler before returning the error response.", "instant_fix": False},
    {"error": "asyncpg.exceptions.InFailedSQLTransactionError: current transaction is aborted, commands ignored until end of transaction block", "error_type": "sqlalchemy_transaction", "fix": "The previous query failed and the transaction is in an error state. Call await db.rollback() before the next query.", "instant_fix": False},
    {"error": "sqlalchemy.exc.InvalidRequestError: Can't reconnect until invalid transaction is rolled back. Please rollback() first.", "error_type": "sqlalchemy_transaction", "fix": "Wrap all DB operations in try/except with await db.rollback() in except and await db.commit() in try.", "instant_fix": False},

    {"error": "N+1 query problem: SELECT * FROM users executed, then N individual SELECT * FROM posts WHERE user_id=? queries inside loop", "error_type": "sqlalchemy_n1", "fix": "Use selectinload: select(User).options(selectinload(User.posts)) to load all posts in one query.", "instant_fix": False, "llm_prompt": "Fix N+1: replace loop over relationships with selectinload() or joinedload() in the initial query."},
    {"error": "Performance: accessing user.posts inside a for loop triggers one SQL query per user (N+1 anti-pattern)", "error_type": "sqlalchemy_n1", "fix": "Eager-load with .options(selectinload(User.posts)) or use a JOIN query to fetch all at once.", "instant_fix": False},

    {"error": "sqlalchemy.exc.IntegrityError: update or delete on table 'users' violates foreign key constraint 'posts_user_id_fkey' on table 'posts'", "error_type": "sqlalchemy_cascade", "fix": "Add cascade='all, delete-orphan' to the relationship: posts = relationship('Post', cascade='all, delete-orphan') or use ondelete='CASCADE' on the FK column.", "instant_fix": False},
    {"error": "ForeignKeyViolationError: DETAIL: Key (id)=(1) is still referenced from table 'comments' — cannot delete user", "error_type": "sqlalchemy_cascade", "fix": "Delete dependent records first or set cascade='all, delete-orphan' on the relationship so SQLAlchemy handles deletion order.", "instant_fix": False},

    # ═══ CATEGORY 13: Advanced FastAPI Errors (40 examples) ═══

    {"error": "ImportError: cannot import name 'get_db' from partially initialized module 'src.database.connection' (most likely due to a circular import)", "error_type": "fastapi_circular_import", "fix": "Break the circular import: move get_db to its own module (src/database/deps.py) and import from there.", "instant_fix": False, "llm_prompt": "Fix circular import by extracting shared dependencies into a separate module."},
    {"error": "ImportError: cannot import name 'User' from partially initialized module 'src.models.user' (most likely due to a circular import)", "error_type": "fastapi_circular_import", "fix": "Use TYPE_CHECKING guard: from __future__ import annotations and if TYPE_CHECKING: import the model.", "instant_fix": False},
    {"error": "circular import: src.routers.users imports src.schemas which imports src.routers.users", "error_type": "fastapi_circular_import", "fix": "Move shared Pydantic schemas to a dedicated src/schemas/ package that no router imports from.", "instant_fix": False},

    {"error": "TypeError: get_db() missing 1 required positional argument: 'db' — dependency not injected", "error_type": "fastapi_dependency", "fix": "Don't call get_db() manually. Use it as a FastAPI dependency: db: AsyncSession = Depends(get_db)", "instant_fix": False},
    {"error": "fastapi.exceptions.FastAPIError: Invalid args for response field! Hint: check that the dependency return type is a valid Pydantic type", "error_type": "fastapi_dependency", "fix": "Ensure the dependency returns a concrete type. Use response_model on the route, not on the dependency.", "instant_fix": False},
    {"error": "RuntimeError: Dependency 'get_current_user' raised an exception — HTTPException 401 propagated through dependency chain", "error_type": "fastapi_dependency", "fix": "HTTPException raised inside a dependency is correctly propagated. Ensure the client sends a valid Authorization header.", "instant_fix": False},
    {"error": "TypeError: get_current_user() got an unexpected keyword argument 'token' — dependency signature mismatch", "error_type": "fastapi_dependency", "fix": "Dependency parameters must be declared as FastAPI path/query params or use OAuth2PasswordBearer. Remove manual kwarg passing.", "instant_fix": False},

    {"error": "Exception in background task: asyncpg.InterfaceError — connection is closed (background task outlived the session)", "error_type": "fastapi_background", "fix": "Background tasks must create their own DB session. Don't pass a request-scoped session to a background task.", "instant_fix": False, "llm_prompt": "Fix background task DB error: create a fresh AsyncSession inside the background task function."},
    {"error": "RuntimeError: Task attached to a different loop — background task started in wrong async context", "error_type": "fastapi_background", "fix": "Use BackgroundTasks.add_task() with a regular async function. Don't manually create asyncio tasks inside FastAPI routes.", "instant_fix": False},
    {"error": "Exception in background task: sqlalchemy.exc.InvalidRequestError: Session is already closed", "error_type": "fastapi_background", "fix": "Create a dedicated session for background work: async with AsyncSessionFactory() as db: ...", "instant_fix": False},

    {"error": "starlette.websockets.WebSocketDisconnect: code=1000 — client disconnected", "error_type": "fastapi_websocket_error", "fix": "Wrap WebSocket receive/send in try/except WebSocketDisconnect and clean up resources on disconnect.", "instant_fix": False, "llm_prompt": "Handle WebSocketDisconnect: wrap websocket.receive_text() in try/except WebSocketDisconnect."},
    {"error": "RuntimeError: Cannot call 'send' once a close message has been sent — websocket already closed", "error_type": "fastapi_websocket_error", "fix": "Check connection state before sending. Set a connected flag to False on disconnect and guard sends with it.", "instant_fix": False},
    {"error": "WebSocketDisconnect: 1001 going away — client navigated away before WebSocket completed", "error_type": "fastapi_websocket_error", "fix": "Always handle WebSocketDisconnect exception in the WebSocket endpoint handler to avoid unhandled exceptions.", "instant_fix": False},

    {"error": "ValueError: content-type header 'application/json' is not multipart/form-data — file upload requires Form()", "error_type": "fastapi_file_upload", "fix": "Use UploadFile and Form() dependencies: async def upload(file: UploadFile, name: str = Form(...))", "instant_fix": False},
    {"error": "AttributeError: 'NoneType' object has no attribute 'filename' — UploadFile parameter not received", "error_type": "fastapi_file_upload", "fix": "Make file required: file: UploadFile (not Optional). Ensure client sends multipart/form-data Content-Type.", "instant_fix": False},
    {"error": "RequestEntityTooLarge: 413 — request body exceeds maximum allowed size", "error_type": "fastapi_file_upload", "fix": "Set max upload size in uvicorn: --limit-concurrency or use a middleware to check Content-Length before reading.", "instant_fix": False},

    {"error": "fastapi.exceptions.ResponseValidationError: 1 validation error for UserResponse — value is not a valid dict (type=type_error.dict)", "error_type": "fastapi_response_model", "fix": "Ensure the route returns data matching the response_model. Add model_config = ConfigDict(from_attributes=True) to the schema.", "instant_fix": False},
    {"error": "pydantic.ValidationError: field 'created_at' required in response model UserResponse but not found in returned dict", "error_type": "fastapi_response_model", "fix": "Include all required fields in the response. Return the full ORM object or use model_validate(orm_obj) if using Pydantic v2.", "instant_fix": False},
    {"error": "ResponseValidationError: Object of type datetime is not JSON serializable in response model", "error_type": "fastapi_response_model", "fix": "Add json_encoders to Pydantic model Config: json_encoders = {datetime: lambda v: v.isoformat()}", "instant_fix": False},

    {"error": "RuntimeError: CORS middleware must be the outermost middleware — add CORSMiddleware before other middleware", "error_type": "fastapi_middleware_error", "fix": "Add CORSMiddleware first: app.add_middleware(CORSMiddleware, ...) before any other middleware.", "instant_fix": False},
    {"error": "GZipMiddleware: Cannot set content-encoding when using GZip compression — middleware order issue", "error_type": "fastapi_middleware_error", "fix": "Add middleware in correct order: GZipMiddleware before CORSMiddleware (innermost registered = outermost executed in Starlette).", "instant_fix": False},

    {"error": "RuntimeError: Application startup failed. Ensure that the lifespan context manager yields exactly once.", "error_type": "fastapi_lifespan_error", "fix": "Add 'yield' inside the lifespan function: @asynccontextmanager async def lifespan(app): await init(); yield; await cleanup()", "instant_fix": False},
    {"error": "Exception in ASGI lifespan startup: asyncpg.exceptions.ConnectionRefusedError — database not ready", "error_type": "fastapi_lifespan_error", "fix": "Add retry logic in lifespan startup or ensure the database is running before starting the server.", "instant_fix": False},

    {"error": "ValueError: Duplicate route path '/api/users' with GET method — check include_router() prefix configuration", "error_type": "fastapi_router_conflict", "fix": "Check router prefixes. If router prefix='/users' and include_router has prefix='/api/users', remove the duplicate.", "instant_fix": False},
    {"error": "starlette.routing.NoMatchFound: No route exists for path '/api/v1/users' — router not included", "error_type": "fastapi_router_conflict", "fix": "Add app.include_router(users_router, prefix='/api/v1') in main.py to register the router.", "instant_fix": False},
    {"error": "AssertionError: APIRouter path '/users/{id}' conflicts with '/users/me' — more specific route must come first", "error_type": "fastapi_router_conflict", "fix": "Define '/users/me' before '/users/{id}' in the router so the literal path takes precedence.", "instant_fix": False},

    {"error": "WARNING: Duplicate operationId 'create_user_api_users_post' in OpenAPI schema — multiple routes have the same ID", "error_type": "fastapi_duplicate_op", "fix": "Add operation_id parameter to the route decorator: @router.post('/users', operation_id='create_user')", "instant_fix": False},
    {"error": "fastapi.exceptions.FastAPIError: operationId 'get_items' is used by more than one route in the OpenAPI schema", "error_type": "fastapi_duplicate_op", "fix": "Set unique operation_id on each route, or rename the function so FastAPI generates distinct IDs.", "instant_fix": False},

    {"error": "pydantic.ValidationError: value is not a valid integer for path parameter 'user_id' — received string 'me'", "error_type": "fastapi_path_param_error", "fix": "Define the '/users/me' route before '/users/{user_id}' so 'me' is matched as a literal before the integer param.", "instant_fix": False},
    {"error": "ValueError: path parameter '{user_id}' is declared in the route but not in the function signature", "error_type": "fastapi_path_param_error", "fix": "Add user_id: int to the function signature: async def get_user(user_id: int, db: AsyncSession = Depends(get_db))", "instant_fix": False},
    {"error": "MissingParamError: Path parameter 'id' not found in route '/users/{user_id}' — name mismatch", "error_type": "fastapi_path_param_error", "fix": "Match the parameter name in the path and function: use {user_id} in path and user_id: int in function.", "instant_fix": False},

    {"error": "RequestValidationError: value is not a valid integer for query parameter 'page' — received 'two'", "error_type": "fastapi_query_type", "fix": "Validate query param type in the signature: page: int = Query(1, ge=1). Client must pass an integer.", "instant_fix": False},
    {"error": "422 Unprocessable Entity: query parameter 'limit' expected int, got 'all'", "error_type": "fastapi_query_type", "fix": "Use Optional[int] = Query(None) if the parameter can be omitted, or document that integers are required.", "instant_fix": False},
    {"error": "RequestValidationError: value could not be parsed to a boolean for query parameter 'active' — use true/false not 1/0", "error_type": "fastapi_query_type", "fix": "FastAPI parses bool query params from 'true'/'false'. If using 1/0, cast manually: active: int and convert to bool.", "instant_fix": False},

    {"error": "422 Unprocessable Entity: header 'x-api-key' field required", "error_type": "fastapi_header_error", "fix": "Add the header dependency: api_key: str = Header(..., alias='x-api-key')", "instant_fix": False},
    {"error": "starlette.exceptions.HTTPException: 401 — Authorization header missing or not in 'Bearer <token>' format", "error_type": "fastapi_header_error", "fix": "Use OAuth2PasswordBearer or manually parse: token = authorization.split('Bearer ')[1] with error handling.", "instant_fix": False},
    {"error": "RequestValidationError: missing required header 'Content-Type: application/json'", "error_type": "fastapi_header_error", "fix": "Client must set Content-Type: application/json for JSON body requests. Check client request configuration.", "instant_fix": False},

    {"error": "422 Unprocessable Entity: Form field 'username' is required but missing — did you send multipart/form-data?", "error_type": "fastapi_form_data", "fix": "Use Form() in the route: username: str = Form(...). Client must send multipart/form-data, not JSON.", "instant_fix": False},
    {"error": "ValueError: parameter 'email' must use Form() — mixing JSON body and form data is not allowed in FastAPI", "error_type": "fastapi_form_data", "fix": "Use all Form() parameters for form submissions. Cannot mix request body JSON with Form() fields.", "instant_fix": False},
    {"error": "RuntimeError: Stream consumed — cannot read form data after request body was already read", "error_type": "fastapi_form_data", "fix": "Read form data only once. Use await request.form() or Form() dependency — do not mix both.", "instant_fix": False},

    # ═══ CATEGORY 14: Advanced React / TypeScript Errors (50 examples) ═══

    {"error": "Warning: Maximum update depth exceeded. This can happen when a component calls setState inside useEffect, but useEffect either doesn't have a dependency array, or one of the dependencies changes on every render.", "error_type": "react_useeffect_loop", "fix": "Add a dependency array to useEffect. Make sure objects/arrays in deps are stable (use useMemo/useCallback).", "instant_fix": False},
    {"error": "useEffect infinite loop: object in dependency array is recreated every render — { id: 1 } !== { id: 1 }", "error_type": "react_useeffect_loop", "fix": "Destructure primitive values from objects: useEffect(() => {...}, [user.id]) instead of [user].", "instant_fix": False},
    {"error": "useEffect infinite loop: function dependency changes on every render because it's defined inline", "error_type": "react_useeffect_loop", "fix": "Wrap the function in useCallback with stable deps, or move it inside useEffect if only used there.", "instant_fix": False},
    {"error": "React: setState called inside useEffect with no dependency array — causes infinite loop", "error_type": "react_useeffect_loop", "fix": "Add [] as dependency array for one-time effects, or add only the values that should trigger re-fetch.", "instant_fix": False},

    {"error": "Stale closure: the counter value in setTimeout is always 0 even after multiple increments", "error_type": "react_stale_closure", "fix": "Use the functional update form: setCount(prev => prev + 1) to avoid capturing stale state in closures.", "instant_fix": False},
    {"error": "Stale closure in async fetch: state value inside async callback is the initial value, not the current one", "error_type": "react_stale_closure", "fix": "Use a ref to track the latest value: const latestValue = useRef(value); latestValue.current = value;", "instant_fix": False},
    {"error": "useCallback returns stale function — the callback captures old state values because deps array is empty", "error_type": "react_stale_closure", "fix": "Add all state values used inside useCallback to its dependency array.", "instant_fix": False},

    {"error": "Error: useContext must be used inside a Provider. useThemeContext returned undefined.", "error_type": "react_context_missing", "fix": "Wrap the component tree with <ThemeProvider>. Ensure the Provider is an ancestor of all consumers.", "instant_fix": False},
    {"error": "TypeError: Cannot destructure property 'user' of undefined — useAuthContext returned undefined (no Provider)", "error_type": "react_context_missing", "fix": "Add <AuthProvider> to App.tsx wrapping all routes that need auth context.", "instant_fix": False},
    {"error": "React context: useContext(CartContext) is undefined — CartContext.Provider is missing in the component tree", "error_type": "react_context_missing", "fix": "Add a default value to createContext() or throw a helpful error when used outside provider.", "instant_fix": False},

    {"error": "TypeError: Cannot read properties of null (reading 'focus') — ref.current is null on first render", "error_type": "react_ref_null", "fix": "Guard ref access: if (ref.current) ref.current.focus(). Refs are null until after the first render.", "instant_fix": False},
    {"error": "Error: ref.current is null — element not yet mounted when useEffect with [] runs synchronously", "error_type": "react_ref_null", "fix": "useLayoutEffect runs synchronously after DOM mutations. Use it instead of useEffect for DOM measurements.", "instant_fix": False},
    {"error": "Cannot read properties of null (reading 'getBoundingClientRect') — the ref is not attached to a DOM element", "error_type": "react_ref_null", "fix": "Ensure the ref is passed as the ref prop to the correct JSX element. Check conditional rendering isn't hiding the element.", "instant_fix": False},

    {"error": "Warning: Function components cannot be given refs. Attempts to access this ref will fail. Did you mean to use React.forwardRef()?", "error_type": "react_forwardref", "fix": "Wrap the child component with forwardRef: const Input = forwardRef((props, ref) => <input ref={ref} {...props} />)", "instant_fix": False},
    {"error": "TypeError: Cannot assign to 'current' property of object — forwardRef ref type is wrong", "error_type": "react_forwardref", "fix": "Type the ref correctly: const inputRef = useRef<HTMLInputElement>(null) and forwardRef<HTMLInputElement, Props>", "instant_fix": False},

    {"error": "Error: A React component suspended while rendering, but no fallback UI was specified. Add a <Suspense fallback={}> component higher in the tree.", "error_type": "react_suspense", "fix": "Wrap lazy-loaded components with <Suspense fallback={<Spinner />}>. Every React.lazy() component needs a Suspense boundary.", "instant_fix": False},
    {"error": "Uncaught Error: Minified React error #185 — component suspended but Suspense boundary not found in tree", "error_type": "react_suspense", "fix": "Add <Suspense fallback={<div>Loading...</div>}> around the component that uses React.lazy() or data-fetching with use().", "instant_fix": False},
    {"error": "SuspenseException: Suspended while rendering — Suspense boundary above does not have a fallback prop", "error_type": "react_suspense", "fix": "Add the fallback prop: <Suspense fallback={<LoadingSpinner />}>. Empty fallback is allowed: fallback={null}", "instant_fix": False},

    {"error": "ErrorBoundary not catching async error — unhandled promise rejection in useEffect is not caught by error boundary", "error_type": "react_error_boundary", "fix": "Error boundaries only catch render errors. Handle async errors separately: setError(err) and throw in render.", "instant_fix": False},
    {"error": "Uncaught TypeError: Cannot read property 'map' of undefined — no ErrorBoundary to catch render crash", "error_type": "react_error_boundary", "fix": "Wrap the component tree with a class-based ErrorBoundary or use react-error-boundary package.", "instant_fix": False},
    {"error": "Error boundary caught: Objects are not valid as a React child — but error not displayed because no fallback UI", "error_type": "react_error_boundary", "fix": "Add a renderError method to ErrorBoundary: static getDerivedStateFromError(error) { return { hasError: true, error }; }", "instant_fix": False},

    {"error": "Warning: A component is changing an uncontrolled input to be controlled. This may be caused by the value changing from undefined to a value.", "error_type": "react_controlled_input", "fix": "Initialize state with empty string, not undefined: const [value, setValue] = useState('') not useState()", "instant_fix": False},
    {"error": "Warning: A component is changing a controlled input to be uncontrolled. The value prop was provided but onChange was not.", "error_type": "react_controlled_input", "fix": "Add an onChange handler: onChange={(e) => setValue(e.target.value)}. Controlled inputs need both value and onChange.", "instant_fix": False},
    {"error": "React: input switches from controlled to uncontrolled when form data becomes undefined after API call", "error_type": "react_controlled_input", "fix": "Use optional chaining with fallback: value={data?.name ?? ''} to keep the input controlled even when data is undefined.", "instant_fix": False},

    {"error": "Warning: Button.defaultProps will be removed from future React versions. Use JavaScript default parameters instead.", "error_type": "react_defaultprops", "fix": "Replace Component.defaultProps = {...} with default params: function Button({ size = 'md', color = 'blue' }: Props)", "instant_fix": False},
    {"error": "React: defaultProps is deprecated for function components and will be removed in a future version", "error_type": "react_defaultprops", "fix": "Use ES6 default parameter values in the function signature instead of static defaultProps.", "instant_fix": False},

    {"error": "React StrictMode: useEffect cleanup function called twice — side effects run twice in development", "error_type": "react_strict_mode", "fix": "This is expected in React 18 StrictMode. Implement proper cleanup in useEffect return function. Don't disable StrictMode.", "instant_fix": False},
    {"error": "Warning: An update to App inside a test was not wrapped in act(). When testing React code, ensure all state updates use act().", "error_type": "react_strict_mode", "fix": "Wrap async operations in act(): await act(async () => { fireEvent.click(button); await somePromise; })", "instant_fix": False},

    {"error": "TS2315: Type 'string' does not satisfy the constraint 'Record<string, unknown>'", "error_type": "typescript_generic", "fix": "Add a type constraint to the generic: function fetchData<T extends Record<string, unknown>>(url: string): Promise<T>", "instant_fix": False},
    {"error": "TS2344: Type 'number' does not satisfy the constraint 'string | number' for generic type parameter", "error_type": "typescript_generic", "fix": "Specify the correct type argument when calling the generic function: getData<string>() or getData<number>()", "instant_fix": False},
    {"error": "TS2322: Type 'Component<Props>' is not assignable to type 'ComponentType<{}>' — generic component type mismatch", "error_type": "typescript_generic", "fix": "Use ComponentPropsWithRef<typeof Component> or cast with 'as React.ComponentType' for dynamic component types.", "instant_fix": False},
    {"error": "TS2345: Argument of type 'T' is not assignable to parameter of type 'string' — generic not constrained", "error_type": "typescript_generic", "fix": "Constrain the generic: <T extends string>(value: T) or use keyof/typeof for object-based constraints.", "instant_fix": False},

    {"error": "TS2339: Property 'name' does not exist on type 'Cat | Dog' — discriminated union not narrowed", "error_type": "typescript_discriminated", "fix": "Use the discriminant field to narrow: if (animal.type === 'cat') { animal.name } — TypeScript narrows the union.", "instant_fix": False},
    {"error": "TS2366: Function lacks ending return statement and return type does not include 'undefined' — non-exhaustive switch on discriminated union", "error_type": "typescript_discriminated", "fix": "Add a default case that throws: default: const _exhaustive: never = value; throw new Error('Not exhaustive')", "instant_fix": False},
    {"error": "TS2345: Argument of type 'Circle | Square' is not assignable to type 'Circle' — discriminated union too wide", "error_type": "typescript_discriminated", "fix": "Narrow the type first: if (shape.kind === 'circle') { handleCircle(shape) } — shape is now typed as Circle.", "instant_fix": False},

    {"error": "TS2339: Property 'name' does not exist on type 'string | User' — type not narrowed after API call", "error_type": "react_stale_closure", "fix": "Narrow the type: if (typeof result === 'object' && result !== null) { result.name } or use a type guard.", "instant_fix": False},

    {"error": "Error: async components are not supported in React. Use useEffect for data fetching instead.", "error_type": "react_async_component", "fix": "Remove async from the component: function MyComponent() { const [data, setData] = useState(); useEffect(() => { fetchData().then(setData); }, []); }", "instant_fix": False},
    {"error": "Warning: An update was not wrapped in act() — async state update after component unmounted", "error_type": "react_async_component", "fix": "Cancel async operations on unmount: let mounted = true; return () => { mounted = false; }; Only setState if mounted.", "instant_fix": False},
    {"error": "Unhandled Rejection: Cannot update state of unmounted component in async useEffect", "error_type": "react_async_component", "fix": "Use AbortController to cancel fetch on unmount: const controller = new AbortController(); return () => controller.abort();", "instant_fix": False},

    {"error": "React Hook useCallback received a function whose dependencies are unknown. Pass an inline function instead.", "error_type": "react_callback_deps", "fix": "Move the function inline into useCallback: useCallback(() => { doThing(dep); }, [dep])", "instant_fix": False},
    {"error": "ESLint: React Hook useCallback has a missing dependency: 'onSubmit'. Either include it or remove the dependency array.", "error_type": "react_callback_deps", "fix": "Add onSubmit to the dependency array: useCallback(() => onSubmit(data), [onSubmit, data])", "instant_fix": False},
    {"error": "React useCallback with empty deps array returns stale function — onSave doesn't reflect latest form values", "error_type": "react_callback_deps", "fix": "Include all values used inside useCallback in the deps: useCallback(() => save(formValues), [formValues, save])", "instant_fix": False},

    {"error": "useMemo is recomputing on every render — the dependency is an object created inline", "error_type": "react_usememo", "fix": "Memoize or extract the object: const config = useMemo(() => ({ key: 'value' }), []). Don't pass inline objects as deps.", "instant_fix": False},
    {"error": "useMemo: expensive calculation still running every render because deps contain a non-primitive value", "error_type": "react_usememo", "fix": "Use primitive deps: useMemo(() => compute(id), [id]) instead of useMemo(() => compute(obj), [obj])", "instant_fix": False},

    {"error": "Warning: Each child in a list should have a unique 'key' prop. Keys should be stable — using array index can cause bugs with re-ordering.", "error_type": "react_key", "fix": "Use a stable unique ID as key: items.map(item => <Item key={item.id} .../>). Avoid index unless list never reorders.", "instant_fix": False},
    {"error": "Warning: React.Fragment cannot have a key prop when using shorthand <> syntax", "error_type": "react_key", "fix": "Use explicit <React.Fragment key={id}> syntax when you need to add keys to fragments.", "instant_fix": False},

    {"error": "TS2322: Type '(e: Event) => void' is not assignable to type 'MouseEventHandler<HTMLButtonElement>'", "error_type": "react_event_type", "fix": "Use the correct React event type: (e: React.MouseEvent<HTMLButtonElement>) => void", "instant_fix": False},
    {"error": "TS2345: Argument of type 'ChangeEvent<HTMLInputElement>' is not assignable to parameter of type 'Event'", "error_type": "react_event_type", "fix": "Use React.ChangeEvent<HTMLInputElement> as the event type, not the native DOM Event.", "instant_fix": False},
    {"error": "TS7006: Parameter 'e' implicitly has an 'any' type in event handler", "error_type": "react_event_type", "fix": "Type the event: (e: React.FormEvent<HTMLFormElement>) => void or (e: React.ChangeEvent<HTMLInputElement>) => void", "instant_fix": False},

    # ═══ CATEGORY 15: Advanced Python Errors (40 examples) ═══

    {"error": "RuntimeWarning: coroutine 'fetch_users' was never awaited — function called without await", "error_type": "python_coroutine_not_awaited", "fix": "Add await: result = await fetch_users() not result = fetch_users()", "instant_fix": False},
    {"error": "RuntimeWarning: Enable tracemalloc to get the object allocation traceback. coroutine 'create_user' was never awaited", "error_type": "python_coroutine_not_awaited", "fix": "Find and await every async function call. Search for calls to async functions without 'await'.", "instant_fix": False},
    {"error": "TypeError: object coroutine can't be used in 'await' expression — double-awaited coroutine", "error_type": "python_coroutine_not_awaited", "fix": "Remove one await. Each async function should be awaited exactly once.", "instant_fix": False},
    {"error": "SyntaxError: 'await' outside async function — await used in regular function", "error_type": "python_coroutine_not_awaited", "fix": "Change def to async def, or call the async function using asyncio.run() from synchronous code.", "instant_fix": False},

    {"error": "RuntimeError: This event loop is already running — calling asyncio.run() inside a running event loop", "error_type": "python_blocking_async", "fix": "In async contexts, await the coroutine directly instead of using asyncio.run(). Use asyncio.get_event_loop().run_until_complete() only from sync code.", "instant_fix": False},
    {"error": "asyncio warning: Executing <Task ...> took too long — blocking call detected in async event loop", "error_type": "python_blocking_async", "fix": "Replace blocking calls with async equivalents: requests → httpx.AsyncClient, time.sleep → asyncio.sleep, open → aiofiles.open", "instant_fix": False},
    {"error": "Event loop blocked: time.sleep(5) called inside async function — use await asyncio.sleep(5) instead", "error_type": "python_blocking_async", "fix": "Replace time.sleep() with await asyncio.sleep(). Never use blocking I/O inside async functions.", "instant_fix": False},
    {"error": "BlockingIOError: synchronous file read detected in async context — use aiofiles for async file I/O", "error_type": "python_blocking_async", "fix": "Use aiofiles: async with aiofiles.open('file.txt') as f: content = await f.read()", "instant_fix": False},

    {"error": "ImportError: cannot import name 'User' from partially initialized module 'src.models' (most likely due to a circular import)", "error_type": "python_circular_import", "fix": "Break the cycle: use TYPE_CHECKING guard, lazy imports inside functions, or restructure modules.", "instant_fix": False},
    {"error": "ImportError: circular import — src.schemas imports src.models which imports src.schemas", "error_type": "python_circular_import", "fix": "Move shared type definitions to a separate src/types.py module that neither models.py nor schemas.py imports from.", "instant_fix": False},
    {"error": "AttributeError: partially initialized module 'src.database.models' has no attribute 'User' (circular import)", "error_type": "python_circular_import", "fix": "Import inside the function body: def get_user(): from src.models import User; return User. Breaks the import-time cycle.", "instant_fix": False},

    {"error": "pydantic.errors.PydanticUserError: 'validator' is removed in Pydantic v2. Use 'field_validator' instead.", "error_type": "python_pydantic_v2", "fix": "Replace @validator with @field_validator('field_name', mode='before') in Pydantic v2.", "instant_fix": False},
    {"error": "AttributeError: 'UserSchema' object has no attribute 'dict' — use model_dump() in Pydantic v2", "error_type": "python_pydantic_v2", "fix": "Replace .dict() with .model_dump() and .json() with .model_dump_json() in Pydantic v2.", "instant_fix": False},
    {"error": "pydantic.errors.PydanticUserError: You must use 'model_validate' instead of 'parse_obj' in Pydantic v2", "error_type": "python_pydantic_v2", "fix": "Replace UserSchema.parse_obj(data) with UserSchema.model_validate(data) in Pydantic v2.", "instant_fix": False},
    {"error": "pydantic.errors.PydanticSchemaGenerationError: class-based config is deprecated in Pydantic v2 — use model_config = ConfigDict(...)", "error_type": "python_pydantic_v2", "fix": "Replace inner class Config: orm_mode = True with model_config = ConfigDict(from_attributes=True) in Pydantic v2.", "instant_fix": False},
    {"error": "ValidationError: Input should be a valid dictionary or instance of UserSchema [type=model_type] — Pydantic v2 stricter validation", "error_type": "python_pydantic_v2", "fix": "Use model_validate(obj, from_attributes=True) when converting ORM objects to Pydantic models in v2.", "instant_fix": False},

    {"error": "KeyError: 'DATABASE_URL' — environment variable not found. Did you forget to call load_dotenv()?", "error_type": "python_dotenv_order", "fix": "Call load_dotenv() at the top of main.py BEFORE importing modules that read env vars.", "instant_fix": False},
    {"error": "os.environ['SECRET_KEY'] raises KeyError — .env file exists but variables not loaded at import time", "error_type": "python_dotenv_order", "fix": "Move from dotenv import load_dotenv; load_dotenv() to the very top of the entry point file before all other imports.", "instant_fix": False},
    {"error": "Settings reads DATABASE_URL=None — dotenv loaded after settings module was imported", "error_type": "python_dotenv_order", "fix": "Ensure load_dotenv() is called before 'from src.config.settings import settings'. The order of imports matters.", "instant_fix": False},

    {"error": "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte", "error_type": "python_encoding", "fix": "Specify encoding when reading files: open('file.txt', 'r', encoding='utf-8', errors='replace')", "instant_fix": False},
    {"error": "UnicodeEncodeError: 'ascii' codec can't encode character '\u2019' — default encoding is ASCII on Windows", "error_type": "python_encoding", "fix": "Set encoding explicitly: open('output.txt', 'w', encoding='utf-8') or set PYTHONIOENCODING=utf-8.", "instant_fix": False},
    {"error": "LookupError: unknown encoding: utf8 — use 'utf-8' (with hyphen) not 'utf8'", "error_type": "python_encoding", "fix": "Use the canonical name: encoding='utf-8' (with hyphen). Python accepts both but some contexts require the dash.", "instant_fix": False},
    {"error": "chardet detected encoding: windows-1252 — file read as wrong encoding causing garbled text", "error_type": "python_encoding", "fix": "Detect encoding: import chardet; enc = chardet.detect(raw)['encoding']; then decode with that encoding.", "instant_fix": False},

    {"error": "TypeError: Object of type UUID is not JSON serializable", "error_type": "python_json_serialize", "fix": "Convert UUID to string before serializing: json.dumps({'id': str(obj.id)}) or use a custom encoder.", "instant_fix": False},
    {"error": "TypeError: Object of type datetime is not JSON serializable", "error_type": "python_json_serialize", "fix": "Convert datetime to ISO string: json.dumps({'created_at': obj.created_at.isoformat()})", "instant_fix": False},
    {"error": "TypeError: Object of type Decimal is not JSON serializable", "error_type": "python_json_serialize", "fix": "Convert Decimal to float or string: json.dumps({'price': float(obj.price)}). Use float for display, string for precision.", "instant_fix": False},
    {"error": "TypeError: Object of type bytes is not JSON serializable — binary data in JSON payload", "error_type": "python_json_serialize", "fix": "Base64-encode bytes: import base64; json.dumps({'data': base64.b64encode(bytes_val).decode()})", "instant_fix": False},

    {"error": "TypeError: can't compare offset-naive and offset-aware datetimes — mixing naive and timezone-aware datetime", "error_type": "python_datetime_tz", "fix": "Make both datetimes timezone-aware: datetime.now(timezone.utc) instead of datetime.utcnow() (which is naive).", "instant_fix": False},
    {"error": "ValueError: astimezone() cannot be applied to a naive datetime — datetime has no timezone info", "error_type": "python_datetime_tz", "fix": "Replace datetime.utcnow() with datetime.now(timezone.utc) throughout the codebase for timezone-aware datetimes.", "instant_fix": False},
    {"error": "sqlalchemy.exc.StatementError: (builtins.ValueError) 'offset-naive datetime' — SQLAlchemy column expects timezone-aware datetime", "error_type": "python_datetime_tz", "fix": "Use DateTime(timezone=True) in the ORM column and datetime.now(timezone.utc) when creating timestamps.", "instant_fix": False},
    {"error": "pytz.exceptions.AmbiguousTimeError — DST transition makes time ambiguous during fallback", "error_type": "python_datetime_tz", "fix": "Pass is_dst=False to localize(): tz.localize(dt, is_dst=False) to resolve ambiguity during DST fallback.", "instant_fix": False},

    {"error": "FloatingPointError: 0.1 + 0.2 == 0.30000000000000004 — float precision error in money calculations", "error_type": "python_decimal_float", "fix": "Use Decimal for money: from decimal import Decimal; total = Decimal('0.1') + Decimal('0.2')", "instant_fix": False},
    {"error": "sqlalchemy.exc.DataError: invalid input syntax for type numeric — float passed where Decimal expected", "error_type": "python_decimal_float", "fix": "Cast float to Decimal before storing: Decimal(str(float_value)) — avoid Decimal(float_value) which inherits float imprecision.", "instant_fix": False},

    {"error": "TypeError: a bytes-like object is required, not 'str' — writing string to binary file", "error_type": "python_bytes_string", "fix": "Encode string to bytes: file.write(text.encode('utf-8')) or open in text mode: open(path, 'w', encoding='utf-8')", "instant_fix": False},
    {"error": "AttributeError: 'bytes' object has no attribute 'split' — bytes returned where string expected", "error_type": "python_bytes_string", "fix": "Decode bytes to string: data.decode('utf-8'). If data may be bytes or str: if isinstance(data, bytes): data = data.decode()", "instant_fix": False},
    {"error": "TypeError: sequence item 0: expected str instance, bytes found — joining mixed bytes and strings", "error_type": "python_bytes_string", "fix": "Decode all items before joining: ''.join(item.decode('utf-8') if isinstance(item, bytes) else item for item in items)", "instant_fix": False},

    {"error": "RuntimeError: no running event loop — asyncio.get_event_loop() deprecated outside of async context in Python 3.10+", "error_type": "python_blocking_async", "fix": "Use asyncio.run(main()) to start the event loop. Inside async functions, use asyncio.get_running_loop().", "instant_fix": False},
    {"error": "DeprecationWarning: There is no current event loop — asyncio.get_event_loop() in Python 3.12 raises an error", "error_type": "python_blocking_async", "fix": "Replace asyncio.get_event_loop().run_until_complete(coro) with asyncio.run(coro) at the entry point.", "instant_fix": False},

    # ═══ CATEGORY 16: Deployment / Production Errors (30 examples) ═══

    {"error": "Error: listen EADDRINUSE: address already in use :::8000 — PORT not set, defaulting to hardcoded 8000 conflicts with Render's assigned port", "error_type": "deploy_port", "fix": "Use PORT env var: port = int(os.getenv('PORT', 8000)); uvicorn.run(app, host='0.0.0.0', port=port)", "instant_fix": True, "fix_action": "fix_port_binding"},
    {"error": "Render: Web service exited with code 1 — app bound to localhost instead of 0.0.0.0", "error_type": "deploy_port", "fix": "Bind to 0.0.0.0 and use $PORT: uvicorn main:app --host 0.0.0.0 --port $PORT", "instant_fix": True, "fix_action": "fix_port_binding"},
    {"error": "Railway: Application failed to listen on port $PORT — uvicorn started on wrong port", "error_type": "deploy_port", "fix": "Set start command: uvicorn main:app --host 0.0.0.0 --port $PORT (Railway injects PORT as env var).", "instant_fix": True, "fix_action": "fix_port_binding"},
    {"error": "Heroku H10 app crashed: web process failed to bind to $PORT within 60 seconds of launch", "error_type": "deploy_port", "fix": "Bind to $PORT: import os; uvicorn.run(app, host='0.0.0.0', port=int(os.environ['PORT']))", "instant_fix": True, "fix_action": "fix_port_binding"},

    {"error": "asyncpg: invalid DSN 'postgres://user:pass@host/db' — asyncpg requires 'postgresql+asyncpg://' scheme", "error_type": "deploy_database_url", "fix": "Replace 'postgres://' with 'postgresql+asyncpg://' in DATABASE_URL. Render/Heroku provide postgres:// URLs by default.", "instant_fix": True, "fix_action": "fix_database_url_scheme"},
    {"error": "sqlalchemy.exc.ArgumentError: Could not parse rfc1738 URL from string 'postgres://...' — use postgresql+asyncpg://", "error_type": "deploy_database_url", "fix": "Update DATABASE_URL: replace the 'postgres://' prefix with 'postgresql+asyncpg://' for SQLAlchemy async.", "instant_fix": True, "fix_action": "fix_database_url_scheme"},
    {"error": "create_async_engine() error: scheme 'postgres' is not recognized — did you mean 'postgresql+asyncpg'?", "error_type": "deploy_database_url", "fix": "DATABASE_URL must start with 'postgresql+asyncpg://'. Convert at load time: url.replace('postgres://', 'postgresql+asyncpg://', 1)", "instant_fix": True, "fix_action": "fix_database_url_scheme"},
    {"error": "Render deployment: DATABASE_URL is set to 'postgres://' but asyncpg requires 'postgresql+asyncpg://'", "error_type": "deploy_database_url", "fix": "Add URL conversion in settings: DATABASE_URL = os.getenv('DATABASE_URL', '').replace('postgres://', 'postgresql+asyncpg://', 1)", "instant_fix": True, "fix_action": "fix_database_url_scheme"},

    {"error": "asyncpg.exceptions.InvalidPasswordError: SSL connection has been closed unexpectedly — production DB requires SSL", "error_type": "deploy_ssl", "fix": "Add SSL to asyncpg connection: create_async_engine(url + '?ssl=require') or engine_args={'ssl': True}", "instant_fix": False},
    {"error": "ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED — server certificate verification failed on production DB", "error_type": "deploy_ssl", "fix": "Pass SSL context: import ssl; ctx = ssl.create_default_context(); engine = create_async_engine(url, connect_args={'ssl': ctx})", "instant_fix": False},
    {"error": "asyncpg: SSL required but server does not support SSL — add ?sslmode=require to DATABASE_URL", "error_type": "deploy_ssl", "fix": "Append sslmode to the URL: DATABASE_URL + '?sslmode=require' or add ssl='require' in connect_args.", "instant_fix": False},

    {"error": "404 Not Found for /static/css/main.chunk.css — static files not configured in FastAPI", "error_type": "deploy_static_files", "fix": "Mount static files: app.mount('/static', StaticFiles(directory='static'), name='static'). Ensure build output is in 'static/'.", "instant_fix": False},
    {"error": "RuntimeError: StaticFiles directory 'frontend/build' does not exist — run npm run build first", "error_type": "deploy_static_files", "fix": "Add build step in Dockerfile/CI: RUN npm run build in frontend/. Then mount the 'frontend/build' directory.", "instant_fix": False},
    {"error": "React app shows blank page on production — index.html not served by FastAPI for non-API routes", "error_type": "deploy_static_files", "fix": "Add catch-all route to serve index.html: @app.get('/{path:path}') async def spa(path): return FileResponse('build/index.html')", "instant_fix": False},

    {"error": "502 Bad Gateway — Nginx upstream error: connect() failed (111: Connection refused) while connecting to upstream", "error_type": "deploy_502", "fix": "App is not running or bound to wrong interface. Ensure uvicorn binds to 0.0.0.0 and the correct PORT.", "instant_fix": False},
    {"error": "Render 502: upstream connect error or disconnect/reset before headers — uvicorn crashed on startup", "error_type": "deploy_502", "fix": "Check deployment logs for startup error. Common causes: DB connection fail, missing env vars, import error.", "instant_fix": False},
    {"error": "502 Bad Gateway on Vercel/Railway — function timeout or process crash after cold start", "error_type": "deploy_502", "fix": "Add startup health check, increase timeout setting, and ensure all env vars (DATABASE_URL, SECRET_KEY) are set.", "instant_fix": False},

    {"error": "OOMKilled: container killed due to memory limit exceeded (512Mi) — Render/Railway free tier", "error_type": "deploy_memory", "fix": "Reduce memory: lower SQLAlchemy pool_size, reduce worker count, or upgrade to paid tier with more RAM.", "instant_fix": False},
    {"error": "JavaScript heap out of memory — Node.js build process exceeded memory limit during npm run build", "error_type": "deploy_memory", "fix": "Set NODE_OPTIONS: export NODE_OPTIONS=--max-old-space-size=4096 before running the build command.", "instant_fix": False},
    {"error": "Python MemoryError: cannot allocate memory in static TLS block — too many workers or large model loaded", "error_type": "deploy_memory", "fix": "Reduce Gunicorn workers: --workers 1 for low-memory deployments. Consider async workers: --worker-class uvicorn.workers.UvicornWorker.", "instant_fix": False},

    {"error": "Heroku H12 Request Timeout: request took longer than 30 seconds — gunicorn worker timed out", "error_type": "deploy_timeout", "fix": "Add --timeout 60 to Gunicorn or move slow operations to background tasks. Heroku allows increasing timeout via config.", "instant_fix": False},
    {"error": "Gateway Timeout 504 — upstream timed out (110: Connection timed out) waiting for response", "error_type": "deploy_timeout", "fix": "Increase Nginx proxy_read_timeout, add request timeout to uvicorn, or optimize slow DB queries causing the delay.", "instant_fix": False},
    {"error": "Cloud Run timeout: request deadline exceeded 60s — function timed out during database migration", "error_type": "deploy_timeout", "fix": "Run migrations separately from app startup. Use a pre-deploy script or a separate migration job/container.", "instant_fix": False},

    {"error": "Build failed: npm run build exited with code 1 — TypeScript compilation error in production build", "error_type": "deploy_build", "fix": "Fix all TypeScript errors locally with 'npx tsc --noEmit' before deploying. Build is stricter than dev mode.", "instant_fix": False},
    {"error": "pip install failed: Could not find a version that satisfies the requirement python-dotenv==2.0.0 — version does not exist", "error_type": "deploy_build", "fix": "Fix requirements.txt: pin to an existing version (python-dotenv==1.0.0) or use >= without upper bound.", "instant_fix": False},
    {"error": "Render build failed: bash: python: command not found — Python not in PATH on build image", "error_type": "deploy_build", "fix": "Use 'python3' instead of 'python' in build/start commands on Linux. Or specify runtime: python-3.11.0 in render.yaml.", "instant_fix": False},

    {"error": "Render: start command 'node server.js' failed — wrong command for a Python FastAPI app", "error_type": "deploy_start", "fix": "Set start command to: uvicorn main:app --host 0.0.0.0 --port $PORT", "instant_fix": False},
    {"error": "Railway: web process failed — Procfile has 'web: python app.py' but port binding is wrong", "error_type": "deploy_start", "fix": "Update Procfile: web: uvicorn main:app --host 0.0.0.0 --port $PORT", "instant_fix": False},
    {"error": "Missing environment variable SECRET_KEY on production — KeyError during startup", "error_type": "deploy_env_missing", "fix": "Set SECRET_KEY in the platform's environment variable settings (Render > Environment, Railway > Variables).", "instant_fix": False},
    {"error": "Production: ANTHROPIC_API_KEY not found — environment variable not set in deployment platform", "error_type": "deploy_env_missing", "fix": "Add all required env vars to the deployment platform. Check .env.example for required variables.", "instant_fix": False},

    # ═══ CATEGORY 17: Authentication / Security Errors (30 examples) ═══

    {"error": "jwt.exceptions.ExpiredSignatureError: Signature has expired — JWT token past its exp claim", "error_type": "auth_jwt_expired", "fix": "Catch ExpiredSignatureError and return HTTP 401. Implement token refresh: issue a new access token using the refresh token.", "instant_fix": False},
    {"error": "jose.ExpiredSignatureError: Signature has expired — access token expired, client should refresh", "error_type": "auth_jwt_expired", "fix": "Return 401 with 'detail': 'token_expired' so the client knows to use the refresh endpoint.", "instant_fix": False},
    {"error": "JWT verification failed: exp claim is in the past — token issued at 2024-01-01 expired at 2024-01-01T01:00:00", "error_type": "auth_jwt_expired", "fix": "Issue tokens with sufficient lifetime: exp = datetime.utcnow() + timedelta(hours=1). Use refresh tokens for longer sessions.", "instant_fix": False},

    {"error": "jwt.exceptions.InvalidSignatureError: Signature verification failed — wrong SECRET_KEY used to verify token", "error_type": "auth_jwt_invalid", "fix": "Ensure the same SECRET_KEY is used for both signing and verifying. Check env vars are consistent across services.", "instant_fix": False},
    {"error": "jose.JWTError: Signature verification failed — token tampered or signed with different secret", "error_type": "auth_jwt_invalid", "fix": "Return HTTP 401. Never expose the error details to the client. Log for investigation but respond with 'invalid token'.", "instant_fix": False},
    {"error": "jwt.exceptions.DecodeError: Not enough segments — malformed JWT token (not 3 dot-separated parts)", "error_type": "auth_jwt_invalid", "fix": "Validate token format before decoding: parts = token.split('.'); if len(parts) != 3: raise HTTPException(401)", "instant_fix": False},

    {"error": "passlib.exc.UnknownHashError: hash could not be identified — bcrypt hash stored incorrectly or truncated", "error_type": "auth_bcrypt", "fix": "Ensure the full hash is stored. bcrypt hashes are 60 characters. Check database column is large enough (Text or VARCHAR(255)).", "instant_fix": False},
    {"error": "ValueError: Invalid salt — bcrypt checkpw() received a non-bcrypt hash string", "error_type": "auth_bcrypt", "fix": "Ensure you stored the hash from bcrypt.hashpw() not the plaintext. Column should store the $2b$... hash string.", "instant_fix": False},

    {"error": "Access-Control-Allow-Origin cannot be '*' when credentials mode is 'include' — CORS credentials + wildcard forbidden", "error_type": "auth_cors_credentials", "fix": "Set a specific origin: allow_origins=['https://myapp.com'] and allow_credentials=True. Wildcard is forbidden with credentials.", "instant_fix": False},
    {"error": "CORS preflight blocked: credentials flag is set but the 'Access-Control-Allow-Credentials' header is missing or false", "error_type": "auth_cors_credentials", "fix": "Add allow_credentials=True to CORSMiddleware AND set a specific (non-wildcard) origin in allow_origins.", "instant_fix": False},
    {"error": "fetch() failed: CORS error with credentials — cookie not sent because withCredentials: true requires matching origin", "error_type": "auth_cors_credentials", "fix": "Set allow_origins=['http://localhost:3000'] (exact match). The backend must echo the specific origin, not '*'.", "instant_fix": False},

    {"error": "Set-Cookie blocked: cookie has SameSite=Strict and request is cross-site — authentication cookie not sent", "error_type": "auth_cookie", "fix": "Use SameSite=None; Secure for cross-origin cookies: response.set_cookie('token', value, samesite='none', secure=True)", "instant_fix": False},
    {"error": "Cookie not sent: Secure flag requires HTTPS — session cookie dropped on HTTP development server", "error_type": "auth_cookie", "fix": "For local development, use SameSite=Lax without Secure. For production, use SameSite=None with Secure over HTTPS.", "instant_fix": False},
    {"error": "Cookie not included in request — browser blocked third-party cookie due to SameSite policy", "error_type": "auth_cookie", "fix": "Use Authorization header with Bearer token instead of cookies for cross-origin API calls.", "instant_fix": False},

    {"error": "HTTP 401 Unauthorized on protected route — Bearer token not being sent by the client", "error_type": "auth_protected_route", "fix": "Add Authorization header in API client: headers: { Authorization: `Bearer ${token}` }. Check token is stored and retrieved correctly.", "instant_fix": False},
    {"error": "Protected FastAPI route returns 403 Forbidden — JWT verified but user lacks required role/permission", "error_type": "auth_protected_route", "fix": "Check role claim in JWT: if user.role not in required_roles: raise HTTPException(403, 'Insufficient permissions')", "instant_fix": False},
    {"error": "React protected route redirects authenticated user to login — auth state not persisted across page reload", "error_type": "auth_protected_route", "fix": "Persist auth token in localStorage and restore on app init: useEffect(() => { const token = localStorage.getItem('token'); if (token) dispatch(setAuth(token)); }, [])", "instant_fix": False},

    {"error": "HTTP 403: user role 'viewer' cannot access admin endpoint — RBAC check failed", "error_type": "auth_rbac", "fix": "Add role-checking dependency: async def require_admin(user = Depends(get_current_user)): if user.role != 'admin': raise HTTPException(403)", "instant_fix": False},
    {"error": "AttributeError: 'User' has no attribute 'permissions' — RBAC permissions not loaded from database", "error_type": "auth_rbac", "fix": "Eager-load permissions: select(User).options(selectinload(User.permissions)).where(User.id == user_id)", "instant_fix": False},
    {"error": "RBAC: JWT contains role='user' but endpoint requires role='admin' — role not updated after promotion", "error_type": "auth_rbac", "fix": "Force re-login after role change, or use short-lived access tokens + role lookup from DB on each request.", "instant_fix": False},

    {"error": "OAuth2 callback error: state mismatch — CSRF protection failed, state parameter doesn't match", "error_type": "auth_oauth2", "fix": "Store state in server session before redirect and validate on callback. Use secrets.token_urlsafe(32) for state.", "instant_fix": False},
    {"error": "OAuth2 error: invalid_grant — authorization code already used or expired", "error_type": "auth_oauth2", "fix": "Authorization codes are single-use. Don't retry on callback. Redirect user back to OAuth login if code expired.", "instant_fix": False},
    {"error": "OAuth2 token exchange failed: redirect_uri mismatch — callback URL not registered with provider", "error_type": "auth_oauth2", "fix": "Register the exact redirect URI in the OAuth provider's dashboard. Use http://localhost:3000/callback for dev.", "instant_fix": False},

    {"error": "CSRF token missing or invalid — POST request blocked by CSRF protection middleware", "error_type": "auth_csrf", "fix": "Include CSRF token in request headers: X-CSRFToken from cookie. Or use JWT Bearer tokens which are CSRF-safe.", "instant_fix": False},
    {"error": "HTTP 429 Too Many Requests — rate limit exceeded for login endpoint", "error_type": "auth_rate_limit", "fix": "Implement exponential backoff on the client. Add slowapi rate limiting: @limiter.limit('5/minute') on the login route.", "instant_fix": False},

    # ═══ CATEGORY 18: WebSocket / Real-time Errors (20 examples) ═══

    {"error": "WebSocket connection failed: Error during WebSocket handshake: Unexpected response code: 404", "error_type": "websocket_refused", "fix": "Check the WebSocket URL path matches the FastAPI @app.websocket('/ws') route. Include the full path.", "instant_fix": False},
    {"error": "WebSocket connection refused: net::ERR_CONNECTION_REFUSED — server not running or wrong port", "error_type": "websocket_refused", "fix": "Verify the backend is running. Use ws://localhost:8000/ws (not http://). Check firewall/proxy settings.", "instant_fix": False},
    {"error": "Failed to connect WebSocket: WebSocket is closed before the connection is established", "error_type": "websocket_refused", "fix": "Server closed the connection immediately — likely a route not found or authentication failed during handshake.", "instant_fix": False},

    {"error": "WebSocket upgrade blocked by CORS — 'Upgrade: websocket' request blocked by browser CORS policy", "error_type": "websocket_cors_error", "fix": "Add the WebSocket origin to CORS allow_origins. WebSocket handshakes respect CORS headers.", "instant_fix": False},
    {"error": "socket.io CORS error: Origin 'http://localhost:3000' not allowed — missing in socketio CORS config", "error_type": "websocket_cors_error", "fix": "Set CORS in socket.io: sio = socketio.AsyncServer(cors_allowed_origins='http://localhost:3000')", "instant_fix": False},

    {"error": "WebSocketDisconnect: code=1006 — abnormal closure, connection dropped without clean close frame", "error_type": "websocket_disconnect", "fix": "Add heartbeat/ping-pong to detect dead connections. Handle WebSocketDisconnect in a try/except block.", "instant_fix": False},
    {"error": "socket.io: client disconnect event not firing — disconnect handler never called on server", "error_type": "websocket_disconnect", "fix": "Register disconnect handler: @sio.on('disconnect') async def disconnect(sid): await cleanup(sid)", "instant_fix": False},
    {"error": "WebSocket: client disconnected but server still sending — RuntimeError: websocket.send on closed connection", "error_type": "websocket_disconnect", "fix": "Track connection state: use try/except WebSocketDisconnect around all send operations.", "instant_fix": False},

    {"error": "Memory leak: WebSocket listeners accumulate — socket.on('update', handler) called multiple times", "error_type": "websocket_memory_leak", "fix": "Remove listeners on cleanup: socket.off('update', handler) in useEffect cleanup function.", "instant_fix": False},
    {"error": "Node.js memory growing: 1000+ active WebSocket connections not closed — leak in connection management", "error_type": "websocket_memory_leak", "fix": "Track active connections in a Set and remove on disconnect. Implement server-side connection timeout.", "instant_fix": False},

    {"error": "socket.io: event 'message' not received by client — emitting to wrong room or namespace", "error_type": "websocket_event", "fix": "Check emit target: sio.emit('message', data, room=sid) for specific client or sio.emit('message', data) for all.", "instant_fix": False},
    {"error": "WebSocket message not received — event name mismatch between server emit and client on() handler", "error_type": "websocket_event", "fix": "Ensure event names match exactly: server emit('user_update') and client socket.on('user_update', handler).", "instant_fix": False},
    {"error": "socket.io: client not joining room — join_room() called before connection established", "error_type": "websocket_event", "fix": "Join room in the 'connect' event handler: @sio.on('connect') async def connect(sid, environ): sio.enter_room(sid, room_id)", "instant_fix": False},

    {"error": "WebSocket binary message not handled — received ArrayBuffer but event listener expects string", "error_type": "websocket_binary", "fix": "Set binaryType: socket.binaryType = 'arraybuffer' and handle as buffer: new Uint8Array(event.data)", "instant_fix": False},
    {"error": "FastAPI WebSocket: received bytes but await websocket.receive_text() expected string", "error_type": "websocket_binary", "fix": "Use receive_bytes() for binary data: data = await websocket.receive_bytes(). Check client is sending the right type.", "instant_fix": False},

    {"error": "WebSocket authentication failed: token not sent during handshake", "error_type": "websocket_auth", "fix": "Pass token as query param: new WebSocket('ws://host/ws?token=Bearer_xxx') and extract in FastAPI: token = websocket.query_params.get('token')", "instant_fix": False},
    {"error": "socket.io: cannot access request headers for auth in WebSocket handshake", "error_type": "websocket_auth", "fix": "Use socket.io auth option: io({ auth: { token: getToken() } }) and server: data = await sio.get_session(sid)['auth']", "instant_fix": False},

    # ═══ CATEGORY 19: File Upload / Storage Errors (20 examples) ═══

    {"error": "HTTP 413 Request Entity Too Large — file exceeds the maximum upload size", "error_type": "upload_too_large", "fix": "Set max body size in uvicorn: --limit-concurrency 100 or add a middleware that checks Content-Length before reading.", "instant_fix": False},
    {"error": "MultiPartException: chunk too large — file upload exceeds python-multipart default limit", "error_type": "upload_too_large", "fix": "Increase multipart limit: set max_fields and max_file_size in the Form() or configure python-multipart settings.", "instant_fix": False},
    {"error": "413 Payload Too Large from Nginx — client_max_body_size exceeded before reaching FastAPI", "error_type": "upload_too_large", "fix": "Add to nginx.conf: client_max_body_size 50M; inside the server {} block.", "instant_fix": False},
    {"error": "upload rejected: file size 52428800 bytes exceeds maximum allowed 10485760 bytes", "error_type": "upload_too_large", "fix": "Add size check: if file.size > MAX_SIZE: raise HTTPException(413, 'File too large'). Set MAX_SIZE = 10 * 1024 * 1024.", "instant_fix": False},

    {"error": "ValueError: content type 'image/png' not allowed — endpoint only accepts PDF files", "error_type": "upload_content_type", "fix": "Check content_type: if file.content_type not in ALLOWED_TYPES: raise HTTPException(400, 'Invalid file type')", "instant_fix": False},
    {"error": "upload error: expected multipart/form-data but received application/octet-stream", "error_type": "upload_content_type", "fix": "Client must send multipart/form-data. In fetch: use FormData() — don't set Content-Type manually, let browser set it.", "instant_fix": False},
    {"error": "UploadFile content_type is 'application/octet-stream' for all files — browser not detecting MIME type", "error_type": "upload_content_type", "fix": "Don't rely on content_type alone. Validate file extension and use python-magic to detect actual file format.", "instant_fix": False},

    {"error": "Security: path traversal attempt detected — filename '../../../etc/passwd' rejected", "error_type": "upload_path_traversal", "fix": "Sanitize filenames: use pathlib.Path(filename).name to strip directory components before saving.", "instant_fix": False},
    {"error": "unsafe filename: '../../config.py' — directory traversal in uploaded file name", "error_type": "upload_path_traversal", "fix": "Use secure_filename from werkzeug or: safe_name = Path(filename).name. Never use raw user-supplied filenames as paths.", "instant_fix": False},
    {"error": "file save path resolves outside upload directory — path traversal blocked", "error_type": "upload_path_traversal", "fix": "Validate the resolved path: full_path = (UPLOAD_DIR / safe_name).resolve(); assert full_path.parent == UPLOAD_DIR.resolve()", "instant_fix": False},

    {"error": "OSError: [Errno 13] Permission denied: '/uploads/photo.jpg' — upload directory not writable", "error_type": "upload_save_error", "fix": "Fix permissions: chmod 755 uploads/ or chown the directory to the app user. In Docker, ensure volume is writable.", "instant_fix": False},
    {"error": "FileNotFoundError: upload directory '/app/uploads' does not exist", "error_type": "upload_save_error", "fix": "Create upload dir on startup: Path('uploads').mkdir(parents=True, exist_ok=True)", "instant_fix": False},
    {"error": "aiofiles write error: no space left on device — disk full during file upload", "error_type": "upload_save_error", "fix": "Add disk space monitoring. Clean up old uploads with a scheduled job. Consider using S3/cloud storage instead.", "instant_fix": False},

    {"error": "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/uploads' — missing upload directory", "error_type": "upload_missing_dir", "fix": "Create upload directory on startup: UPLOAD_DIR = Path('/tmp/uploads'); UPLOAD_DIR.mkdir(parents=True, exist_ok=True)", "instant_fix": True, "fix_action": "create_upload_dir", "file_to_fix": "backend/main.py"},
    {"error": "PermissionError: upload folder '/var/www/uploads' does not exist or is not writable", "error_type": "upload_missing_dir", "fix": "Create and set permissions: mkdir -p /var/www/uploads && chown -R appuser:appuser /var/www/uploads", "instant_fix": False},

    {"error": "upload rejected: file extension '.exe' is not in allowed extensions [jpg, png, pdf, docx]", "error_type": "upload_validation", "fix": "Check extension: ext = Path(file.filename).suffix.lower(); if ext not in ALLOWED_EXTS: raise HTTPException(400, 'File type not allowed')", "instant_fix": False},
    {"error": "upload validation failed: file appears to be a ZIP archive disguised as JPEG (magic bytes mismatch)", "error_type": "upload_validation", "fix": "Use python-magic to verify actual file type: import magic; mime = magic.from_buffer(await file.read(2048), mime=True)", "instant_fix": False},
    {"error": "boto3 upload error: NoCredentialsError — AWS credentials not configured for S3 upload", "error_type": "upload_validation", "fix": "Set AWS credentials: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars, or use IAM role in production.", "instant_fix": False},
    {"error": "S3 upload failed: BucketNotFound — bucket name in env var does not match existing bucket", "error_type": "upload_validation", "fix": "Verify bucket name in AWS console matches S3_BUCKET_NAME env var. Bucket names are globally unique.", "instant_fix": False},

    # ═══ CATEGORY 20: Performance / Optimization Issues (20 examples) ═══

    {"error": "N+1 query: fetching 100 posts triggers 100 additional SELECT queries for post.author", "error_type": "perf_n1_query", "fix": "Use selectinload: select(Post).options(selectinload(Post.author)). One query for posts, one for all authors.", "instant_fix": False, "llm_prompt": "Fix N+1: add selectinload() or joinedload() to eager-load the relationship."},
    {"error": "SQLAlchemy warning: 100 SQL queries executed for a single endpoint — likely N+1 problem in serialization", "error_type": "perf_n1_query", "fix": "Profile with sqlalchemy.event.listen('after_cursor_execute'). Use joinedload() for relationships accessed in response serialization.", "instant_fix": False},
    {"error": "Slow API: /users endpoint takes 5s — each user triggers a separate DB query for profile data", "error_type": "perf_n1_query", "fix": "Use a JOIN query: select(User, Profile).join(Profile, User.id == Profile.user_id) to fetch both in one query.", "instant_fix": False},
    {"error": "database slow query log: 'SELECT * FROM products WHERE category_id = ?' executed 500 times per request", "error_type": "perf_n1_query", "fix": "Cache categories in memory or prefetch: result = await db.execute(select(Category).where(Category.id.in_(ids)))", "instant_fix": False},

    {"error": "Slow query: sequential scan on 'orders' table (10M rows) — missing index on user_id column", "error_type": "perf_missing_index", "fix": "Add index: CREATE INDEX idx_orders_user_id ON orders(user_id); or in SQLAlchemy: mapped_column(ForeignKey('users.id'), index=True)", "instant_fix": False},
    {"error": "PostgreSQL EXPLAIN shows Seq Scan on 'messages' — created_at column used in WHERE but not indexed", "error_type": "perf_missing_index", "fix": "Add index: CREATE INDEX idx_messages_created_at ON messages(created_at DESC); Add index=True to SQLAlchemy column.", "instant_fix": False},
    {"error": "slow API response: ORDER BY created_at takes 3s on 1M row table — no index on sort column", "error_type": "perf_missing_index", "fix": "Add a covering index: CREATE INDEX idx_posts_created ON posts(created_at DESC) INCLUDE (id, title, user_id)", "instant_fix": False},

    {"error": "webpack warning: bundle size exceeds 244 KiB for chunk 'vendors' — consider code splitting", "error_type": "perf_bundle_size", "fix": "Use React.lazy() for route-level code splitting: const Dashboard = lazy(() => import('./pages/Dashboard'))", "instant_fix": False},
    {"error": "Large bundle: lodash adds 72KB — only 3 functions used from the entire library", "error_type": "perf_bundle_size", "fix": "Import only what you need: import debounce from 'lodash/debounce' instead of import _ from 'lodash'", "instant_fix": False},
    {"error": "bundle-analyzer: moment.js locale data adds 256KB — locales not tree-shaken", "error_type": "perf_bundle_size", "fix": "Replace moment.js with date-fns (tree-shakeable): import { format } from 'date-fns' only imports what you use.", "instant_fix": False},

    {"error": "Memory leak: component subscribes to WebSocket in useEffect but never unsubscribes — event listeners accumulate", "error_type": "perf_memory_leak_react", "fix": "Return cleanup from useEffect: return () => { socket.off('message', handler); socket.disconnect(); }", "instant_fix": False},
    {"error": "Memory leak: setInterval in useEffect not cleared — timer fires on unmounted component causing state update warnings", "error_type": "perf_memory_leak_react", "fix": "Clear interval in cleanup: const id = setInterval(tick, 1000); return () => clearInterval(id);", "instant_fix": False},
    {"error": "Memory leak: event listener on window not removed — document.addEventListener('resize', handler) never cleaned up", "error_type": "perf_memory_leak_react", "fix": "Remove listener in useEffect cleanup: return () => window.removeEventListener('resize', handler);", "instant_fix": False},

    {"error": "React DevTools: 50+ re-renders per second in parent component — context value object recreated every render", "error_type": "perf_rerender", "fix": "Memoize context value: const value = useMemo(() => ({ user, login, logout }), [user]) to prevent unnecessary re-renders.", "instant_fix": False},
    {"error": "Excessive re-renders: 100 child components re-render when parent state changes — missing React.memo()", "error_type": "perf_rerender", "fix": "Wrap child component: export default React.memo(ChildComponent). Also memoize callback props with useCallback.", "instant_fix": False},
    {"error": "React Profiler: 'ProductList' renders 30 times in 1 second — useEffect with no deps triggers on every parent render", "error_type": "perf_rerender", "fix": "Add dependency array to useEffect. If no deps: useEffect(() => { init(); }, []). If deps: list only changed values.", "instant_fix": False},

    {"error": "API endpoint /products returns all 50,000 records — no pagination causing 30s response time", "error_type": "perf_missing_index", "fix": "Add pagination: select(Product).limit(page_size).offset((page-1)*page_size). Add ?page=1&limit=20 query params.", "instant_fix": False},
    {"error": "no HTTP caching: browser re-fetches same static data on every page load — missing Cache-Control headers", "error_type": "perf_missing_index", "fix": "Add cache headers: response.headers['Cache-Control'] = 'public, max-age=3600' for static or infrequently-changing data.", "instant_fix": False},

    # ═══ REAL SESSION ERRORS (Collected from actual NexusFlow development) ═══

    {"error": "404 Not Found - GET https://registry.npmjs.org/@react-bits%2freact - Not found", "error_type": "fake_package", "fix": "Remove @react-bits/react from package.json. It is a non-existent package. Use lucide-react for icons instead.", "package_to_remove": "@react-bits/react", "replacement_package": "lucide-react", "replacement_version": "^0.263.0", "instant_fix": True, "fix_action": "remove_fake_package", "notes": "This package was hallucinated by the LLM. Never use @react-bits/* packages."},
    {"error": "TS2561: Object literal may only specify known properties, but 'insetX' does not exist in type 'Properties<string | number, string & {}>'. Did you mean to write 'inset'?", "error_type": "typescript_css", "fix": "Replace 'insetX: 0' with 'left: 0, right: 0' in React inline style objects. insetX is a CSS logical property not supported in React style objects.", "pattern": "insetX: 0", "replacement": "left: 0, right: 0", "instant_fix": True, "fix_action": "replace_pattern", "notes": "React style objects use camelCase but don't support all CSS logical properties."},
    {"error": "TS7016: Could not find a declaration file for module 'three'. 'node_modules/three/build/three.js' implicitly has an 'any' type.", "error_type": "typescript_missing_types", "fix": "Add // @ts-ignore before the three import line. OR install @types/three@0.152.0 specifically (not latest - incompatible with TypeScript 4.x).", "instant_fix": True, "fix_action": "add_ts_ignore", "package_to_add": "@types/three", "version": "0.152.0", "notes": "Latest @types/three uses const type parameters incompatible with TypeScript 4.x used by react-scripts."},
    {"error": "TS1139: Type parameter declaration expected. at @types/three/src/nodes/accessors/ReferenceNode.d.ts", "error_type": "typescript_version", "fix": "Add skipLibCheck: true to tsconfig.json compilerOptions. This skips type checking of declaration files in node_modules.", "instant_fix": True, "fix_action": "add_skip_lib_check", "notes": "Latest @types/three uses const type parameters which require TypeScript 5.x but react-scripts uses TypeScript 4.x."},
    {"error": "FileNotFoundError: [WinError 2] The system cannot find the file specified — 'py'", "error_type": "wrong_python_command", "fix": "Replace 'py' or 'py -3.11' with sys.executable in all subprocess calls. On Linux/Mac the command is 'python3' not 'py'.", "pattern": "['py', '-3.11'", "replacement": "[sys.executable", "instant_fix": True, "fix_action": "replace_py_with_sys_executable", "notes": "The 'py' command only exists on Windows. Use sys.executable for cross-platform compatibility."},
    {"error": "Access to fetch at 'https://nexusflow-k1hw.onrender.com/analyze' from origin 'https://nexus-flow-ai-dashboard.vercel.app' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present", "error_type": "cors_error", "fix": "Set allow_origins=['*'] and allow_credentials=False in CORSMiddleware. When using wildcard origins, credentials must be False.", "instant_fix": False, "llm_prompt": "Fix CORS in FastAPI: set allow_origins=['*'] with allow_credentials=False in CORSMiddleware configuration.", "notes": "Cannot use allow_credentials=True with allow_origins=['*']. Choose one or the other."},
    {"error": "Import error, can't find file: ./index.css in StackBlitz preview", "error_type": "stackblitz_missing_css", "fix": "Create src/index.css file. In StackBlitz, index.tsx imports './index.css' by default. Also fix any imports pointing to './styles/globals.css' - change them to './index.css'.", "instant_fix": True, "fix_action": "create_index_css", "file_to_fix": "src/index.css", "fix_content": "body {\n  margin: 0;\n  padding: 0;\n  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n}\n\n*, *::before, *::after {\n  box-sizing: border-box;\n}\n\na { text-decoration: none; color: inherit; }\nbutton { cursor: pointer; }\nimg { max-width: 100%; }", "notes": "StackBlitz uses create-react-app template which expects index.css at src/index.css"},
    {"error": "Can't find packages: react-router-dom axios in StackBlitz", "error_type": "stackblitz_missing_package", "fix": "Scan all TSX/TS files for import statements. For each imported package not in package.json, add it with the correct version. react-router-dom: ^6.8.0, axios: ^1.3.0", "instant_fix": True, "fix_action": "scan_and_add_packages", "notes": "StackBlitz installs packages from package.json. All imports must be listed there."},
    {"error": "TS2304: Cannot find name 'ShieldCheck'. lucide-react import missing", "error_type": "typescript_missing_import", "fix": "Add the missing icon name to the lucide-react import statement at the top of the file.", "instant_fix": False, "llm_prompt": "Add the missing icon 'ShieldCheck' to the lucide-react import: import { ..., ShieldCheck } from 'lucide-react'", "notes": "Always check that all used lucide-react icons are imported."},
    {"error": "NotImplementedError: asyncio subprocess not supported on Windows with default event loop", "error_type": "windows_asyncio", "fix": "Use subprocess.Popen instead of asyncio.create_subprocess_exec on Windows. Run blocking subprocess.run in loop.run_in_executor(None, lambda: subprocess.run(...)).", "instant_fix": False, "llm_prompt": "Replace asyncio.create_subprocess_exec with subprocess.Popen for Windows compatibility. Use asyncio.get_event_loop().run_in_executor for blocking calls.", "notes": "Windows requires WindowsProactorEventLoopPolicy for asyncio subprocess but uvicorn overrides this."},
    {"error": "npm error ERESOLVE could not resolve — typescript@5.x found but react-scripts needs typescript@^3.2.1 || ^4", "error_type": "npm_peer_dependency", "fix": "Run npm install with --legacy-peer-deps flag. Add 'legacy-peer-deps=true' to .npmrc file.", "instant_fix": True, "fix_action": "add_legacy_peer_deps", "file_to_fix": ".npmrc", "fix_content": "legacy-peer-deps=true", "notes": "react-scripts 5.0.1 requires TypeScript 4.x but newer projects use TypeScript 5.x."},
    {"error": "Cannot find module 'ajv/dist/compile/codegen' — webpack dev server startup fails", "error_type": "ajv_version_conflict", "fix": "Run: npm install ajv@^8 --legacy-peer-deps to fix the AJV version conflict.", "instant_fix": False, "llm_prompt": "Install compatible AJV version: npm install ajv@^8 --legacy-peer-deps", "notes": "react-scripts has a peer dependency conflict with AJV versions."},
    {"error": "error: metadata-generation-failed — watchfiles requires Rust toolchain on Python 3.14", "error_type": "render_build_failure", "fix": "Remove watchfiles from requirements.txt. Add .python-version file with content '3.11.9' to force Python 3.11 on Render.", "instant_fix": True, "fix_action": "remove_package_from_requirements", "package_to_remove": "watchfiles", "file_to_fix": ".python-version", "fix_content": "3.11.9", "notes": "watchfiles requires Rust to build on Python 3.14. Not needed in production (only for --reload)."},
    {"error": "Objects are not valid as a React child (found: object with keys {style}) — rendering API response object directly", "error_type": "react_invalid_child", "fix": "Convert object values to strings before rendering. Use a safeStr helper: const safeStr = (val) => typeof val === 'object' ? Object.values(val).filter(Boolean).join(', ') : String(val || '')", "instant_fix": False, "llm_prompt": "The component is trying to render an object directly as a React child. Convert the object to a string using JSON.stringify() or access specific string properties.", "notes": "API responses often contain nested objects. Always extract string values before rendering."},
    {"error": "html, body, #root { height: 100% } causes landing page scroll to be trapped", "error_type": "css_scroll_trap", "fix": "Change #root from height: 100% to min-height: 100%. Keep html and body at height: 100% but #root must use min-height to allow content taller than viewport.", "pattern": "#root {\n  height: 100%", "replacement": "#root {\n  min-height: 100%", "instant_fix": True, "fix_action": "replace_pattern", "notes": "height: 100% on #root creates a fixed-height scroll container that traps scroll events."},
    {"error": "502 Bad Gateway from Render after deployment", "error_type": "render_502", "fix": "Check Render logs for startup errors. Common causes: missing environment variables, wrong start command, import errors on startup, database connection failure.", "instant_fix": False, "llm_prompt": "The Render deployment returns 502. Check: 1) All env vars are set 2) Start command is correct 3) No import errors 4) Database is accessible", "notes": "502 means the app started but crashed immediately. Always check Render logs."},
    {"error": "GET /list-projects takes too long — scanning node_modules", "error_type": "performance_node_modules", "fix": "Exclude node_modules, __pycache__, and .git from file scanning: if 'node_modules' not in f.parts and '__pycache__' not in f.parts", "pattern": "for f in entry.rglob('*') if f.is_file()", "replacement": "for f in entry.rglob('*') if f.is_file() and 'node_modules' not in f.parts and '__pycache__' not in f.parts", "instant_fix": True, "fix_action": "replace_pattern", "notes": "node_modules contains thousands of files. Always exclude from recursive scans."},
    {"error": "Failed to execute 'insertBefore' on 'Node' — React DOM conflict with StackBlitz SDK", "error_type": "stackblitz_dom_conflict", "fix": "Use a useRef flag to ensure sdk.embedProject() is only called once. Never let React re-render the container div that StackBlitz injects into.", "instant_fix": False, "llm_prompt": "Fix React DOM conflict with StackBlitz: use hasEmbedded = useRef(false) to call sdk.embedProject only once. Keep the embed container div stable.", "notes": "StackBlitz SDK manipulates DOM directly. React's virtual DOM conflicts with this."},

    # ═══ SAAS PATTERN EXAMPLES (quality_score=1.0) ═══

    {
        "error": "Build JWT auth in FastAPI",
        "error_type": "saas_pattern",
        "fix": '''from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
from database import get_db
from models import User
from schemas import UserCreate, UserResponse, Token
import os

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token=Depends(oauth2_scheme), db=Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None: raise credentials_exception
    return user

@router.post("/auth/register")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none(): raise HTTPException(status_code=400, detail="Email already registered")
    db_user = User(email=user.email, hashed_password=hash_password(user.password), name=user.name)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@router.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me")
async def get_me(current_user=Depends(get_current_user)):
    return current_user''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build React login form with TailwindCSS",
        "error_type": "saas_pattern",
        "fix": '''import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/auth/login`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Invalid credentials");
      const data = await res.json();
      localStorage.setItem("token", data.access_token);
      navigate("/dashboard");
    } catch (err) { setError("Invalid email or password"); }
    finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back</h1>
        <p className="text-gray-500 mb-8">Sign in to your account</p>
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors" />
          </div>
          <button type="submit" disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 rounded-lg transition-colors shadow-md">
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>
        <p className="mt-6 text-center text-gray-500">No account? <Link to="/register" className="text-blue-600 font-semibold hover:underline">Sign up</Link></p>
      </div>
    </div>
  );
}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build FastAPI CRUD with pagination",
        "error_type": "saas_pattern",
        "fix": '''from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from database import get_db
from models import Item
from schemas import ItemCreate, ItemResponse

router = APIRouter()

@router.get("/items")
async def get_items(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100), search: str = Query(""), db: AsyncSession = Depends(get_db)):
    offset = (page - 1) * limit
    query = select(Item)
    if search:
        query = query.where(or_(Item.title.ilike(f"%{search}%"), Item.description.ilike(f"%{search}%")))
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    items = (await db.execute(query.offset(offset).limit(limit))).scalars().all()
    return {"items": items, "total": total, "page": page, "pages": (total + limit - 1) // limit}

@router.post("/items")
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    db_item = Item(**item.model_dump())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@router.put("/items/{item_id}")
async def update_item(item_id: int, item: ItemCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Item).where(Item.id == item_id))
    db_item = result.scalar_one_or_none()
    if not db_item: raise HTTPException(status_code=404, detail="Item not found")
    for key, value in item.model_dump().items():
        setattr(db_item, key, value)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@router.delete("/items/{item_id}")
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Item).where(Item.id == item_id))
    db_item = result.scalar_one_or_none()
    if not db_item: raise HTTPException(status_code=404, detail="Item not found")
    await db.delete(db_item)
    await db.commit()
    return {"message": "Deleted successfully"}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build React data table with search",
        "error_type": "saas_pattern",
        "fix": '''import React, { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

export default function DataTable({ endpoint, columns, title }) {
  const [data, setData] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const limit = 10;

  useEffect(() => { fetchData(); }, [page, search]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page, limit, search });
      const res = await fetch(`${API_URL}/api/${endpoint}?${params}`, { headers: { Authorization: `Bearer ${localStorage.getItem("token")}` } });
      const json = await res.json();
      setData(json.items);
      setTotal(json.total);
    } catch (err) { console.error(err); }
    finally { setLoading(false); }
  };

  const pages = Math.ceil(total / limit);
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <input value={search} onChange={e => { setSearch(e.target.value); setPage(1); }}
          placeholder="Search..." className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500" />
      </div>
      {loading ? <div className="flex justify-center py-12"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" /></div> : (
        <table className="w-full">
          <thead className="bg-gray-50"><tr>{columns.map(col => <th key={col.key} className="text-left px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">{col.label}</th>)}</tr></thead>
          <tbody className="divide-y divide-gray-100">
            {data.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50 transition-colors">
                {columns.map(col => <td key={col.key} className="px-6 py-4 text-sm text-gray-700">{col.render ? col.render(row[col.key], row) : String(row[col.key] ?? "")}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
        <span className="text-sm text-gray-500">{total} results</span>
        <div className="flex gap-2">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="px-3 py-1 rounded border text-sm disabled:opacity-50">Previous</button>
          <span className="px-3 py-1 text-sm text-gray-700">Page {page} of {pages || 1}</span>
          <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page >= pages} className="px-3 py-1 rounded border text-sm disabled:opacity-50">Next</button>
        </div>
      </div>
    </div>
  );
}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build SaaS dashboard layout",
        "error_type": "saas_pattern",
        "fix": '''import React, { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const navItems = [
  { label: "Dashboard", path: "/dashboard", icon: "📊" },
  { label: "Projects", path: "/projects", icon: "📁" },
  { label: "Settings", path: "/settings", icon: "⚙️" },
];

export default function DashboardLayout({ children }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(true);
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const logout = () => { localStorage.clear(); navigate("/login"); };

  return (
    <div className="flex h-screen bg-gray-50">
      <aside className={`${open ? "w-64" : "w-16"} bg-white border-r border-gray-200 flex flex-col transition-all duration-300`}>
        <div className="p-4 border-b flex items-center justify-between">
          {open && <span className="font-bold text-xl text-blue-600">App</span>}
          <button onClick={() => setOpen(!open)} className="p-2 rounded-lg hover:bg-gray-100">☰</button>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map(item => (
            <Link key={item.path} to={item.path}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${location.pathname === item.path ? "bg-blue-50 text-blue-600 font-semibold" : "text-gray-600 hover:bg-gray-100"}`}>
              <span>{item.icon}</span>{open && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>
        <div className="p-4 border-t">
          {open && <p className="text-sm text-gray-500 mb-2 truncate">{user.email}</p>}
          <button onClick={logout} className="text-red-500 hover:text-red-700 text-sm font-medium flex items-center gap-2">
            <span>🚪</span>{open && "Logout"}
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-800">{navItems.find(i => i.path === location.pathname)?.label || "Dashboard"}</h1>
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
            {user.name?.[0]?.toUpperCase() || "U"}
          </div>
        </header>
        <div className="p-6">{children}</div>
      </main>
    </div>
  );
}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build Stripe payment integration",
        "error_type": "saas_pattern",
        "fix": '''import stripe
import os
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
router = APIRouter()

@router.post("/payments/create-checkout")
async def create_checkout_session(price_id: str, user_id: int, db: AsyncSession = Depends(get_db)):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="http://localhost:5173/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:5173/pricing",
            metadata={"user_id": str(user_id)},
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/payments/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    if event["type"] == "checkout.session.completed":
        user_id = event["data"]["object"]["metadata"]["user_id"]
        # Update user subscription in DB
    return {"status": "ok"}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build WebSocket real-time chat",
        "error_type": "saas_pattern",
        "fix": '''from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room: str):
        await websocket.accept()
        self.active_connections.setdefault(room, []).append(websocket)

    def disconnect(self, websocket: WebSocket, room: str):
        if room in self.active_connections:
            self.active_connections[room].remove(websocket)

    async def broadcast(self, message: dict, room: str):
        for conn in self.active_connections.get(room, []):
            await conn.send_text(json.dumps(message))

manager = ConnectionManager()

@router.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    await manager.connect(websocket, room)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"message": json.loads(data), "room": room}, room)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
        await manager.broadcast({"message": "A user left", "room": room}, room)''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build file upload FastAPI",
        "error_type": "saas_pattern",
        "fix": '''import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "application/pdf"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {file.content_type}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
    (UPLOAD_DIR / filename).write_bytes(contents)
    return {"filename": filename, "url": f"/uploads/{filename}", "size": len(contents)}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build React stats cards",
        "error_type": "saas_pattern",
        "fix": '''import React from "react";

export default function StatsCards({ stats }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {stats.map((stat, i) => (
        <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <span className={`text-2xl p-2 rounded-lg ${stat.color}`}>{stat.icon}</span>
            {stat.trend && (
              <span className={`text-sm font-medium ${stat.trendUp ? "text-green-600" : "text-red-600"}`}>
                {stat.trendUp ? "↑" : "↓"} {stat.trend}
              </span>
            )}
          </div>
          <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
          <p className="text-sm text-gray-500 mt-1">{stat.label}</p>
        </div>
      ))}
    </div>
  );
}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },

    {
        "error": "Build React auth context",
        "error_type": "saas_pattern",
        "fix": '''import React, { createContext, useContext, useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";
const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (token) { fetchUser(); }
    else { setIsLoading(false); }
  }, [token]);

  const fetchUser = async () => {
    try {
      const res = await fetch(`${API_URL}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error("Unauthorized");
      setUser(await res.json());
    } catch { logout(); }
    finally { setIsLoading(false); }
  };

  const login = async (email, password) => {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);
    const res = await fetch(`${API_URL}/api/auth/login`, { method: "POST", body: formData });
    if (!res.ok) throw new Error("Invalid credentials");
    const { access_token } = await res.json();
    localStorage.setItem("token", access_token);
    setToken(access_token);
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
  };

  return <AuthContext.Provider value={{ user, token, login, logout, isLoading }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}''',
        "instant_fix": False,
        "quality_score": 1.0,
    },
]


async def seed_training_data(db) -> None:
    """Seed the database with curated training examples.

    Safe to call on every startup: deduplicates by input_prompt so re-running
    only inserts examples not already present.
    """
    from src.database.models import TrainingExample
    from sqlalchemy import select

    result = await db.execute(select(TrainingExample))
    existing = result.scalars().all()

    # Deduplicate by input_prompt — never re-insert what's already there
    existing_errors = {ex.input_prompt for ex in existing}
    new_items = [item for item in TRAINING_DATA if item.get("error", "") not in existing_errors]

    for item in new_items:
        example = TrainingExample(
            input_prompt=item.get("error", ""),
            error_context=item.get("error_type", "unknown"),
            correct_output=item.get("fix", ""),
            example_type=item.get("error_type", "unknown"),
            quality_score=1.0 if item.get("instant_fix") else 0.8,
        )
        db.add(example)

    if new_items:
        await db.commit()
