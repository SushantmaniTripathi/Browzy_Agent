# Browzy

Browzy is a Chromium extension with a FastAPI backend and optional local Ollama
integration.

## Browser task architecture

Browser tasks use a shared task runtime and a single Playwright browser session.
The runtime serializes browser access, records task state in the local database,
and reports queued/running state to the side panel over WebSocket.

Known commands use deterministic LangGraph routes for low latency. Tasks that
need page interpretation use the local Ollama model to decide which action to
take; Playwright performs the action. Both paths share the same browser session
and task lock. The model receives compact page observations rather than a full
DOM dump. After a page change or failed action, Browzy stops the current batch,
observes the current page again, and continues within bounded action and retry
limits. A task is reported as complete only when its flow can verify the result;
otherwise Browzy reports that it could not verify completion.

The browser agent is implemented with Browzy's Ollama tool-calling loop and
Playwright. It does not require the separate `browser-use` Python package.

## Windows setup

Open PowerShell in this directory and run:

```powershell
.\scripts\setup_venv.ps1
.\scripts\run_backend.ps1
```

The setup script creates or repairs `venv`, installs
`browzy/backend/requirements.txt`, and verifies the backend imports.

The backend listens on <http://localhost:8764>. Its health endpoint is
<http://localhost:8764/health>.

## Configuration

Copy `browzy/.env.example` to `browzy/.env` and add a speech-to-text API key
when voice input is needed. The `.env` file is local-only and is ignored by
Git.

For local chat, install Ollama and pull a supported model, for example:

```powershell
ollama pull llama3.2:3b
```

## Project layout

```text
browzy/
  backend/       FastAPI composition, task runtime, browser session,
                 observation, semantic interactions, and Python services
  extension/     Chromium extension assets
  models/        Ollama Modelfile
runtime/
  chrome/        Bundled Chromium runtime files
scripts/         Windows setup and run helpers
venv/            Local Python virtual environment (ignored)
```
