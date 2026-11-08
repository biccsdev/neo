# NEO

NEO is a conversational computer-operations plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It provides one mission-control interface for coding specialists, Blender work, browser navigation, files, previews and reusable Hermes capabilities.

The core architecture is portable. **macOS is the initially validated platform.** Windows and Linux are intended but have not been validated.

## What it does

- Runs durable tasks in parallel with independent status, logs, cancellation and retry.
- Resumes the exact recorded Codex or Claude session for coding follow-ups.
- Uses detached Hermes workers for reusable capabilities without recursive NEO calls.
- Provides a `/neo` Desktop page, task-scoped previews and an approval inbox.
- Verifies expected artifacts before completion and presents successful results automatically.
- Recovers honestly after restart; a lost process is never reported as completed work.
- Requires exact-name, exact-commit consent before installing another capability.

NEO is an orchestrator, not another coding model, realtime-voice stack, browser engine or Blender MCP server.

## Install

Requires Hermes Agent 0.22.6 or later on macOS.

```bash
hermes plugins install biccsdev/neo --enable
hermes plugins doctor neo --ci
```

For a local checkout:

```bash
hermes plugins install "file://$(pwd)" --enable
```

See [docs/install.md](docs/install.md) for isolated testing, executor setup and data locations.

## Use

Ask Hermes naturally:

- “Fix the failing parser tests in this repository.”
- “Create a small Blender scene and save the render.”
- “Open the documentation page in a visible browser.”
- “How is it going?”
- “Refine the last task without duplicating it.”
- “Cancel that task.”

The plugin exposes seven `neo_*` tools and a read-only task-context hook. The operator and worker skills are bundled in `skills/`.

## Current evidence

- 672 automated system tests pass; 12 API tests skip under a system Python without FastAPI.
- 115 API, presentation and security tests pass in the development virtual environment.
- Hermes Plugin Doctor loads seven tools and one hook.
- Hermes package validation and the security scan pass.
- Installation and enablement have been exercised in an isolated Hermes home.

Live Codex, Claude, Blender, Desktop and microphone acceptance remain environment-dependent gates. Their status and an executable demo script are recorded in [docs/demo.md](docs/demo.md). NEO reports unavailable capabilities honestly instead of claiming success.

## Security and privacy

Workers use argv-only spawning, scoped tokens, conversation-scoped state and bounded/redacted public task projections. Browser approval does not publish, send, upload, purchase or log in. Blender requires the official pinned Hermes plugin and fails closed when its installed revision cannot be verified.

Read [docs/security.md](docs/security.md) and [docs/privacy.md](docs/privacy.md) before running real workloads.

## Repository layout

The repository root is the installable Hermes plugin package. It contains the Python agent plugin, FastAPI backend, Desktop page, bundled skills, resources and tests.

## Licence

MIT. See [LICENSE](LICENSE).
