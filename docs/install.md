# Install and configure NEO

## Requirements

- macOS; this is the initially validated platform.
- Hermes Agent 0.22.6 or later.
- For coding tasks: the Codex or Claude CLI installed and authenticated, with a model the account can actually run.
- For browser tasks: an enabled browser plugin or the enabled `browser` skill.
- For Blender tasks: the official Hermes `blender` plugin at the reviewed commit, plus a host capability context that exposes its installed revision.

NEO's own Python runtime uses the standard library. FastAPI and httpx are supplied by Hermes for the Dashboard API; they are needed locally only when running the API tests outside Hermes.

## Install from GitHub

```bash
hermes plugins install biccsdev/neo --enable
hermes plugins doctor neo --ci
```

Restart the Hermes gateway or Desktop session when Hermes requests it. Confirm that Doctor reports seven tools and one hook.

## Install a local checkout

From the repository root:

```bash
hermes plugins install "file://$(pwd)" --enable
```

For an immutable public revision:

```bash
hermes plugins install https://github.com/biccsdev/neo.git \
  --ref EXACT_40_CHARACTER_COMMIT --enable
```

Do not use a moving branch or `latest` when installing a reviewed release.

## Isolated installation test

This creates a throwaway Hermes home and leaves your active profile unchanged:

```bash
NEO_SRC="$(pwd)"
TEST_HOME="$(mktemp -d "${TMPDIR:-/tmp}/neo-hermes.XXXXXX")"
env HERMES_HOME="$TEST_HOME" hermes -p default plugins install \
  "file://$NEO_SRC" --enable --yes-deps
env HERMES_HOME="$TEST_HOME" hermes -p default plugins enable neo
env HERMES_HOME="$TEST_HOME" hermes -p default plugins doctor neo --ci
```

Remove the temporary home after the test. Do not point this command at your normal Hermes home.

## Executors

Set the preferred coding executor before starting Hermes:

```bash
export NEO_CODING_EXECUTOR=codex  # or claude
```

Verify the underlying CLI independently before asking NEO to use it. NEO preserves native session identifiers and resumes those exact sessions; it does not silently replace a failed specialist with another executor.

The detached-worker concurrency limit defaults to four:

```bash
export NEO_MAX_PARALLEL=4
```

Only one Blender task can own port 9876 at a time. Additional Blender work queues rather than sharing the unauthenticated port.

### Reduced worker environment

By default, workers inherit Hermes's environment so existing provider authentication works. To restrict inherited variables:

```bash
export NEO_WORKER_ENV_ALLOWLIST="PATH,HOME,OPENAI_API_KEY"
```

Use only the provider variables your chosen executor requires. NEO adds its own task and recursion-guard variables. An empty or malformed allowlist is rejected.

## Capability consent

Catalog discovery is read-only. A NEO capability installation requires a recorded decision for the exact package name and immutable commit. A disclosure or installation request is not execution consent by itself.

Blender readiness requires the official plugin, enabled state and a verified installed revision. NEO does not probe readiness by opening Blender or connecting unauthenticated to port 9876.

Browser navigation can run with a recognised enabled capability. Publication, sending, uploading, purchasing and login remain outside NEO's execution boundary.

## Data and upgrades

NEO stores its SQLite registry, workspaces, previews and logs in Hermes's per-plugin data directory for the active profile. Data survives updates and restarts. Uninstalling the package does not automatically erase it.

Back up or remove that directory deliberately when resetting NEO. Do not copy the database between profiles while tasks are active.

## Verification

From a checkout:

```bash
python3 -m unittest discover -s tests
hermes plugins doctor . --ci
hermes plugins validate . --install-deps
```

For API tests outside Hermes, create a local virtual environment and install `requirements-dev.txt`, then run the API, presentation and security modules with that interpreter. See [docs/demo.md](demo.md) for the acceptance matrix and current live-only gates.
