# Security model

## Worker authentication

Each submission creates a random, task-scoped worker token. Only its SHA-256 digest is stored. Bridge updates require the raw token and an expected task version, and both the state transition and its event are committed in one transaction. Invalid, expired or stale tokens do not mutate state.

The raw token exists in the owning subprocess environment for its lifetime. It is not a general administrator credential and cannot be used to read another task, install a capability or submit arbitrary work.

## Process execution

NEO uses argv-based process spawning with `shell=False`. Prompts are written to files and passed by filename; they are not interpolated into shell commands. Detached workers belong to their own process group, and cancellation targets that group rather than an unrelated process.

The supervisor owns processes, not lifecycle truth. SQLite remains authoritative. Restarts reconcile running tasks against captured process metadata and executor output; a dead worker is not silently reported as completed.

By default the child inherits the Hermes environment because CLI providers commonly keep authentication there. Set `NEO_WORKER_ENV_ALLOWLIST` to a comma-separated list of environment-variable names to restrict what workers receive. NEO always adds its internal task ID, raw worker token, data directory, task directory, timeout and recursion guard. Keep only provider variables required by your executor in the allowlist.

## Capability installation

Discovery is read-only. Installation requires a consent record for both the exact package name and the immutable reviewed commit. Moving branches, tags and `latest` are not accepted as consent. NEO does not execute `curl | sh`, silently install catalog code or start Blender to discover whether it is ready.

Blender requires the official Hermes plugin pinned to commit `9b11ffeda84b077bea9de52d22733c7a41ca581a`. A local Blender executable or an open TCP port is not proof of readiness. The host must also expose the installed revision through its public capability context; if revision lookup is unavailable, NEO fails closed rather than assuming the pin. Only one task at a time owns the unauthenticated Blender port.

Browser and Hermes-worker execution requires an installed and enabled capability. Blender, browser and Hermes workers consult the active Hermes plugin context; NEO never guesses profile paths or reads a global profile directly.

## Actions and approvals

The browser slice navigates visibly and captures task-scoped screenshots. It does not publish, send, upload, purchase or log in. If such an action is requested, NEO stores a payload describing the action, account, audience, reason and any referenced screenshot path, then emits an approval request. Denial or timeout leaves the action undone. Approving one action does not authorise a different action, account or audience.

## Files and artifacts

Task and preview containment checks resolve real paths and reject traversal. Previews use an extension allowlist, byte caps, monotonic revisions and task-scoped paths. An expected artifact must exist as a file inside the task workspace and match the executor's verified artifact list before the task can be opened automatically.

A verified artifact is evidence of the expected file, not proof that an external action succeeded. Completion remains separate from verification and requires executor evidence. Automatic presentation opens the recorded artifact once per `(task_id, version)` and claims it durably before invoking the opener.

## Public data and logs

The task API exposes only allowlisted fields. It excludes raw tokens, token digests, absolute workdir paths and arbitrary approval fields. Log access is conversation- and task-scoped and rejects foreign log names, traversal and symlink escapes. Returned log tails are capped and redacted. Malformed internal payloads are not reflected into public responses.

Redaction cannot recognise every possible secret. Keep credentials out of prompts and source files, review worker logs when necessary, and use the environment allowlist for untrusted work.

## Host middleware

NEO registers only the documented plugin interfaces: `plugin.yaml`, Desktop plugin loading, the Dashboard API and an optional tool-manifest extension. It does not patch Hermes source files. Middleware appended to a turn is visible to the user and may read that turn's Hermes-provided context; it cannot inspect model weights, change safety messages, register persistent bypasses or recover raw worker tokens from storage.
