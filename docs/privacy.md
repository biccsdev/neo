# Privacy

NEO is a local operations plugin. It stores task state on the same Mac that runs Hermes.

## Stored data

NEO keeps its data under Hermes's per-plugin data directory for the active profile:

- task intents, states, timestamps, executor names and native session identifiers;
- conversation-scoped events and progress;
- expected and verified artifact names;
- bounded preview metadata and images;
- process identifiers, exit receipts and worker logs;
- approval decisions and a durable record that a result was already presented.

Worker tokens are stored only as SHA-256 digests. Raw worker tokens are passed to the owning subprocess in its environment and are not intentionally persisted. NEO never reads or stores Hermes provider credentials directly.

Task workspaces, logs, previews and the SQLite database survive plugin updates. Uninstalling the plugin does not automatically delete this data; remove the NEO plugin-data directory for the relevant profile after uninstalling if you want it erased.

## External services

A task can contact external services through the chosen executor:

- Codex and Claude receive the task intent and any source/context they read.
- Detached Hermes workers send prompts to their configured model provider.
- Browser workers navigate the requested URLs.
- The official Blender worker may send scene information read by its model provider.
- Installing a capability downloads the pinned package from the catalog source.

NEO does not send task data to a separate NEO server. Model-provider, CLI, MCP, browser and Blender privacy policies still apply.

## UI and logs

The Desktop API projects an allowlisted task shape. It does not return token digests or absolute workdir paths. Workspace names are shown as basenames. Log tails are size-capped and filtered for common credential shapes, private-key blocks, bearer tokens and absolute user paths.

Redaction is defence in depth, not a secrecy guarantee. A worker that deliberately prints a secret in an unrecognised format can still put it in its own log. Do not ask workers to print credentials, and use `NEO_WORKER_ENV_ALLOWLIST` when a reduced environment is practical.

Approval payloads are persisted for audit, but arbitrary payload fields are not exposed by the public task API. Browser publication is not implemented: NEO records approval or denial without publishing, sending, uploading or purchasing anything.

## Conversation scope

Tasks and events carry a conversation key. Status, list, preview and follow-up operations are restricted to the owning conversation. The middleware injects a compact summary of active tasks into the current turn only; it does not alter Hermes system or safety messages and does not persist that injected copy.
