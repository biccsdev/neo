# Blender capability

NEO routes Blender work through the official `blender` plugin and a detached
Hermes worker. This page records exactly what that path promises, what it needs
on the Mac, and what has not been proven yet.

## Prerequisites

- macOS with Blender 5.1.2 installed at `/Applications/Blender.app`. The apply
  probe runs the app bundle's `Contents/MacOS/Blender --version` as an argv
  array (never a shell) and records the reported version and path.
- Hermes with the official `blender` plugin **installed and enabled**. NEO's
  readiness answer is `ready` only when the app is present *and* the plugin is
  installed *and* enabled. Anything else is `needs_approval`.
- A free `127.0.0.1:9876`. That port is the plugin's local bridge: it has no
  authentication while Blender is open, and it accepts one task at a time.

App presence is not scene readiness. NEO reports `scene_ready: false` at submit
time; a scene is only known to be live after the worker connects and
`get_objects_summary` returns. Discovery never stands in for that check.

## Disclosure before consent

Before anything is installed, the user sees the pinned identity and every risk:

| Field | Value |
| --- | --- |
| Name | `blender` |
| Version | `0.4.1` |
| SHA | `9b11ffeda84b077bea9de52d22733c7a41ca581a` |
| Tier | official |
| Subdir | `blender` |

The disclosure states, in plain words:

- the worker's tools execute arbitrary Python inside Blender, unsandboxed;
- the Blender session is online, so the model provider sees task text and the
  scene data the worker reads;
- the local bridge listens on **unauthenticated** `127.0.0.1:9876` while
  Blender is open, so any local process can drive that session;
- installing downloads the pinned plugin package; the download size is not
  pinned in advance and is reported by the installer.

## Installing

Installing requires an exact `ConsentRecord` for `blender` at that SHA. A record
for a different name, a different revision, a short SHA or no record at all is
refused before the installer is touched. The only request shape is:

```
hermes plugins install blender --ref 9b11ffeda84b077bea9de52d22733c7a41ca581a --enable
```

NEO never passes `--force`, `--allow-removed`, a repository URL, a ref other
than the pinned SHA, or unapproved dependencies. After the call the installed
revision is read back and must equal the approved SHA; the plugin must also be
enabled. A mismatch is reported as failure, never as success.

Installing is a separate, explicit action. Submitting a Blender task never
installs anything: with the plugin missing or disabled, the answer is
`needs_approval` plus the disclosure, and no task is created.

## Running a task

A prepared Blender task carries:

- `kind`: `blender`
- `executor`: `hermes_worker`
- `resource_key`: `blender:9876`
- skills: `blender:blender` and `neo-worker`
- argv: one `hermes chat --oneshot -Q ... --query-file <file> --in <workdir>`
  array with `--skills blender:blender,neo-worker`. The intent travels in the
  mode-`0600` query file, never on a command line, and no shell is used.
- expected artifact: `scene.blend`

The prompt tells the worker to read the scene with `get_objects_summary` first,
build with low-poly primitives, publish a task-scoped screenshot after each
modelling phase through `neo_task_publish_preview`, verify the result with
`get_objects_summary` again, and save `scene.blend` inside the task workdir.
It also requires approval before installing add-ons or publishing anything
outside the Mac, and forbids a nested NEO task or a second worker. It never
contains the worker token; the executor refuses a prompt carrying credential
material.

## Concurrency

The bridge port allows one scene at a time, so every Blender task takes the same
resource key, `blender:9876`. The registry reserves that key in the same
transaction that creates the task: a second Blender task is created `queued`,
not started, and is promoted by the existing queue drain when the first
finishes.

Port 9876 listening is only a conflict when NEO did not start the listener. If a
Blender instance outside NEO holds the port, submission returns
`blender_port_busy` with the host and port. NEO does not attach to that session,
does not retry, and never spawns a second instance on the same port.

## Verification and completion

Completion requires a **real** `.blend` inside the task's own workdir:

- the path must resolve beneath the workdir, with no symlinked components;
- it must be a regular file, non-empty, and start like a Blender save
  (`BLENDER`, gzip or zstd);
- every expected artifact — typically a phase screenshot — must verify through
  the artifact allowlist as well.

A missing, empty, truncated or non-Blender file prevents completion. A process
exit or a claim never counts: completion evidence is built from verified
artifacts and the executor's own terminal result. The saved `.blend` is the
artifact auto-presentation opens; screenshots arrive earlier as task-scoped
preview revisions.

## Demo steps

1. Install Blender 5.1.2 at the default location and confirm the plugin is
   installed and enabled in the active profile.
2. Ask NEO for a low-poly object in Blender. The first submission shows the
   pinned disclosure until consent is recorded; after consent, the pinned
   install runs and is verified by SHA.
3. Watch the mission-control task: `get_objects_summary` runs before modelling,
   each phase publishes a preview revision, and the worker saves
   `scene.blend`.
4. Completion is refused if `scene.blend` is absent. With it present, NEO opens
   the `.blend` result and shows the final screenshot.
5. Submit a second Blender task while the first runs: it appears `queued` on
   `blender:9876`.
6. Open Blender manually and submit again: NEO reports the busy port honestly
   instead of retrying into it.

## Known limitations

- **Not tested against live Blender in this change.** The slice's unit tests use
  injected probes, ports, installers and discovery. The live acceptance gate
  (P0-D: pinned install, MCP connect, scripted cube, screenshot, `.blend`) has
  not been run here, and no claim of live acceptance is made.
- The plugin bridge protocol itself (tool names and payloads) is the official
  plugin's contract; NEO only names `get_objects_summary` in the prompt and
  cannot verify tool schemas from this side.
- Scene verification is structural, not visual: NEO checks that a real `.blend`
  exists and was saved; it cannot judge whether the model looks right.
- The runtime hook is opt-in: a `NeoRuntime` built without a `BlenderSlice`
  keeps the previous worker route (and that route does not check readiness).
  Wiring the slice into the plugin's service graph is a composition-time step.
- `neo_task_frame_follow_up` still uses the worker route directly; it does not
  re-apply the slice's readiness or port checks.
