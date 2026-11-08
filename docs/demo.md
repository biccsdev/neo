# NEO acceptance demo

This is the executable demo script for the current NEO build. It separates two
kinds of evidence:

- **Automated evidence** — reproducible on this machine with no live model, no
  Blender, no Desktop app and no microphone. It exercises the real NEO
  registry, task service, process supervisor, conversational tools and browser
  slice; executors are controlled recorders, and the only process started is a
  short-lived local Python child used to prove cancellation and restart
  honesty.
- **Live-only gates** — they need real Codex/Claude credentials, the official
  Blender plugin, the Hermes Desktop app or a microphone. They are **not
  proven yet**. No screenshot or video exists. Nothing below claims a live
  gate passed.

Run every command from the repository root:

```
cd /Users/user/code/work/neo-public
```

Dates and counts in this file were recorded on 2026-11-07 against this checkout.

---

## 1. Automated acceptance

### 1.1 End-to-end operator flow (Task 24)

```
python3 -m unittest tests.test_e2e -v
```

Expected visible result, ten tests and no failures:

```
test_approval_records_the_exact_action_and_publishes_nothing ... ok
test_cancel_kills_the_supervised_group_and_records_the_decision ... ok
test_a_repeated_request_keeps_one_task_and_one_launch ... ok
test_the_same_request_in_another_conversation_is_a_new_task ... ok
test_active_follow_up_queues_into_the_exact_session_without_a_new_task ... ok
test_completed_follow_up_creates_one_child_on_the_exact_session ... ok
test_an_artifact_that_no_longer_verifies_is_not_opened ... ok
test_verified_artifact_is_presented_exactly_once ... ok
test_restart_fails_a_lost_worker_and_never_infers_completion ... ok
test_status_and_list_are_accurate_and_never_mutate ... ok

Ran 10 tests in 0.2s

OK
```

What each scenario proves:

| Test | Visible assertion |
| --- | --- |
| `test_status_and_list_are_accurate_and_never_mutate` | `neo_task_status` reports state, kind, executor, intent, progress and needs; row, version, task list and event stream are byte-identical afterwards. |
| `test_a_repeated_request_keeps_one_task_and_one_launch` | Two equivalent submissions return one `task_id`, `deduplicated: true` on the second, and exactly one recorded argv. |
| `test_the_same_request_in_another_conversation_is_a_new_task` | Deduplication is conversation-scoped; the other conversation gets its own task. |
| `test_active_follow_up_queues_into_the_exact_session_without_a_new_task` | Follow-up argv is `codex queue --thread <session> --message ...`; task count is unchanged and the parent stays `running`. |
| `test_completed_follow_up_creates_one_child_on_the_exact_session` | One child task, `parent_id` set, the same `executor_session_id`, and `codex exec ... resume <session>`; never `--last`. |
| `test_cancel_kills_the_supervised_group_and_records_the_decision` | A real supervised child receives the group signal, is gone, a receipt records the signal/exit, and the task is `cancelled`; a repeat cancel changes nothing. |
| `test_verified_artifact_is_presented_exactly_once` | The verified `report.md` is opened once; a second call is skipped with `already presented`. |
| `test_an_artifact_that_no_longer_verifies_is_not_opened` | Deleting the file after completion makes presentation skip with `artifact verification failed`; the opener is never called. |
| `test_approval_records_the_exact_action_and_publishes_nothing` | Pending approval stores action, account, audience, fingerprint; `approve` returns the task to `running`, records `publication_not_executed`, and launches nothing. |
| `test_restart_fails_a_lost_worker_and_never_infers_completion` | A fresh graph over the same database reconciles the orphan as `failed` / `worker_lost` with "never reported completion". |

The suite spawns no Codex, Claude, browser or Blender process and uses no
network. Its own short-lived child is killed in cleanup; verify with:

```
ps -axo pid,command | grep "time.sleep(30)" | grep -v grep
```

Expected: no output.

### 1.2 Full system suite

```
python3 -m unittest discover -s tests
```

Expected visible result:

```
Ran 672 tests in 15.8s

OK (skipped=12)
```

The 12 skips are the FastAPI/httpx API surfaces: the system Python has no
FastAPI. They run under the project venv in 1.3.

### 1.3 API, presentation and security under the project venv

```
./.venv/bin/python -m unittest tests.test_api tests.test_presentation tests.test_security
```

Expected visible result:

```
Ran 115 tests in 2.8s

OK
```

### 1.4 Package validation and plugin doctor

```
hermes plugins validate .
```

Expected tail:

```
✓ security scan — safe
✓ no core override — no runtime rebinds of Hermes core
✓ desktop surface — stays inside the plugin SDK surface
◆ Isolation — runs in the plugin host (plugins.isolation: host)

Validation passed.
```

```
hermes plugins doctor .
```

Expected visible result:

```
Plugin Doctor: /Users/user/code/work/neo-public
  manifest: neo 0.1.0 (standalone)
  OK: runtime discovery, manifest parsing, import, and registration passed
  registrations: 7 tool(s), 1 hook(s)
```

### 1.5 Clean install into a throwaway HERMES_HOME

This is the only command in the demo that writes outside the repository, and
only beneath a temporary home. It never touches the operator's real profile
(`~/.hermes`).

```
NEO_SRC="/Users/user/code/work/neo-public"
HERMES_HOME="$(mktemp -d /tmp/neo-hermes-home.XXXXXX)"
env HERMES_HOME="$HERMES_HOME" hermes plugins install "file://$NEO_SRC" --enable --yes-deps
env HERMES_HOME="$HERMES_HOME" hermes plugins list
env HERMES_HOME="$HERMES_HOME" hermes plugins doctor neo
```

Expected visible results:

- `plugins install` reports the local `file://` source, installs `neo`, and
  auto-enables it (`--enable`).
- `plugins list` shows `neo` with status `enabled`.
- `plugins doctor neo` prints
  `registrations: 7 tool(s), 1 hook(s)`.

Clean up when finished:

```
rm -rf "$HERMES_HOME"
```

Status: this isolated install was executed successfully during Task 24 setup
(plugin enabled; Doctor reported 7 tools and 1 hook). Re-run the block above to
reproduce it. It is automated evidence, not a live-model gate.

---

## 2. Live-only gates

These require real external components. **None has been run for this demo.**
There is no screenshot or video yet; every item below stays `PENDING` until
evidence is captured.

### 2.1 Codex coding task

Prerequisites:

- `codex` CLI on `PATH`;
- a configured model and working authentication for that model (the Codex
  provider login is valid and the account can run a turn);
- Hermes running with the NEO plugin enabled in a non-worker process.

Scenario:

1. Ask NEO to implement a small change in a scratch repository.
2. Confirm `neo_task_status` moves `starting` → `running` → `completed`.
3. Ask for a follow-up ("also add a test").
4. Confirm the follow-up argv resumes the **same** reported thread id and no
   new top-level task is created by the active queue path.

Expected visible result: one Codex window/CLI run, files changed in the task
workspace, a completion summary and a resumed thread. Evidence required: task
id, thread id, status transcript and changed files. **PENDING.**

### 2.2 Claude coding task

Prerequisites:

- `claude` CLI on `PATH` and authenticated;
- a Hermes profile with NEO enabled.

Scenario: submit a coding task with the Claude executor selected, then refine
the completed task. Expected: the follow-up uses `--resume <uuid>` for the
recorded session and never `--fork-session`. **PENDING.**

### 2.3 Blender

Prerequisites:

- official Hermes Blender plugin installed and enabled at the reviewed commit
  `9b11ffeda84b077bea9de52d22733c7a41ca581a`;
- the host exposes the installed revision through its public capability
  context (revision lookup available; otherwise NEO fails closed);
- Blender running with its port available (the slice owns port 9876
  exclusively).

Scenario: request a small scene, confirm the capability disclosure and
approval appear before any launch, then confirm the Blender session produces
the expected scene/render artifact. Expected: `needs_approval` first, then one
Blender worker owning the port. **PENDING.**

### 2.4 Hermes Desktop

Prerequisite: Hermes Desktop installed and signed in.

Scenario: open the NEO page, confirm the mission-control view lists tasks,
status updates appear while a task runs, and an approval card can be answered.
Expected: live task list, previews, and a visible approval decision.
**PENDING.**

### 2.5 Microphone / voice turn

Prerequisites:

- a working microphone and speech input;
- the voice model configured for spoken turns.

Scenario: say a chat request, then an operations request. Expected: chat is
answered in two or three spoken sentences; an operations request opens one
visible CLI window with the task pre-loaded and no manual prompts. **PENDING.**

### 2.6 Screenshots and video

No screenshots or screen recording exist for this build. Every visual gate
above remains unproven until they are captured alongside the task ids.
**PENDING.**

---

## 3. Evidence checklist

| # | Gate | Kind | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1 | End-to-end operator flow | automated | PASS | `python3 -m unittest tests.test_e2e -v` → 10 tests, OK |
| 2 | Full system suite | automated | PASS | 672 tests, OK (12 skips) |
| 3 | API/presentation/security | automated | PASS | 115 tests, OK under `.venv` |
| 4 | Package validation | automated | PASS | `hermes plugins validate .` → Validation passed |
| 5 | Plugin doctor | automated | PASS | 7 tool(s), 1 hook(s) |
| 6 | Isolated install + enable | automated | RECORDED | temporary `HERMES_HOME`, `file://` install, enabled, Doctor 7/1; re-run 1.5 |
| 7 | Live Codex task | live | PENDING | needs model/auth; no evidence yet |
| 8 | Live Codex exact-thread follow-up | live | PENDING | needs model/auth; no evidence yet |
| 9 | Live Claude resume | live | PENDING | needs Claude auth; no evidence yet |
| 10 | Live Blender scene | live | PENDING | needs official plugin + revision lookup + Blender |
| 11 | Hermes Desktop mission control | live | PENDING | needs Desktop |
| 12 | Microphone voice routing | live | PENDING | needs microphone |
| 13 | Screenshots/video | live | PENDING | none captured |

Rule for this checklist: a row becomes `PASS` only when the named command or
scenario has been run in this checkout and its output is attached. A planned
or plausible result is `PENDING`. Row 6 was executed before this document was
written and its output was not saved here, so it is `RECORDED`; re-running 1.5
in this checkout promotes it to `PASS`.

---

## 4. Suggested demo order

1. Run 1.1–1.4 and keep the terminal output; this proves the automated
   contract without any external service.
2. Run 1.5 in a throwaway `HERMES_HOME` to prove packaging and registration.
3. Only then attempt the live gates in 2.1–2.5, in that order, capturing task
   ids and status output for each.
4. Update the checklist only with captured evidence. Leave the rest
   `PENDING`.

## 5. What the automated flow does not prove

- It does not prove a real Codex or Claude turn can run with the operator's
  credentials.
- It does not prove a real browser can navigate or that Blender can render.
- It does not prove the Desktop UI renders or that the microphone works.
- It does not prove publication is impossible in every live browser context;
  it proves NEO's slice records approval and executes no publication itself.
- A process exit, a receipt or a successful `open` is never treated as proof
  of work; completion still requires executor evidence and verified artifacts.
