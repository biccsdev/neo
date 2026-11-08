# Browser capability

NEO routes browser work through Hermes' browser toolset in a detached `hermes`
worker. This page records exactly what the demo proves, where approval stops the
worker, and what is deliberately out of scope.

## What the demo does

A browser task navigates one harmless page — `https://example.com/` — in a
**visible** browser window. The worker is told this task is not headless and
must never claim headless-only browsing. It does not use a shell; it calls the
browser tools directly.

After each navigation the worker writes two files inside its own task workdir:

| File | Contents |
| --- | --- |
| `browser-state.json` | `url` (final http/https URL), `title` (text), `status` (HTTP status or `ok`/`loaded`/`done`) |
| `browser-screenshot.png` | a real screenshot image, published with `neo_task_publish_preview` |

The state file is verified as JSON; the URL must be `http` or `https` with a
host, the title must be text, and the status must be an HTTP code or a known
status word. The screenshot must be an allowlisted image, non-empty, starting
with a real image header, and contained beneath the task workdir. A path, a
claim or a process exit is never navigation evidence.

## The approval boundary

Before any publish, send, post, email, upload, purchase, deploy, comment, share
or account action, the worker must stop and report `needs_approval`. The request
records the exact action, the account it would use, the audience it would reach,
an action fingerprint and the task version.

`classify_publication_action()` is deliberately conservative:

- recognised publication verbs name their category (`publish`, `send`, `post`,
  `upload`, `purchase`, `deploy`, `account`);
- plain navigation/reading vocabulary is allowed without approval;
- anything it cannot recognise as navigation — including empty text — is
  `unknown_external_action` and requires approval.

The fingerprint binds action, account and audience. An approval for one draft
cannot authorise another draft, account or audience.

## Decisions

`approve` moves the task back to `running` and **executes nothing**. `deny`
cancels it. Both are idempotent: an identical repeat returns the same row and
emits no second event; a conflicting decision is refused. A decision is bound to
the conversation that owns the task, so another conversation cannot decide it.

A missing decision and a timeout both leave the task in `needs_approval`. A
timeout is reported, never treated as consent. NEO cannot bypass the waiting
state through a worker update or a completion call; only an explicit recorded
decision or a cancellation leaves it.

## Completion

Completion is refused unless `browser-state.json` is valid and a task-scoped
screenshot verifies. If the navigation state claims a publication, the recorded
approval must be `approve` **and** its fingerprint must match the claimed
action, account and audience.

Even then the completion is recorded as `navigation_only` with
`publication_executed: false`. This slice never performs the publication, with
or without approval, and never launches a follow-up worker to do it.

## What is proven, and what is not

Proven by the tests in `tests/test_browser_slice.py`:

- navigation state and screenshot verification, including refusal of invalid
  state, missing images and images outside the workdir;
- the classified publication table and the unknown-action default;
- the prompt and the prompt-file argv: `browser,neo-worker`, no shell, no
  credential or login instruction, no nested NEO;
- approval request/decision/idempotence/conversation binding/timeout;
- completion refusal without a matching approved decision, and
  `publication_executed: false` with one;
- conversation routing through the slice, with the legacy worker route intact
  when no slice is injected.

Not proven, and out of scope for this task:

- **No live browser run is claimed.** The tests use a fake launcher and no
  network. A live navigation demo is a later acceptance step.
- **Actual publication is out of scope.** Nothing here sends, posts, uploads,
  purchases, deploys or changes an account.
- No login, authentication or credential handling is supported or required.
- Approval does not queue a follow-up publication worker in this slice; it only
  records the decision.
