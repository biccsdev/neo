"""The browser vertical slice: navigation evidence, approval, completion.

A browser task is the first NEO slice whose natural next step can leave the
machine. So this module keeps three things apart that are easy to blur:

* **Navigation.** A worker drives Hermes' browser tools in a visible browser and
  proves each navigation with two files inside its own workdir:
  ``browser-state.json`` (the final url, the title and the response status) and
  a task-scoped screenshot image. Prose is not evidence.
* **Publication.** Publish, send, post, email, upload, purchase, deploy,
  comment, share and account actions are external side effects. NEO classifies
  them conservatively and anything it cannot recognise as plain navigation
  defaults to requiring approval. This slice never performs the publication
  itself, with or without approval.
* **Approval.** A pending request is recorded in ``approval_json`` with the
  exact action, account, audience, an action fingerprint and the task version.
  ``approve`` moves the task back to ``running`` and executes nothing; ``deny``
  cancels it. A missing decision or a timeout leaves the task in
  ``needs_approval``. Decisions are idempotent, bound to their conversation and
  cannot be bypassed by a worker update or completion call.

Completion needs a valid navigation state, a real screenshot and, when the
state claims a publication, an approved decision whose fingerprint matches the
claimed action. Even then the completion is recorded as navigation only;
``publication_executed`` stays ``False``.

The slice is injected with its runtime, launcher and approval callback. It
spawns nothing implicitly and never opens a network connection.

Sections 4.2, 4.3, 4.6 and 4.7 of the implementation plan (Task 20).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from artifacts import (
    DEFAULT_ALLOWED_EXTENSIONS,
    ArtifactError,
    verify_artifact,
)
from executors.hermes_worker import HermesWorkerExecutor
from state_machine import CompletionEvidence

STATE_ARTIFACT = "browser-state.json"
"""The navigation record a worker writes after each navigation."""

SCREENSHOT_ARTIFACT = "browser-screenshot.png"
"""The task-scoped screenshot a worker captures and publishes."""

STATE_SCHEMA = "neo.browser.state.v1"
APPROVAL_SCHEMA = "neo.browser.approval.v1"

WORKER_SKILLS: Tuple[str, ...] = ("browser", "neo-worker")
"""Hermes preloads the browser toolset skill and the NEO reporting skill."""

WORKER_EXECUTOR = "hermes_worker"
CAPABILITY = "browser"
HARMLESS_URL = "https://example.com/"

DEFAULT_APPROVAL_TIMEOUT = 900.0
"""Fifteen minutes; a decision after this is reported as timed out, not taken."""

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

_MAX_INTENT = 8000
_MAX_STATE_BYTES = 1024 * 1024
_MAX_ACTION = 2000

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)


# -- errors -------------------------------------------------------------------


class BrowserSliceError(Exception):
    """Base class for an honest browser refusal."""


class BrowserApprovalError(BrowserSliceError):
    """An approval request or decision was refused or conflicts with state."""


class BrowserCompletionError(BrowserSliceError):
    """A browser task cannot be completed from the evidence it offers."""


# -- publication classification ----------------------------------------------


_PUBLICATION_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "publish",
        ("publish", "publishes", "publishing", "published", "publisher"),
    ),
    (
        "send",
        (
            "send",
            "sends",
            "sending",
            "sent",
            "email",
            "emails",
            "emailing",
            "emailed",
            "message",
            "messages",
            "dm",
            "sms",
            "reply",
            "forward",
        ),
    ),
    (
        "post",
        (
            "post",
            "posts",
            "posting",
            "posted",
            "tweet",
            "tweets",
            "tweeting",
            "comment",
            "comments",
            "commenting",
            "share",
            "shares",
            "sharing",
            "repost",
            "announce",
        ),
    ),
    (
        "upload",
        ("upload", "uploads", "uploading", "uploaded", "attach", "attachment"),
    ),
    (
        "purchase",
        (
            "purchase",
            "purchases",
            "purchasing",
            "buy",
            "buys",
            "buying",
            "bought",
            "order",
            "orders",
            "checkout",
            "pay",
            "pays",
            "payment",
            "subscribe",
            "subscription",
            "bid",
            "booking",
        ),
    ),
    (
        "deploy",
        (
            "deploy",
            "deploys",
            "deploying",
            "deployment",
            "release",
            "releases",
            "rollout",
            "ship",
            "shipping",
            "push",
            "merge",
        ),
    ),
    (
        "account",
        (
            "account",
            "accounts",
            "sign in",
            "signin",
            "log in",
            "login",
            "register",
            "password",
            "delete",
            "deletes",
            "deleting",
            "remove",
            "removes",
            "revoke",
            "transfer",
            "unsubscribe",
            "follow",
            "unfollow",
            "profile",
        ),
    ),
)

_COMPILED_CATEGORIES = tuple(
    (
        name,
        re.compile(
            r"\b(?:%s)\b" % "|".join(re.escape(term) for term in terms),
            re.IGNORECASE,
        ),
        tuple(terms),
    )
    for name, terms in _PUBLICATION_CATEGORIES
)

_SAFE_TOKENS = frozenset(
    {
        "navigate",
        "navigates",
        "open",
        "opens",
        "visit",
        "goto",
        "browse",
        "read",
        "reads",
        "view",
        "scroll",
        "screenshot",
        "capture",
        "inspect",
        "extract",
        "list",
        "check",
        "find",
        "report",
        "search",
        "look",
        "wait",
        "click",
        "page",
        "pages",
        "url",
        "urls",
        "link",
        "links",
        "local",
        "title",
        "test",
        "demo",
    }
)

_WORD = re.compile(r"[a-z0-9]+")


def classify_publication_action(text: object) -> dict:
    """Classify one action as plain navigation or an approval-worthy effect.

    The answer is deliberately conservative. A recognised publication verb
    names its categories and matched terms. Text that contains no publication
    verb but also no plain navigation vocabulary is *unknown* and requires
    approval, because NEO cannot prove it is harmless. Empty or non-text input
    is unknown too.
    """

    raw = text if isinstance(text, str) else ""
    cleaned = " ".join(raw.split()).strip()
    lowered = cleaned.lower()
    action = cleaned[:_MAX_ACTION]

    categories = []
    terms = []
    for name, pattern, vocabulary in _COMPILED_CATEGORIES:
        matches = pattern.findall(lowered)
        if not matches:
            continue
        categories.append(name)
        for match in matches:
            token = match.lower()
            if token not in terms and token in vocabulary:
                terms.append(token)

    if categories:
        return {
            "action": action,
            "requires_approval": True,
            "categories": tuple(categories),
            "terms": tuple(terms),
            "reason": "the action can publish, send or change something outside NEO",
        }

    words = set(_WORD.findall(lowered))
    if words & _SAFE_TOKENS:
        return {
            "action": action,
            "requires_approval": False,
            "categories": (),
            "terms": (),
            "reason": "plain navigation or reading",
        }

    return {
        "action": action,
        "requires_approval": True,
        "categories": ("unknown_external_action",),
        "terms": (),
        "reason": (
            "this action is not recognisable as plain navigation, so NEO treats "
            "it as a potentially destructive external action"
        ),
    }


def publication_fingerprint(
    action: object, account: object = None, audience: object = None
) -> str:
    """Return the stable SHA-256 identity of one exact publication request.

    The fingerprint binds the action text, the account and the audience, so an
    approval for one draft cannot authorise a different draft, account or
    audience.
    """

    def normalise(value: object) -> str:
        return " ".join(str(value if value is not None else "").split()).strip().lower()

    material = "\n".join(
        (normalise(action), normalise(account), normalise(audience))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# -- prompt -------------------------------------------------------------------


def build_worker_prompt(intent: str, workdir: str) -> str:
    """Render the browser work order: navigation, evidence, approval, limits."""

    text = intent.strip() if isinstance(intent, str) else ""
    if not text or len(text) > _MAX_INTENT:
        raise ValueError("a browser task needs a non-empty intent")
    if not isinstance(workdir, str) or not workdir.strip():
        raise ValueError("a browser task needs an absolute workdir")
    return "\n".join(
        (
            "## Browser work order",
            "",
            "Intent: %s" % text,
            "",
            "Work only inside %s. Drive a visible browser through Hermes' browser "
            "tools (skill `browser`) and report through the `neo-worker` bridge. "
            "This task is not headless: use a visible browser window, navigate "
            "visibly, and never claim headless-only browsing. Do not use a shell; "
            "call the browser tools directly." % workdir,
            "",
            "### Steps",
            "1. Navigate to a harmless page, for example %s, in the visible "
            "browser. Never log in, never authenticate, and never handle "
            "credentials of any kind." % HARMLESS_URL,
            "2. After each navigation, write `%s` inside %s with exactly these "
            "fields: `url` (the final http or https URL), `title` (the page "
            "title) and `status` (the response status)." % (STATE_ARTIFACT, workdir),
            "3. Capture `%s` inside %s after that navigation and publish it with "
            "`neo_task_publish_preview`, so the preview is scoped to this task "
            "and belongs to this task alone." % (SCREENSHOT_ARTIFACT, workdir),
            "4. Report the URL, the title and the status after every navigation. "
            "The two files are the evidence; a description is not.",
            "5. Finish with `neo_task_complete` only when both files exist inside "
            "%s and their contents match what you actually saw." % workdir,
            "",
            "### Approval boundary",
            "Stop before any publish, send, post, email, upload, purchase, deploy, "
            "comment, share or account action. Do not start it, do not queue it "
            "and do not prepare it for later.",
            "When the next step is one of those actions, report state "
            "`needs_approval` and state the exact action, the account it would "
            "use and the audience it would reach. Stop there and wait.",
            "If approval is denied or the request times out, or its timeout "
            "elapses, the task stays unpublished; report that honestly and do "
            "not retry the action. An approval is a decision recorded by NEO, "
            "not a permission for this worker to publish anything itself.",
            "",
            "### Recursion boundary",
            "Never create a nested NEO task, a second worker or another Hermes "
            "session. One task, one worker. Report the need and stop instead.",
            "",
            "### Reporting",
            "Use the NEO bridge tools for progress, previews, approval requests, "
            "completion and failure. Never print, paste or write the task token.",
        )
    )


# -- helpers ------------------------------------------------------------------


def _decode_mapping(raw: object) -> dict:
    """Decode a JSON object without trusting its shape."""

    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (str, bytes)):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _decoded_expected(raw: object) -> Tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        values = raw
    elif isinstance(raw, (str, bytes)):
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return ()
    else:
        return ()
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(item for item in values if isinstance(item, str) and item.strip())


def _valid_state_url(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    parsed = urlparse(text)
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    return text


_STATUS_WORDS = frozenset(
    {"ok", "loaded", "complete", "completed", "success", "navigated", "ready", "done"}
)


def _valid_state_status(value: object) -> Optional[object]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 100 <= value <= 599 else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _STATUS_WORDS:
            return text
    return None


def _looks_like_image(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    if any(header.startswith(magic) for magic in _IMAGE_MAGIC):
        return True
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return True
    return False


# -- the slice ----------------------------------------------------------------


class BrowserSlice:
    """Prepare, approve and verify one browser task; never publish anything."""

    def __init__(
        self,
        *,
        runtime: Any = None,
        launcher: Optional[Callable[..., object]] = None,
        approval: Optional[Callable[[dict], object]] = None,
        clock: Optional[Callable[[], float]] = None,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> None:
        self.runtime = runtime
        self.launcher = launcher
        self.approval = approval
        self.clock = clock or (lambda: 0.0)
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("approval timeout must be a number of seconds")
        if timeout_seconds < 0:
            raise ValueError("approval timeout cannot be negative")
        self.timeout_seconds = float(timeout_seconds)

    # -- preparation ------------------------------------------------------

    def prepare_task(
        self,
        intent: str,
        conversation_key: str,
        workdir: Optional[str] = None,
        *,
        task_id: Optional[str] = None,
    ) -> dict:
        """Return browser task fields and, with a task id, the worker argv.

        The caller owns task creation and the launch, so the registry keeps its
        single-writer rules. Nothing is spawned, installed or published here.
        """

        text = intent.strip() if isinstance(intent, str) else ""
        if not text or len(text) > _MAX_INTENT:
            raise ValueError("a browser task needs a non-empty intent")
        if not isinstance(conversation_key, str) or not conversation_key.strip():
            raise ValueError("a browser task needs a conversation key")
        directory = self._validated_workdir(workdir)

        prompt = build_worker_prompt(text, directory)
        fields: dict = {
            "kind": "browser",
            "capability": CAPABILITY,
            "executor": WORKER_EXECUTOR,
            "resource_key": None,
            "skills": WORKER_SKILLS,
            "intent": text,
            "prompt": prompt,
            "workdir": directory,
            "expected_artifacts": (STATE_ARTIFACT, SCREENSHOT_ARTIFACT),
            "headless": False,
            "visible": True,
            "prompt_path": None,
            "argv": (),
        }
        if task_id is not None:
            if self.runtime is None:
                raise BrowserSliceError(
                    "a browser task needs the NEO runtime to build a prompt file"
                )
            prompt_path = self.runtime.prompt_file(task_id, prompt, directory)
            executor = HermesWorkerExecutor(skills=WORKER_SKILLS)
            fields["prompt_path"] = prompt_path
            fields["argv"] = tuple(
                executor.build_initial_argv(prompt_path, workdir=directory)
            )
        return fields

    def launch(self, task_id: str, fields: Mapping[str, Any]) -> object:
        """Launch the prepared argv through the injected launcher, never a shell."""

        argv = list(fields.get("argv") or ())
        if not argv:
            raise BrowserSliceError("the browser slice did not build a worker argv")
        if self.launcher is not None:
            return self.launcher(task_id, argv, role="task")
        runtime_launch = getattr(self.runtime, "launch", None)
        if callable(runtime_launch):
            return runtime_launch(task_id, argv, role="task")
        raise BrowserSliceError("a browser task needs a launcher")

    def _validated_workdir(self, workdir: Optional[str]) -> str:
        if not isinstance(workdir, str) or not workdir.strip():
            raise ValueError("a browser task needs an absolute existing workdir")
        text = workdir.strip()
        if "\x00" in text or not os.path.isabs(text):
            raise ValueError("a browser task needs an absolute existing workdir")
        try:
            resolved = Path(text).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("a browser task needs an absolute existing workdir") from None
        if not resolved.is_dir():
            raise ValueError("a browser task workdir must be a directory")
        return str(resolved)

    # -- approval ---------------------------------------------------------

    def request_approval(
        self,
        task: Mapping[str, Any],
        *,
        reason: str,
        action: str,
        account: Optional[str] = None,
        audience: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        conversation_key: Optional[str] = None,
    ) -> dict:
        """Record one pending approval request and enter ``needs_approval``.

        The row always carries its exact action, account, audience, fingerprint
        and task version *before* the state changes, so the state is never
        waiting without a record of what it waits for.
        """

        row = self._task_row(task)
        self._check_conversation(row, conversation_key)
        self._require_runtime()

        text = action.strip() if isinstance(action, str) else ""
        if not text or len(text) > _MAX_ACTION:
            raise BrowserApprovalError("an approval request needs an exact action")
        reason_text = reason.strip() if isinstance(reason, str) else ""
        if not reason_text:
            raise BrowserApprovalError("an approval request needs a reason")
        account_text = account.strip() if isinstance(account, str) else None
        audience_text = audience.strip() if isinstance(audience, str) else None
        payload_dict = dict(payload) if isinstance(payload, Mapping) else {}

        classification = classify_publication_action(text)
        existing = _decode_mapping(row.get("approval_json"))
        record = {
            "schema": APPROVAL_SCHEMA,
            "decision": None,
            "reason": reason_text,
            "action": text,
            "account": account_text,
            "audience": audience_text,
            "payload": payload_dict,
            "fingerprint": publication_fingerprint(text, account_text, audience_text),
            "categories": list(classification["categories"]),
            "publication": bool(classification["requires_approval"]),
            "requires_approval": True,
            "task_version": row.get("version"),
            "requested_at": self.clock(),
            "timeout_seconds": self.timeout_seconds,
            "publication_executed": False,
        }

        if row.get("state") == "needs_approval":
            if existing.get("fingerprint") == record["fingerprint"]:
                return dict(row)
            raise BrowserApprovalError(
                "this task is already waiting for a different approval request"
            )
        if row.get("state") not in ("running", "waiting"):
            raise BrowserApprovalError(
                "only a running browser task can request approval"
            )

        registry = self.runtime.registry
        service = self.runtime.service
        stored = registry.update_task(
            row["id"],
            expected_version=row["version"],
            approval_json=json.dumps(record, sort_keys=True),
        )
        try:
            updated = service.transition(
                row["id"],
                "needs_approval",
                expected_version=stored["version"],
                event_extra={
                    "action": text,
                    "account": account_text,
                    "audience": audience_text,
                    "reason": reason_text,
                    "fingerprint": record["fingerprint"],
                },
            )
        except Exception:
            # Put the row back the way it was; a failed transition must not
            # leave a request stored on a task that is not waiting for it.
            try:
                registry.update_task(
                    row["id"],
                    expected_version=stored["version"],
                    approval_json=row.get("approval_json"),
                )
            except Exception:
                pass
            raise
        record["task_version"] = int(updated["version"]) + 1
        final = registry.update_task(
            row["id"],
            expected_version=updated["version"],
            approval_json=json.dumps(record, sort_keys=True),
        )
        if self.approval is not None:
            try:
                self.approval(dict(record))
            except Exception:
                pass
        return dict(final)

    def record_approval(
        self,
        task_id: str,
        decision: str,
        *,
        reason: Optional[str] = None,
        conversation_key: Optional[str] = None,
    ) -> dict:
        """Record one approval decision; an identical repeat is a no-op.

        ``approve`` moves the task to ``running`` and ``deny`` cancels it. No
        publication, browser action or worker launch happens here: this slice
        records the decision and stops.
        """

        if decision not in ("approve", "deny"):
            raise BrowserApprovalError("an approval decision must be approve or deny")
        self._require_runtime()
        row = self.runtime.registry.get_task(task_id)
        if row is None:
            raise BrowserApprovalError("no browser task with that id is visible")
        self._check_conversation(row, conversation_key)

        approval = _decode_mapping(row.get("approval_json"))
        if not approval or not approval.get("fingerprint"):
            raise BrowserApprovalError("this task has no recorded approval request")

        if approval.get("decision") == decision:
            return {
                "task": dict(row),
                "decision": decision,
                "idempotent": True,
                "fingerprint": approval.get("fingerprint"),
                "publication_executed": False,
            }
        if approval.get("decision") in ("approve", "deny"):
            raise BrowserApprovalError("a conflicting approval decision was refused")
        if row.get("state") != "needs_approval":
            raise BrowserApprovalError("this task is not waiting for approval")

        target = "running" if decision == "approve" else "cancelled"
        updated = self.runtime.service.transition(
            task_id,
            target,
            expected_version=row["version"],
            event_extra={"decision": decision, "fingerprint": approval.get("fingerprint")},
        )
        reason_text = reason if isinstance(reason, str) and reason.strip() else approval.get(
            "reason", ""
        )
        approval.update(
            {
                "decision": decision,
                "reason": reason_text,
                "decided_at": self.clock(),
                "task_version": updated["version"],
                "publication_executed": False,
                "outcome": "publication_not_executed",
            }
        )
        final = self.runtime.registry.update_task(
            task_id,
            expected_version=updated["version"],
            approval_json=json.dumps(approval, sort_keys=True),
        )
        return {
            "task": dict(final),
            "decision": decision,
            "idempotent": False,
            "fingerprint": approval.get("fingerprint"),
            "publication_executed": False,
        }

    def approval_status(
        self, task: Mapping[str, Any], *, now: Optional[float] = None
    ) -> dict:
        """Report the recorded decision and whether the request has timed out."""

        row = self._task_row(task)
        approval = _decode_mapping(row.get("approval_json"))
        decision = approval.get("decision")
        requested_at = approval.get("requested_at")
        timeout = approval.get("timeout_seconds", self.timeout_seconds)
        current = self.clock() if now is None else float(now)
        timed_out = False
        if (
            row.get("state") == "needs_approval"
            and decision is None
            and isinstance(requested_at, (int, float))
            and isinstance(timeout, (int, float))
        ):
            timed_out = (current - float(requested_at)) >= float(timeout)
        return {
            "task_id": row.get("id"),
            "state": row.get("state"),
            "decision": decision,
            "pending": row.get("state") == "needs_approval" and decision is None,
            "timed_out": timed_out,
            "fingerprint": approval.get("fingerprint"),
            "action": approval.get("action"),
            "account": approval.get("account"),
            "audience": approval.get("audience"),
            "publication_executed": False,
        }

    def expire_approval(self, task_id: str, *, now: Optional[float] = None) -> dict:
        """Report an elapsed approval request without changing the task.

        A timeout is not a decision: the task remains in ``needs_approval`` and
        nothing is published.
        """

        self._require_runtime()
        row = self.runtime.registry.get_task(task_id)
        if row is None:
            raise BrowserApprovalError("no browser task with that id is visible")
        return self.approval_status(row, now=now)

    def guard_worker_update(self, task: Mapping[str, Any], target: object) -> bool:
        """Refuse a worker update or completion that would clear approval.

        The state machine allows ``needs_approval -> running``, so this guard is
        the slice's own boundary: only an explicit recorded decision or a
        cancellation may leave the waiting state.
        """

        row = self._task_row(task)
        if row.get("state") != "needs_approval":
            return True
        approval = _decode_mapping(row.get("approval_json"))
        if approval.get("decision") == "approve":
            return True
        value = getattr(target, "value", target)
        if value == "cancelled":
            return True
        raise BrowserApprovalError(
            "a task waiting for approval cannot move to %r without a decision" % (value,)
        )

    # -- navigation evidence ---------------------------------------------

    def verify_navigation(
        self, task: Mapping[str, Any], *, screenshot: Optional[str] = None
    ) -> dict:
        """Verify ``browser-state.json`` and a task-scoped screenshot.

        The state must name an http/https URL, a text title and a status; the
        screenshot must be a real image inside the task workdir.
        """

        row = self._task_row(task)
        root = self._workdir(row)
        state_path = self._verify_state(root)
        state = self._read_state(state_path)

        url = _valid_state_url(state.get("url"))
        if url is None:
            raise BrowserCompletionError(
                "browser-state.json must name an http or https url"
            )
        title = state.get("title")
        if not isinstance(title, str):
            raise BrowserCompletionError("browser-state.json must carry a text title")
        status = _valid_state_status(state.get("status"))
        if status is None:
            raise BrowserCompletionError(
                "browser-state.json must carry an http status or a status word"
            )

        name = self._verify_screenshot(root, state, screenshot)
        return {
            "url": url,
            "title": title,
            "status": status,
            "screenshot": name,
            "artifacts": [STATE_ARTIFACT, name],
            "publication": bool(state.get("publication")),
            "action": state.get("action"),
            "account": state.get("account"),
            "audience": state.get("audience"),
        }

    def verify_completion(self, task: Mapping[str, Any]) -> CompletionEvidence:
        """Return completion evidence, or refuse when navigation is unproven."""

        evidence, _navigation = self._verify(task)
        return evidence

    def completion_record(self, task: Mapping[str, Any]) -> dict:
        """Return what the completion actually proves: navigation, not publication."""

        _evidence, navigation = self._verify(task)
        return {
            "task_id": self._task_row(task).get("id"),
            "outcome": "navigation_only",
            "publication_executed": False,
            "navigation": navigation,
            "verified_artifacts": navigation["artifacts"],
        }

    # -- internals --------------------------------------------------------

    def _verify(self, task: Mapping[str, Any]) -> Tuple[CompletionEvidence, dict]:
        row = self._task_row(task)
        try:
            self.guard_worker_update(row, "completed")
        except BrowserApprovalError as error:
            raise BrowserCompletionError(str(error)) from error
        navigation = self.verify_navigation(row)
        expected = _decoded_expected(row.get("expected_json"))

        verified: list = [STATE_ARTIFACT, navigation["screenshot"]]
        root = self._workdir(row)
        for name in expected:
            if name in verified:
                continue
            try:
                verify_artifact(root, name, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS)
            except ArtifactError as error:
                raise BrowserCompletionError(str(error)) from error
            verified.append(name)

        self._check_publication(row, navigation)
        return (
            CompletionEvidence(
                executor_confirmed=True,
                expected_artifacts=tuple(expected),
                verified_artifacts=tuple(verified),
            ),
            navigation,
        )

    def _check_publication(self, row: Mapping[str, Any], navigation: Mapping[str, Any]) -> None:
        """Require an approved, matching fingerprint for a publication claim."""

        approval = _decode_mapping(row.get("approval_json"))
        claimed = bool(navigation.get("publication")) or bool(
            row.get("approval_json") and approval.get("publication")
        )
        if not claimed:
            return
        if approval.get("decision") != "approve":
            raise BrowserCompletionError(
                "this completion claims a publication without an approved decision"
            )
        action = navigation.get("action") or approval.get("action")
        account = navigation.get("account")
        audience = navigation.get("audience")
        if account is None:
            account = approval.get("account")
        if audience is None:
            audience = approval.get("audience")
        if approval.get("fingerprint") != publication_fingerprint(action, account, audience):
            raise BrowserCompletionError(
                "the approved fingerprint does not match the claimed publication"
            )

    def _verify_state(self, root: Path) -> Path:
        try:
            artifact = verify_artifact(
                str(root), STATE_ARTIFACT, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS
            )
        except ArtifactError as error:
            raise BrowserCompletionError(
                "completion requires a readable %s inside the task workdir: %s"
                % (STATE_ARTIFACT, error)
            ) from error
        try:
            if artifact.path.stat().st_size > _MAX_STATE_BYTES:
                raise BrowserCompletionError(
                    "%s is too large to be a navigation record" % STATE_ARTIFACT
                )
        except OSError:
            raise BrowserCompletionError("%s cannot be read" % STATE_ARTIFACT) from None
        return artifact.path

    def _read_state(self, path: Path) -> dict:
        try:
            raw = path.read_text(encoding="utf-8")
            value = json.loads(raw)
        except (OSError, UnicodeError, ValueError):
            raise BrowserCompletionError(
                "%s is not valid JSON navigation state" % STATE_ARTIFACT
            ) from None
        if not isinstance(value, Mapping):
            raise BrowserCompletionError(
                "%s must contain a JSON object" % STATE_ARTIFACT
            )
        return dict(value)

    def _verify_screenshot(
        self, root: Path, state: Mapping[str, Any], explicit: Optional[str]
    ) -> str:
        candidates: list = []
        if isinstance(explicit, str) and explicit.strip():
            candidates.append(explicit.strip())
        elif isinstance(state.get("screenshot"), str) and state["screenshot"].strip():
            candidates.append(state["screenshot"].strip())
        else:
            named = root / SCREENSHOT_ARTIFACT
            if named.is_file():
                candidates.append(SCREENSHOT_ARTIFACT)
            else:
                for candidate in sorted(root.iterdir()):
                    if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.is_file():
                        candidates.append(candidate.name)
                        break

        for name in candidates:
            try:
                artifact = verify_artifact(
                    str(root), name, allowed_extensions=IMAGE_EXTENSIONS
                )
            except ArtifactError as error:
                raise BrowserCompletionError(
                    "the task-scoped screenshot is not a verifiable image: %s" % error
                ) from error
            try:
                if artifact.path.stat().st_size <= 0:
                    raise BrowserCompletionError("the screenshot is empty")
            except OSError:
                raise BrowserCompletionError("the screenshot cannot be read") from None
            if not _looks_like_image(artifact.path):
                raise BrowserCompletionError(
                    "the screenshot does not start like an image file"
                )
            return name
        raise BrowserCompletionError(
            "completion requires a task-scoped screenshot image inside the workdir"
        )

    def _workdir(self, row: Mapping[str, Any]) -> Path:
        workdir = row.get("workdir")
        if not isinstance(workdir, str) or not workdir.strip():
            raise BrowserCompletionError("the task has no workdir to verify")
        try:
            root = Path(workdir).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise BrowserCompletionError("the task workdir is not readable") from None
        if not root.is_dir():
            raise BrowserCompletionError("the task workdir is not a directory")
        return root

    def _task_row(self, task: object) -> dict:
        if not isinstance(task, Mapping):
            raise BrowserApprovalError("a browser task row is required")
        row = dict(task)
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise BrowserApprovalError("a browser task row needs an id")
        return row

    def _check_conversation(
        self, row: Mapping[str, Any], conversation_key: Optional[str]
    ) -> None:
        if conversation_key is None:
            return
        if not isinstance(conversation_key, str) or not conversation_key.strip():
            raise BrowserApprovalError("a conversation key must be non-empty")
        if row.get("conversation_key") != conversation_key:
            raise BrowserApprovalError(
                "another conversation cannot decide this task's approval"
            )

    def _require_runtime(self) -> None:
        if self.runtime is None:
            raise BrowserSliceError("a browser task needs the NEO runtime")


__all__ = [
    "APPROVAL_SCHEMA",
    "CAPABILITY",
    "DEFAULT_APPROVAL_TIMEOUT",
    "HARMLESS_URL",
    "IMAGE_EXTENSIONS",
    "SCREENSHOT_ARTIFACT",
    "STATE_ARTIFACT",
    "STATE_SCHEMA",
    "WORKER_EXECUTOR",
    "WORKER_SKILLS",
    "BrowserApprovalError",
    "BrowserCompletionError",
    "BrowserSlice",
    "BrowserSliceError",
    "build_worker_prompt",
    "classify_publication_action",
    "publication_fingerprint",
]
