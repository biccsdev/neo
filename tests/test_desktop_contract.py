"""Static contract tests for the NEO Desktop plugin (plan Tasks 15 and 16).

The Desktop half is an uncompiled ESM file evaluated in the renderer realm, so
most of its contract can be checked without loading Hermes Desktop: the module
must parse as ESM, import only the three sanctioned specifiers, register the
required contribution areas under host-namespaced ids, keep every timer and
listener disposable, and speak only to the NEO backend routes that exist.

The pure helpers exported for those checks live in a marked, SDK-free region of
``desktop/plugin.js``. This suite extracts that region, imports it in Node, and
exercises the state mapping, grouping, URL parsing, preview URI and error text
directly, so the tests prove behaviour rather than the presence of a string.

Run with the system interpreter::

    python3 -m unittest tests.test_desktop_contract -v
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "desktop" / "plugin.js"
BACKEND_PATH = ROOT / "dashboard" / "plugin_api.py"
NODE = shutil.which("node")

PURE_START = "// <pure-helpers>"
PURE_END = "// </pure-helpers>"

ALLOWED_SPECIFIERS = {"@hermes/plugin-sdk", "react", "react/jsx-runtime"}


def _strip_comments(source: str) -> str:
    """Blank out // and /* */ comments, leaving strings and line numbers intact."""
    pattern = re.compile(
        r'("(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`)'
        r"|/\*.*?\*/|(?<![:\\w])//[^\n]*",
        re.DOTALL,
    )
    return pattern.sub(lambda m: m.group(1) or "\n" * m.group(0).count("\n"), source)


class DesktopPluginContractTest(unittest.TestCase):
    """The Desktop half is a single, dependency-free, disposable ESM module."""

    @classmethod
    def setUpClass(cls) -> None:
        if not PLUGIN_PATH.exists():
            raise AssertionError(f"missing desktop plugin: {PLUGIN_PATH}")
        cls.source = PLUGIN_PATH.read_text(encoding="utf-8")
        cls.code = _strip_comments(cls.source)

    # -- module shape ------------------------------------------------------

    def test_module_parses_as_esm(self) -> None:
        """Node parses the file as an ES module without resolving any import."""
        if NODE is None:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [NODE, "--input-type=module", "--check", "-"],
            input=self.source,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_specifiers_are_resolved_without_the_sdk(self) -> None:
        """A SourceTextModule parse exposes exactly the three allowed specifiers."""
        if NODE is None:
            self.skipTest("node is not installed")
        script = textwrap.dedent(
            """
            import vm from 'node:vm'
            import fs from 'node:fs'
            const module = new vm.SourceTextModule(fs.readFileSync(process.argv[2], 'utf8'))
            console.log(JSON.stringify(module.dependencySpecifiers))
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            driver = Path(tmp) / "parse.mjs"
            driver.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [NODE, "--experimental-vm-modules", str(driver), str(PLUGIN_PATH)],
                text=True,
                capture_output=True,
                timeout=60,
            )
        if result.returncode != 0:
            self.skipTest(f"vm.SourceTextModule unavailable: {result.stderr.strip()[:200]}")
        specifiers = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(sorted(specifiers), sorted(ALLOWED_SPECIFIERS))

    def test_default_export_is_the_neo_plugin(self) -> None:
        self.assertRegex(
            self.code,
            r"export\s+default\s*\{[^}]*\bid:\s*['\"]neo['\"]",
            "default export must declare id 'neo'",
        )
        self.assertRegex(self.code, r"name:\s*['\"]NEO['\"]")
        self.assertRegex(self.code, r"register\s*\(\s*ctx\s*\)|register\s*:\s*\(?\s*ctx")

    def test_all_imports_are_static_and_allowed(self) -> None:
        specifiers = re.findall(
            r"^\s*import\s+(?:[^;'\"]*?\bfrom\s*)?['\"]([^'\"]+)['\"]",
            self.code,
            re.MULTILINE,
        )
        self.assertTrue(specifiers, "the plugin must import the SDK")
        self.assertEqual(set(specifiers), ALLOWED_SPECIFIERS)
        self.assertNotRegex(self.code, r"\bimport\s*\(")
        self.assertNotRegex(self.code, r"\brequire\s*\(")

    def test_no_jsx_syntax(self) -> None:
        self.assertNotIn("</", self.code, "closing JSX tags cannot parse in a disk plugin")
        self.assertNotRegex(self.code, r"<[A-Za-z][\w.]*\s*/>")
        self.assertNotRegex(self.code, r"<[A-Z][\w.]*\s+[^<>]*>")
        self.assertIn("jsx(", self.code)

    def test_pure_helpers_are_exported(self) -> None:
        for name in (
            "normalizeState",
            "groupTasks",
            "buildPreviewUri",
            "taskIdFromRoute",
            "describeRestError",
            "activeTaskCount",
        ):
            self.assertRegex(
                self.code,
                rf"\bexport\s+(?:function\s+{name}\b|const\s+{name}\b)",
                f"{name} must be exported for tests",
            )

    # -- contributions -----------------------------------------------------

    def test_required_areas_are_registered(self) -> None:
        for token in (
            "ROUTES_AREA",
            "SIDEBAR_NAV_AREA",
            "STATUSBAR_AREAS",
            "PALETTE_AREA",
            "TOOL_MEDIA_RENDERER_AREA",
        ):
            self.assertIn(f"sdk.{token}", self.code, f"{token} must come from the SDK")
        # Fallback literals keep the plugin loadable on a host that predates an export.
        for literal in (
            "'routes'",
            "'sidebar.nav'",
            "'statusBar.right'",
            "'palette'",
            "'transcript.toolMedia'",
        ):
            self.assertIn(literal, self.code, f"missing host fallback {literal}")
        self.assertIn("registerMany(", self.code)

    def test_contributions_use_local_ids(self) -> None:
        for local_id in ("page", "nav", "chip", "open", "media"):
            self.assertRegex(
                self.code,
                rf"\bid:\s*['\"]{local_id}['\"]",
                f"contribution id '{local_id}' must be registered",
            )
        self.assertNotRegex(
            self.code,
            r"\bid:\s*['\"]neo:(?:page|nav|chip|open|media)['\"]",
            "the host namespaces contribution ids; write the local id only",
        )

    def test_page_route_and_nav_point_at_neo(self) -> None:
        self.assertRegex(self.code, r"data:\s*\{\s*path:\s*['\"]/neo['\"]")
        self.assertRegex(self.code, r"path:\s*routePath\s*\|\|\s*['\"]/neo['\"]|path:\s*['\"]/neo['\"]")
        self.assertRegex(self.code, r"label:\s*['\"]NEO['\"]")
        self.assertRegex(self.code, r"codicon:\s*['\"][\w-]+['\"]")

    def test_palette_command_navigates_to_mission_control(self) -> None:
        self.assertRegex(self.code, r"id:\s*['\"]neo\.open['\"]")
        self.assertRegex(self.code, r"host\.navigate\(\s*['\"]/neo['\"]\s*\)")

    def test_tool_media_renderer_names_neo_tools(self) -> None:
        self.assertIn("neo_preview_publish", self.code)
        self.assertIn("neo_task_complete", self.code)
        self.assertRegex(self.code, r"tools:\s*NEO_MEDIA_TOOLS")
        self.assertIn("resolveUrl(", self.code)

    # -- forbidden surfaces ------------------------------------------------

    def test_no_forbidden_desktop_surfaces(self) -> None:
        forbidden = {
            "direct fetch": r"\bfetch\s*\(",
            "XMLHttpRequest": r"\bXMLHttpRequest\b",
            "eval": r"(?<![\w$.])eval\s*\(",
            "new Function": r"\bnew\s+Function\s*\(",
            "localStorage": r"\blocalStorage\b",
            "sessionStorage": r"\bsessionStorage\b",
            "DOM selector": r"\bdocument\s*\.\s*(?:querySelector|querySelectorAll|getElementById|getElementsBy)",
            "synthetic keyboard event": r"\bKeyboardEvent\b",
            "synthetic event dispatch": r"\bdispatchEvent\b",
            "innerHTML": r"\binnerHTML\b",
            "bare timer": r"(?<!ctx\.)\b(?:setInterval|setTimeout)\s*\(",
            "bare window listener": r"\bwindow\s*\.\s*(?:setInterval|setTimeout|addEventListener|dispatchEvent|open)\b",
            "script element": r"\bcreateElement\s*\(\s*['\"]script",
            "global stylesheet": r"\b(?:insertRule|adoptedStyleSheets|document\s*\.\s*head)\b",
            "style tag": r"<style\b",
        }
        hits = [label for label, pattern in forbidden.items() if re.search(pattern, self.code)]
        self.assertEqual(hits, [], f"forbidden Desktop surface used: {hits}")

    def test_polling_and_events_are_disposable(self) -> None:
        self.assertRegex(self.code, r"function\s+usePolling\s*\(", "timers must be owned by one hook")
        self.assertIn("ctx.setInterval", self.code)
        self.assertRegex(
            self.code,
            r"return\s*\(\)\s*=>\s*\{[^}]*stop\(\)",
            "the polling hook must dispose its interval on unmount",
        )
        self.assertEqual(
            len(re.findall(r"^\s+usePolling\(ctx,", self.code, re.MULTILINE)),
            3,
            "page list, selected preview and chip each poll once",
        )
        self.assertIn("onEvent('*'", self.code)
        self.assertIn("plugin.neo.task.", self.code)
        self.assertRegex(
            self.code,
            r"return\s*\(\)\s*=>\s*\{[^}]*typeof stop === 'function'[^}]*stop\(\)",
            "the event subscription must be disposed on unmount",
        )
        self.assertEqual(self.code.count("3000"), 1, "the preview poll interval is 3 s")

    # -- appearance --------------------------------------------------------

    def test_no_hardcoded_colours(self) -> None:
        hits = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\s*\(|\bhsla?\s*\(", self.code)
        self.assertEqual(hits, [], f"hardcoded colour values: {hits}")
        palette_classes = re.findall(
            r"\b(?:bg|text|border|ring)-(?:white|black|slate|gray|zinc|red|green|blue|emerald|amber)-\d{2,3}\b",
            self.code,
        )
        self.assertEqual(palette_classes, [], f"palette colour classes: {palette_classes}")

    def test_layout_is_two_pane_and_responsive(self) -> None:
        self.assertIn("min-[47.5rem]:flex-row", self.code)
        self.assertIn("flex-col", self.code)
        self.assertNotRegex(self.code, r"\b(?:min-h-screen|h-screen|bg-background)\b")

    # -- backend contract --------------------------------------------------

    def test_rest_paths_match_existing_backend_routes(self) -> None:
        calls = re.findall(r"ctx\.rest\(\s*`([^`]+)`", self.code)
        self.assertTrue(calls, "actions must use ctx.rest")
        normalised = set()
        for call in calls:
            path = call.split("?")[0]
            path = re.sub(r"\$\{[^}]*\}", "{id}", path)
            normalised.add(path)
        expected = {
            "/health",
            "/tasks",
            "/tasks/{id}",
            "/tasks/{id}/cancel",
            "/tasks/{id}/retry",
            "/tasks/{id}/preview",
            "/tasks/{id}/log",
            "/tasks/{id}/capabilities",
            "/tasks/{id}/artifact",
            "/tasks/{id}/approve",
            "/tasks/{id}/present",
            "/capabilities",
            "/capabilities/consent",
            "/capabilities/install",
        }
        self.assertEqual(normalised, expected, f"unexpected ctx.rest routes: {sorted(normalised)}")

        backend = BACKEND_PATH.read_text(encoding="utf-8")
        backend_routes = {
            path.replace("{task_id}", "{id}")
            for path in re.findall(r"@router\.(?:get|post)\(\s*[\"']([^\"']+)[\"']", backend)
        }
        missing = sorted(route for route in normalised if route not in backend_routes)
        self.assertEqual(missing, [], f"plugin calls routes the backend does not serve: {missing}")

    def test_actions_cover_cancel_retry_open_approve_install(self) -> None:
        for fragment in (
            "/cancel",
            "/retry",
            "/artifact",
            "/approve",
            "/capabilities/consent",
            "/capabilities/install",
        ):
            self.assertIn(fragment, self.code)
        self.assertIn("decision: 'approve'", self.code)
        self.assertIn("decision: 'deny'", self.code)
        self.assertIn("aria-busy", self.code)
        self.assertRegex(self.code, r"disabled:\s*pending")

    def test_detail_fields_are_rendered(self) -> None:
        for label in (
            "Intent",
            "Capability",
            "Executor",
            "Native session",
            "Workspace",
            "Progress",
            "Approval",
            "Log tail",
            "Artifacts",
            "Preview",
        ):
            self.assertIn(label, self.code, f"detail pane must show {label}")
        self.assertIn("evidence", self.code.lower(), "absent fields must be reported honestly")

    def test_preview_is_base64_data_uri_and_lazy(self) -> None:
        self.assertIn("data_base64", self.code)
        self.assertIn("data:${", self.code)
        self.assertRegex(self.code, r"loading:\s*['\"]lazy['\"]")
        self.assertIn("previewToken", self.code, "only the selected task's preview may be applied")
        self.assertIn("available", self.code)

    # -- pure helper behaviour --------------------------------------------

    def _helpers_source(self) -> str:
        start = self.source.find(PURE_START)
        end = self.source.find(PURE_END)
        self.assertNotEqual(start, -1, "pure helper region start marker is missing")
        self.assertGreater(end, start, "pure helper region end marker is missing")
        return self.source[start + len(PURE_START) : end]

    def _run_helpers(self, body: str) -> dict:
        if NODE is None:
            self.skipTest("node is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            module_path = Path(tmp) / "helpers.mjs"
            module_path.write_text(self._helpers_source(), encoding="utf-8")
            driver_path = Path(tmp) / "driver.mjs"
            driver_path.write_text(
                "import * as helpers from " + json.dumps(Path(module_path).as_uri()) + "\n" + body,
                encoding="utf-8",
            )
            result = subprocess.run(
                [NODE, str(driver_path)],
                text=True,
                capture_output=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_state_mapping_covers_every_internal_state(self) -> None:
        report = self._run_helpers(
            """
const raw = ['queued', 'starting', 'running', 'waiting', 'needs_approval',
  'needs_input', 'cancelling', 'unknown', 'completed', 'failed', 'cancelled', 'not-a-state']
console.log(JSON.stringify({
  order: helpers.STATE_ORDER,
  labels: helpers.STATE_LABELS,
  mapped: raw.map(state => [state, helpers.normalizeState(state)]),
  unknownLabel: helpers.rawStateLabel('unknown')
}))
"""
        )
        self.assertEqual(
            report["order"],
            ["queued", "working", "needs-you", "blocked", "reviewing", "completed", "failed"],
        )
        self.assertEqual(
            report["labels"],
            {
                "queued": "Queued",
                "working": "Working",
                "needs-you": "Needs you",
                "blocked": "Blocked",
                "reviewing": "Reviewing",
                "completed": "Completed",
                "failed": "Failed",
            },
        )
        mapping = {state: bucket for state, bucket in report["mapped"]}
        self.assertEqual(mapping["queued"], "queued")
        self.assertEqual(mapping["starting"], "working")
        self.assertEqual(mapping["running"], "working")
        self.assertEqual(mapping["cancelling"], "working")
        self.assertEqual(mapping["needs_approval"], "needs-you")
        self.assertEqual(mapping["needs_input"], "needs-you")
        self.assertEqual(mapping["waiting"], "blocked")
        self.assertEqual(mapping["unknown"], "blocked")
        self.assertEqual(mapping["completed"], "completed")
        self.assertEqual(mapping["failed"], "failed")
        self.assertEqual(mapping["cancelled"], "failed")
        self.assertEqual(mapping["not-a-state"], "blocked")
        self.assertIn("Unknown", report["unknownLabel"])

    def test_grouping_uses_the_seven_buckets_and_keeps_newest_first(self) -> None:
        report = self._run_helpers(
            """
const tasks = [
  { id: 'a', state: 'running', created_at: 30 },
  { id: 'b', state: 'needs_approval', created_at: 20 },
  { id: 'c', state: 'waiting', created_at: 10 },
  { id: 'd', state: 'waiting', finished_at: 40, created_at: 5 },
  { id: 'e', state: 'completed', created_at: 50 },
  { id: 'f', state: 'running', created_at: 60 },
  null,
  { id: 'g', state: 'bogus' }
]
console.log(JSON.stringify({
  groups: helpers.groupTasks(tasks).map(group => [group.state, group.tasks.map(task => task.id)]),
  active: helpers.activeTaskCount(tasks)
}))
"""
        )
        groups = dict((state, ids) for state, ids in report["groups"])
        self.assertEqual(
            list(groups),
            ["queued", "working", "needs-you", "blocked", "reviewing", "completed", "failed"],
        )
        self.assertEqual(groups["working"], ["f", "a"])
        self.assertEqual(groups["needs-you"], ["b"])
        self.assertEqual(groups["blocked"], ["c", "g"])
        self.assertEqual(groups["reviewing"], ["d"])
        self.assertEqual(groups["completed"], ["e"])
        # An unknown raw state is honest: it counts as blocked, never as a success.
        self.assertEqual(groups["blocked"].count("undefined"), 0)
        self.assertEqual(report["active"], 6)

    def test_preview_uri_is_built_only_from_available_image_bytes(self) -> None:
        report = self._run_helpers(
            """
const cases = [
  ['png', helpers.buildPreviewUri({ available: true, mime: 'image/png', data_base64: 'QUJD' })],
  ['jpeg', helpers.buildPreviewUri({ available: true, mime: 'image/jpeg', data_base64: 'QUJD' })],
  ['unavailable', helpers.buildPreviewUri({ available: false, mime: 'image/png', data_base64: null, reason: 'no preview' })],
  ['html', helpers.buildPreviewUri({ available: true, mime: 'text/html', data_base64: 'QUJD' })],
  ['garbage', helpers.buildPreviewUri({ available: true, mime: 'image/png', data_base64: '<script>' })],
  ['empty', helpers.buildPreviewUri({ available: true, mime: 'image/png', data_base64: '' })],
  ['null', helpers.buildPreviewUri(null)]
]
console.log(JSON.stringify({
  cases,
  placeholder: helpers.previewPlaceholder({ available: false, reason: 'preview is unavailable' })
}))
"""
        )
        cases = dict((name, value) for name, value in report["cases"])
        self.assertEqual(cases["png"], "data:image/png;base64,QUJD")
        self.assertEqual(cases["jpeg"], "data:image/jpeg;base64,QUJD")
        for name in ("unavailable", "html", "garbage", "empty", "null"):
            self.assertIsNone(cases[name], f"{name} must not become a data URI")
        self.assertIn("unavailable", report["placeholder"])

    def test_route_task_id_is_parsed_and_validated(self) -> None:
        report = self._run_helpers(
            """
const routes = ['#/neo?task=abc_123', '/neo?x=1&task=t-9', '/neo', '', '#/neo?task=../../etc',
  '/neo?task=', '/neo?task=' + 'a'.repeat(65)]
console.log(JSON.stringify(routes.map(route => helpers.taskIdFromRoute(route))))
"""
        )
        self.assertEqual(
            report,
            ["abc_123", "t-9", None, None, None, None, None],
        )

    def test_rest_errors_are_reported_honestly(self) -> None:
        report = self._run_helpers(
            """
const cases = [
  '404: {"detail":"task not found"}',
  '409: {"detail":"only a terminal task can be retried"}',
  '422: {"detail":"commit SHA is not a full 40-character SHA"}',
  '500: {"detail":"neo services are unavailable"}',
  'network down'
]
console.log(JSON.stringify(cases.map(message => helpers.parseRestError(message))))
"""
        )
        parsed = report
        self.assertEqual([item["status"] for item in parsed], [404, 409, 422, 500, None])
        self.assertEqual(parsed[0]["detail"], "task not found")
        self.assertEqual(parsed[1]["detail"], "only a terminal task can be retried")
        self.assertIn("SHA", parsed[2]["detail"])
        self.assertEqual(parsed[3]["status"], 500)
        self.assertIsNone(parsed[4]["status"])


    def test_completion_identity_and_quiet_setting_are_pure(self) -> None:
        report = self._run_helpers(
            """
const tasks = [
  { id: 'a', state: 'completed', version: 3 },
  { id: 'b', state: 'running', version: 1 },
  { id: 'a', state: 'completed', version: 3 },
  { id: 'c', state: 'completed', version: 0 },
  { id: 'bad id', state: 'completed', version: 1 },
  { id: 'd', state: 'completed', version: 2 },
  null
]
const seen = new Set(['a:3'])
const first = helpers.newlyCompleted(tasks, seen)
const grown = new Set([...seen, 'c:0', 'd:2'])
const second = helpers.newlyCompleted(tasks, grown)
console.log(JSON.stringify({
  keys: tasks.map(task => helpers.completionKey(task)),
  fresh: first.map(task => task.id),
  again: second.map(task => task.id),
  quiet: [true, '1', 'true', 'quiet', 'on', false, '0', '', null].map(value => helpers.isQuietValue(value))
}))
"""
        )
        self.assertEqual(
            report["keys"],
            ["a:3", None, "a:3", "c:0", None, "d:2", None],
        )
        self.assertEqual(report["fresh"], ["c", "d"])
        self.assertEqual(report["again"], [], "an observed version must not present twice")
        self.assertEqual(
            report["quiet"],
            [True, True, True, True, True, False, False, False, False],
        )

    def test_preview_polling_touches_only_the_preview_route_every_three_seconds(self) -> None:
        self.assertIn("PREVIEW_POLL_MS = 3000", self.code)
        self.assertRegex(
            self.code, r"usePolling\(ctx,\s*pollPreview,\s*PREVIEW_POLL_MS\)"
        )
        block = re.search(
            r"const pollPreview = useCallback\((.*?)\n  \}, \[ctx\]\)",
            self.code,
            re.DOTALL,
        )
        self.assertIsNotNone(block, "the selected preview needs its own poll")
        body = block.group(1)
        self.assertIn("/preview", body)
        self.assertNotIn("/log", body)
        self.assertNotIn("/capabilities", body)
        self.assertNotRegex(body, r"/tasks/\$\{encodeURIComponent\(id\)\}`")

    def test_proactive_presentation_is_storage_backed_and_quiet_aware(self) -> None:
        self.assertRegex(
            self.code,
            r"ctx\.rest\(`/tasks/\$\{encodeURIComponent\([^`]+\)\}/present`",
            "proactive presentation must call the present route",
        )
        self.assertIn("ctx.storage", self.code, "presented keys must persist where possible")
        self.assertIn("neo.presented", self.code)
        self.assertIn("neo.quiet", self.code)
        self.assertIn("/neo?task=", self.code, "completion must navigate to the task")
        self.assertIn("host.navigate(", self.code)
        self.assertIn("placeholder", self.code.lower())
        self.assertIn("completion", self.code.lower())
        # Proactive work may present and navigate; it must never approve or publish.
        self.assertNotRegex(
            self.code,
            r"present`[^`]*/approve|/approve[^`]*present`",
        )


if __name__ == "__main__":
    unittest.main()
