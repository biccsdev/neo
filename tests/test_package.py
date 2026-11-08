"""Task 2 — the neo package manifest and its clean-load contract.

These tests treat the repository root as the installable package, exactly as
Hermes does: one ``plugin.yaml`` beside one ``__init__.py`` that exposes
``register(ctx)``.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "plugin.yaml"
ENTRYPOINT_PATH = REPO_ROOT / "__init__.py"
REPOSITORY_URL = "https://github.com/biccsdev/neo"

MODULE_NAME = "neo_package_under_test"


def load_manifest() -> dict:
    """Return the parsed root manifest, or fail with the reason it is unusable."""
    if not MANIFEST_PATH.is_file():
        raise AssertionError("plugin.yaml is missing at the repository root")
    data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("plugin.yaml must contain a YAML mapping")
    return data


def load_entrypoint():
    """Import the root ``__init__.py`` as a package, the way Hermes loads a plugin directory."""
    if not ENTRYPOINT_PATH.is_file():
        raise AssertionError("__init__.py is missing at the repository root")
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ENTRYPOINT_PATH,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise AssertionError("__init__.py is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


class RecordingContext:
    """Minimal stand-in for Hermes' plugin context: records instead of wiring anything."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.hooks: list[str] = []
        self.skills: list[str] = []

    def register_tool(self, name: str | None = None, **_kwargs) -> None:
        self.tools.append(str(name))

    def register_hook(self, hook_name: str | None = None, _callback=None, **_kwargs) -> None:
        self.hooks.append(str(hook_name))

    def register_skill(self, name: str | None = None, _path=None, **_kwargs) -> None:
        self.skills.append(str(name))


class ManifestMetadataTests(unittest.TestCase):
    def test_manifest_name_is_neo(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest.get("name"), "neo")

    def test_manifest_metadata_for_initial_macos_release(self) -> None:
        manifest = load_manifest()

        version = str(manifest.get("version", ""))
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(version, "0.1.0", "the initial macOS-validated release starts at 0.1.0")

        self.assertTrue(str(manifest.get("description", "")).strip())
        self.assertEqual(manifest.get("platforms"), ["macos"])

        self.assertEqual(manifest.get("manifest_version"), 2)
        self.assertEqual(manifest.get("api_version"), 1)

    def test_manifest_declares_repository_and_licence(self) -> None:
        manifest = load_manifest()

        self.assertEqual(manifest.get("license"), "MIT")
        # ``homepage`` is Hermes' manifest field for the project URL; a
        # ``repository`` key would be an unknown field and warn on every load.
        self.assertEqual(manifest.get("homepage"), REPOSITORY_URL)

    def test_manifest_requests_no_privileged_capabilities(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest.get("capabilities") or [], [])

    def test_manifest_declares_the_conversational_tools_and_hook(self) -> None:
        manifest = load_manifest()
        self.assertEqual(
            list(manifest.get("provides_tools") or []),
            [
                "neo_submit_task",
                "neo_task_status",
                "neo_task_list",
                "neo_task_refine",
                "neo_task_cancel",
                "neo_task_frame_follow_up",
                "neo_task_open",
            ],
        )
        self.assertEqual(list(manifest.get("provides_hooks") or []), ["pre_llm_call"])


class RegisterEntrypointTests(unittest.TestCase):
    def test_register_succeeds_with_recording_context(self) -> None:
        module = load_entrypoint()
        self.addCleanup(sys.modules.pop, MODULE_NAME, None)

        register = getattr(module, "register", None)
        if not callable(register):
            self.fail("__init__.py must expose register(ctx)")

        ctx = RecordingContext()
        self.assertIsNone(register(ctx))
        self.assertEqual(
            ctx.tools,
            [
                "neo_submit_task",
                "neo_task_status",
                "neo_task_list",
                "neo_task_refine",
                "neo_task_cancel",
                "neo_task_frame_follow_up",
                "neo_task_open",
            ],
        )
        self.assertEqual(ctx.hooks, ["pre_llm_call"])
        self.assertEqual(ctx.skills, ["neo-operator", "neo-worker"])

    def test_package_version_matches_manifest(self) -> None:
        module = load_entrypoint()
        self.addCleanup(sys.modules.pop, MODULE_NAME, None)
        self.assertEqual(module.__version__, str(load_manifest()["version"]))


if __name__ == "__main__":
    unittest.main()
