"""Task 13 — capability discovery and consent.

Mission control must be able to *look* without acting: opening the page performs
no network access and no install. Catalog search is a separate, explicit
action; installed checks go through the injected public lookups; and installing
anything requires a consent record bound to the exact capability name and the
exact 40-character commit SHA.

These tests pin the four-step flow from the plan: probe, present disclosure,
explicit consent, standard pinned install/enable. The installer and command
runner are always injected fakes, so no test touches the network or the real
Hermes installation.

Sections 4.2, 4.4, 4.7 and 5 of the implementation plan.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capabilities import (  # noqa: E402
    CATALOG_TIERS,
    OFFICIAL_BLENDER,
    Capability,
    CapabilityDiscovery,
    ConsentRecord,
    ConsentRequiredError,
    ConsentStoreError,
    InMemoryConsentStore,
    InstallRefusedError,
    JsonConsentStore,
    build_install_request,
    parse_catalog_output,
    validate_capability,
)

BLENDER_SHA = "9b11ffeda84b077bea9de52d22733c7a41ca581a"
OTHER_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f001122334"

CATALOG_PAYLOAD = {
    "query": "blender",
    "results": [
        {
            "name": "blender",
            "repo": "https://github.com/NousResearch/hermes-official-plugins",
            "sha": BLENDER_SHA,
            "tier": "official",
            "description": "Official Blender Lab MCP server.",
            "version": "0.4.1",
            "subdir": "blender",
            "maintainer": "NousResearch",
            "category": "tools",
            "capabilities": {"provides_tools": [], "provides_hooks": []},
            "needs_newer_hermes": None,
        }
    ],
}


def catalog_json(**overrides: object) -> str:
    payload = json.loads(json.dumps(CATALOG_PAYLOAD))
    payload["results"][0].update(overrides)
    return json.dumps(payload)


class FakeRunner:
    """A command runner that records argv and returns canned output."""

    def __init__(
        self,
        stdout: object = "",
        returncode: int = 0,
        stderr: str = "",
        raises: object = None,
    ) -> None:
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises

    def __call__(self, argv):
        self.calls.append(list(argv))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            list(argv), self.returncode, stdout=self.stdout, stderr=self.stderr
        )


class FakeInstaller:
    """An installer that records requests and reports a post-install revision."""

    def __init__(
        self,
        verified_sha: object = None,
        receipt: object = None,
        raises: object = None,
        omit_verification: bool = False,
    ) -> None:
        self.requests = []
        self.verified_sha = verified_sha
        self.receipt = receipt
        self.raises = raises
        self.omit_verification = omit_verification
        self._installed = None

    def install(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        self._installed = request.sha
        if self.receipt is not None:
            return self.receipt
        return {"ok": True, "name": request.name, "sha": request.sha}

    def installed_sha(self, name):
        if self.omit_verification:
            raise AssertionError("installed_sha must not be called in this test")
        if self.verified_sha is not None:
            return self.verified_sha
        return self._installed


def parsed_blender() -> Capability:
    capabilities, unavailable = parse_catalog_output(catalog_json())
    assert not unavailable, unavailable
    return capabilities[0]


class DiscoveryInstalledTests(unittest.TestCase):
    def test_installed_only_lists_plugins_and_skills_offline(self):
        runner = FakeRunner(raises=AssertionError("network must not be used"))
        discovery = CapabilityDiscovery(
            run_command=runner,
            has_plugin=lambda name: name == "alpha",
            list_plugins=lambda: ["alpha", "beta"],
            is_plugin_enabled=lambda name: name == "alpha",
            find_skills=lambda: [{"name": "neo-worker"}, "notes"],
        )
        snapshot = discovery.list_available(installed_only=True)
        self.assertEqual(runner.calls, [])
        self.assertEqual(discovery.network_calls, 0)
        self.assertFalse(snapshot.network_used)
        self.assertFalse(snapshot.catalog_requested)
        self.assertEqual(snapshot.catalog, ())
        self.assertEqual(
            {(item.name, item.kind) for item in snapshot.installed},
            {("alpha", "plugin"), ("neo-worker", "skill"), ("notes", "skill")},
        )
        self.assertNotIn("beta", {item.name for item in snapshot.installed})

    def test_installed_only_ignores_a_catalog_query(self):
        runner = FakeRunner(stdout=catalog_json())
        discovery = CapabilityDiscovery(run_command=runner, has_plugin=lambda n: False)
        snapshot = discovery.list_available(query="blender", installed_only=True)
        self.assertEqual(runner.calls, [])
        self.assertEqual(snapshot.unavailable, ())
        self.assertEqual(snapshot.installed, ())

    def test_explicit_catalog_search_runs_the_documented_command(self):
        runner = FakeRunner(stdout=catalog_json())
        discovery = CapabilityDiscovery(run_command=runner, has_plugin=lambda n: False)
        snapshot = discovery.search_catalog("blender")
        self.assertEqual(
            runner.calls, [["hermes", "plugins", "search", "blender", "--json"]]
        )
        self.assertTrue(snapshot.catalog_requested)
        self.assertTrue(snapshot.network_used)
        self.assertEqual(len(snapshot.catalog), 1)
        self.assertEqual(snapshot.catalog[0].name, "blender")
        self.assertEqual(snapshot.catalog[0].sha, BLENDER_SHA)

    def test_list_available_with_catalog_is_explicit(self):
        runner = FakeRunner(stdout=catalog_json())
        discovery = CapabilityDiscovery(run_command=runner, has_plugin=lambda n: False)
        snapshot = discovery.list_available(query="blender", installed_only=False)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(snapshot.catalog), 1)

    def test_layout_is_json_safe(self):
        runner = FakeRunner(stdout=catalog_json())
        discovery = CapabilityDiscovery(run_command=runner, has_plugin=lambda n: False)
        payload = discovery.search_catalog("blender").to_dict()
        self.assertEqual(payload["catalog"][0]["name"], "blender")
        json.dumps(payload)


class CatalogParsingTests(unittest.TestCase):
    def test_real_cli_shape_parses_with_extra_fields(self):
        capabilities, unavailable = parse_catalog_output(catalog_json())
        self.assertEqual(unavailable, ())
        capability = capabilities[0]
        self.assertEqual(capability.name, "blender")
        self.assertEqual(capability.repo, OFFICIAL_BLENDER.repo)
        self.assertEqual(capability.sha, BLENDER_SHA)
        self.assertEqual(capability.tier, "official")
        self.assertEqual(capability.version, "0.4.1")
        self.assertEqual(capability.subdir, "blender")
        self.assertEqual(capability.dependencies, ())

    def test_bytes_and_whitespace_are_accepted(self):
        capabilities, unavailable = parse_catalog_output(
            ("\n" + catalog_json() + "\n").encode("utf-8")
        )
        self.assertEqual(unavailable, ())
        self.assertEqual(len(capabilities), 1)

    def test_malformed_json_is_unavailable(self):
        capabilities, unavailable = parse_catalog_output("{not json")
        self.assertEqual(capabilities, ())
        self.assertEqual(len(unavailable), 1)
        self.assertIn("JSON", unavailable[0])

    def test_missing_results_array_is_unavailable(self):
        capabilities, unavailable = parse_catalog_output(json.dumps({"query": "x"}))
        self.assertEqual(capabilities, ())
        self.assertEqual(len(unavailable), 1)

    def test_partial_entry_is_unavailable(self):
        for missing in ("name", "repo", "sha", "tier"):
            payload = json.loads(catalog_json())
            del payload["results"][0][missing]
            capabilities, unavailable = parse_catalog_output(json.dumps(payload))
            self.assertEqual(capabilities, (), missing)
            self.assertEqual(len(unavailable), 1, missing)

    def test_invalid_fields_are_unavailable(self):
        cases = {
            "name": "Bad Name",
            "sha": BLENDER_SHA[:12],
            "tier": "gold",
            "version": "a version with spaces",
            "repo": "http://github.com/example/plugin",
            "subdir": "../escape",
        }
        for field, value in cases.items():
            capabilities, unavailable = parse_catalog_output(catalog_json(**{field: value}))
            self.assertEqual(capabilities, (), field)
            self.assertEqual(len(unavailable), 1, field)

    def test_one_bad_entry_does_not_hide_a_good_one(self):
        payload = json.loads(catalog_json())
        bad = dict(payload["results"][0])
        bad.update({"name": "broken", "sha": "not-a-sha"})
        payload["results"].append(bad)
        capabilities, unavailable = parse_catalog_output(json.dumps(payload))
        self.assertEqual([c.name for c in capabilities], ["blender"])
        self.assertEqual(len(unavailable), 1)

    def test_duplicate_names_keep_the_first_entry(self):
        payload = json.loads(catalog_json())
        duplicate = dict(payload["results"][0])
        duplicate["version"] = "9.9.9"
        payload["results"].append(duplicate)
        capabilities, unavailable = parse_catalog_output(json.dumps(payload))
        self.assertEqual(len(capabilities), 1)
        self.assertEqual(capabilities[0].version, "0.4.1")
        self.assertEqual(len(unavailable), 1)

    def test_search_failure_is_unavailable(self):
        runner = FakeRunner(stdout="", returncode=2, stderr="boom")
        discovery = CapabilityDiscovery(run_command=runner)
        snapshot = discovery.search_catalog("blender")
        self.assertEqual(snapshot.catalog, ())
        self.assertEqual(len(snapshot.unavailable), 1)

    def test_runner_exception_is_unavailable(self):
        runner = FakeRunner(raises=OSError("no hermes"))
        discovery = CapabilityDiscovery(run_command=runner)
        snapshot = discovery.search_catalog("blender")
        self.assertEqual(snapshot.catalog, ())
        self.assertEqual(len(snapshot.unavailable), 1)
        self.assertEqual(discovery.network_calls, 1)

    def test_flag_like_queries_are_refused_without_running(self):
        runner = FakeRunner(stdout=catalog_json())
        discovery = CapabilityDiscovery(run_command=runner)
        snapshot = discovery.search_catalog("--allow-removed")
        self.assertEqual(runner.calls, [])
        self.assertEqual(snapshot.catalog, ())
        self.assertEqual(len(snapshot.unavailable), 1)


class BlenderRoutingTests(unittest.TestCase):
    def test_route_ready_only_when_installed_and_enabled(self):
        discovery = CapabilityDiscovery(
            has_plugin=lambda name: name == "blender",
            is_plugin_enabled=lambda name: True,
        )
        decision = discovery.route_blender()
        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["route"], "hermes_worker")
        self.assertEqual(decision["skill"], "blender:blender")
        self.assertFalse(decision["requires_approval"])

    def test_missing_plugin_returns_pinned_disclosure(self):
        runner = FakeRunner(raises=AssertionError("network must not be used"))
        discovery = CapabilityDiscovery(run_command=runner, has_plugin=lambda n: False)
        decision = discovery.route_blender()
        self.assertEqual(discovery.network_calls, 0)
        self.assertEqual(decision["status"], "needs_approval")
        self.assertIsNone(decision["route"])
        self.assertTrue(decision["requires_approval"])
        disclosure = decision["disclosure"]
        self.assertEqual(disclosure["name"], "blender")
        self.assertEqual(disclosure["sha"], BLENDER_SHA)
        self.assertEqual(disclosure["tier"], "official")
        self.assertEqual(disclosure["version"], "0.4.1")
        self.assertTrue(disclosure["installable"])

    def test_disabled_plugin_returns_needs_approval(self):
        discovery = CapabilityDiscovery(
            has_plugin=lambda name: name == "blender",
            is_plugin_enabled=lambda name: False,
        )
        decision = discovery.route_blender()
        self.assertEqual(decision["status"], "needs_approval")
        self.assertEqual(decision["disclosure"]["sha"], BLENDER_SHA)

    def test_official_pin_is_valid(self):
        validate_capability(OFFICIAL_BLENDER)
        self.assertEqual(OFFICIAL_BLENDER.tier, "official")
        self.assertIn(OFFICIAL_BLENDER.tier, CATALOG_TIERS)


class ConsentRecordTests(unittest.TestCase):
    def test_granted_record_is_exact_and_carries_a_nonce(self):
        discovery = CapabilityDiscovery()
        capability = parsed_blender()
        record = discovery.grant_consent(capability, source="mission_control")
        self.assertIsInstance(record, ConsentRecord)
        self.assertEqual(record.capability, "blender")
        self.assertEqual(record.sha, BLENDER_SHA)
        self.assertEqual(record.version, "0.4.1")
        self.assertEqual(record.source, "mission_control")
        self.assertGreaterEqual(len(record.nonce), 16)
        self.assertTrue(record.granted_at)
        self.assertEqual(record.to_dict()["sha"], BLENDER_SHA)
        self.assertEqual(discovery.consent_store.find("blender", BLENDER_SHA), record)

    def test_grants_use_distinct_nonces(self):
        discovery = CapabilityDiscovery()
        first = discovery.grant_consent(parsed_blender(), source="test")
        second = discovery.grant_consent(
            Capability(
                name="other",
                repo="https://github.com/example/other",
                sha=OTHER_SHA,
                tier="community",
                description="Other.",
            ),
            source="test",
        )
        self.assertNotEqual(first.nonce, second.nonce)

    def test_consent_does_not_transfer_to_another_sha_or_name(self):
        discovery = CapabilityDiscovery()
        record = discovery.grant_consent(parsed_blender(), source="test")
        self.assertIsNone(discovery.consent_store.find("blender", OTHER_SHA))
        self.assertIsNone(discovery.consent_store.find("other", BLENDER_SHA))
        self.assertTrue(record.matches("blender", BLENDER_SHA))
        self.assertFalse(record.matches("blender", OTHER_SHA))
        self.assertFalse(record.matches("other", BLENDER_SHA))

    def test_revoked_consent_is_gone(self):
        discovery = CapabilityDiscovery()
        discovery.grant_consent(parsed_blender(), source="test")
        self.assertTrue(discovery.revoke_consent("blender", BLENDER_SHA))
        self.assertIsNone(discovery.consent_store.find("blender", BLENDER_SHA))
        self.assertFalse(discovery.revoke_consent("blender", BLENDER_SHA))


class InstallRefusalTests(unittest.TestCase):
    def setUp(self):
        self.capability = parsed_blender()

    def test_install_without_consent_never_calls_the_installer(self):
        installer = FakeInstaller(verified_sha=self.capability.sha)
        discovery = CapabilityDiscovery(installer=installer)
        with self.assertRaises(ConsentRequiredError):
            discovery.install_capability(self.capability)
        self.assertEqual(installer.requests, [])

    def test_consent_for_another_sha_does_not_authorise_install(self):
        discovery = CapabilityDiscovery()
        discovery.grant_consent(
            Capability(
                name="blender",
                repo=self.capability.repo,
                sha=OTHER_SHA,
                tier="official",
                version="0.4.1",
            ),
            source="test",
        )
        installer = FakeInstaller(verified_sha=self.capability.sha)
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=discovery.consent_store
        )
        with self.assertRaises(ConsentRequiredError):
            discovery.install_capability(self.capability)
        self.assertEqual(installer.requests, [])

    def test_consent_for_another_name_does_not_authorise_install(self):
        discovery = CapabilityDiscovery()
        discovery.grant_consent(
            Capability(
                name="other",
                repo=self.capability.repo,
                sha=self.capability.sha,
                tier="official",
                version="0.4.1",
            ),
            source="test",
        )
        installer = FakeInstaller(verified_sha=self.capability.sha)
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=discovery.consent_store
        )
        with self.assertRaises(ConsentRequiredError):
            discovery.install_capability(self.capability)
        self.assertEqual(installer.requests, [])

    def test_forbidden_flags_are_refused(self):
        for flag in ("--force", "-f", "--allow-removed"):
            with self.subTest(flag=flag):
                with self.assertRaises(InstallRefusedError) as raised:
                    build_install_request(self.capability, extra_args=[flag])
                self.assertIn(flag, str(raised.exception))

    def test_any_extra_installer_flag_is_refused(self):
        with self.assertRaises(InstallRefusedError):
            build_install_request(self.capability, extra_args=["--yes-deps"])

    def test_arbitrary_git_urls_are_refused(self):
        bad_repos = (
            "git@github.com:example/plugin.git",
            "http://github.com/example/plugin",
            "ssh://git@github.com/example/plugin",
            "file:///tmp/plugin",
            "/tmp/plugin",
            "https://user:pass@github.com/example/plugin",
            "https://localhost/plugin",
            "https://127.0.0.1/plugin",
            "https://github.com/example/plugin extra",
        )
        for repo in bad_repos:
            with self.subTest(repo=repo):
                capability = Capability(
                    name="plugin",
                    repo=repo,
                    sha=BLENDER_SHA,
                    tier="community",
                )
                with self.assertRaises(InstallRefusedError):
                    build_install_request(capability)

    def test_unpinned_refs_are_refused(self):
        for sha in ("main", "v0.4.1", BLENDER_SHA[:12], "", BLENDER_SHA.upper() + "x"):
            with self.subTest(sha=sha):
                capability = Capability(
                    name="plugin",
                    repo="https://github.com/example/plugin",
                    sha=sha,
                    tier="community",
                )
                with self.assertRaises(InstallRefusedError):
                    build_install_request(capability)

    def test_unapproved_dependencies_are_refused(self):
        with self.assertRaises(InstallRefusedError) as raised:
            build_install_request(
                self.capability,
                dependencies=["numpy", "requests"],
                approved_dependencies=["numpy"],
            )
        self.assertIn("requests", str(raised.exception))

    def test_approved_dependencies_are_recorded(self):
        request = build_install_request(
            self.capability,
            dependencies=["numpy"],
            approved_dependencies=["numpy"],
        )
        self.assertEqual(request.dependencies, ("numpy",))

    def test_invalid_capability_is_refused(self):
        capability = Capability(
            name="plugin",
            repo="https://github.com/example/plugin",
            sha=BLENDER_SHA,
            tier="unreviewed",
        )
        with self.assertRaises(InstallRefusedError):
            build_install_request(capability)


class InstallFlowTests(unittest.TestCase):
    def setUp(self):
        self.capability = parsed_blender()
        self.discovery = CapabilityDiscovery(
            installer=FakeInstaller(verified_sha=self.capability.sha)
        )
        self.discovery.grant_consent(self.capability, source="test")

    def test_install_builds_a_pinned_request_and_verifies_the_result(self):
        installer = FakeInstaller(verified_sha=self.capability.sha)
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertTrue(result.installed)
        self.assertTrue(result.verified)
        self.assertTrue(result.installer_called)
        self.assertEqual(result.name, "blender")
        self.assertEqual(result.sha, BLENDER_SHA)
        self.assertEqual(len(installer.requests), 1)
        request = installer.requests[0]
        self.assertEqual(request.name, "blender")
        self.assertEqual(request.repo, OFFICIAL_BLENDER.repo)
        self.assertEqual(request.sha, BLENDER_SHA)
        self.assertEqual(request.version, "0.4.1")
        self.assertEqual(request.subdir, "blender")
        self.assertEqual(
            request.argv,
            (
                "hermes",
                "plugins",
                "install",
                "blender",
                "--ref",
                BLENDER_SHA,
                "--enable",
            ),
        )
        for forbidden in ("--force", "-f", "--allow-removed"):
            self.assertNotIn(forbidden, request.argv)

    def test_mismatched_installed_sha_is_reported_honestly(self):
        installer = FakeInstaller(verified_sha=OTHER_SHA)
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)
        self.assertFalse(result.verified)
        self.assertTrue(result.installer_called)
        self.assertIn("did not match", result.reason)

    def test_missing_verification_is_reported_honestly(self):
        installer = FakeInstaller(omit_verification=True)

        def missing(name):
            return None

        installer.installed_sha = missing
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)
        self.assertFalse(result.verified)

    def test_receipt_failure_is_reported_honestly(self):
        installer = FakeInstaller(receipt={"ok": False, "reason": "blocked by policy"})
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)
        self.assertTrue(result.installer_called)

    def test_receipt_with_another_name_is_a_failure(self):
        installer = FakeInstaller(
            receipt={"ok": True, "name": "impostor", "sha": BLENDER_SHA}
        )
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)

    def test_installer_exception_is_reported_honestly(self):
        installer = FakeInstaller(raises=RuntimeError("git clone exploded"))
        discovery = CapabilityDiscovery(
            installer=installer, consent_store=self.discovery.consent_store
        )
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)
        self.assertTrue(result.installer_called)
        self.assertIn("installer failed", result.reason)

    def test_no_installer_configured_refuses(self):
        discovery = CapabilityDiscovery(consent_store=self.discovery.consent_store)
        result = discovery.install_capability(self.capability)
        self.assertFalse(result.installed)
        self.assertFalse(result.installer_called)


class ConsentStoreTests(unittest.TestCase):
    def test_default_store_is_in_memory_and_writes_nothing(self):
        discovery = CapabilityDiscovery()
        self.assertIsInstance(discovery.consent_store, InMemoryConsentStore)
        record = discovery.grant_consent(parsed_blender(), source="test")
        self.assertIsNone(discovery.consent_store.path)
        self.assertIs(discovery.consent_store.find("blender", BLENDER_SHA), record)

    def test_json_store_roundtrip_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plugin-data" / "neo" / "consent.json"
            store = JsonConsentStore(path)
            record = store.save(
                ConsentRecord(
                    capability="blender",
                    sha=BLENDER_SHA,
                    version="0.4.1",
                    granted_at="2026-11-07T12:00:00Z",
                    source="mission_control",
                    nonce="a" * 32,
                )
            )
            self.assertEqual(record.nonce, "a" * 32)
            self.assertTrue(path.is_file())
            self.assertFalse(stat.S_IMODE(os.stat(path).st_mode) & 0o077)
            self.assertEqual(
                [p.name for p in path.parent.iterdir() if p.name != path.name], []
            )
            reloaded = JsonConsentStore(path)
            self.assertEqual(reloaded.find("blender", BLENDER_SHA), record)
            self.assertTrue(reloaded.revoke("blender", BLENDER_SHA))
            self.assertIsNone(JsonConsentStore(path).find("blender", BLENDER_SHA))

    def test_json_store_refuses_malformed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consent.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConsentStoreError):
                JsonConsentStore(path)

    def test_json_store_does_not_accept_partial_records(self):
        for payload in (
            {"version": 1, "records": [{"capability": "blender"}]},
            {"version": 1, "records": [{"capability": "blender", "sha": "short"}]},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "consent.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ConsentStoreError):
                    JsonConsentStore(path)


if __name__ == "__main__":
    unittest.main()
