"""Capability discovery, consent, and pinned install for NEO.

Mission control may *show* capabilities without acting. Opening the page calls
:meth:`CapabilityDiscovery.list_available` with ``installed_only=True`` (the
default), which consults the injected ``has_plugin`` / skill lookups and the
injected installed-name source only. It never runs a command and never installs
anything. Catalog search is a separate explicit call,
:meth:`CapabilityDiscovery.search_catalog`, which runs
``hermes plugins search <query> --json`` through the injected runner.

Catalog output is untrusted input. Every entry is validated against the real
CLI shape (``{"query": ..., "results": [...]}`` with ``name``, ``repo``,
``sha``, ``tier``, ``description``, ``version`` and ``subdir``); a malformed or
partial entry is reported as unavailable and never becomes an installable
capability.

Installing is a four-step flow: probe, present disclosure, explicit consent,
standard pinned install/enable. A :class:`ConsentRecord` binds one capability
name to one verified 40-character commit SHA (plus optional version, granting
timestamp, source and a random nonce); a record for one name/SHA never
authorises another. Requests are built by :func:`build_install_request`, which
refuses arbitrary Git URLs, unpinned refs, mismatched SHAs, unapproved
dependencies and any installer flag such as ``--force`` or
``--allow-removed``. Only after a matching consent record is found is the
*injected* installer called; its post-install revision is then verified by name
and SHA and any mismatch is reported honestly rather than claimed as success.

Consent persistence uses a small injected store protocol: the default is
in-memory (nothing is written anywhere), or an explicit
:class:`JsonConsentStore` path written atomically with owner-only permissions.
The module never edits profile configuration.

Sections 4.2, 4.4, 4.7 and 5 of the implementation plan (Task 13).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union
from urllib.parse import urlsplit

CATALOG_TIERS = ("official", "verified", "community")
"""Tier names accepted by the Hermes plugin catalog."""

FORBIDDEN_INSTALL_FLAGS = frozenset({"--force", "-f", "--allow-removed"})
"""Flags that must never reach an installer, whatever the caller asks."""

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
_DEPENDENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SUBDIR_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_MAX_REASON = 200
_MAX_QUERY = 128
_MAX_NONCE = 128


class InstallRefusedError(Exception):
    """A refused install request; ``reason`` is short and path-free."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ConsentRequiredError(InstallRefusedError):
    """No consent record exists for the exact capability name and SHA."""


class ConsentStoreError(RuntimeError):
    """The persisted consent store is unreadable or malformed."""


class ConsentStore(Protocol):
    """The minimal persistence surface consent needs.

    ``save`` is idempotent for an existing ``(capability, sha)`` pair, ``find``
    is an exact match on both fields, and ``revoke`` removes one exact pair.
    """

    path: Optional[Path]

    def save(self, record: "ConsentRecord") -> "ConsentRecord":
        ...

    def find(self, capability: str, sha: str) -> Optional["ConsentRecord"]:
        ...

    def records(self) -> Tuple["ConsentRecord", ...]:
        ...

    def revoke(self, capability: str, sha: str) -> bool:
        ...


@dataclass(frozen=True)
class Capability:
    """One validated catalog capability, pinned to an exact commit SHA."""

    name: str
    repo: str
    sha: str
    tier: str
    description: str = ""
    version: str = ""
    subdir: str = ""
    dependencies: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def install_identifier(self) -> str:
        return "%s#%s" % (self.repo, self.subdir) if self.subdir else self.repo

    @property
    def installable(self) -> bool:
        try:
            validate_capability(self)
        except InstallRefusedError:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "repo": self.repo,
            "sha": self.sha,
            "tier": self.tier,
            "description": self.description,
            "version": self.version,
            "subdir": self.subdir,
            "dependencies": list(self.dependencies),
            "installable": self.installable,
        }


def _official_blender() -> Capability:
    """The pinned official Blender entry shipped with this module.

    Snapshot of the curated Hermes catalog entry (official tier). Overridable
    through ``CapabilityDiscovery(official_blender=...)`` so a newer pin can be
    presented without editing this module.
    """
    return Capability(
        name="blender",
        repo="https://github.com/NousResearch/hermes-official-plugins",
        sha="9b11ffeda84b077bea9de52d22733c7a41ca581a",
        tier="official",
        description=(
            "Official Blender Lab MCP server and workflow skill. Its local bridge "
            "on port 9876 is unauthenticated while Blender is open."
        ),
        version="0.4.1",
        subdir="blender",
    )


OFFICIAL_BLENDER = _official_blender()
"""Pinned official ``blender`` capability used for disclosure and routing."""

BLENDER_DISCLOSURE_POINTS = (
    "The Blender worker's tools execute arbitrary Python inside Blender; NEO "
    "does not sandbox that code or limit which files Blender can reach.",
    "The Blender session is online: the worker's model provider receives the "
    "task text and whatever scene data the worker reads.",
    "The plugin's local bridge listens on unauthenticated 127.0.0.1:9876 while "
    "Blender is open, so any local process can drive that session.",
    "Installing downloads the pinned plugin package from the catalog. The "
    "download size is not pinned in advance; the installer reports it.",
)
"""Everything the user must see before consenting to the Blender capability."""


def blender_disclosure(capability: Optional[Capability] = None) -> dict:
    """Return the pinned Blender disclosure: identity, pin and every risk."""

    cap = capability or OFFICIAL_BLENDER
    data = cap.to_dict()
    data["pinned"] = True
    data["consent_required"] = True
    data["points"] = list(BLENDER_DISCLOSURE_POINTS)
    return data



@dataclass(frozen=True)
class InstalledCapability:
    """One capability already present in the Hermes installation."""

    name: str
    kind: str  # "plugin" or "skill"
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "enabled": self.enabled}


@dataclass(frozen=True)
class DiscoverySnapshot:
    """A read-only view of installed and (optionally) catalog capabilities."""

    installed: Tuple[InstalledCapability, ...] = ()
    catalog: Tuple[Capability, ...] = ()
    unavailable: Tuple[str, ...] = ()
    network_used: bool = False
    catalog_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "installed": [item.to_dict() for item in self.installed],
            "catalog": [item.to_dict() for item in self.catalog],
            "unavailable": list(self.unavailable),
            "network_used": self.network_used,
            "catalog_requested": self.catalog_requested,
        }


@dataclass(frozen=True)
class ConsentRecord:
    """Exact, non-transferable consent for one capability name and SHA."""

    capability: str
    sha: str
    granted_at: str
    source: str
    nonce: str
    version: str = ""

    def __post_init__(self) -> None:
        _validate_identity(self.capability, self.sha)
        if not isinstance(self.granted_at, str) or not self.granted_at.strip():
            raise ValueError("consent timestamp is required")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("consent source is required")
        if (
            not isinstance(self.nonce, str)
            or not self.nonce.strip()
            or len(self.nonce) > _MAX_NONCE
        ):
            raise ValueError("consent nonce is required")
        if self.version and not _VERSION_RE.match(self.version):
            raise ValueError("consent version is malformed")

    def matches(self, capability: str, sha: str) -> bool:
        return self.capability == capability and self.sha == sha.lower()

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "sha": self.sha,
            "version": self.version,
            "granted_at": self.granted_at,
            "source": self.source,
            "nonce": self.nonce,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ConsentRecord":
        if not isinstance(data, Mapping):
            raise ConsentStoreError("consent record must be a mapping")
        try:
            return cls(
                capability=data["capability"],
                sha=data["sha"],
                version=data.get("version") or "",
                granted_at=data["granted_at"],
                source=data["source"],
                nonce=data["nonce"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConsentStoreError("consent record is malformed: %s" % exc) from None


def _validate_identity(capability: Any, sha: Any) -> None:
    if not isinstance(capability, str) or not _NAME_RE.match(capability):
        raise ValueError("capability name is malformed")
    if capability.startswith("-"):
        raise ValueError("capability name is malformed")
    if not isinstance(sha, str) or not _SHA_RE.match(sha.lower()):
        raise ValueError("sha must be a full 40-character commit SHA")


class InMemoryConsentStore:
    """The default store: process-local, writes nothing outside memory."""

    path: Optional[Path] = None

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records = {}

    def save(self, record: ConsentRecord) -> ConsentRecord:
        if not isinstance(record, ConsentRecord):
            raise TypeError("record must be a ConsentRecord")
        key = (record.capability, record.sha)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            self._records[key] = record
            return record

    def find(self, capability: str, sha: str) -> Optional[ConsentRecord]:
        with self._lock:
            return self._records.get((capability, str(sha).lower()))

    def records(self) -> Tuple[ConsentRecord, ...]:
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda r: (r.capability, r.sha)))

    def revoke(self, capability: str, sha: str) -> bool:
        with self._lock:
            return self._records.pop((capability, str(sha).lower()), None) is not None


class JsonConsentStore:
    """An explicit JSON-file store written atomically with owner-only permissions."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._records = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            info = os.lstat(self.path)
        except OSError as exc:
            raise ConsentStoreError("consent store is unreadable") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ConsentStoreError("consent store must be a regular file")
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ConsentStoreError("consent store is malformed") from exc
        if not isinstance(data, Mapping) or data.get("version") != 1:
            raise ConsentStoreError("consent store has an unsupported shape")
        entries = data.get("records")
        if not isinstance(entries, list):
            raise ConsentStoreError("consent store has an unsupported shape")
        records = {}
        for entry in entries:
            record = ConsentRecord.from_dict(entry)
            records[(record.capability, record.sha)] = record
        return records

    def _write(self, records: Mapping[Tuple[str, str], ConsentRecord]) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "version": 1,
            "records": [
                record.to_dict()
                for record in sorted(records.values(), key=lambda r: (r.capability, r.sha))
            ],
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            dir=str(parent), prefix=".%s." % self.path.name, suffix=".tmp"
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, str(self.path))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        try:
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def save(self, record: ConsentRecord) -> ConsentRecord:
        if not isinstance(record, ConsentRecord):
            raise TypeError("record must be a ConsentRecord")
        key = (record.capability, record.sha)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            updated = dict(self._records)
            updated[key] = record
            self._write(updated)
            self._records = updated
            return record

    def find(self, capability: str, sha: str) -> Optional[ConsentRecord]:
        with self._lock:
            return self._records.get((capability, str(sha).lower()))

    def records(self) -> Tuple[ConsentRecord, ...]:
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda r: (r.capability, r.sha)))

    def revoke(self, capability: str, sha: str) -> bool:
        with self._lock:
            key = (capability, str(sha).lower())
            if key not in self._records:
                return False
            updated = dict(self._records)
            del updated[key]
            self._write(updated)
            self._records = updated
            return True


@dataclass(frozen=True)
class InstallRequest:
    """A validated, pinned install/enable request for an injected installer."""

    name: str
    repo: str
    sha: str
    version: str
    subdir: str
    dependencies: Tuple[str, ...]
    argv: Tuple[str, ...]

    @property
    def install_identifier(self) -> str:
        return "%s#%s" % (self.repo, self.subdir) if self.subdir else self.repo

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "repo": self.repo,
            "sha": self.sha,
            "version": self.version,
            "subdir": self.subdir,
            "dependencies": list(self.dependencies),
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class InstallResult:
    """The honest outcome of one install attempt."""

    installed: bool
    name: str
    sha: str
    reason: str = ""
    verified: bool = False
    installer_called: bool = False

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "name": self.name,
            "sha": self.sha,
            "reason": self.reason,
            "verified": self.verified,
            "installer_called": self.installer_called,
        }


class Installer(Protocol):
    """The only way an installation happens; injected, never imported."""

    def install(self, request: InstallRequest) -> Any:
        ...

    def installed_sha(self, name: str) -> Optional[str]:
        ...


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _short_text(value: Any, limit: int = _MAX_REASON) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _repo_problem(repo: Any) -> Optional[str]:
    if not isinstance(repo, str) or not repo:
        return "repo must be an https:// URL"
    if repo != repo.strip() or _CONTROL_RE.search(repo) or any(ch.isspace() for ch in repo):
        return "repo must be an https:// URL"
    if "\\" in repo:
        return "repo must be an https:// URL"
    parts = urlsplit(repo)
    if parts.scheme != "https" or not parts.hostname:
        return "repo must be an https:// URL"
    if parts.username or parts.password:
        return "repo must not embed credentials"
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return "repo host is not a public repository host"
    if _IPV4_RE.match(host) or ":" in host:
        return "repo host is not a public repository host"
    if "." not in host:
        return "repo host is not a public repository host"
    return None


def _subdir_problem(subdir: Any) -> Optional[str]:
    if subdir in (None, ""):
        return None
    if not isinstance(subdir, str):
        return "subdir must be a relative path"
    if not _SUBDIR_RE.match(subdir):
        return "subdir must be a relative path"
    if any(part in (".", "..") for part in subdir.split("/")):
        return "subdir must not escape the repository"
    return None


def _dependency_problem(name: Any) -> Optional[str]:
    if not isinstance(name, str) or not _DEPENDENCY_RE.match(name):
        return "dependency name is malformed"
    return None


def _coerce_capability(capability: Any) -> Capability:
    if isinstance(capability, Capability):
        return capability
    if isinstance(capability, Mapping):
        data: Mapping[str, Any] = capability
        raw_dependencies = data.get("dependencies") or ()
        if isinstance(raw_dependencies, (str, bytes)):
            raise InstallRefusedError("dependencies must be a list")
        try:
            dependencies = tuple(raw_dependencies)
        except TypeError:
            raise InstallRefusedError("dependencies must be a list") from None
        name = data.get("name")
        repo = data.get("repo")
        sha = data.get("sha")
        tier = data.get("tier")
        description = data.get("description")
        version = data.get("version")
        subdir = data.get("subdir")
        return Capability(
            # Non-string fields become empty strings here and are then rejected
            # by validate_capability, so a mapping can never bypass validation.
            name=name if isinstance(name, str) else "",
            repo=repo if isinstance(repo, str) else "",
            sha=sha if isinstance(sha, str) else "",
            tier=tier if isinstance(tier, str) else "",
            description=description if isinstance(description, str) else "",
            version=version if isinstance(version, str) else "",
            subdir=subdir if isinstance(subdir, str) else "",
            dependencies=dependencies,
        )
    raise InstallRefusedError("capability must be a Capability")


def validate_capability(capability: Any) -> Capability:
    """Return the capability or raise :class:`InstallRefusedError`.

    One validator serves both catalog parsing and install requests, so nothing
    malformed can slip from one into the other.
    """

    cap = _coerce_capability(capability)
    if not isinstance(cap.name, str) or not _NAME_RE.match(cap.name):
        raise InstallRefusedError("capability name is malformed")
    if cap.name.startswith("-"):
        raise InstallRefusedError("capability name is malformed")
    problem = _repo_problem(cap.repo)
    if problem:
        raise InstallRefusedError(problem)
    if not isinstance(cap.sha, str) or not _SHA_RE.match(cap.sha.lower()):
        raise InstallRefusedError("sha must be a full 40-character commit SHA")
    if not isinstance(cap.tier, str) or cap.tier not in CATALOG_TIERS:
        raise InstallRefusedError("tier must be one of %s" % "/".join(CATALOG_TIERS))
    if cap.version == "":
        pass
    elif not isinstance(cap.version, str) or not _VERSION_RE.match(cap.version):
        raise InstallRefusedError("version is malformed")
    if not isinstance(cap.description, str):
        raise InstallRefusedError("description must be text")
    problem = _subdir_problem(cap.subdir)
    if problem:
        raise InstallRefusedError(problem)
    raw_dependencies = cap.dependencies or ()
    if isinstance(raw_dependencies, (str, bytes)):
        raise InstallRefusedError("dependencies must be a list")
    try:
        dependencies = tuple(raw_dependencies)
    except TypeError:
        raise InstallRefusedError("dependencies must be a list") from None
    for dependency in dependencies:
        problem = _dependency_problem(dependency)
        if problem:
            raise InstallRefusedError(problem)
    return Capability(
        name=cap.name,
        repo=cap.repo,
        sha=cap.sha.lower(),
        tier=cap.tier,
        description=cap.description,
        version=cap.version,
        subdir=cap.subdir or "",
        dependencies=dependencies,
    )


def build_install_request(
    capability: Any,
    *,
    dependencies: Sequence[str] = (),
    approved_dependencies: Iterable[str] = (),
    extra_args: Sequence[str] = (),
    hermes_bin: str = "hermes",
) -> InstallRequest:
    """Build the single request shape the injected installer may receive."""

    cap = validate_capability(capability)
    if not isinstance(hermes_bin, str) or not hermes_bin or _CONTROL_RE.search(hermes_bin):
        raise InstallRefusedError("hermes binary is malformed")

    if extra_args:
        for token in extra_args:
            if token in FORBIDDEN_INSTALL_FLAGS:
                raise InstallRefusedError(
                    "installer flag %s is not permitted" % _short_text(token, 40)
                )
        raise InstallRefusedError(
            "installer flags are not permitted: %s" % _short_text(tuple(extra_args)[0], 40)
        )

    approved = set()
    for name in approved_dependencies or ():
        if isinstance(name, str):
            approved.add(name)
    requested = []
    for dependency in dependencies or ():
        problem = _dependency_problem(dependency)
        if problem:
            raise InstallRefusedError(problem)
        if dependency not in approved:
            raise InstallRefusedError("dependency %s is not approved" % dependency)
        if dependency not in requested:
            requested.append(dependency)

    argv = (
        hermes_bin,
        "plugins",
        "install",
        cap.name,
        "--ref",
        cap.sha,
        "--enable",
    )
    return InstallRequest(
        name=cap.name,
        repo=cap.repo,
        sha=cap.sha,
        version=cap.version,
        subdir=cap.subdir,
        dependencies=tuple(requested),
        argv=argv,
    )


def _parse_entry(entry: Any) -> Tuple[Optional[Capability], Optional[str]]:
    if not isinstance(entry, Mapping):
        return None, "entry must be an object"
    data: Mapping[str, Any] = entry
    name = data.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name) or name.startswith("-"):
        return None, "invalid name"
    repo = data.get("repo")
    problem = _repo_problem(repo)
    if problem:
        return None, problem
    sha = data.get("sha")
    if not isinstance(sha, str):
        return None, "sha must be a full 40-character commit SHA"
    sha = sha.strip().lower()
    if not _SHA_RE.match(sha):
        return None, "sha must be a full 40-character commit SHA"
    tier = data.get("tier")
    if not isinstance(tier, str) or tier not in CATALOG_TIERS:
        return None, "tier must be one of %s" % "/".join(CATALOG_TIERS)
    version = data.get("version")
    if version is None:
        version = ""
    if not isinstance(version, str):
        return None, "version is malformed"
    version = version.strip()
    if version and not _VERSION_RE.match(version):
        return None, "version is malformed"
    subdir = data.get("subdir")
    if subdir is None:
        subdir = ""
    problem = _subdir_problem(subdir)
    if problem:
        return None, problem
    description = data.get("description")
    if description is None:
        description = ""
    if not isinstance(description, str):
        return None, "description must be text"
    return (
        Capability(
            name=name,
            repo=repo,
            sha=sha,
            tier=tier,
            description=description,
            version=version,
            subdir=subdir,
            dependencies=(),
        ),
        None,
    )


def parse_catalog_output(payload: Union[str, bytes, Mapping, None]) -> Tuple[Tuple[Capability, ...], Tuple[str, ...]]:
    """Parse ``hermes plugins search --json`` output without ever raising.

    Returns ``(capabilities, unavailable_reasons)``. A malformed root, a
    missing ``results`` array, or any single bad/partial entry yields a reason
    instead of an installable capability.
    """

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return (), ("catalog output was empty",)
        try:
            data = json.loads(text)
        except ValueError:
            return (), ("catalog output was not valid JSON",)
    else:
        data = payload
    if not isinstance(data, Mapping):
        return (), ("catalog output has an unsupported shape",)
    results = data.get("results")
    if not isinstance(results, list):
        return (), ("catalog output had no results array",)

    capabilities = []
    unavailable = []
    seen = set()
    for index, entry in enumerate(results, start=1):
        capability, reason = _parse_entry(entry)
        if reason is not None or capability is None:
            unavailable.append(
                "entry %d: %s" % (index, reason or "entry is unavailable")
            )
            continue
        if capability.name in seen:
            unavailable.append("entry %d: duplicate capability name" % index)
            continue
        seen.add(capability.name)
        capabilities.append(capability)
    return tuple(capabilities), tuple(unavailable)


class CapabilityDiscovery:
    """Read-only discovery plus the consent-gated install flow."""

    def __init__(
        self,
        *,
        run_command: Optional[Callable[[Sequence[str]], Any]] = None,
        has_plugin: Optional[Callable[[str], Any]] = None,
        is_plugin_enabled: Optional[Callable[[str], Any]] = None,
        list_plugins: Optional[Callable[[], Iterable[Any]]] = None,
        plugin_names: Iterable[str] = (),
        find_skills: Optional[Callable[[], Iterable[Any]]] = None,
        consent_store: Optional[ConsentStore] = None,
        installer: Optional[Installer] = None,
        approved_dependencies: Iterable[str] = (),
        official_blender: Optional[Capability] = OFFICIAL_BLENDER,
        hermes_bin: str = "hermes",
        timeout: float = 30.0,
    ) -> None:
        self._runner = run_command
        self._has_plugin = has_plugin if has_plugin is not None else (lambda name: False)
        self._is_plugin_enabled = is_plugin_enabled
        self._list_plugins = list_plugins
        self._plugin_names = tuple(plugin_names or ())
        self._find_skills = find_skills if find_skills is not None else (lambda: ())
        self._store = consent_store if consent_store is not None else InMemoryConsentStore()
        self._installer = installer
        self._approved_dependencies = tuple(approved_dependencies or ())
        self._official_blender = official_blender
        self._hermes_bin = hermes_bin
        self._timeout = timeout
        self._network_calls = 0

    # -- read-only installed discovery ------------------------------------

    @property
    def consent_store(self) -> ConsentStore:
        return self._store

    @property
    def network_calls(self) -> int:
        """How many times this instance has invoked the command runner."""

        return self._network_calls

    def list_installed(self, names: Optional[Iterable[str]] = None) -> DiscoverySnapshot:
        """List installed plugins and skills using only injected lookups."""

        installed, unavailable = self._installed(names)
        return DiscoverySnapshot(
            installed=installed,
            unavailable=unavailable,
            network_used=False,
            catalog_requested=False,
        )

    def list_available(
        self,
        query: str = "",
        *,
        installed_only: bool = True,
        include_catalog: Optional[bool] = None,
        names: Optional[Iterable[str]] = None,
    ) -> DiscoverySnapshot:
        """List capabilities; the default never runs a command.

        ``installed_only=True`` is the mission-control path: no subprocess, no
        network, no install. Passing ``installed_only=False`` (or
        ``include_catalog=True``) is an explicit catalog search.
        """

        if include_catalog is None:
            include_catalog = not installed_only
        if installed_only and include_catalog:
            raise ValueError("installed_only conflicts with a catalog search")
        installed, unavailable = self._installed(names)
        if not include_catalog:
            return DiscoverySnapshot(
                installed=installed,
                unavailable=unavailable,
                network_used=False,
                catalog_requested=False,
            )
        catalog_snapshot = self.search_catalog(query)
        return DiscoverySnapshot(
            installed=installed,
            catalog=catalog_snapshot.catalog,
            unavailable=unavailable + catalog_snapshot.unavailable,
            network_used=catalog_snapshot.network_used,
            catalog_requested=True,
        )

    def search_catalog(self, query: str = "") -> DiscoverySnapshot:
        """Explicitly search the curated catalog with one read-only CLI call."""

        if (
            not isinstance(query, str)
            or len(query) > _MAX_QUERY
            or _CONTROL_RE.search(query)
            or query.strip().startswith("-")
        ):
            return DiscoverySnapshot(
                unavailable=("catalog query is malformed",),
                catalog_requested=True,
            )

        argv = [self._hermes_bin, "plugins", "search"]
        if query:
            argv.append(query)
        argv.append("--json")

        self._network_calls += 1
        try:
            raw = self._call_runner(argv)
        except Exception:
            return DiscoverySnapshot(
                unavailable=("catalog search failed",),
                network_used=True,
                catalog_requested=True,
            )
        returncode, stdout = raw
        if returncode != 0:
            return DiscoverySnapshot(
                unavailable=("catalog search failed (exit %s)" % returncode,),
                network_used=True,
                catalog_requested=True,
            )
        capabilities, unavailable = parse_catalog_output(stdout)
        return DiscoverySnapshot(
            catalog=capabilities,
            unavailable=unavailable,
            network_used=True,
            catalog_requested=True,
        )

    def _call_runner(self, argv: Sequence[str]) -> Tuple[int, str]:
        runner = self._runner
        if runner is None:
            result = subprocess.run(
                list(argv),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        elif callable(runner):
            result = runner(list(argv))
        elif hasattr(runner, "run"):
            result = runner.run(list(argv))
        else:
            raise TypeError("command runner must be callable")

        if isinstance(result, bytes):
            return 0, result.decode("utf-8", errors="replace")
        if isinstance(result, str):
            return 0, result
        if isinstance(result, Mapping):
            return int(result.get("returncode", result.get("rc", 0))), _as_text(result.get("stdout"))
        returncode = getattr(result, "returncode", None)
        if returncode is None:
            returncode = getattr(result, "rc", None)
        if returncode is None:
            raise TypeError("command runner returned an unsupported result")
        return int(returncode), _as_text(getattr(result, "stdout", ""))

    def _plugin_candidates(self, unavailable: list) -> list:
        raw: Iterable[Any]
        try:
            if self._list_plugins is not None:
                raw = self._list_plugins()
            else:
                raw = self._plugin_names
            return sorted(_normalise_names(raw, unavailable, "plugin"))
        except Exception:
            unavailable.append("installed plugin lookup failed")
            return []

    def _installed(self, names: Optional[Iterable[str]]) -> Tuple[Tuple[InstalledCapability, ...], Tuple[str, ...]]:
        unavailable = []
        if names is None:
            candidates = self._plugin_candidates(unavailable)
        else:
            candidates = sorted(_normalise_names(names, unavailable, "plugin"))

        installed = []
        for name in candidates:
            try:
                present = bool(self._has_plugin(name))
            except Exception:
                unavailable.append("plugin lookup failed for %s" % _short_text(name, 64))
                continue
            if not present:
                continue
            try:
                if self._is_plugin_enabled is None:
                    enabled = True
                else:
                    enabled = bool(self._is_plugin_enabled(name))
            except Exception:
                enabled = False
                unavailable.append("plugin state unknown for %s" % _short_text(name, 64))
            installed.append(InstalledCapability(name=name, kind="plugin", enabled=enabled))

        try:
            skills = self._find_skills()
        except Exception:
            skills = ()
            unavailable.append("installed skill lookup failed")
        for skill in _skill_entries(skills, unavailable):
            installed.append(skill)

        deduped = {}
        for item in installed:
            deduped.setdefault((item.kind, item.name), item)
        return (
            tuple(sorted(deduped.values(), key=lambda i: (i.kind, i.name))),
            tuple(unavailable),
        )

    # -- routing -----------------------------------------------------------

    def route_blender(self) -> dict:
        """Route only to an installed *and enabled* official ``blender`` plugin."""

        installed = False
        enabled = False
        try:
            installed = bool(self._has_plugin("blender"))
            if installed:
                enabled = (
                    True
                    if self._is_plugin_enabled is None
                    else bool(self._is_plugin_enabled("blender"))
                )
        except Exception:
            installed = False
            enabled = False

        if installed and enabled:
            return {
                "status": "ready",
                "route": "hermes_worker",
                "capability": "blender",
                "skill": "blender:blender",
                "installed": True,
                "enabled": True,
                "requires_approval": False,
                "disclosure": None,
            }

        disclosure = None
        if self._official_blender is not None:
            disclosure = blender_disclosure(self._official_blender)
        return {
            "status": "needs_approval",
            "route": None,
            "capability": "blender",
            "skill": None,
            "installed": installed,
            "enabled": enabled,
            "requires_approval": True,
            "reason": (
                "the official blender plugin is not installed and enabled"
                if not installed
                else "the official blender plugin is installed but disabled"
            ),
            "disclosure": disclosure,
        }

    # -- consent -----------------------------------------------------------

    def grant_consent(
        self,
        capability: Any,
        *,
        source: str = "mission_control",
        now: Optional[Union[str, datetime]] = None,
        nonce: Optional[str] = None,
    ) -> ConsentRecord:
        """Record explicit consent for one exact capability name and SHA."""

        cap = validate_capability(capability)
        existing = self._store.find(cap.name, cap.sha)
        if existing is not None:
            return existing
        if not isinstance(source, str) or not source.strip():
            raise ValueError("consent source is required")
        if now is None:
            granted_at = _utc_now()
        elif isinstance(now, datetime):
            granted_at = now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        elif isinstance(now, str) and now.strip():
            granted_at = now.strip()
        else:
            raise ValueError("consent timestamp is malformed")
        record = ConsentRecord(
            capability=cap.name,
            sha=cap.sha,
            version=cap.version,
            granted_at=granted_at,
            source=source.strip(),
            nonce=nonce if nonce is not None else secrets.token_hex(16),
        )
        return self._store.save(record)

    def revoke_consent(self, capability: str, sha: str) -> bool:
        return self._store.revoke(capability, str(sha).lower())

    # -- install -----------------------------------------------------------

    def install_capability(
        self,
        capability: Any,
        *,
        dependencies: Sequence[str] = (),
        extra_args: Sequence[str] = (),
    ) -> InstallResult:
        """Install only with exact consent, then verify the installed revision.

        Refusals (bad request, missing/mismatched consent) raise
        :class:`InstallRefusedError` before the installer is touched. Failures
        after the call are returned as ``installed=False`` with a reason.
        """

        cap = validate_capability(capability)
        request = build_install_request(
            cap,
            dependencies=dependencies,
            approved_dependencies=self._approved_dependencies,
            extra_args=extra_args,
            hermes_bin=self._hermes_bin,
        )
        consent = self._store.find(request.name, request.sha)
        if consent is None or not consent.matches(request.name, request.sha):
            raise ConsentRequiredError(
                "consent is required for %s at %s" % (request.name, request.sha)
            )
        if self._installer is None:
            return InstallResult(
                installed=False, name=request.name, sha=request.sha,
                reason="no installer is configured", installer_called=False,
            )

        try:
            outcome = self._installer.install(request)
        except Exception as exc:
            detail = _short_text(exc) or exc.__class__.__name__
            return InstallResult(
                installed=False,
                name=request.name,
                sha=request.sha,
                reason="installer failed: %s" % detail,
                verified=False,
                installer_called=True,
            )

        receipt_problem = _receipt_problem(outcome, request)
        if receipt_problem is not None:
            return InstallResult(
                installed=False,
                name=request.name,
                sha=request.sha,
                reason=receipt_problem,
                verified=False,
                installer_called=True,
            )

        verify = getattr(self._installer, "installed_sha", None)
        if not callable(verify):
            return InstallResult(
                installed=False,
                name=request.name,
                sha=request.sha,
                reason="installer cannot verify the installed revision",
                verified=False,
                installer_called=True,
            )
        try:
            actual = verify(request.name)
        except Exception:
            return InstallResult(
                installed=False,
                name=request.name,
                sha=request.sha,
                reason="installer verification failed",
                verified=False,
                installer_called=True,
            )
        if not isinstance(actual, str) or actual.strip().lower() != request.sha:
            return InstallResult(
                installed=False,
                name=request.name,
                sha=request.sha,
                reason="installed revision did not match the approved commit",
                verified=False,
                installer_called=True,
            )
        return InstallResult(
            installed=True,
            name=request.name,
            sha=request.sha,
            reason="",
            verified=True,
            installer_called=True,
        )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalise_names(raw: Any, unavailable: list, kind: str) -> list:
    if isinstance(raw, Mapping):
        raw = raw.get(kind + "s") or raw.get("items") or ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        unavailable.append("%s list is malformed" % kind)
        return []
    names = []
    for entry in raw:
        name = _entry_name(entry)
        if name is None:
            unavailable.append("%s entry is malformed" % kind)
            continue
        names.append(name)
    return names


def _entry_name(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        candidate = entry
    elif isinstance(entry, Mapping):
        candidate = entry.get("name")
    else:
        candidate = getattr(entry, "name", None)
    if not isinstance(candidate, str):
        return None
    candidate = candidate.strip()
    if not candidate or _CONTROL_RE.search(candidate) or len(candidate) > 128:
        return None
    return candidate


def _skill_entries(raw: Any, unavailable: list) -> list:
    if isinstance(raw, Mapping):
        raw = raw.get("skills") or raw.get("items") or ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        unavailable.append("installed skill lookup returned an unsupported shape")
        return []
    skills = []
    for entry in raw:
        name = _entry_name(entry)
        if name is None:
            unavailable.append("skill entry is malformed")
            continue
        enabled = True
        if isinstance(entry, Mapping) and entry.get("enabled") is False:
            enabled = False
        skills.append(InstalledCapability(name=name, kind="skill", enabled=enabled))
    return skills


def _receipt_problem(outcome: Any, request: InstallRequest) -> Optional[str]:
    """Return a reason when the installer's own receipt contradicts success."""

    if outcome is None:
        return None
    if isinstance(outcome, Mapping):
        ok = outcome.get("ok")
        name = outcome.get("name")
        sha = outcome.get("sha")
        reason = outcome.get("reason")
    else:
        ok = getattr(outcome, "ok", None)
        name = getattr(outcome, "name", None)
        sha = getattr(outcome, "sha", None)
        reason = getattr(outcome, "reason", None)
    if ok is False:
        detail = _short_text(reason, 120)
        return "installer reported failure" + (": %s" % detail if detail else "")
    if name is not None and name != request.name:
        return "installer reported another capability"
    if sha is not None and (not isinstance(sha, str) or sha.strip().lower() != request.sha):
        return "installer reported another revision"
    return None
