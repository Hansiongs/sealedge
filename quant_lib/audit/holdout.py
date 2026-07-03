"""
Phase 4: Sealed Holdout Set Manager.

Framework principles:
  - The holdout is separated BEFORE Phase 1 starts, never touched until Phase 4.
  - Only one test -- once the holdout is used, it dies as a holdout.
  - Violation = invalidates all results.

C-2 fix: ``seal()`` now REQUIRES a real ``data_hash`` (SHA256 hex
digest) so the no-peek guarantee is enforced. The framework's
ResearchSession computes this hash from the actual holdout data
at session creation; tests can use a deterministic fake.

B0.1 fix: the seal is now signed with **HMAC-SHA256** over the
canonical seal state using a secret loaded from the
``QUANT_LIB_HMAC_SECRET`` environment variable. Without the
secret, the seal cannot be forged (an attacker cannot construct
a valid ``signature`` for a tampered state) and cannot have its
``broken_at`` field silently reset. The secret is **required**:
if the environment variable is not set, all seal operations
raise ``RuntimeError``. There is no insecure fallback.

The ``data_hash`` is a SHA256 fingerprint of the holdout data
alone (for the no-peek guarantee); the ``signature`` is an
HMAC of the seal's metadata (``start``, ``end``, ``sealed_at``,
``broken_at``, ``data_hash``) using the secret. Together they
provide two independent integrity guarantees: data integrity
(no-peek) and state integrity (no tampering with broken/sealed
status).

Usage:
    >>> import os
    >>> os.environ["QUANT_LIB_HMAC_SECRET"] = "<32+ char random string>"
    >>> hs = HoldoutSet(
    ...     name="final_test_v1",
    ...     start="2025-06-01",
    ...     end="2026-05-31",
    ...     seal_path="holdout_seal.json"
    ... )
    >>> hs.seal(data_hash="<sha256-of-holdout-data>")
    >>> hs.is_sealed()
    True
    >>> hs.verify()  # Re-computes HMAC; fails if seal was tampered
    True
    >>> hs.commit_break(
    ...     real_data_hash="<sha256-of-holdout-data>",
    ...     description="Phase 4 final test"
    ... )
    (True, "<hash-before>", "<hash-after>")
"""

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional


# Environment variable holding the HMAC secret. Must be set in the
# environment before any seal operation. There is intentionally no
# default: a missing secret is a configuration error, not a fallback.
_HMAC_SECRET_ENV = "QUANT_LIB_HMAC_SECRET"

# Minimum acceptable secret length (bytes). Short keys are easier to
# brute-force. NIST SP 800-107 recommends >= 112 bits (14 bytes) for
# HMAC-SHA256; we enforce 32 bytes (256 bits) for a generous margin.
_MIN_SECRET_BYTES = 32


def get_hmac_secret() -> bytes:
    """Return the HMAC secret from the environment.

    Cached on first call so the env var is read at most once per
    process (allows test fixtures to set the env var before any
    seal operation).

    Returns
    -------
    bytes
        The decoded secret (UTF-8).

    Raises
    ------
    RuntimeError
        If ``QUANT_LIB_HMAC_SECRET`` is not set, is empty, or is
        shorter than ``_MIN_SECRET_BYTES`` characters. There is no
        insecure fallback -- the framework refuses to operate seals
        without a valid secret.
    """
    cached = getattr(get_hmac_secret, "_cached", None)
    if cached is not None:
        return cached
    raw = os.environ.get(_HMAC_SECRET_ENV, "")
    if not raw:
        raise RuntimeError(
            f"HMAC seal secret is not configured. Set the "
            f"{_HMAC_SECRET_ENV!r} environment variable to a "
            f"random string of at least {_MIN_SECRET_BYTES} "
            f"characters before creating or verifying seals."
        )
    if len(raw) < _MIN_SECRET_BYTES:
        raise RuntimeError(
            f"HMAC seal secret is too short "
            f"({len(raw)} chars; minimum {_MIN_SECRET_BYTES}). "
            f"Use a cryptographically random string "
            f"(e.g. ``python -c \"import secrets; print(secrets.token_urlsafe(32))\"``)."
        )
    secret = raw.encode("utf-8")
    setattr(get_hmac_secret, "_cached", secret)
    return secret


def _reset_hmac_secret_cache() -> None:
    """Clear the cached HMAC secret. Used by tests to switch secrets
    within a single process. Production code does not need this.
    """
    if hasattr(get_hmac_secret, "_cached"):
        delattr(get_hmac_secret, "_cached")


def _canonical_payload(state: dict) -> bytes:
    """Serialize a seal-state dict to canonical bytes for signing.

    Uses ``json.dumps`` with sorted keys and tight separators so the
    resulting byte string is deterministic across Python versions
    and platforms. This is the canonical form over which the HMAC
    is computed.

    Parameters
    ----------
    state : dict
        Seal state dict. Must NOT contain a ``signature`` key
        (caller must strip it before signing).
    """
    return json.dumps(
        state, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def compute_seal_signature(state: dict) -> str:
    """Compute the HMAC-SHA256 signature of a seal state dict.

    Parameters
    ----------
    state : dict
        Seal state dict. The ``signature`` key (if present) is
        excluded from the signed payload.

    Returns
    -------
    str
        64-char hex digest of the HMAC-SHA256.

    Raises
    ------
    RuntimeError
        If the HMAC secret is not configured.
    """
    # Strip signature if present (defensive; callers should already do so).
    payload_state = {k: v for k, v in state.items() if k != "signature"}
    payload = _canonical_payload(payload_state)
    secret = get_hmac_secret()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def verify_seal_signature(state: dict) -> bool:
    """Verify the ``signature`` field of a seal state dict.

    Constant-time comparison via ``hmac.compare_digest`` to avoid
    timing attacks.

    Parameters
    ----------
    state : dict
        The full seal state dict (with ``signature`` key).

    Returns
    -------
    bool
        True if the signature is present and matches the recomputed
        HMAC over the remaining fields. False if signature is
        missing, or if it does not match (tamper detected).
    """
    sig = state.get("signature")
    if not sig or not isinstance(sig, str):
        return False
    expected = compute_seal_signature(state)
    return hmac.compare_digest(sig, expected)


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class HoldoutSeal:
    """In-memory representation of a sealed holdout.

    ``signature`` holds the HMAC of the on-disk state. It is set
    at seal/commit time and verified at every ``verify()`` call.

    The dataclass does NOT include ``seal_file`` in its signed
    payload -- ``seal_file`` is a path on the local filesystem and
    not part of the integrity contract.
    """
    start: str
    end: str
    sealed_at: Optional[str] = None
    broken_at: Optional[str] = None
    data_hash: Optional[str] = None
    signature: Optional[str] = None
    seal_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize the in-memory state to a JSON-friendly dict.

        ``signature`` is included only if it has been computed --
        callers that want to write a signed seal use ``to_dict()``
        to get the unsigned payload, then add the signature field
        explicitly (see ``_save_seal``).
        """
        d = {
            "start": self.start,
            "end": self.end,
            "sealed_at": self.sealed_at,
            "broken_at": self.broken_at,
            "data_hash": self.data_hash,
        }
        if self.signature is not None:
            d["signature"] = self.signature
        return d

    @staticmethod
    def from_dict(d: dict) -> "HoldoutSeal":
        """Reconstruct from a JSON dict (with or without signature)."""
        return HoldoutSeal(
            start=d["start"],
            end=d["end"],
            sealed_at=d.get("sealed_at"),
            broken_at=d.get("broken_at"),
            data_hash=d.get("data_hash"),
            signature=d.get("signature"),
            seal_file=d.get("seal_file"),
        )


class HoldoutSet:
    """Manages a sealed holdout period with HMAC-signed integrity.

    The seal file is a JSON document containing:
        - ``start``, ``end``: holdout boundary
        - ``sealed_at``: ISO 8601 timestamp when seal was applied
        - ``broken_at``: ISO 8601 timestamp when seal was broken (None until then)
        - ``data_hash``: SHA256 hex digest of the holdout raw data (no-peek)
        - ``signature``: HMAC-SHA256 of the above fields using
          ``QUANT_LIB_HMAC_SECRET``

    Parameters
    ----------
    name : str
        Holdout identifier.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    seal_path : str, optional
        Path to save the seal JSON. If None, seal lives in memory only.

    Example
    -------
    >>> import os
    >>> os.environ["QUANT_LIB_HMAC_SECRET"] = "x" * 32
    >>> hs = HoldoutSet("test", "2025-06-01", "2026-05-31")
    >>> hs.seal(data_hash="0" * 64)
    >>> hs.is_sealed()
    True
    >>> hs.verify()
    True
    >>> hs.commit_break(real_data_hash="0" * 64)
    (True, '0...0', '0...0')
    """

    def __init__(self, name: str, start: str, end: str, seal_path: Optional[str] = None):
        self.name = name
        self._seal = HoldoutSeal(start=start, end=end, seal_file=seal_path)
        self._tampered = False
        # ── One-shot holdout invariant (Phase 4.x Blocker fix) ──
        # If a sealed or broken seal file already exists at ``seal_path``,
        # load it into in-memory state so subsequent ``seal()`` /
        # ``is_sealed()`` / ``commit_break()`` see the prior state.
        # Without this, a second ``ResearchSession(...)`` against the
        # same holdout_period would silently overwrite the broken seal
        # file (contradicting "Only one test -- once the holdout is
        # used, it dies as a holdout").
        #
        # We only load when (a) the file exists, (b) it parses as JSON,
        # and (c) the HMAC signature is valid under the current secret.
        # If any check fails, we silently fall through to fresh
        # in-memory state -- this preserves the existing tamper-
        # detection behavior (an invalid signature must NOT be
        # trusted; the verify() path still raises _tampered).
        if seal_path and os.path.exists(seal_path):
            try:
                with open(seal_path, "r") as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, OSError):
                saved = None
            if saved is not None and verify_seal_signature(saved):
                # Valid signature under current secret: replicate state.
                self._seal.sealed_at = saved.get("sealed_at")
                self._seal.broken_at = saved.get("broken_at")
                self._seal.data_hash = saved.get("data_hash")
                self._seal.signature = saved.get("signature")

    def seal(self, data_hash: str) -> None:
        """Lock the holdout set with an HMAC-signed seal.

        Records the current time and data hash, then writes a signed
        seal to disk. After this, the holdout should not be accessed
        until ``commit_break()`` is called.

        Parameters
        ----------
        data_hash : str
            SHA256 hash of the holdout data at sealing time. REQUIRED.
            Passing a placeholder defeats the no-peek guarantee enforced
            by ResearchSession. Tests that don't care about the hash can
            pass a deterministic fake (e.g. "0" * 64).

        Raises
        ------
        ValueError
            If data_hash is None or empty.
        RuntimeError
            If the holdout was already broken, or if the HMAC
            secret is not configured.
        """
        if self._seal.broken_at is not None:
            raise RuntimeError(
                f"Holdout '{self.name}' was already broken at "
                f"{self._seal.broken_at}. Cannot re-seal."
            )
        if not data_hash:
            raise ValueError(
                "seal() requires a non-empty data_hash. "
                "Pass a SHA256 hex digest of the holdout data. "
                "Tests can use a deterministic fake (e.g. '0' * 64)."
            )

        # Get the secret here so a missing/short secret fails loudly
        # BEFORE we mutate any state.
        get_hmac_secret()

        self._seal.sealed_at = datetime.now(timezone.utc).isoformat()
        self._seal.data_hash = data_hash
        # Clear any prior signature so _save_seal recomputes it.
        self._seal.signature = None
        self._save_seal()
        self._tampered = False

    def is_sealed(self) -> bool:
        """Check if the holdout is currently sealed."""
        return self._seal.sealed_at is not None and self._seal.broken_at is None

    def is_broken(self) -> bool:
        """Check if the holdout has already been used."""
        return self._seal.broken_at is not None

    def verify(self) -> bool:
        """Verify the seal integrity.

        Loads the seal file (if saved) and checks that:
        - The holdout hasn't been broken
        - The HMAC signature is present and valid (B0.1)
        - The data_hash and broken_at fields haven't been tampered with

        The signature check is the strongest: an attacker who can
        modify the on-disk JSON cannot forge a valid signature
        without the HMAC secret.

        Returns
        -------
        bool
            True if the seal is intact and untampered.
        """
        if self._tampered:
            return False

        if self._seal.broken_at is not None:
            return False

        if self._seal.seal_file and self._file_exists():
            try:
                with open(self._seal.seal_file, "r") as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._tampered = True
                return False

            # B0.1: signature check first -- if the signature is
            # missing/invalid, we don't trust ANY field in the file.
            if not verify_seal_signature(saved):
                self._tampered = True
                return False

            # Business-logic integrity checks (after signature).
            if saved.get("data_hash") != self._seal.data_hash:
                self._tampered = True
                return False
            if saved.get("broken_at") is not None:
                self._seal.broken_at = saved["broken_at"]
                return False

        return True

    def commit_break(
        self, real_data_hash: str, description: str = "Phase 4 final test"
    ) -> tuple[bool, str, str]:
        """Break the seal with an externally-computed data hash.

        Use this when the caller has already computed a SHA256 of the
        actual holdout data and wants to record that hash as the seal's
        authoritative fingerprint before breaking. Unlike ``break_seal``,
        this method does NOT recompute a hash from an arbitrary DataFrame
        passed in (which is error-prone and often a placeholder).

        The seal is atomically updated with the new ``data_hash``,
        ``broken_at``, and a fresh HMAC signature -- all in a single
        disk write to avoid crash-window races.

        Parameters
        ----------
        real_data_hash : str
            SHA256 hex digest of the holdout data, computed by the caller.
        description : str
            Description of this test (logged for audit).

        Returns
        -------
        tuple of (was_intact: bool, hash_before: str, hash_after: str)
            was_intact: True if the seal was intact before breaking.
            hash_before: The hash recorded at sealing time (may be placeholder).
            hash_after: The real_data_hash passed in.

        Sprint 1 fix: re-verify the on-disk seal before breaking.
        Without this call, an in-memory ``_tampered`` flag could be
        stale if disk tampering occurred between the last ``verify()``
        call and this ``commit_break()`` (e.g., another process
        modified the seal file after session creation). The re-verify
        guarantees the integrity contract is enforced at the moment
        the seal is irreversibly broken.
        """
        if self._tampered:
            return False, self._seal.data_hash or "", real_data_hash

        # Re-verify disk state (may set self._tampered and update
        # self._seal.broken_at from disk if previously broken by a
        # parallel process). In-memory-only seals (seal_file=None)
        # pass through verify() unchanged.
        if not self.verify():
            return False, self._seal.data_hash or "", real_data_hash

        was_intact = self._seal.broken_at is None
        if not was_intact:
            return False, self._seal.data_hash or "", real_data_hash

        hash_before = self._seal.data_hash or ""
        # Set ALL fields atomically before saving -- single disk write
        # to avoid crash-window race (where the file on disk has a
        # partially-updated seal with broken_at=None but new data_hash).
        self._seal.data_hash = real_data_hash
        self._seal.broken_at = datetime.now(timezone.utc).isoformat()
        self._seal.signature = None  # recompute in _save_seal

        from quant_lib.core._logging import log
        log.warning(
            f"HOLDOUT SEAL BROKEN: '{self.name}' ({self._seal.start} to "
            f"{self._seal.end}) at {self._seal.broken_at}. "
            f"Description: {description}. "
            f"This holdout is now INVALIDATED for future use."
        )

        # Single atomic save (Phase 3.3 D1 fix: was 2 saves, race risk).
        self._save_seal()
        return was_intact, hash_before, real_data_hash

    def boundary(self) -> tuple:
        """Return the (start, end) boundary of this holdout."""
        return (self._seal.start, self._seal.end)

    def summary(self) -> str:
        """Return human-readable status."""
        status = "BROKEN" if self.is_broken() else ("SEALED" if self.is_sealed() else "UNSEALED")
        parts = [
            f"Holdout: {self.name} ({status})",
            f"  Period : {self._seal.start} -> {self._seal.end}",
        ]
        if self._seal.sealed_at:
            parts.append(f"  Sealed : {self._seal.sealed_at}")
        if self._seal.broken_at:
            parts.append(f"  Broken : {self._seal.broken_at}")
        if self._seal.signature:
            parts.append(f"  Sig    : {self._seal.signature[:32]}...")
        if self._tampered:
            parts.append("  [!]  TAMPER DETECTED")
        return "\n".join(parts)

    # -- Private helpers --

    def _file_exists(self) -> bool:
        import os
        return self._seal.seal_file is not None and os.path.exists(self._seal.seal_file)

    def _save_seal(self) -> None:
        """Atomically write the seal to disk with a fresh HMAC signature.

        Phase 3.2: Atomic write via tempfile + os.replace. A crash
        mid-write leaves the original seal intact (the old seal's
        signature remains valid against the old state). Previously
        used direct ``open()`` which could produce a half-written
        JSON on crash, causing verify() to fail despite no actual
        tampering.
        """
        if self._seal.seal_file:
            payload = self._seal.to_dict()
            payload["signature"] = compute_seal_signature(payload)
            self._seal.signature = payload["signature"]

            import tempfile
            dir_name = os.path.dirname(self._seal.seal_file) or "."
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_name, prefix=".seal_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, self._seal.seal_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise


__all__ = [
    "HoldoutSeal",
    "HoldoutSet",
    "get_hmac_secret",
    "compute_seal_signature",
    "verify_seal_signature",
    "_reset_hmac_secret_cache",
]
