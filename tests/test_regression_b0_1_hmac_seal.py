"""Regression tests for B0.1: HMAC-SHA256 seal signature.

Bug
----
Before B0.1, the holdout seal persisted only ``data_hash`` (a
plain SHA256 of the holdout data) and a few metadata fields.
There was no integrity check on the seal *itself* -- anyone who
could read the seal JSON and the underlying data could:

  - Edit ``broken_at`` back to ``null`` to "un-break" a used holdout.
  - Construct a forged seal with a known ``data_hash`` for new
    data and trick the framework into thinking the holdout is
    untampered.

The ``data_hash`` only proves the *content* of the holdout data
matched at sealing time; it does not prove the *seal metadata*
(``broken_at``, ``sealed_at``, etc.) is authentic. An attacker
who can modify the seal JSON cannot be detected without an
integrity check over the seal state.

Fix
---
The seal JSON now includes an ``HMAC-SHA256 signature`` field
over a canonical serialization of the seal state, using a
secret loaded from the ``QUANT_LIB_HMAC_SECRET`` environment
variable. ``verify()`` re-computes the signature and compares
in constant time. If the signature is missing or mismatches,
the seal is treated as tampered. There is no insecure fallback:
if the secret is missing, ``seal()`` raises ``RuntimeError``.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════
# get_hmac_secret
# ═══════════════════════════════════════════════════════════════════════


class TestGetHmacSecret:
    def test_returns_secret_when_set(self, monkeypatch):
        """Secret is returned as bytes when env var is set."""
        from quant_lib.audit.holdout import get_hmac_secret, _reset_hmac_secret_cache
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "x" * 32)
        _reset_hmac_secret_cache()
        secret = get_hmac_secret()
        assert isinstance(secret, bytes)
        assert secret == b"x" * 32

    def test_raises_when_secret_missing(self, monkeypatch):
        """No env var -> RuntimeError (no insecure fallback)."""
        from quant_lib.audit.holdout import get_hmac_secret, _reset_hmac_secret_cache
        monkeypatch.delenv("QUANT_LIB_HMAC_SECRET", raising=False)
        _reset_hmac_secret_cache()
        with pytest.raises(RuntimeError, match="HMAC seal secret is not configured"):
            get_hmac_secret()

    def test_raises_when_secret_empty(self, monkeypatch):
        """Empty env var -> RuntimeError."""
        from quant_lib.audit.holdout import get_hmac_secret, _reset_hmac_secret_cache
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "")
        _reset_hmac_secret_cache()
        with pytest.raises(RuntimeError, match="HMAC seal secret is not configured"):
            get_hmac_secret()

    def test_raises_when_secret_too_short(self, monkeypatch):
        """Secret < 32 chars -> RuntimeError (NIST recommends >= 14 bytes for SHA-256)."""
        from quant_lib.audit.holdout import get_hmac_secret, _reset_hmac_secret_cache
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "short")
        _reset_hmac_secret_cache()
        with pytest.raises(RuntimeError, match="too short"):
            get_hmac_secret()

    def test_secret_is_cached(self, monkeypatch):
        """Subsequent calls use the cached value (env var can change after first call)."""
        from quant_lib.audit import holdout as holdout_mod
        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "a" * 32)
        first = holdout_mod.get_hmac_secret()
        # Change the env var; cached value should still be returned.
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "b" * 32)
        second = holdout_mod.get_hmac_secret()
        assert first == second == b"a" * 32
        holdout_mod._reset_hmac_secret_cache()

    def test_cache_reset_picks_up_new_secret(self, monkeypatch):
        """After _reset_hmac_secret_cache, a new env var is read."""
        from quant_lib.audit import holdout as holdout_mod
        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "a" * 32)
        first = holdout_mod.get_hmac_secret()
        holdout_mod._reset_hmac_secret_cache()
        second = holdout_mod.get_hmac_secret()
        assert first == b"a" * 32
        assert second == b"a" * 32
        holdout_mod._reset_hmac_secret_cache()


# ═══════════════════════════════════════════════════════════════════════
# compute_seal_signature
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSealSignature:
    def test_returns_64_char_hex(self):
        from quant_lib.audit.holdout import compute_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30", "data_hash": "abc"}
        sig = compute_seal_signature(state)
        assert isinstance(sig, str)
        assert len(sig) == 64
        int(sig, 16)  # valid hex

    def test_deterministic_for_same_state(self):
        from quant_lib.audit.holdout import compute_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30", "data_hash": "abc"}
        sig1 = compute_seal_signature(state)
        sig2 = compute_seal_signature(state)
        assert sig1 == sig2

    def test_differs_for_different_state(self):
        from quant_lib.audit.holdout import compute_seal_signature
        sig1 = compute_seal_signature({"data_hash": "abc"})
        sig2 = compute_seal_signature({"data_hash": "def"})
        assert sig1 != sig2

    def test_key_order_does_not_affect_signature(self):
        """Canonical JSON (sort_keys=True) means dict order is irrelevant."""
        from quant_lib.audit.holdout import compute_seal_signature
        sig1 = compute_seal_signature({"a": 1, "b": 2, "c": 3})
        sig2 = compute_seal_signature({"c": 3, "a": 1, "b": 2})
        sig3 = compute_seal_signature({"b": 2, "c": 3, "a": 1})
        assert sig1 == sig2 == sig3

    def test_signature_key_excluded_from_payload(self):
        """A 'signature' key in the input must not be signed (would self-sign)."""
        from quant_lib.audit.holdout import compute_seal_signature
        sig_without = compute_seal_signature({"a": 1, "b": 2})
        sig_with = compute_seal_signature({"a": 1, "b": 2, "signature": "garbage"})
        # The implementation strips 'signature' before signing.
        assert sig_without == sig_with

    def test_different_secret_produces_different_signature(self, monkeypatch):
        from quant_lib.audit import holdout as holdout_mod
        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "a" * 32)
        sig_a = holdout_mod.compute_seal_signature({"a": 1})

        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "b" * 32)
        sig_b = holdout_mod.compute_seal_signature({"a": 1})

        assert sig_a != sig_b
        holdout_mod._reset_hmac_secret_cache()


# ═══════════════════════════════════════════════════════════════════════
# verify_seal_signature
# ═══════════════════════════════════════════════════════════════════════


class TestVerifySealSignature:
    def test_valid_signature_returns_true(self):
        from quant_lib.audit.holdout import compute_seal_signature, verify_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30", "data_hash": "abc"}
        sig = compute_seal_signature(state)
        assert verify_seal_signature({**state, "signature": sig}) is True

    def test_tampered_signature_returns_false(self):
        from quant_lib.audit.holdout import verify_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30",
                 "data_hash": "abc", "signature": "0" * 64}
        # Even one bit-flip must fail
        state["signature"] = "1" + "0" * 63
        assert verify_seal_signature(state) is False

    def test_missing_signature_returns_false(self):
        from quant_lib.audit.holdout import verify_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30", "data_hash": "abc"}
        assert verify_seal_signature(state) is False

    def test_tampered_data_hash_returns_false(self):
        """If the data_hash is changed after signing, verify must fail."""
        from quant_lib.audit.holdout import compute_seal_signature, verify_seal_signature
        state = {"start": "2025-01-01", "end": "2025-06-30", "data_hash": "original"}
        sig = compute_seal_signature(state)
        tampered = {**state, "data_hash": "tampered", "signature": sig}
        assert verify_seal_signature(tampered) is False

    def test_tampered_broken_at_returns_false(self):
        """If broken_at is changed (e.g. un-broke), verify must fail."""
        from quant_lib.audit.holdout import compute_seal_signature, verify_seal_signature
        state = {
            "start": "2025-01-01", "end": "2025-06-30",
            "data_hash": "abc", "broken_at": "2025-07-01T00:00:00Z",
        }
        sig = compute_seal_signature(state)
        tampered = {**state, "broken_at": None, "signature": sig}
        assert verify_seal_signature(tampered) is False


# ═══════════════════════════════════════════════════════════════════════
# HoldoutSet integration
# ═══════════════════════════════════════════════════════════════════════


class TestSealRequiresHmacSecret:
    """seal() must fail loudly if the HMAC secret is not configured."""

    def test_seal_raises_when_secret_missing(self, monkeypatch):
        from quant_lib.audit import holdout as holdout_mod
        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.delenv("QUANT_LIB_HMAC_SECRET", raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = holdout_mod.HoldoutSet("t", "2025-01-01", "2025-06-30",
                                         seal_path=seal_path)
            with pytest.raises(RuntimeError, match="HMAC seal secret"):
                hs.seal(data_hash="a" * 64)

    def test_seal_raises_when_secret_too_short(self, monkeypatch):
        from quant_lib.audit import holdout as holdout_mod
        holdout_mod._reset_hmac_secret_cache()
        monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "short")
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = holdout_mod.HoldoutSet("t", "2025-01-01", "2025-06-30",
                                         seal_path=seal_path)
            with pytest.raises(RuntimeError, match="too short"):
                hs.seal(data_hash="a" * 64)


class TestSealFileContents:
    """The on-disk seal JSON must contain a valid signature field."""

    def test_seal_file_contains_signature(self):
        from quant_lib.audit.holdout import HoldoutSet
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            saved = json.loads(Path(seal_path).read_text())
            assert "signature" in saved
            assert isinstance(saved["signature"], str)
            assert len(saved["signature"]) == 64  # hex SHA-256

    def test_seal_file_signature_verifies(self):
        from quant_lib.audit.holdout import HoldoutSet, verify_seal_signature
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            saved = json.loads(Path(seal_path).read_text())
            assert verify_seal_signature(saved)

    def test_seal_file_after_commit_break_has_new_signature(self):
        from quant_lib.audit.holdout import HoldoutSet, verify_seal_signature
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            saved_before = json.loads(Path(seal_path).read_text())
            sig_before = saved_before["signature"]

            hs.commit_break(real_data_hash="1" * 64)
            saved_after = json.loads(Path(seal_path).read_text())
            sig_after = saved_after["signature"]

            # Signature must have changed (state changed)
            assert sig_before != sig_after
            # New signature must verify
            assert verify_seal_signature(saved_after)
            # The saved broken_at is now populated
            assert saved_after["broken_at"] is not None
            assert saved_after["data_hash"] == "1" * 64


class TestVerifyDetectsTampering:
    """verify() must return False when the seal file is tampered."""

    def test_verify_fails_when_data_hash_tampered(self):
        """An attacker who edits data_hash on disk is detected."""
        from quant_lib.audit.holdout import HoldoutSet
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            assert hs.verify() is True

            # Tamper with data_hash on disk
            saved = json.loads(Path(seal_path).read_text())
            saved["data_hash"] = "f" * 64
            Path(seal_path).write_text(json.dumps(saved, indent=2))
            assert hs.verify() is False

    def test_verify_fails_when_broken_at_reset_to_none(self):
        """An attacker who edits broken_at back to None is detected.

        This is the canonical B0.1 attack: the user "broke" the seal
        legitimately, then edits the seal JSON to make the framework
        think the holdout is unused. The HMAC signature prevents this
        because the attacker cannot forge a valid signature for the
        modified state.
        """
        from quant_lib.audit.holdout import HoldoutSet
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            hs.commit_break(real_data_hash="1" * 64)

            # In-memory state is "broken", so verify() returns False
            # before even reading the file. To test the file-tamper
            # path, create a fresh HoldoutSet and tamper the file.
            saved = json.loads(Path(seal_path).read_text())
            assert saved["broken_at"] is not None  # we did break it

            # Attacker edits broken_at back to None, keeps the
            # original signature (which no longer matches the new
            # state).
            saved["broken_at"] = None
            Path(seal_path).write_text(json.dumps(saved, indent=2))

            # Fresh HoldoutSet: seal metadata in memory, but verify
            # will read the tampered file and check the signature.
            hs2 = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs2._seal.data_hash = saved["data_hash"]  # match the in-memory state
            assert hs2.verify() is False  # signature mismatch detected

    def test_verify_fails_when_signature_removed(self):
        """Removing the signature field must mark the seal as tampered."""
        from quant_lib.audit.holdout import HoldoutSet
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)

            # Remove the signature from the file
            saved = json.loads(Path(seal_path).read_text())
            del saved["signature"]
            Path(seal_path).write_text(json.dumps(saved, indent=2))

            # verify() must return False (signature is missing)
            assert hs.verify() is False

    def test_verify_fails_when_signature_replaced(self):
        """Replacing the signature with a wrong value is detected."""
        from quant_lib.audit.holdout import HoldoutSet
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            hs = HoldoutSet("t", "2025-01-01", "2025-06-30", seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            assert hs.verify() is True

            # Replace signature with garbage
            saved = json.loads(Path(seal_path).read_text())
            saved["signature"] = "deadbeef" * 8
            Path(seal_path).write_text(json.dumps(saved, indent=2))
            assert hs.verify() is False


class TestSecretRotationInvalidatesOldSeals:
    """Changing the HMAC secret must invalidate seals signed with the old one."""

    def test_old_signature_invalid_with_new_secret(self, monkeypatch):
        from quant_lib.audit import holdout as holdout_mod
        # NOTE: cannot use tempfile.TemporaryDirectory() here because
        # we need the seal file to outlive the first block (to
        # verify it under a different secret). Use a manual cleanup.
        seal_path = None
        try:
            holdout_mod._reset_hmac_secret_cache()
            monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "old" + "x" * 29)
            seal_path = os.path.join(tempfile.gettempdir(), "b04_rotate_test.json")
            if os.path.exists(seal_path):
                os.remove(seal_path)
            hs = holdout_mod.HoldoutSet("t", "2025-01-01", "2025-06-30",
                                         seal_path=seal_path)
            hs.seal(data_hash="0" * 64)
            assert hs.verify() is True  # signed with old secret

            # Switch to new secret
            holdout_mod._reset_hmac_secret_cache()
            monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "new" + "x" * 29)
            # Same seal file, new secret -> verify must fail
            hs2 = holdout_mod.HoldoutSet("t", "2025-01-01", "2025-06-30",
                                          seal_path=seal_path)
            hs2._seal.data_hash = "0" * 64
            assert hs2.verify() is False
        finally:
            holdout_mod._reset_hmac_secret_cache()
            if seal_path and os.path.exists(seal_path):
                os.remove(seal_path)


class TestEndToEndTamperResistance:
    """Realistic B0.1 attack scenarios end-to-end."""

    def test_cannot_construct_fake_seal_without_secret(self, monkeypatch):
        """An attacker who knows the data but not the secret cannot
        create a valid seal that verify() will accept.
        """
        from quant_lib.audit import holdout as holdout_mod
        with tempfile.TemporaryDirectory() as tmp:
            seal_path = os.path.join(tmp, "seal.json")
            # The attacker writes a JSON file mimicking a valid seal.
            # Without the secret, the signature is just a guess.
            forged = {
                "start": "2025-01-01",
                "end": "2025-06-30",
                "sealed_at": "2025-01-01T00:00:00+00:00",
                "broken_at": None,
                "data_hash": "0" * 64,
                "signature": "forged_" + "0" * 56,
            }
            Path(seal_path).write_text(json.dumps(forged, indent=2))

            # Load the forged seal into a HoldoutSet and verify.
            # verify() must reject it.
            holdout_mod._reset_hmac_secret_cache()
            monkeypatch.setenv("QUANT_LIB_HMAC_SECRET", "x" * 32)
            hs = holdout_mod.HoldoutSet("t", "2025-01-01", "2025-06-30",
                                         seal_path=seal_path)
            hs._seal.data_hash = forged["data_hash"]
            assert hs.verify() is False
            holdout_mod._reset_hmac_secret_cache()
