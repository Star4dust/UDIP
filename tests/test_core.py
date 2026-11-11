import pytest
import os
import tempfile
from pathlib import Path
from modules.core import encrypt, decrypt, safe_write, _get_machine_key

def test_encryption_round_trip():
    """Test that encrypt/decrypt works correctly."""
    plaintext = "test_groq_key_123"
    encrypted = encrypt(plaintext)
    decrypted = decrypt(encrypted)
    assert decrypted == plaintext

def test_decrypt_invalid():
    """Test decrypt with invalid ciphertext returns empty string."""
    assert decrypt("invalid") == ""

def test_machine_key_consistent():
    """Test that machine key is consistent."""
    key1 = _get_machine_key()
    key2 = _get_machine_key()
    assert key1 == key2
    assert len(key1) == 32  # SHA256

def test_safe_write():
    """Test safe_write creates file correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        content = "test content"
        safe_write(path, content)
        assert path.read_text() == content