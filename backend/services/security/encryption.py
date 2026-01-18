from __future__ import annotations

import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_bytes(key: bytes, data: bytes) -> bytes:
    aes = AESGCM(key)
    nonce, ct = data[:12], data[12:]
    return aes.decrypt(nonce, ct, None)
