"""
duplicate_detector.py

Detects duplicate and near-duplicate documents.

Techniques used

1. SHA-256
   Exact duplicate detection

2. SimHash
   Near duplicate detection
"""

import hashlib
import re
from collections import Counter


# -------------------------------------------------------
# Exact Duplicate Detection
# -------------------------------------------------------

def generate_sha256(text: str) -> str:
    """
    Generate SHA256 hash of a document.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------

def tokenize(text: str):

    text = text.lower()

    words = re.findall(r"[a-zA-Z0-9]+", text)

    return words


# -------------------------------------------------------
# SimHash
# -------------------------------------------------------

def simhash(text: str, bits=64):

    words = tokenize(text)

    frequency = Counter(words)

    vector = [0] * bits

    for word, weight in frequency.items():

        h = int(
            hashlib.md5(
                word.encode("utf-8")
            ).hexdigest(),
            16
        )

        for i in range(bits):

            bit = (h >> i) & 1

            if bit:

                vector[i] += weight

            else:

                vector[i] -= weight

    fingerprint = 0

    for i in range(bits):

        if vector[i] > 0:

            fingerprint |= 1 << i

    return fingerprint


# -------------------------------------------------------
# Hamming Distance
# -------------------------------------------------------

def hamming_distance(hash1, hash2):

    return bin(hash1 ^ hash2).count("1")


# -------------------------------------------------------
# Near Duplicate
# -------------------------------------------------------

def is_near_duplicate(
        text1,
        text2,
        threshold=5):

    hash1 = simhash(text1)

    hash2 = simhash(text2)

    distance = hamming_distance(hash1, hash2)

    return distance <= threshold


# -------------------------------------------------------
# Similarity Score
# -------------------------------------------------------

def similarity_percentage(text1, text2):

    hash1 = simhash(text1)

    hash2 = simhash(text2)

    distance = hamming_distance(hash1, hash2)

    similarity = (64 - distance) / 64

    return similarity * 100