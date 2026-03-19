"""KB noise utilities for robustness experiments."""

from __future__ import annotations

import random


def drop_top_chunk(chunks: list[dict], drop_prob: float) -> list[dict]:
    """Drop the top chunk with probability drop_prob.

    Args:
      chunks: retrieved chunks sorted by similarity (best first).
      drop_prob: probability in [0,1] of dropping the top chunk.
    Returns:
      New list of chunks.
    """
    if not chunks:
        return []
    if drop_prob <= 0.0:
        return list(chunks)
    if drop_prob >= 1.0:
        return list(chunks[1:])
    if random.random() < drop_prob:
        return list(chunks[1:])
    return list(chunks)


def inject_offtopic_noise(
    chunks: list[dict],
    global_kb: list[dict],
    noise_ratio: float,
) -> list[dict]:
    """Replace a fraction of chunks with random off-topic KB entries.

    Args:
      chunks: relevant retrieved chunks.
      global_kb: full KB list to sample noise from.
      noise_ratio: fraction of chunks to replace with random entries.
    Returns:
      Mixed list of chunks with some replaced by noise.
    """
    if not chunks:
        return []
    if not global_kb:
        return list(chunks)
    if noise_ratio <= 0.0:
        return list(chunks)

    n = len(chunks)
    k = int(round(n * noise_ratio))
    k = max(0, min(k, n))
    if k == 0:
        return list(chunks)

    indices = random.sample(range(n), k)
    out = list(chunks)
    for idx in indices:
        out[idx] = random.choice(global_kb)
    return out


def limit_k(chunks: list[dict], k: int) -> list[dict]:
    """Limit retrieved chunks to top-k."""
    if k <= 0:
        return []
    return list(chunks[:k])
