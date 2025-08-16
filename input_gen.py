import numpy as np
from typing import List, Dict


def make_sequence_tokens(S: int, L: int) -> List[str]:
    """Return token IDs Sx_Ty for S sequences of length L."""
    tokens: List[str] = []
    for s in range(S):
        for t in range(L):
            tokens.append(f"S{s}_T{t}")
    return tokens


def build_token_sdrs(
    tokens: List[str],
    input_size: int,
    on_bits: int,
    overlap_pct: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Build SDRs with controlled within-sequence overlap.

    Each sequence (prefix Sx_) receives a shared core bitset sized by
    ``overlap_pct`` of ``on_bits``. Tokens in that sequence contain the core
    plus disjoint unique bits. Cores are disjoint across sequences to ensure
    no sequence shares its overlap bits with another.
    """
    # Group tokens by sequence prefix
    seq_tokens: Dict[str, List[str]] = {}
    for tok in tokens:
        seq = tok.split("_")[0]
        seq_tokens.setdefault(seq, []).append(tok)

    core_size = int(on_bits * overlap_pct / 100)
    unique_size = on_bits - core_size

    available_bits = set(range(input_size))
    seq_cores: Dict[str, set] = {}
    for seq in seq_tokens:
        if core_size > 0:
            chosen = rng.choice(np.array(list(available_bits)), size=core_size, replace=False)
            seq_cores[seq] = set(chosen.tolist())
            available_bits -= seq_cores[seq]
        else:
            seq_cores[seq] = set()

    all_cores = set().union(*seq_cores.values()) if seq_cores else set()

    token_map: Dict[str, np.ndarray] = {}
    base_pool = list(set(range(input_size)) - all_cores)

    for seq, toks in seq_tokens.items():
        used: set = set()
        for tok in toks:
            avail = list(set(base_pool) - used)
            if unique_size > 0:
                uniq = rng.choice(np.array(avail), size=unique_size, replace=False).tolist()
            else:
                uniq = []
            used.update(uniq)
            bits = seq_cores[seq].union(uniq)
            assert len(bits) == on_bits, f"{tok} SDR has {len(bits)} bits, expected {on_bits}"
            token_map[tok] = np.array(sorted(bits), dtype=np.int32)
    return token_map
