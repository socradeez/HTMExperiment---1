import numpy as np
from typing import List, Dict, Optional, Tuple


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


def generate_noisy_stream(
    V: int,
    N_total: int,
    sequences: List[List[int]],
    gap_dist: Tuple[str, float],
    noise_vocab: str = "in_dist",
    p_intra: float = 0.0,
    min_gap: int = 0,
    occurrence_weights: Optional[List[float]] = None,
    seed: Optional[int] = None,
):
    """Generate a token stream with random noise gaps between sequences.

    Returns a dictionary with keys:
        tokens, is_noise, seq_id, seq_pos, occurrence_id, phase
    """
    rng = np.random.default_rng(seed)

    if gap_dist[0] == "geometric":
        def draw_gap():
            g = int(rng.geometric(gap_dist[1]))
            return g
    elif gap_dist[0] == "poisson":
        def draw_gap():
            return int(rng.poisson(gap_dist[1]))
    else:
        raise ValueError("gap_dist must be ('geometric', p) or ('poisson', lam)")

    if occurrence_weights is not None:
        weights = np.array(occurrence_weights, dtype=float)
        weights = weights / weights.sum()
    else:
        weights = None

    noise_low, noise_high = (0, V) if noise_vocab == "in_dist" else (V, 2 * V)

    tokens: List[str] = []
    is_noise: List[int] = []
    seq_ids: List[int] = []
    seq_pos: List[int] = []
    occ_ids: List[int] = []
    phases: List[float] = []

    occ_counter = 0
    while len(tokens) < N_total:
        g = draw_gap()
        while g < min_gap:
            g = draw_gap()
        for _ in range(g):
            if len(tokens) >= N_total:
                break
            t = int(rng.integers(noise_low, noise_high))
            tokens.append(str(t))
            is_noise.append(1)
            seq_ids.append(-1)
            seq_pos.append(-1)
            occ_ids.append(-1)
            phases.append(-1.0)
        if len(tokens) >= N_total:
            break
        s_idx = int(rng.choice(len(sequences), p=weights))
        seq = sequences[s_idx]
        occ_counter += 1
        for pos, tok in enumerate(seq):
            if p_intra > 0 and rng.random() < p_intra:
                k = draw_gap()
                for _ in range(k):
                    if len(tokens) >= N_total:
                        break
                    t = int(rng.integers(noise_low, noise_high))
                    tokens.append(str(t))
                    is_noise.append(1)
                    seq_ids.append(-1)
                    seq_pos.append(-1)
                    occ_ids.append(-1)
                    phases.append(-1.0)
            if len(tokens) >= N_total:
                break
            tokens.append(str(tok))
            is_noise.append(0)
            seq_ids.append(s_idx)
            seq_pos.append(pos)
            occ_ids.append(occ_counter)
            phases.append((pos + 1) / len(seq))
        if len(tokens) >= N_total:
            break

    # truncate to N_total
    tokens = tokens[:N_total]
    is_noise = is_noise[:N_total]
    seq_ids = seq_ids[:N_total]
    seq_pos = seq_pos[:N_total]
    occ_ids = occ_ids[:N_total]
    phases = phases[:N_total]

    return {
        "tokens": tokens,
        "is_noise": is_noise,
        "seq_id": seq_ids,
        "seq_pos": seq_pos,
        "occurrence_id": occ_ids,
        "phase": phases,
    }


def build_token_sdrs_between_sequences(
    tokens: List[str],
    input_size: int,
    on_bits: int,
    overlap_pct: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Build SDRs with a shared core across all sequences.

    ``overlap_pct`` controls the fraction of active bits that every token
    shares with every other token. The remaining bits are sampled
    independently per token from the unused pool.
    """
    shared_size = int(on_bits * overlap_pct / 100)
    if shared_size > 0:
        shared = rng.choice(input_size, size=shared_size, replace=False)
    else:
        shared = np.array([], dtype=np.int32)
    unique_size = on_bits - shared_size
    pool = np.setdiff1d(np.arange(input_size), shared)
    token_map: Dict[str, np.ndarray] = {}
    for tok in tokens:
        if unique_size > 0:
            uniq = rng.choice(pool, size=unique_size, replace=False)
            bits = np.concatenate([shared, uniq])
        else:
            bits = shared.copy()
        token_map[tok] = np.sort(bits).astype(np.int32)
    return token_map
