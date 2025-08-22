import logging
import numpy as np
from typing import List, Dict


def make_sequence_tokens(S: int, L: int) -> List[str]:
    """Return token IDs Sx_Ty for S sequences of length L."""
    tokens: List[str] = []
    for s in range(S):
        for t in range(L):
            tokens.append(f"S{s}_T{t}")
    return tokens


logger = logging.getLogger(__name__)


def build_token_sdrs(
    tokens: List[str],
    input_size: int,
    on_bits: int,
    overlap_pct: int,
    rng: np.random.Generator,
    cross_sequence_reuse: bool = True,
    enforce_global_unique: bool = False,
) -> Dict[str, np.ndarray]:
    """Build SDRs with controlled within-sequence overlap.

    Parameters
    ----------
    cross_sequence_reuse : bool, default True
        Each sequence draws from its own local bit pool 0..``input_size``-1 so
        bits may repeat across sequences. Overlap/uniqueness are enforced only
        within a sequence.
    enforce_global_unique : bool, default False
        Preserve the previous global uniqueness behaviour. Raises an error if
        the requested configuration cannot be satisfied without reusing bits.
    """
    # Group tokens by sequence prefix
    seq_tokens: Dict[str, List[str]] = {}
    for tok in tokens:
        seq = tok.split("_")[0]
        seq_tokens.setdefault(seq, []).append(tok)

    core_size = int(on_bits * overlap_pct / 100)
    unique_size = on_bits - core_size

    S = len(seq_tokens)
    L = len(next(iter(seq_tokens.values()))) if seq_tokens else 0

    token_map: Dict[str, np.ndarray] = {}

    if enforce_global_unique:
        # Old behaviour: no reuse across sequences
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
        base_pool = list(set(range(input_size)) - all_cores)

        for seq, toks in seq_tokens.items():
            logger.info(
                f"[input_gen] seq={seq} k={on_bits} overlap={overlap_pct}% "
                f"unique_per_token≈{unique_size} pool={input_size}"
            )
            used: set = set()
            for tok in toks:
                avail = list(set(base_pool) - used)
                replace_needed = unique_size > len(avail)
                if replace_needed and enforce_global_unique:
                    raise ValueError(
                        f"Infeasible: need {unique_size} unique bits but only {len(avail)} available "
                        f"(N={input_size}, L={L}, S={S}, k={on_bits}, overlap={overlap_pct}%)"
                    )
                uniq = (
                    rng.choice(np.array(avail), size=unique_size, replace=replace_needed).tolist()
                    if unique_size > 0
                    else []
                )
                used.update(uniq)
                bits = seq_cores[seq].union(uniq)
                assert len(bits) == on_bits, f"{tok} SDR has {len(bits)} bits, expected {on_bits}"
                token_map[tok] = np.array(sorted(bits), dtype=np.int32)
        return token_map

    # cross-sequence reuse path: independent bit pools per sequence
    for seq, toks in seq_tokens.items():
        logger.info(
            f"[input_gen] seq={seq} k={on_bits} overlap={overlap_pct}% "
            f"unique_per_token≈{unique_size} pool={input_size}"
        )
        local_pool = list(range(input_size))
        core = (
            set(rng.choice(np.array(local_pool), size=core_size, replace=False).tolist())
            if core_size > 0
            else set()
        )
        used: set = set(core)
        for tok in toks:
            avail = list(set(local_pool) - used)
            replace_needed = unique_size > len(avail)
            if replace_needed and enforce_global_unique:
                raise ValueError(
                    f"Infeasible: need {unique_size} unique bits but only {len(avail)} available "
                    f"(N={input_size}, L={L}, S={S}, k={on_bits}, overlap={overlap_pct}%)"
                )
            uniq = (
                rng.choice(np.array(avail), size=unique_size, replace=replace_needed).tolist()
                if unique_size > 0
                else []
            )
            used.update(uniq)
            bits = core.union(uniq)
            assert len(bits) == on_bits, f"{tok} SDR has {len(bits)} bits, expected {on_bits}"
            token_map[tok] = np.array(sorted(bits), dtype=np.int32)
    return token_map
