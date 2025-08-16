
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from config import ModelConfig

def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def pick_k_best(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k >= len(scores):
        return np.arange(len(scores))
    jitter = rng.uniform(0, 1e-6, size=scores.shape)
    idx = np.argpartition(scores + jitter, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx

@dataclass
class SpatialPooler:
    cfg: ModelConfig
    rng: np.random.Generator
    proximal_idx: np.ndarray
    proximal_perm: np.ndarray

    @classmethod
    def create(cls, cfg: ModelConfig, rng: np.random.Generator) -> 'SpatialPooler':
        idx = rng.integers(0, cfg.input_size, size=(cfg.num_columns, cfg.synapses_per_column), endpoint=False)
        perm = rng.normal(loc=cfg.init_perm_mean, scale=cfg.init_perm_sd, size=(cfg.num_columns, cfg.synapses_per_column))
        perm = np.clip(perm, 0.0, 1.0)
        return cls(cfg=cfg, rng=rng, proximal_idx=idx, proximal_perm=perm)

    def compute_overlap(self, inp: np.ndarray) -> np.ndarray:
        connected = self.proximal_perm >= self.cfg.perm_connected
        inp_active = inp[self.proximal_idx]
        overlap = np.sum(connected & (inp_active > 0), axis=1).astype(np.float32)
        return overlap

    def k_wta(self, overlap: np.ndarray) -> np.ndarray:
        return pick_k_best(overlap, self.cfg.k_active_columns, self.rng)

    def learn(self, inp: np.ndarray, active_columns: np.ndarray):
        if active_columns.size == 0:
            return
        idx = self.proximal_idx[active_columns]
        perm = self.proximal_perm[active_columns]
        active_mask = (inp[idx] > 0)
        perm = perm + self.cfg.perm_inc * active_mask - self.cfg.perm_dec * (~active_mask)
        np.clip(perm, 0.0, 1.0, out=perm)
        self.proximal_perm[active_columns] = perm

@dataclass
class Segment:
    presyn_cells: np.ndarray
    permanences: np.ndarray

    def connected_mask(self, thr: float) -> np.ndarray:
        return self.permanences >= thr

    def activation_count(self, active_cells_prev: Set[int], thr: float) -> int:
        mask = self.connected_mask(thr)
        if mask.sum() == 0:
            return 0
        pres = self.presyn_cells[mask]
        return int(np.isin(pres, list(active_cells_prev)).sum())

@dataclass
class TemporalMemory:
    cfg: ModelConfig
    rng: np.random.Generator
    segments: Dict[int, List[Segment]]

    @classmethod
    def create(cls, cfg: ModelConfig, rng: np.random.Generator) -> 'TemporalMemory':
        return cls(cfg=cfg, rng=rng, segments={})

    def _cell_id(self, col: int, cell: int) -> int:
        return col * self.cfg.cells_per_column + cell

    def _column_range(self, col: int) -> List[int]:
        base = col * self.cfg.cells_per_column
        return list(range(base, base + self.cfg.cells_per_column))

    def compute_predictive_cells(self, active_cells_prev: Set[int]) -> Set[int]:
        predictive: Set[int] = set()
        if not active_cells_prev:
            return predictive
        thr = self.cfg.segment_activation_threshold
        for cell_id, segs in self.segments.items():
            for seg in segs:
                if seg.activation_count(active_cells_prev, self.cfg.perm_connected) >= thr:
                    predictive.add(cell_id)
                    break
        return predictive

    def activate_cells(self, active_columns: np.ndarray, predictive_prev: Set[int]) -> Tuple[Set[int], Dict[int, List[Segment]]]:
        active_cells: Set[int] = set()
        active_segments: Dict[int, List[Segment]] = {}
        for c in active_columns:
            col_cells = self._column_range(c)
            preds = [cell for cell in col_cells if cell in predictive_prev]
            if preds:
                for cell in preds:
                    active_cells.add(cell)
                    if cell in self.segments:
                        segs = []
                        for seg in self.segments[cell]:
                            if seg.activation_count(active_cells_prev=active_cells_prev_global, thr=self.cfg.perm_connected) >= self.cfg.segment_activation_threshold:
                                segs.append(seg)
                        if segs:
                            active_segments[cell] = segs
            else:
                active_cells.update(col_cells)
        return active_cells, active_segments

    def _best_matching_cell(self, col: int, active_cells_prev: Set[int]) -> int:
        best_cell = None
        best_score = -1
        for cell in self._column_range(col):
            score = 0
            for seg in self.segments.get(cell, []):
                score = max(score, seg.activation_count(active_cells_prev, self.cfg.perm_connected))
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            best_cell = self._column_range(col)[self.rng.integers(0, self.cfg.cells_per_column)]
        return best_cell

    def _grow_new_segment(self, cell_id: int, active_cells_prev: Set[int]):
        if not active_cells_prev:
            return
        pres = np.array(list(active_cells_prev), dtype=np.int32)
        if pres.size == 0:
            return
        if pres.size > self.cfg.distal_synapses_per_segment:
            pres = self.rng.choice(pres, size=self.cfg.distal_synapses_per_segment, replace=False)
        perms = self.rng.normal(self.cfg.new_segment_init_perm_mean, self.cfg.new_segment_init_perm_sd, size=pres.shape[0])
        perms = np.clip(perms, 0.0, 1.0)
        seg = Segment(presyn_cells=pres.astype(np.int32), permanences=perms.astype(np.float32))
        self.segments.setdefault(cell_id, []).append(seg)

    def learn(self,
              active_cells_prev: Set[int],
              active_columns: np.ndarray,
              active_cells: Set[int],
              active_segments: Dict[int, List[Segment]],
              predictive_prev: Set[int]):
        """Update synapses based on activity at the current timestep.

        Parameters
        ----------
        active_cells_prev : Set[int]
            Cells active at t−1.
        active_columns : np.ndarray
            Columns that won inhibition at t.
        active_cells : Set[int]
            Cells active at t after applying predictions (non-bursting cells suppressed).
        active_segments : Dict[int, List[Segment]]
            Segments that were active at t.
        predictive_prev : Set[int]
            Cells that were predicted for t based on state at t−1. Used to
            determine which columns were truly predicted versus bursting.
        """
        predicted_cols = {cell // self.cfg.cells_per_column for cell in predictive_prev}
        bursting_cols = set(active_columns) - predicted_cols

        for cell_id, segs in active_segments.items():
            for seg in segs:
                if seg.presyn_cells.size == 0:
                    continue
                is_prev = np.isin(seg.presyn_cells, list(active_cells_prev))
                seg.permanences[is_prev] += self.cfg.perm_inc
                seg.permanences[~is_prev] -= self.cfg.perm_dec
                np.clip(seg.permanences, 0.0, 1.0, out=seg.permanences)

        for col in bursting_cols:
            winner_cell = self._best_matching_cell(col, active_cells_prev)
            self._grow_new_segment(winner_cell, active_cells_prev)

active_cells_prev_global: Set[int] = set()
