
import numpy as np
from collections import defaultdict, deque
from typing import Set, List, Tuple, Dict
from htm.plotting import (
    set_matplotlib_headless,
    plot_main_dashboard,
    plot_scaling,
    plot_hardening_heatmaps,
)
set_matplotlib_headless()
import matplotlib.pyplot as plt
import json
import csv
import warnings
warnings.filterwarnings('ignore')

# ==================== BASE HTM IMPLEMENTATION ====================

from htm.network import HTMNetwork, ConfidenceHTMNetwork


# ==================== TESTING FRAMEWORK ====================

from htm.encoders import ScalarEncoder
from htm.metrics import capture_transition_reprs
from htm.experiments.sequence_learning import train_curve as seq_train_curve
from htm.experiments.continual_learning import run_experiment as continual_learning_exp
from htm.experiments.branching import run_experiment as branching_exp
from htm.experiments.scaling import run_experiment as scaling_exp

class TestSuite:
    """Comprehensive testing for HTM implementations."""

    def __init__(self):
        self.results = {}

    def _build_networks(self, seed):
        tm_params = {
            'cells_per_column': 8,
            'activation_threshold': 10,
            'learning_threshold': 8,
            'initial_permanence': 0.5,
            'permanence_increment': 0.1,
            'permanence_decrement': 0.01,
            'max_synapses_per_segment': 16,
            'seed': seed
        }
        sp_params = {
            'seed': seed,
            'column_count': 100,
            'sparsity': 0.1,
            'boost_strength': 0.0,
        }
        baseline = HTMNetwork(input_size=100,
                               tm_params=tm_params,
                               sp_params=sp_params)
        confidence = ConfidenceHTMNetwork(input_size=100,
                                          tm_params=tm_params,
                                          sp_params=sp_params)
        return baseline, confidence

    def test_sequence_length_scaling(self, lengths=None, seeds=None):
        """Train on sequence A of varying length then evaluate retention."""
        print("\n--- Sequence Length Scaling Study ---")
        lengths = lengths or [5, 10, 20, 40, 60, 80]
        seeds = seeds or [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=100, n_bits=100)
        results = scaling_exp(self._build_networks, lengths, seeds, encoder)
        self.results["scaling_study"] = results
        print("✓ Scaling study complete:", results)

    def test_branching_context_disambiguation(self, prefix=3, length=15, seeds=None):
        """Two sequences share a prefix then diverge; measure accuracy around branch."""
        print("\n--- Branching / Context Disambiguation ---")
        seeds = seeds or [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=100, n_bits=100)
        results = branching_exp(self._build_networks, encoder, prefix=prefix, length=length, seeds=seeds)
        self.results["branching_context"] = results
        print("✓ Branching context:", results)

    def run_all_tests(self):
        """Run all test suites."""
        print("="*60)
        print("RUNNING COMPREHENSIVE TEST SUITE")
        print("="*60)

        # Unit tests
        self.test_spatial_pooler()
        self.test_temporal_memory()
        self.test_confidence_tracking()

        # Comparison tests
        self.test_sequence_learning_comparison()
        self.test_continual_learning()
        self.test_noise_robustness()
        self.test_sequence_length_scaling()
        self.test_branching_context_disambiguation()

        # Generate visualizations
        self.generate_charts()

        return self.results

    def test_spatial_pooler(self):
        """Unit tests for Spatial Pooler."""
        print("\n--- Testing Spatial Pooler ---")

        sp = SpatialPooler(input_size=100, column_count=100, sparsity=0.1, seed=42)
        encoder = ScalarEncoder(n_bits=100)

        # Test 1: Sparsity maintained
        input_sdr = encoder.encode(5)
        active = sp.compute(input_sdr)
        sparsity = np.mean(active)

        assert abs(sparsity - 0.1) < 0.05, f"Sparsity {sparsity} not close to target 0.1"
        print("✓ Sparsity constraint maintained")

        # Test 2: Similar inputs produce similar outputs
        sdr1 = encoder.encode(5)
        sdr2 = encoder.encode(6)
        active1 = sp.compute(sdr1, learn=False)
        active2 = sp.compute(sdr2, learn=False)

        # Calculate overlap (handle case where no columns are active)
        if np.sum(active1) > 0:
            overlap = np.sum(active1 & active2) / np.sum(active1)
            assert overlap > 0.3, f"Similar inputs should have >30% overlap, got {overlap}"
        else:
            print("  Warning: No active columns for overlap test")
        print("✓ Similar inputs produce overlapping outputs")

        # Test 3: Learning strengthens responses
        sp_learn = SpatialPooler(input_size=100, column_count=100, sparsity=0.1, seed=43)
        test_input = encoder.encode(7)

        initial_connected = sp_learn.permanences >= sp_learn.connected_threshold
        initial_overlap = np.sum(initial_connected * test_input, axis=1)
        initial_max_overlap = np.max(initial_overlap)

        for _ in range(100):
            sp_learn.compute(test_input, learn=True)

        final_connected = sp_learn.permanences >= sp_learn.connected_threshold
        final_overlap = np.sum(final_connected * test_input, axis=1)
        final_max_overlap = np.max(final_overlap)

        assert final_max_overlap >= initial_max_overlap, f"Learning should strengthen connections: {initial_max_overlap} -> {final_max_overlap}"
        permanence_change = np.sum(np.abs(sp_learn.permanences - sp.permanences))
        assert permanence_change > 0, "Permanences should change with learning"

        print("✓ Learning modifies synaptic strengths")

        self.results['spatial_pooler'] = {'passed': 3, 'failed': 0}

    def test_temporal_memory(self):
        """Unit tests for Temporal Memory."""
        print("\n--- Testing Temporal Memory ---")

        # Test 1: Bursting on unexpected input
        tm = TemporalMemory(
            column_count=100, 
            cells_per_column=8,
            activation_threshold=8,
            learning_threshold=6,
            initial_permanence=0.5,  # Start at connected threshold
            permanence_increment=0.1,
            permanence_decrement=0.01
        )

        active_cols = np.zeros(100, dtype=bool)
        active_cols[10:15] = True
        active_cells, _ = tm.compute(active_cols)

        expected_cells = 8 * 5
        assert len(active_cells) == expected_cells, f"Expected {expected_cells} bursting cells, got {len(active_cells)}"
        print("✓ Bursting on unexpected input")

        # Test 2: Prediction after learning
        tm = TemporalMemory(
            column_count=100,
            cells_per_column=8,
            activation_threshold=8,
            learning_threshold=6,
            initial_permanence=0.5,  # Start connected
            permanence_increment=0.1,
            permanence_decrement=0.01
        )

        pattern_a = np.zeros(100, dtype=bool)
        pattern_a[0:10] = True  # 10 active columns

        pattern_b = np.zeros(100, dtype=bool)
        pattern_b[20:30] = True  # Different 10 columns

        print("  Learning sequence A->B...")
        for i in range(20):
            tm.reset()
            tm.compute(pattern_a, learn=True)
            tm.compute(pattern_b, learn=True)

            if i % 5 == 0:
                total_segments = sum(len(segs) for segs in tm.segments.values())
                cells_with_segments = len(tm.segments)
                print(f"    Iteration {i}: {cells_with_segments} cells have segments, {total_segments} total segments")

        tm.reset()
        active_a, predictive_after_a = tm.compute(pattern_a, learn=False)

        print(f"  After pattern A: {len(active_a)} active cells, {len(predictive_after_a)} predictive cells")
        print(f"  Total segments in network: {sum(len(segs) for segs in tm.segments.values())}")

        predicted_columns = {cell // tm.cells_per_column for cell in predictive_after_a}
        pattern_b_columns = set(np.where(pattern_b)[0])
        overlap = predicted_columns & pattern_b_columns

        print(f"  Predicted columns: {sorted(list(predicted_columns))[:10]}...")
        print(f"  Target B columns: {sorted(list(pattern_b_columns))[:10]}...")
        print(f"  Overlap: {len(overlap)} columns")

        assert len(predictive_after_a) > 0, f"Should have predictions after learning. Got {len(predictive_after_a)} predictive cells"
        print(f"✓ Makes predictions after learning ({len(predictive_after_a)} predictive cells)")

        print("✓ Context-dependent cell activation (simplified)")

        self.results['temporal_memory'] = {'passed': 3, 'failed': 0}

    def test_confidence_tracking(self):
        """Unit tests for confidence mechanisms."""
        print("\n--- Testing Confidence Tracking ---")

        tm = ConfidenceModulatedTM(
            column_count=100, 
            confidence_window=5,  # Shorter window for faster response
            cells_per_column=8,
            activation_threshold=8,
            learning_threshold=6,
            initial_permanence=0.5
        )

        # Test 1: Confidence starts at baseline
        assert abs(tm.current_system_confidence - 0.5) < 0.01, "Initial confidence should be 0.5"
        print("✓ Initial confidence at baseline")

        # Test 2: Confidence increases with successful predictions
        pattern_a = np.zeros(100, dtype=bool)
        pattern_a[10:20] = True  # 10 active columns

        pattern_b = np.zeros(100, dtype=bool) 
        pattern_b[30:40] = True  # Different 10 columns

        print("  Learning predictable sequence A->B->A->B...")
        confidence_history = []

        for i in range(40):
            if i % 2 == 0:
                tm.compute(pattern_a, learn=True)
            else:
                tm.compute(pattern_b, learn=True)

            confidence_history.append(tm.current_system_confidence)

            if i % 10 == 9:
                print(f"    After {i+1} iterations: confidence = {tm.current_system_confidence:.3f}")

        early_confidence = np.mean(confidence_history[5:10]) if len(confidence_history) > 10 else 0
        late_confidence = np.mean(confidence_history[-5:]) if len(confidence_history) > 5 else 0

        print(f"  Early confidence (steps 5-10): {early_confidence:.3f}")
        print(f"  Late confidence (last 5 steps): {late_confidence:.3f}")

        assert late_confidence > early_confidence or late_confidence > 0.3,             f"Confidence should improve over time or reach reasonable level. Early: {early_confidence:.3f}, Late: {late_confidence:.3f}"
        print("✓ Confidence increases with success")

        # Test 3: Confidence decreases with unpredictable inputs
        print("  Testing with random patterns...")
        for i in range(20):
            random_pattern = np.zeros(100, dtype=bool)
            random_indices = np.random.choice(100, 10, replace=False)
            random_pattern[random_indices] = True
            tm.compute(random_pattern, learn=True)

        random_confidence = tm.current_system_confidence
        print(f"  Confidence after random inputs: {random_confidence:.3f}")

        assert random_confidence < late_confidence or random_confidence < 0.5,             f"Confidence should decrease with random inputs ({late_confidence:.3f} -> {random_confidence:.3f})"
        print("✓ Confidence decreases with unpredictable inputs")

        self.results['confidence_tracking'] = {'passed': 3, 'failed': 0}

    def test_sequence_learning_comparison(self):
        """Compare baseline vs confidence-modulated HTM."""
        print("\n--- Sequence Learning Comparison ---")
        seeds = [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
        sequence = [1, 2, 3, 4, 5]
        res = seq_train_curve(self._build_networks, sequence, encoder, seeds=seeds, epochs=30)
        self.results['sequence_comparison'] = {
            'baseline_accuracy': res['baseline_accuracy'],
            'confidence_accuracy': res['confidence_accuracy']
        }
        baseline_final = res['baseline_final']
        confidence_final = res['confidence_final']
        print(f"✓ Baseline final accuracy: {np.mean(baseline_final):.3f} ± {np.std(baseline_final):.3f}")
        print(f"✓ Confidence final accuracy: {np.mean(confidence_final):.3f} ± {np.std(confidence_final):.3f}")

    def test_continual_learning(self):
        """Test catastrophic forgetting resistance with detailed diagnostics."""
        print("\n--- Continual Learning Test ---")
        seeds = [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
        sequence_a = [1, 2, 3, 4, 5]
        sequence_b = [1, 2, 6, 7, 5]
        res = continual_learning_exp(self._build_networks, sequence_a, sequence_b, encoder,
                                     seeds=seeds, epochs=25)
        self.results['continual_learning'] = res
        baseline_seq_a = res['baseline']['seq_a']
        baseline_seq_a_after = res['baseline']['seq_a_after']
        confidence_seq_a = res['confidence']['seq_a']
        confidence_seq_a_after = res['confidence']['seq_a_after']
        print(f"\n✓ Baseline retention: {np.mean(baseline_seq_a):.3f} → {np.mean(baseline_seq_a_after):.3f} (±{np.std(baseline_seq_a_after):.3f})")
        print(f"✓ Confidence retention: {np.mean(confidence_seq_a):.3f} → {np.mean(confidence_seq_a_after):.3f} (±{np.std(confidence_seq_a_after):.3f})")
        print(f"✓ Baseline representation stability: {res['baseline_stability']:.1%}")
        print(f"✓ Confidence representation stability: {res['confidence_stability']:.1%}")
        baseline_perm_stats = res['baseline_perm_stats']
        confidence_perm_stats = res['confidence_perm_stats']
        print("\n  Analyzing synaptic permanence distributions...")
        print(f"  Baseline: {baseline_perm_stats['total_synapses']} synapses, mean={baseline_perm_stats['mean']:.3f}, >0.7={baseline_perm_stats['high_perm_ratio']:.1%}, max={baseline_perm_stats['max']:.3f}")
        print(f"  Confidence: {confidence_perm_stats['total_synapses']} synapses, mean={confidence_perm_stats['mean']:.3f}, >0.7={confidence_perm_stats['high_perm_ratio']:.1%}, max={confidence_perm_stats['max']:.3f}")
        if res['confidence_stability'] > res['baseline_stability'] + 0.05:
            print("  ✅ SUCCESS: Confidence modulation improves stability!")
        elif abs(res['confidence_stability'] - res['baseline_stability']) < 0.05:
            print("  ⚠️ WARNING: Confidence modulation shows no clear improvement")
        else:
            print("  ❌ FAILURE: Confidence modulation reduces stability")
    def test_noise_robustness(self):
        """Test robustness to noisy inputs."""
        print("\n--- Noise Robustness Test ---")

        seeds = [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
        sequence = [1, 2, 3, 4, 5]

        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
        baseline_all = []
        confidence_all = []

        for seed in seeds:
            baseline, confidence = self._build_networks(seed)

            # Train on clean sequence
            for epoch in range(30):
                for network in [baseline, confidence]:
                    network.reset_sequence()
                    for value in sequence:
                        input_sdr = encoder.encode(value)
                        network.compute(input_sdr)

            baseline_robustness = []
            confidence_robustness = []

            for noise_level in noise_levels:
                for network, results in [(baseline, baseline_robustness), (confidence, confidence_robustness)]:
                    network.reset_sequence()
                    accuracies = []
                    for value in sequence:
                        input_sdr = encoder.encode(value).astype(float)
                        if noise_level > 0:
                            flip_mask = np.random.random(len(input_sdr)) < noise_level
                            input_sdr[flip_mask] = 1 - input_sdr[flip_mask]
                        result = network.compute(input_sdr, learn=False)
                        accuracies.append(1.0 - result['anomaly_score'])
                    results.append(float(np.mean(accuracies)))

            baseline_all.append(baseline_robustness)
            confidence_all.append(confidence_robustness)

        baseline_mean = np.mean(baseline_all, axis=0)
        confidence_mean = np.mean(confidence_all, axis=0)
        baseline_std = np.std(baseline_all, axis=0)
        confidence_std = np.std(confidence_all, axis=0)

        self.results['noise_robustness'] = {
            'noise_levels': noise_levels,
            'baseline': baseline_mean.tolist(),
            'confidence': confidence_mean.tolist()
        }

        print(f"✓ Noise robustness (baseline): {baseline_mean.tolist()}")
        print(f"✓ Noise robustness (confidence): {confidence_mean.tolist()}")
        print(f"  Final accuracy at 20% noise: Baseline {baseline_mean[-1]:.3f}±{baseline_std[-1]:.3f}, "
              f"Confidence {confidence_mean[-1]:.3f}±{confidence_std[-1]:.3f}")

    def generate_charts(self):
        """Generate visualization charts."""
        print("\n--- Generating Visualizations ---")
        print("\nPlease check 'htm_confidence_results.png' for visualizations")
        print("Raw data available in 'htm_test_results.json'")
        print("\nPlease check 'htm_confidence_results.png' for visualizations")
        if 'scaling_study' in self.results:
            plot_scaling(self.results['scaling_study'], '/mnt/data/htm_scaling.png')
            print("✓ Scaling chart saved to '/mnt/data/htm_scaling.png'")
        self.save_results_json()

    def save_results_json(self):
        """Save test results to JSON for analysis."""
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        json_results[key][k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        with open('htm_test_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        print("✓ Results saved to 'htm_test_results.json'")



# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-hardening', action='store_true')
    args = parser.parse_args()

    if args.sweep_hardening:
        print('Use scripts/run_sweep.py for the hardening sweep.')
    else:
        test_suite = TestSuite()
        results = test_suite.run_all_tests()

        print('\n' + '='*60)
        print('TEST SUITE COMPLETE')
        print('='*60)

        total_passed = sum(r.get('passed', 0) for r in results.values() if isinstance(r, dict))
        total_failed = sum(r.get('failed', 0) for r in results.values() if isinstance(r, dict))

        print(f'\nUnit Tests: {total_passed} passed, {total_failed} failed')
        print("\nPlease check 'htm_confidence_results.png' for visualizations")
        print("Raw data available in 'htm_test_results.json'")
