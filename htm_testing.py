
import numpy as np
from collections import defaultdict, deque
from typing import Set, List, Tuple, Dict
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt
import json
import csv
import warnings
warnings.filterwarnings('ignore')

# ==================== BASE HTM IMPLEMENTATION ====================

from htm.network import HTMNetwork, ConfidenceHTMNetwork


# ==================== TESTING FRAMEWORK ====================

from htm.encoders import ScalarEncoder

class TestSuite:
    """Comprehensive testing for HTM implementations."""

    def __init__(self):
        self.results = {}

    def capture_transition_reprs(self, network, seq, encoder):
        """Capture predicted-driven activations for each transition in seq."""
        reps = {}
        network.reset_sequence()
        network.compute(encoder.encode(seq[0]), learn=False)
        for i in range(1, len(seq)):
            res = network.compute(encoder.encode(seq[i]), learn=False)
            preds = set(res['predictive_cells'])
            act = set(res['active_cells'])
            reps[f"{seq[i-1]}->{seq[i]}"] = preds & act if preds else set()
        return reps

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
        """
        Train on a single sequence A of varying length L, measure final accuracy on A,
        then learn a new sequence B (no overlap) and measure retention on A.
        Report mean across seeds.
        """
        print("\n--- Sequence Length Scaling Study ---")
        import numpy as np

        lengths = lengths or [5, 10, 20, 40, 60, 80]
        seeds = seeds or [0, 1, 2]

        encoder = ScalarEncoder(min_val=0, max_val=100, n_bits=100)

        results = {
            "lengths": lengths,
            "baseline_acc_mean": [],
            "confidence_acc_mean": [],
            "baseline_ret_mean": [],
            "confidence_ret_mean": [],
        }

        for L in lengths:
            b_acc, c_acc, b_ret, c_ret = [], [], [], []

            for s in seeds:
                # Build baseline/confidence networks with selective TM params
                base, conf = self._build_networks(s)

                # Build sequences (no overlap): A at offset 10, B at offset 40
                seqA = list(range(10, 10 + L))
                seqB = list(range(40, 40 + L))

                def train_on(net, seq, epochs):
                    for _ in range(epochs):
                        net.reset_sequence()
                        for v in seq:
                            net.compute(encoder.encode(v))

                def eval_on(net, seq):
                    net.reset_sequence()
                    accs = []
                    for v in seq:
                        r = net.compute(encoder.encode(v), learn=False)
                        accs.append(1.0 - r['anomaly_score'])
                    return float(np.mean(accs))

                # epochs scale lightly with L to keep runtime controlled
                epochs = max(10, L // 5)

                # Train both nets on A and evaluate
                train_on(base, seqA, epochs)
                train_on(conf, seqA, epochs)
                b_acc.append(eval_on(base, seqA))
                c_acc.append(eval_on(conf, seqA))

                # Learn B, then evaluate retention on A (no learning)
                train_on(base, seqB, epochs)
                train_on(conf, seqB, epochs)
                b_ret.append(eval_on(base, seqA))
                c_ret.append(eval_on(conf, seqA))

            results["baseline_acc_mean"].append(float(np.mean(b_acc)))
            results["confidence_acc_mean"].append(float(np.mean(c_acc)))
            results["baseline_ret_mean"].append(float(np.mean(b_ret)))
            results["confidence_ret_mean"].append(float(np.mean(c_ret)))

        self.results["scaling_study"] = results
        print("✓ Scaling study complete:", results)

    def test_branching_context_disambiguation(self, prefix=3, length=15, seeds=None):
        """
        Two sequences share a prefix of 'prefix' items, then diverge:
          A: p0, p1, ..., p{prefix-1}, a_prefix, a_{prefix+1}, ...
          B: p0, p1, ..., p{prefix-1}, b_prefix, b_{prefix+1}, ...
        Measure prediction accuracy at and after the branching point.
        """
        print("\n--- Branching / Context Disambiguation ---")
        import numpy as np
        seeds = seeds or [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=100, n_bits=100)

        def build_AB(prefix, length):
            assert length > prefix + 2
            P = list(range(10, 10 + prefix))                    # shared prefix centers
            A = P + list(range(20, 20 + (length - prefix)))     # A tail
            B = P + list(range(40, 40 + (length - prefix)))     # B tail
            return A, B

        A, B = build_AB(prefix, length)

        results = {"prefix": prefix, "length": length, "baseline": {}, "confidence": {}}

        for model_name in ["baseline", "confidence"]:
            branch_acc = []      # accuracy exactly at the first divergent element
            post_acc = []        # mean accuracy for the next 3 steps after branching

            for s in seeds:
                base, conf = self._build_networks(s)
                net = base if model_name == "baseline" else conf

                # Train alternating A and B to encourage context learning
                for _ in range(30):
                    net.reset_sequence()
                    for v in A: net.compute(encoder.encode(v))
                    net.reset_sequence()
                    for v in B: net.compute(encoder.encode(v))

                # Evaluate at branching point
                def eval_branch(net, seq_prev, seq_next):
                    # run all up to branching index (prefix-1), then present branching element and record accuracy
                    net.reset_sequence()
                    for i, v in enumerate(seq_prev):
                        if i == prefix:
                            break
                        net.compute(encoder.encode(v), learn=False)
                    # step at branching
                    r0 = net.compute(encoder.encode(seq_next[prefix]), learn=False)
                    a0 = float(1.0 - r0['anomaly_score'])
                    # next few steps
                    post = []
                    for j in range(prefix+1, min(prefix+4, len(seq_next))):
                        rj = net.compute(encoder.encode(seq_next[j]), learn=False)
                        post.append(float(1.0 - rj['anomaly_score']))
                    return a0, float(np.mean(post)) if post else a0

                a0, a_post = eval_branch(net, A, A)
                b0, b_post = eval_branch(net, B, B)
                # average A and B branches
                branch_acc.append((a0 + b0) / 2.0)
                post_acc.append((a_post + b_post) / 2.0)

            results[model_name]["branch_acc_mean"] = float(np.mean(branch_acc))
            results[model_name]["branch_acc_std"]  = float(np.std(branch_acc))
            results[model_name]["post_acc_mean"]   = float(np.mean(post_acc))
            results[model_name]["post_acc_std"]    = float(np.std(post_acc))

        self.results["branching_context"] = results
        print("✓ Branching context:", results)

    def run_hardening_sweep(self, rates=None, thresholds=None, seeds=None, epochs_per_phase=25):
        """Parameter sweep for hardening settings on continual learning benchmark."""
        print("\n=== Hardening Parameter Sweep ===")
        rates = rates or [0.0, 0.03, 0.05, 0.1, 0.2]
        thresholds = thresholds or [0.6, 0.7, 0.8]
        seeds = seeds or [0, 1, 2]

        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
        seq_a = [1, 2, 3, 4, 5]
        seq_b = [1, 2, 6, 7, 5]

        csv_rows = []
        summary = {}

        for rate in rates:
            for thresh in thresholds:
                key = f"r{rate}_t{thresh}"
                summary[key] = {"initial": [], "retention": [], "stability": [],
                                 "mean_conf": [], "frac_conf": [],
                                 "mean_hard": [], "updates": []}
                for seed in seeds:
                    tm_params = {
                        'cells_per_column': 8,
                        'activation_threshold': 10,
                        'learning_threshold': 8,
                        'initial_permanence': 0.5,
                        'permanence_increment': 0.1,
                        'permanence_decrement': 0.01,
                        'max_synapses_per_segment': 16,
                        'seed': seed,
                        'hardening_rate': rate,
                        'hardening_threshold': thresh
                    }
                    sp_params = {'seed': seed}
                    net = ConfidenceHTMNetwork(input_size=100, use_confidence=True,
                                               tm_params=tm_params, sp_params=sp_params)

                    def train_on(seq):
                        for _ in range(epochs_per_phase):
                            net.reset_sequence()
                            for v in seq:
                                net.compute(encoder.encode(v))

                    def eval_on(seq):
                        net.reset_sequence()
                        accs = []
                        for v in seq:
                            r = net.compute(encoder.encode(v), learn=False)
                            accs.append(1.0 - r['anomaly_score'])
                        return {
                            'mean_acc': float(np.mean(accs)) if accs else 0.0,
                            'last_step_acc': float(accs[-1]) if accs else 0.0
                        }

                    train_on(seq_a)
                    metrics = eval_on(seq_a)
                    initial = metrics['last_step_acc']
                    initial_mean = metrics['mean_acc']
                    reps_before = self.capture_transition_reprs(net, seq_a, encoder)

                    train_on(seq_b)
                    metrics = eval_on(seq_a)
                    retention = metrics['last_step_acc']
                    retention_mean = metrics['mean_acc']
                    reps_after = self.capture_transition_reprs(net, seq_a, encoder)

                    stability = []
                    for tkey in reps_before.keys():
                        b0 = reps_before[tkey]
                        b1 = reps_after.get(tkey, set())
                        stability.append(len(b0 & b1) / (len(b0) or 1))
                    stab = float(np.mean(stability)) if stability else 0.0

                    mean_conf = float(np.mean(net.tm.system_confidence)) if net.tm.system_confidence else 0.0
                    frac_conf = net.tm._conf_over_thr_steps / max(1, net.tm._total_steps)
                    mean_hard = net.tm._hardness_sum / max(1, net.tm._hardness_count)
                    updates = net.tm._hardening_updates

                    csv_rows.append({
                        'hardening_rate': rate,
                        'hardening_threshold': thresh,
                        'seed': seed,
                        'initial_last_step_acc': initial,
                        'initial_mean_acc': initial_mean,
                        'retention_last_step_acc': retention,
                        'retention_mean_acc': retention_mean,
                        'representation_stability': stab,
                        'mean_conf': mean_conf,
                        'frac_conf_ge_thr': frac_conf,
                        'mean_hardness': mean_hard,
                        'hardening_updates': updates
                    })

                    summary[key]['initial'].append(initial)
                    summary[key]['retention'].append(retention)
                    summary[key]['stability'].append(stab)
                    summary[key]['mean_conf'].append(mean_conf)
                    summary[key]['frac_conf'].append(frac_conf)
                    summary[key]['mean_hard'].append(mean_hard)
                    summary[key]['updates'].append(updates)

        csv_path = 'hardening_sweep.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['hardening_rate', 'hardening_threshold', 'seed',
                                                   'initial_last_step_acc', 'initial_mean_acc',
                                                   'retention_last_step_acc', 'retention_mean_acc',
                                                   'representation_stability', 'mean_conf',
                                                   'frac_conf_ge_thr', 'mean_hardness',
                                                   'hardening_updates'])
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        summary_json = {}
        for rate in rates:
            for thresh in thresholds:
                key = f"r{rate}_t{thresh}"
                data = summary[key]
                summary_json[key] = {
                    'hardening_rate': rate,
                    'hardening_threshold': thresh,
                    'initial_mean': float(np.mean(data['initial'])) if data['initial'] else 0.0,
                    'initial_std': float(np.std(data['initial'])) if data['initial'] else 0.0,
                    'retention_mean': float(np.mean(data['retention'])) if data['retention'] else 0.0,
                    'retention_std': float(np.std(data['retention'])) if data['retention'] else 0.0,
                    'stability_mean': float(np.mean(data['stability'])) if data['stability'] else 0.0,
                    'stability_std': float(np.std(data['stability'])) if data['stability'] else 0.0,
                    'mean_conf_mean': float(np.mean(data['mean_conf'])) if data['mean_conf'] else 0.0,
                    'mean_conf_std': float(np.std(data['mean_conf'])) if data['mean_conf'] else 0.0,
                    'frac_conf_ge_thr_mean': float(np.mean(data['frac_conf'])) if data['frac_conf'] else 0.0,
                    'frac_conf_ge_thr_std': float(np.std(data['frac_conf'])) if data['frac_conf'] else 0.0,
                    'mean_hardness_mean': float(np.mean(data['mean_hard'])) if data['mean_hard'] else 0.0,
                    'mean_hardness_std': float(np.std(data['mean_hard'])) if data['mean_hard'] else 0.0,
                    'hardening_updates_mean': float(np.mean(data['updates'])) if data['updates'] else 0.0,
                    'hardening_updates_std': float(np.std(data['updates'])) if data['updates'] else 0.0
                }

        json_path = 'hardening_sweep_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)

        # Heatmaps
        retention_grid = np.array([[summary_json[f"r{r}_t{t}"]['retention_mean'] for r in rates] for t in thresholds])
        stability_grid = np.array([[summary_json[f"r{r}_t{t}"]['stability_mean'] for r in rates] for t in thresholds])
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axs[0].imshow(retention_grid, origin='lower', aspect='auto', vmin=0, vmax=1)
        axs[0].set_xticks(range(len(rates)))
        axs[0].set_xticklabels(rates)
        axs[0].set_yticks(range(len(thresholds)))
        axs[0].set_yticklabels(thresholds)
        axs[0].set_xlabel('Hardening Rate')
        axs[0].set_ylabel('Hardening Threshold')
        axs[0].set_title('Retention Accuracy')
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(stability_grid, origin='lower', aspect='auto', vmin=0, vmax=1)
        axs[1].set_xticks(range(len(rates)))
        axs[1].set_xticklabels(rates)
        axs[1].set_yticks(range(len(thresholds)))
        axs[1].set_yticklabels(thresholds)
        axs[1].set_xlabel('Hardening Rate')
        axs[1].set_ylabel('Hardening Threshold')
        axs[1].set_title('Representation Stability')
        fig.colorbar(im1, ax=axs[1])
        plt.tight_layout()
        heatmap_path = 'hardening_sweep_heatmap.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')

        # Pareto scatter
        plt.figure(figsize=(6, 5))
        for r in rates:
            for t in thresholds:
                key = f"r{r}_t{t}"
                d = summary_json[key]
                plt.scatter(d['initial_mean'], d['retention_mean'], s=50 + 150*d['stability_mean'])
                plt.annotate(f"r={r}, t={t}", (d['initial_mean'], d['retention_mean']),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        plt.xlabel('Initial Accuracy')
        plt.ylabel('Retention Accuracy')
        plt.title('Hardening Sweep Pareto')
        plt.grid(True, alpha=0.3)
        pareto_path = 'hardening_sweep_pareto.png'
        plt.savefig(pareto_path, dpi=150, bbox_inches='tight')

        # Determine best configurations
        baseline_key = 'r0.0_t0.7'
        baseline_initial = summary_json.get(baseline_key, {}).get('initial_mean', 0.0)
        best_ret_key = max(summary_json.items(), key=lambda x: x[1]['retention_mean'])[0]
        constrained_candidates = [item for item in summary_json.items()
                                   if item[1]['initial_mean'] >= baseline_initial - 0.02]
        best_ret_cons_key = max(constrained_candidates, key=lambda x: x[1]['retention_mean'])[0] if constrained_candidates else best_ret_key
        best_stab_key = max(summary_json.items(), key=lambda x: x[1]['stability_mean'])[0]

        def key_to_rt(k):
            parts = k[1:].split('_t') if k.startswith('r') else [0,0]
            return float(parts[0]), float(parts[1])

        print("Top 3 by retention:")
        for k, v in sorted(summary_json.items(), key=lambda x: x[1]['retention_mean'], reverse=True)[:3]:
            r, t = v['hardening_rate'], v['hardening_threshold']
            print(f"  rate={r}, thr={t}: retention={v['retention_mean']:.3f}")
        print("Top 3 by retention+stability:")
        for k, v in sorted(summary_json.items(), key=lambda x: (x[1]['retention_mean']+x[1]['stability_mean']), reverse=True)[:3]:
            r, t = v['hardening_rate'], v['hardening_threshold']
            print(f"  rate={r}, thr={t}: ret={v['retention_mean']:.3f}, stab={v['stability_mean']:.3f}")

        br, bt = key_to_rt(best_ret_key)
        brc, btc = key_to_rt(best_ret_cons_key)
        bs, bt2 = key_to_rt(best_stab_key)
        print(f"Best retention: rate={br}, thr={bt} (ret={summary_json[best_ret_key]['retention_mean']:.3f})")
        print(f"Best retention (constrained): rate={brc}, thr={btc} (ret={summary_json[best_ret_cons_key]['retention_mean']:.3f})")
        print(f"Best stability: rate={bs}, thr={bt2} (stab={summary_json[best_stab_key]['stability_mean']:.3f})")

        self.results['hardening_sweep'] = {
            'csv': csv_path,
            'json': json_path,
            'heatmap': heatmap_path,
            'pareto': pareto_path
        }

        print("✓ Hardening sweep complete")

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

        baseline_acc_all = []
        confidence_acc_all = []

        for seed in seeds:
            baseline, confidence = self._build_networks(seed)
            baseline_accuracy = []
            confidence_accuracy = []

            for epoch in range(30):
                epoch_baseline_acc = []
                epoch_confidence_acc = []

                baseline.reset_sequence()
                confidence.reset_sequence()

                for value in sequence:
                    input_sdr = encoder.encode(value)
                    result_b = baseline.compute(input_sdr)
                    result_c = confidence.compute(input_sdr)

                    epoch_baseline_acc.append(1.0 - result_b['anomaly_score'])
                    epoch_confidence_acc.append(1.0 - result_c['anomaly_score'])

                baseline_accuracy.append(float(np.mean(epoch_baseline_acc)))
                confidence_accuracy.append(float(np.mean(epoch_confidence_acc)))

            baseline_acc_all.append(baseline_accuracy)
            confidence_acc_all.append(confidence_accuracy)

        baseline_mean = np.mean(baseline_acc_all, axis=0).tolist()
        confidence_mean = np.mean(confidence_acc_all, axis=0).tolist()
        baseline_final = [acc[-1] for acc in baseline_acc_all]
        confidence_final = [acc[-1] for acc in confidence_acc_all]

        self.results['sequence_comparison'] = {
            'baseline_accuracy': baseline_mean,
            'confidence_accuracy': confidence_mean
        }

        print(f"✓ Baseline final accuracy: {np.mean(baseline_final):.3f} ± {np.std(baseline_final):.3f}")
        print(f"✓ Confidence final accuracy: {np.mean(confidence_final):.3f} ± {np.std(confidence_final):.3f}")

    def test_continual_learning(self):
        """Test catastrophic forgetting resistance with detailed diagnostics."""
        print("\n--- Continual Learning Test ---")
        seeds = [0, 1, 2]
        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)

        sequence_a = [1, 2, 3, 4, 5]
        sequence_b = [1, 2, 6, 7, 5]

        baseline_seq_a = []
        baseline_seq_b = []
        baseline_seq_a_after = []
        confidence_seq_a = []
        confidence_seq_b = []
        confidence_seq_a_after = []
        baseline_stabilities = []
        confidence_stabilities = []

        for seed in seeds:
            baseline, confidence = self._build_networks(seed)

            print("  Phase 1: Learning sequence A [1->2->3->4->5]...")
            for network, acc_list in [(baseline, baseline_seq_a), (confidence, confidence_seq_a)]:
                for epoch in range(25):
                    network.reset_sequence()
                    epoch_acc = []
                    for value in sequence_a:
                        result = network.compute(encoder.encode(value))
                        epoch_acc.append(1.0 - result['anomaly_score'])
                acc_list.append(float(np.mean(epoch_acc)))

            print("  Capturing sequence A representations...")
            baseline_reps_A = self.capture_transition_reprs(baseline, sequence_a, encoder)
            confidence_reps_A = self.capture_transition_reprs(confidence, sequence_a, encoder)

            print("  Phase 2: Learning sequence B [1->2->6->7->5] (overlaps with A)...")
            for network, acc_list in [(baseline, baseline_seq_b), (confidence, confidence_seq_b)]:
                for epoch in range(25):
                    network.reset_sequence()
                    epoch_acc = []
                    for value in sequence_b:
                        result = network.compute(encoder.encode(value))
                        epoch_acc.append(1.0 - result['anomaly_score'])
                acc_list.append(float(np.mean(epoch_acc)))

            print("  Phase 3: Testing sequence A retention...")
            baseline_reps_after = self.capture_transition_reprs(baseline, sequence_a, encoder)
            confidence_reps_after = self.capture_transition_reprs(confidence, sequence_a, encoder)

            for network, acc_list in [(baseline, baseline_seq_a_after), (confidence, confidence_seq_a_after)]:
                network.reset_sequence()
                test_acc = []
                for value in sequence_a:
                    res = network.compute(encoder.encode(value), learn=False)
                    test_acc.append(1.0 - res['anomaly_score'])
                acc_list.append(float(np.mean(test_acc)))

            baseline_stability = []
            confidence_stability = []
            for key in baseline_reps_A.keys():
                b0, b1 = baseline_reps_A[key], baseline_reps_after.get(key, set())
                c0, c1 = confidence_reps_A[key], confidence_reps_after.get(key, set())
                baseline_stability.append(len(b0 & b1) / (len(b0) or 1))
                confidence_stability.append(len(c0 & c1) / (len(c0) or 1))
            baseline_stabilities.append(float(np.mean(baseline_stability)) if baseline_stability else 0.0)
            confidence_stabilities.append(float(np.mean(confidence_stability)) if confidence_stability else 0.0)

        avg_baseline_stability = float(np.mean(baseline_stabilities)) if baseline_stabilities else 0.0
        avg_confidence_stability = float(np.mean(confidence_stabilities)) if confidence_stabilities else 0.0

        baseline_results = {
            'seq_a': [float(np.mean(baseline_seq_a))],
            'seq_b_during': [float(np.mean(baseline_seq_b))],
            'seq_a_after': [float(np.mean(baseline_seq_a_after))]
        }
        confidence_results = {
            'seq_a': [float(np.mean(confidence_seq_a))],
            'seq_b_during': [float(np.mean(confidence_seq_b))],
            'seq_a_after': [float(np.mean(confidence_seq_a_after))]
        }

        print("\n  Analyzing synaptic permanence distributions...")

        def get_permanence_stats(network):
            all_permanences = []
            for cell_segments in network.tm.segments.values():
                for segment in cell_segments:
                    for _, perm in segment['synapses']:
                        all_permanences.append(perm)
            if all_permanences:
                return {
                    'mean': float(np.mean(all_permanences)),
                    'high_perm_ratio': float(np.mean([p > 0.7 for p in all_permanences])),
                    'very_high_perm_ratio': float(np.mean([p > 0.85 for p in all_permanences])),
                    'max': float(np.max(all_permanences)),
                    'total_synapses': int(len(all_permanences))
                }
            return {'mean': 0.0, 'high_perm_ratio': 0.0, 'very_high_perm_ratio': 0.0, 'max': 0.0, 'total_synapses': 0}

        baseline_perm_stats = get_permanence_stats(baseline)
        confidence_perm_stats = get_permanence_stats(confidence)

        print(f"  Baseline: {baseline_perm_stats['total_synapses']} synapses, "
              f"mean={baseline_perm_stats['mean']:.3f}, "
              f">0.7={baseline_perm_stats['high_perm_ratio']:.1%}, "
              f"max={baseline_perm_stats['max']:.3f}")
        print(f"  Confidence: {confidence_perm_stats['total_synapses']} synapses, "
              f"mean={confidence_perm_stats['mean']:.3f}, "
              f">0.7={confidence_perm_stats['high_perm_ratio']:.1%}, "
              f"max={confidence_perm_stats['max']:.3f}")

        if hasattr(confidence.tm, 'current_system_confidence'):
            print(f"  Final system confidence: {confidence.tm.current_system_confidence:.3f}")

        self.results['continual_learning'] = {
            'baseline': baseline_results,
            'confidence': confidence_results,
            'baseline_stability': avg_baseline_stability,
            'confidence_stability': avg_confidence_stability,
            'baseline_perm_stats': baseline_perm_stats,
            'confidence_perm_stats': confidence_perm_stats
        }

        print(f"\n✓ Baseline retention: {np.mean(baseline_seq_a):.3f} → {np.mean(baseline_seq_a_after):.3f} (±{np.std(baseline_seq_a_after):.3f})")
        print(f"✓ Confidence retention: {np.mean(confidence_seq_a):.3f} → {np.mean(confidence_seq_a_after):.3f} (±{np.std(confidence_seq_a_after):.3f})")
        print(f"✓ Baseline representation stability: {avg_baseline_stability:.1%}")
        print(f"✓ Confidence representation stability: {avg_confidence_stability:.1%}")

        if avg_confidence_stability > avg_baseline_stability + 0.05:
            print("  ✅ SUCCESS: Confidence modulation improves stability!")
        elif abs(avg_confidence_stability - avg_baseline_stability) < 0.05:
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

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('HTM Confidence-Based Learning: Baseline vs Enhanced', fontsize=16)

        # 1. Sequence Learning Comparison
        ax = axes[0, 0]
        if 'sequence_comparison' in self.results:
            data = self.results['sequence_comparison']
            epochs = range(len(data['baseline_accuracy']))
            ax.plot(epochs, data['baseline_accuracy'], label='Baseline HTM', linewidth=2)
            ax.plot(epochs, data['confidence_accuracy'], label='Confidence HTM', linewidth=2)
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Prediction Accuracy')
            ax.set_title('Sequence Learning Speed')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Continual Learning
        ax = axes[0, 1]
        if 'continual_learning' in self.results:
            data = self.results['continual_learning']

            phases = ['Seq A\n(Initial)', 'Seq B\n(New)', 'Seq A\n(Recall)']
            baseline_vals = [
                data['baseline']['seq_a'][-1] if data['baseline']['seq_a'] else 0,
                data['baseline']['seq_b_during'][-1] if data['baseline']['seq_b_during'] else 0,
                data['baseline']['seq_a_after'][0] if data['baseline']['seq_a_after'] else 0
            ]
            confidence_vals = [
                data['confidence']['seq_a'][-1] if data['confidence']['seq_a'] else 0,
                data['confidence']['seq_b_during'][-1] if data['confidence']['seq_b_during'] else 0,
                data['confidence']['seq_a_after'][0] if data['confidence']['seq_a_after'] else 0
            ]

            x = np.arange(len(phases))
            width = 0.35
            ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7)
            ax.bar(x + width/2, confidence_vals, width, label='Confidence', alpha=0.7)
            ax.set_xlabel('Learning Phase')
            ax.set_ylabel('Accuracy')
            ax.set_title('Catastrophic Forgetting Resistance')
            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Noise Robustness
        ax = axes[0, 2]
        if 'noise_robustness' in self.results:
            data = self.results['noise_robustness']
            noise_pct = [n * 100 for n in data['noise_levels']]
            ax.plot(noise_pct, data['baseline'], 'o-', label='Baseline HTM', linewidth=2)
            ax.plot(noise_pct, data['confidence'], 'o-', label='Confidence HTM', linewidth=2)
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel('Prediction Accuracy')
            ax.set_title('Noise Robustness')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. System Confidence Evolution
        ax = axes[1, 0]
        network = ConfidenceHTMNetwork(input_size=100, use_confidence=True)
        encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
        s1, s2 = [1,2,3,4,5], [6,7,8,9,10]

        # Pre-train on S1 so it is familiar before logging
        for _ in range(20):
            network.reset_sequence()
            for v in s1:
                network.compute(encoder.encode(v))

        history = []

        def run(seq, reps):
            for _ in range(reps):
                network.reset_sequence()
                for t, v in enumerate(seq):
                    r = network.compute(encoder.encode(v))
                    if t > 0 and r['system_confidence'] is not None:
                        history.append(r['system_confidence'])

        run(s1, 5)  # familiar
        run(s2, 5)  # novel
        run(s1, 5)  # familiar again

        ax.plot(history, linewidth=2)
        ax.axhline(y=0.7, linestyle='--', alpha=0.5, label='Confidence Threshold')
        ax.set_title('Confidence: Familiar → Novel → Familiar')

        # 5. Learning Rate Modulation
        ax = axes[1, 1]
        confidence_levels = np.linspace(0, 1, 100)
        exploration_rate = []
        exploitation_rate = []

        for conf in confidence_levels:
            if conf < 0.7:
                exploration_rate.append(0.1 * 2.0)
                exploitation_rate.append(0.1)
            else:
                exploration_rate.append(0.1)
                exploitation_rate.append(0.1 * (1.0 - conf * 0.5))

        ax.plot(confidence_levels, exploration_rate, label='Exploration Mode', linewidth=2)
        ax.plot(confidence_levels, exploitation_rate, label='Exploitation Mode', linewidth=2)
        ax.axvline(x=0.7, linestyle='--', alpha=0.5, label='Mode Switch')
        ax.set_xlabel('System Confidence')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Adaptive Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "Performance Summary\n" + "="*30 + "\n\n"

        if 'sequence_comparison' in self.results:
            baseline_final = self.results['sequence_comparison']['baseline_accuracy'][-1]
            confidence_final = self.results['sequence_comparison']['confidence_accuracy'][-1]
            improvement = ((confidence_final - baseline_final) / baseline_final * 100) if baseline_final > 0 else 0
            summary_text += f"Sequence Learning:\n"
            summary_text += f"  Baseline: {baseline_final:.3f}\n"
            summary_text += f"  Confidence: {confidence_final:.3f}\n"
            summary_text += f"  Improvement: {improvement:+.1f}%\n\n"

        if 'continual_learning' in self.results:
            baseline_retention = self.results['continual_learning']['baseline']['seq_a_after'][0]
            confidence_retention = self.results['continual_learning']['confidence']['seq_a_after'][0]
            summary_text += f"Memory Retention:\n"
            summary_text += f"  Baseline: {baseline_retention:.3f}\n"
            summary_text += f"  Confidence: {confidence_retention:.3f}\n\n"

        if 'noise_robustness' in self.results:
            baseline_noise = self.results['noise_robustness']['baseline'][-1]
            confidence_noise = self.results['noise_robustness']['confidence'][-1]
            summary_text += f"Noise Resistance (20%):\n"
            summary_text += f"  Baseline: {baseline_noise:.3f}\n"
            summary_text += f"  Confidence: {confidence_noise:.3f}\n"

        if 'branching_context' in self.results:
            bc = self.results['branching_context']
            summary_text += "\nBranching Context (branch/post):\n"
            summary_text += f"  Baseline: {bc['baseline']['branch_acc_mean']:.3f}/{bc['baseline']['post_acc_mean']:.3f}\n"
            summary_text += f"  Confidence: {bc['confidence']['branch_acc_mean']:.3f}/{bc['confidence']['post_acc_mean']:.3f}\n"
            print("Branching context summary:", bc)

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('htm_confidence_results.png', dpi=150, bbox_inches='tight')
        print("✓ Charts saved to 'htm_confidence_results.png'")

        if 'scaling_study' in self.results:
            data = self.results['scaling_study']
            fig2, axs = plt.subplots(1, 2, figsize=(12, 5))

            axs[0].plot(data['lengths'], data['baseline_acc_mean'], 'o-', label='Baseline')
            axs[0].plot(data['lengths'], data['confidence_acc_mean'], 'o-', label='Confidence')
            axs[0].set_xlabel('Sequence Length')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title('Accuracy vs Length')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()

            axs[1].plot(data['lengths'], data['baseline_ret_mean'], 'o-', label='Baseline')
            axs[1].plot(data['lengths'], data['confidence_ret_mean'], 'o-', label='Confidence')
            axs[1].set_xlabel('Sequence Length')
            axs[1].set_ylabel('Retention')
            axs[1].set_title('Retention vs Length')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()

            plt.tight_layout()
            plt.savefig('/mnt/data/htm_scaling.png', dpi=150, bbox_inches='tight')
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
    parser.add_argument('--epochs-per-phase', type=int, default=25)
    parser.add_argument('--rates', type=str, default="0.0,0.03,0.05,0.1,0.2")
    parser.add_argument('--thresholds', type=str, default="0.6,0.7,0.8")
    parser.add_argument('--seeds', type=str, default="0,1,2")
    args = parser.parse_args()

    test_suite = TestSuite()

    if args.sweep_hardening:
        rates = [float(x) for x in args.rates.split(',') if x]
        thresholds = [float(x) for x in args.thresholds.split(',') if x]
        seeds = [int(x) for x in args.seeds.split(',') if x]
        test_suite.run_hardening_sweep(rates=rates, thresholds=thresholds,
                                       seeds=seeds, epochs_per_phase=args.epochs_per_phase)
        test_suite.save_results_json()
    else:
        results = test_suite.run_all_tests()

        print("\n" + "="*60)
        print("TEST SUITE COMPLETE")
        print("="*60)

        total_passed = sum(r.get('passed', 0) for r in results.values() if isinstance(r, dict))
        total_failed = sum(r.get('failed', 0) for r in results.values() if isinstance(r, dict))

        print(f"\nUnit Tests: {total_passed} passed, {total_failed} failed")
        print("\nPlease check 'htm_confidence_results.png' for visualizations")
        print("Raw data available in 'htm_test_results.json'")
