import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Any, Sequence


def set_matplotlib_headless() -> None:
    """Configure matplotlib to use a headless backend."""
    matplotlib.use("Agg", force=True)


def _annotate_heatmap(ax, data: np.ndarray) -> None:
    """Write numeric values in each cell of a heatmap."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "nan"
            else:
                text = f"{val:.2f}"
            # choose contrasting color
            vmax = np.nanmax(data)
            vmin = np.nanmin(data)
            mid = (vmax + vmin) / 2.0
            color = "white" if not np.isnan(val) and val < mid else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)


def plot_main_dashboard(results: Dict[str, Any], save_path: str) -> None:
    """Plot the main dashboard summarizing experiment results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HTM Confidence-Based Learning: Baseline vs Enhanced', fontsize=16)

    # 1. Sequence Learning Comparison
    ax = axes[0, 0]
    if 'sequence_comparison' in results:
        data = results['sequence_comparison']
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
    if 'continual_learning' in results:
        data = results['continual_learning']
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
    if 'noise_robustness' in results:
        data = results['noise_robustness']
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
    from .network import ConfidenceHTMNetwork
    from .encoders import ScalarEncoder
    tm_params = {
        "cells_per_column": 8,
        "activation_threshold": 10,
        "learning_threshold": 8,
        "initial_permanence": 0.5,
        "permanence_increment": 0.02,
        "permanence_decrement": 0.005,
        "max_synapses_per_segment": 16,
        "seed": 0,
        "hardening_rate": 0.0,
        "hardening_threshold": 0.7,
    }
    sp_params = {
        "seed": 0,
        "column_count": 100,
        "sparsity": 0.1,
        "boost_strength": 0.0,
    }
    network = ConfidenceHTMNetwork(
        input_size=100, tm_params=tm_params, sp_params=sp_params
    )
    encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
    s1, s2 = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
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
    run(s1, 5)
    run(s2, 5)
    run(s1, 5)
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
    if 'sequence_comparison' in results:
        baseline_final = results['sequence_comparison']['baseline_accuracy'][-1]
        confidence_final = results['sequence_comparison']['confidence_accuracy'][-1]
        improvement = ((confidence_final - baseline_final) / baseline_final * 100) if baseline_final > 0 else 0
        summary_text += f"Sequence Learning:\n"
        summary_text += f"  Baseline: {baseline_final:.3f}\n"
        summary_text += f"  Confidence: {confidence_final:.3f}\n"
        summary_text += f"  Improvement: {improvement:+.1f}%\n\n"
    if 'continual_learning' in results:
        baseline_retention = results['continual_learning']['baseline']['seq_a_after'][0]
        confidence_retention = results['continual_learning']['confidence']['seq_a_after'][0]
        summary_text += f"Memory Retention:\n"
        summary_text += f"  Baseline: {baseline_retention:.3f}\n"
        summary_text += f"  Confidence: {confidence_retention:.3f}\n\n"
    if 'noise_robustness' in results:
        baseline_noise = results['noise_robustness']['baseline'][-1]
        confidence_noise = results['noise_robustness']['confidence'][-1]
        summary_text += f"Noise Resistance (20%):\n"
        summary_text += f"  Baseline: {baseline_noise:.3f}\n"
        summary_text += f"  Confidence: {confidence_noise:.3f}\n"
    if 'branching_context' in results:
        bc = results['branching_context']
        summary_text += "\nBranching Context (branch/post):\n"
        summary_text += f"  Baseline: {bc['baseline']['branch_acc_mean']:.3f}/{bc['baseline']['post_acc_mean']:.3f}\n"
        summary_text += f"  Confidence: {bc['confidence']['branch_acc_mean']:.3f}/{bc['confidence']['post_acc_mean']:.3f}\n"
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_scaling(study: Dict[str, Sequence[float]], save_path: str) -> None:
    """Plot accuracy and retention versus sequence length."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(study['lengths'], study['baseline_acc_mean'], 'o-', label='Baseline')
    axs[0].plot(study['lengths'], study['confidence_acc_mean'], 'o-', label='Confidence')
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracy vs Length')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[1].plot(study['lengths'], study['baseline_ret_mean'], 'o-', label='Baseline')
    axs[1].plot(study['lengths'], study['confidence_ret_mean'], 'o-', label='Confidence')
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('Retention')
    axs[1].set_title('Retention vs Length')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_hardening_heatmaps(
    ret_mat: np.ndarray,
    stab_mat: np.ndarray,
    rates: Sequence[float],
    thresholds: Sequence[float],
    save_path: str,
    *,
    add_delta: bool = True,
    annotate: bool = True,
    scale_to_data: bool = True,
) -> None:
    """Plot retention/stability heatmaps and optional delta plot."""
    panels = 3 if add_delta else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))

    def _scale(mat):
        if scale_to_data:
            return dict(vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))
        return dict(vmin=0.0, vmax=1.0)

    im0 = axes[0].imshow(ret_mat, origin='lower', aspect='auto', **_scale(ret_mat))
    axes[0].set_xticks(range(len(rates)))
    axes[0].set_xticklabels(rates)
    axes[0].set_yticks(range(len(thresholds)))
    axes[0].set_yticklabels(thresholds)
    axes[0].set_xlabel('Hardening Rate')
    axes[0].set_ylabel('Hardening Threshold')
    axes[0].set_title('Retention Accuracy')
    fig.colorbar(im0, ax=axes[0])
    if annotate:
        _annotate_heatmap(axes[0], ret_mat)

    im1 = axes[1].imshow(stab_mat, origin='lower', aspect='auto', **_scale(stab_mat))
    axes[1].set_xticks(range(len(rates)))
    axes[1].set_xticklabels(rates)
    axes[1].set_yticks(range(len(thresholds)))
    axes[1].set_yticklabels(thresholds)
    axes[1].set_xlabel('Hardening Rate')
    axes[1].set_ylabel('Hardening Threshold')
    axes[1].set_title('Representation Stability')
    fig.colorbar(im1, ax=axes[1])
    if annotate:
        _annotate_heatmap(axes[1], stab_mat)

    if add_delta and 0.0 in rates:
        baseline = ret_mat[:, rates.index(0.0)]
        delta = ret_mat - baseline[:, None]
        im2 = axes[2].imshow(delta, origin='lower', aspect='auto', **_scale(delta))
        axes[2].set_xticks(range(len(rates)))
        axes[2].set_xticklabels(rates)
        axes[2].set_yticks(range(len(thresholds)))
        axes[2].set_yticklabels(thresholds)
        axes[2].set_xlabel('Hardening Rate')
        axes[2].set_ylabel('Hardening Threshold')
        axes[2].set_title('ΔRetention vs Baseline')
        fig.colorbar(im2, ax=axes[2])
        if annotate:
            _annotate_heatmap(axes[2], delta)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
