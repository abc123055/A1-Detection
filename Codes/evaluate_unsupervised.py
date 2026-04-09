"""
Unsupervised evaluation module for datasets WITHOUT ground truth annotations.

Analyzes PSNR (prediction error) distributions from inference output to:
  1. Assess model training quality (convergence, stability)
  2. Detect anomalous frames via statistical thresholds
  3. Generate per-video reports and visualizations

Usage:
    python evaluate_unsupervised.py -f /path/to/psnr_result.pkl
    python evaluate_unsupervised.py -f /path/to/psnr_dir/
    python evaluate_unsupervised.py -f /path/to/psnr_result.pkl --threshold_sigma 2.5
    python evaluate_unsupervised.py -f /path/to/psnr_result.pkl --plot
"""

import os
import argparse
import pickle
import numpy as np


DECIDABLE_IDX = 4


def load_psnr(loss_file):
    with open(loss_file, 'rb') as f:
        results = pickle.load(f)
    return results['dataset'], results['psnr']


def analyze_video(psnr, sigma=3.0):
    """Analyze a single video's PSNR sequence.

    Lower PSNR = higher prediction error = more likely anomalous.

    Returns a dict with statistics and detected anomaly frame indices.
    """
    valid = psnr[DECIDABLE_IDX:]
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    median = float(np.median(valid))
    psnr_min = float(np.min(valid))
    psnr_max = float(np.max(valid))

    # Anomaly: frames where PSNR drops below (mean - sigma * std)
    threshold = mean - sigma * std
    anomaly_mask = valid < threshold
    anomaly_indices = np.where(anomaly_mask)[0] + DECIDABLE_IDX
    anomaly_ratio = float(np.sum(anomaly_mask)) / len(valid)

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': psnr_min,
        'max': psnr_max,
        'threshold': threshold,
        'anomaly_indices': anomaly_indices,
        'anomaly_ratio': anomaly_ratio,
        'num_frames': len(psnr),
        'num_valid': len(valid),
    }


def training_quality_score(all_stats):
    """Heuristic score (0~100) indicating how well the model has trained.

    Based on:
      - Mean PSNR level (higher is better, model learned to predict)
      - Stability (lower std across videos is better)
      - Consistency (small spread between video means)
    """
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]

    avg_psnr = np.mean(means)
    avg_std = np.mean(stds)
    mean_spread = np.std(means) if len(means) > 1 else 0.0

    # PSNR component: typical good models get 25~40 dB; poor < 20
    psnr_score = np.clip((avg_psnr - 15) / 25 * 60, 0, 60)
    # Stability: lower avg_std is better
    stability_score = np.clip((1 - avg_std / 10) * 25, 0, 25)
    # Consistency: lower spread is better
    consistency_score = np.clip((1 - mean_spread / 10) * 15, 0, 15)

    total = float(psnr_score + stability_score + consistency_score)
    return total, {
        'avg_psnr': float(avg_psnr),
        'avg_std': float(avg_std),
        'mean_spread': float(mean_spread),
        'psnr_score': float(psnr_score),
        'stability_score': float(stability_score),
        'consistency_score': float(consistency_score),
    }


def format_report(dataset, all_stats, quality_score, quality_detail, video_names):
    """Format a human-readable text report."""
    SEP = '=' * 60
    lines = []
    lines.append(SEP)
    lines.append('  Unsupervised Evaluation Report')
    lines.append('  Dataset: {}'.format(dataset))
    lines.append(SEP)
    lines.append('')

    # Training quality
    lines.append('--- Training Quality ---')
    lines.append('  Score          : {:.1f} / 100'.format(quality_score))
    lines.append('  Avg PSNR       : {:.2f} dB'.format(quality_detail['avg_psnr']))
    lines.append('  Avg Std        : {:.2f}'.format(quality_detail['avg_std']))
    if len(all_stats) > 1:
        lines.append('  Cross-video spread : {:.2f}'.format(quality_detail['mean_spread']))
    lines.append('')

    # Score interpretation
    if quality_score >= 75:
        lines.append('  [GOOD] Model has learned strong prediction patterns.')
    elif quality_score >= 50:
        lines.append('  [OK] Model has learned basic patterns. Consider more iterations.')
    elif quality_score >= 25:
        lines.append('  [WEAK] Model is underfitting. Increase training iterations or adjust hyperparameters.')
    else:
        lines.append('  [POOR] Model has not converged. Check data, learning rate, and training setup.')
    lines.append('')

    # Per-video summary
    lines.append('--- Per-Video Summary ---')
    lines.append('{:<10} {:>10} {:>8} {:>8} {:>8} {:>10} {:>10}'.format(
        'Video', 'Frames', 'Mean', 'Std', 'Min', 'Threshold', 'Anomaly%'))
    lines.append('-' * 74)

    for i, stats in enumerate(all_stats):
        vname = video_names[i] if i < len(video_names) else '{:02d}'.format(i + 1)
        lines.append('{:<10} {:>10} {:>8.2f} {:>8.2f} {:>8.2f} {:>10.2f} {:>9.1f}%'.format(
            vname, stats['num_frames'], stats['mean'], stats['std'],
            stats['min'], stats['threshold'], stats['anomaly_ratio'] * 100))
    lines.append('')

    # Anomaly details
    lines.append('--- Detected Anomaly Segments ---')
    for i, stats in enumerate(all_stats):
        vname = video_names[i] if i < len(video_names) else '{:02d}'.format(i + 1)
        indices = stats['anomaly_indices']
        if len(indices) == 0:
            lines.append('  {}: No anomalies detected'.format(vname))
            continue

        segments = _indices_to_segments(indices)
        seg_strs = ['[{}-{}]'.format(s, e) if s != e else '[{}]'.format(s) for s, e in segments]
        lines.append('  {}: {} anomalous frame(s) in {} segment(s)'.format(
            vname, len(indices), len(segments)))
        lines.append('    Segments: {}'.format(', '.join(seg_strs)))
    lines.append('')
    lines.append(SEP)

    return '\n'.join(lines)


def _indices_to_segments(indices):
    """Convert [1,2,3,7,8,12] -> [(1,3),(7,8),(12,12)]."""
    if len(indices) == 0:
        return []
    segments = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((int(start), int(prev)))
            start = idx
            prev = idx
    segments.append((int(start), int(prev)))
    return segments


def save_csv(out_path, dataset, all_stats, video_names):
    """Save per-frame PSNR and anomaly flags to CSV files."""
    os.makedirs(out_path, exist_ok=True)
    for i, stats in enumerate(all_stats):
        vname = video_names[i] if i < len(video_names) else '{:02d}'.format(i + 1)
        csv_path = os.path.join(out_path, '{}_psnr.csv'.format(vname))
        with open(csv_path, 'w') as f:
            f.write('frame_id,psnr,is_anomaly\n')
            anomaly_set = set(stats['anomaly_indices'].tolist())
            # Write from DECIDABLE_IDX since earlier frames are not predicted
            for j in range(DECIDABLE_IDX, stats['num_frames']):
                psnr_val = stats.get('_psnr_raw', None)
                if psnr_val is not None:
                    f.write('{},{:.4f},{}\n'.format(j, float(psnr_val[j]), int(j in anomaly_set)))
    return out_path


def plot_psnr(psnr_records, all_stats, video_names, out_dir):
    """Generate PSNR curve plots per video."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib not installed, skipping plots.')
        return

    os.makedirs(out_dir, exist_ok=True)

    for i, (psnr, stats) in enumerate(zip(psnr_records, all_stats)):
        vname = video_names[i] if i < len(video_names) else '{:02d}'.format(i + 1)

        fig, ax = plt.subplots(figsize=(14, 4))
        frames = np.arange(len(psnr))
        ax.plot(frames, psnr, linewidth=0.5, color='steelblue', label='PSNR')
        ax.axhline(y=stats['mean'], color='green', linestyle='--', linewidth=0.8,
                    label='Mean ({:.2f})'.format(stats['mean']))
        ax.axhline(y=stats['threshold'], color='red', linestyle='--', linewidth=0.8,
                    label='Threshold ({:.2f})'.format(stats['threshold']))

        # Highlight anomaly regions
        anomaly_indices = stats['anomaly_indices']
        if len(anomaly_indices) > 0:
            segments = _indices_to_segments(anomaly_indices)
            for s, e in segments:
                ax.axvspan(s, e, alpha=0.3, color='red')

        ax.set_xlabel('Frame')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('{} - PSNR Curve'.format(vname))
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, len(psnr))

        fig.tight_layout()
        save_path = os.path.join(out_dir, '{}_psnr.png'.format(vname))
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print('  Plot saved: {}'.format(save_path))


def evaluate_unsupervised(loss_file, sigma=3.0, do_plot=False):
    """Main entry: evaluate a single psnr pickle file without ground truth."""
    dataset, psnr_records = load_psnr(loss_file)
    num_videos = len(psnr_records)

    video_names = ['{:02d}'.format(i + 1) for i in range(num_videos)]

    all_stats = []
    for i, psnr in enumerate(psnr_records):
        stats = analyze_video(psnr, sigma=sigma)
        stats['_psnr_raw'] = psnr
        all_stats.append(stats)

    quality_score, quality_detail = training_quality_score(all_stats)

    report = format_report(dataset, all_stats, quality_score, quality_detail, video_names)
    print(report)

    # Save report
    out_dir = os.path.join(os.path.dirname(loss_file), 'unsupervised_results')
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report + '\n')
    print('Report saved: {}'.format(report_path))

    # Save CSV
    save_csv(out_dir, dataset, all_stats, video_names)
    print('CSV files saved: {}'.format(out_dir))

    # Plot
    if do_plot:
        plot_dir = os.path.join(out_dir, 'plots')
        plot_psnr(psnr_records, all_stats, video_names, plot_dir)

    return quality_score


def main():
    parser = argparse.ArgumentParser(description='Unsupervised evaluation for unlabeled datasets.')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Path to psnr pickle file or directory of pickle files.')
    parser.add_argument('--threshold_sigma', type=float, default=3.0,
                        help='Number of standard deviations below mean to flag as anomaly (default: 3.0).')
    parser.add_argument('--plot', action='store_true',
                        help='Generate PSNR curve plots (requires matplotlib).')

    args = parser.parse_args()

    if os.path.isdir(args.file):
        files = sorted([os.path.join(args.file, f) for f in os.listdir(args.file)
                         if not os.path.isdir(os.path.join(args.file, f))])
        for f in files:
            print('\n>>> Evaluating: {}'.format(f))
            evaluate_unsupervised(f, sigma=args.threshold_sigma, do_plot=args.plot)
    else:
        evaluate_unsupervised(args.file, sigma=args.threshold_sigma, do_plot=args.plot)


if __name__ == '__main__':
    main()
