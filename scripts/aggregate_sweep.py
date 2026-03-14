#!/usr/bin/env python3
"""
Standalone sweep aggregation script.

Re-reads all epoch-level best_test_results.txt files under a sweep root
and writes model-level best_results_{ft,lp,combined}.txt with majority-vote
best epoch selection.

Usage:
    python scripts/aggregate_sweep.py runs/20260227_093832/
    python scripts/aggregate_sweep.py runs/20260227_093832/ --watch
"""

import argparse
import collections
import datetime
import os
import re
import sys
import time

# Add project root to path so we can import MASTER_RESULTS_COLUMNS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from baseline.abstract.trainer import MASTER_RESULTS_COLUMNS


def parse_best_test_results(path: str) -> list[dict]:
    """Parse a best_test_results.txt TSV file into list of dicts."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if parts == MASTER_RESULTS_COLUMNS:
                continue
            if len(parts) != len(MASTER_RESULTS_COLUMNS):
                continue
            row = dict(zip(MASTER_RESULTS_COLUMNS, parts))
            rows.append(row)
    return rows


def collect_sweep_records(sweep_root: str) -> list[dict]:
    """Read all epoch-level best_test_results.txt files."""
    records = []
    for entry in sorted(os.listdir(sweep_root)):
        epoch_match = re.match(r'epoch_(\d+)$', entry)
        if not epoch_match:
            continue
        pretrained_epoch = int(epoch_match.group(1))
        epoch_dir = os.path.join(sweep_root, entry)

        for method in ['ft', 'lp']:
            results_file = os.path.join(epoch_dir, method, 'best_test_results.txt')
            if not os.path.isfile(results_file):
                continue
            rows = parse_best_test_results(results_file)
            for row in rows:
                row['pretrained_epoch'] = pretrained_epoch
                row['method'] = method
                records.append(row)
    return records


def write_sweep_summary(sweep_root: str, method: str, records: list[dict]):
    """Write model-level summary with majority-vote best epoch selection."""
    # Group: dataset -> list of (pretrained_epoch, score, row)
    ds_best: dict[str, list[tuple[int, float, dict]]] = collections.defaultdict(list)
    for r in records:
        ds_name = r.get('dataset', '')
        try:
            score = float(r.get('score', '-inf'))
        except (ValueError, TypeError):
            score = float('-inf')
        ds_best[ds_name].append((r['pretrained_epoch'], score, r))

    # For each dataset, find which pretrained epoch had the best score
    epoch_wins: dict[int, int] = collections.Counter()
    epoch_scores: dict[int, list[float]] = collections.defaultdict(list)
    ds_winner: dict[str, tuple[int, float, dict]] = {}

    for ds_name, entries in ds_best.items():
        best_entry = max(entries, key=lambda e: e[1])
        ds_winner[ds_name] = best_entry
        epoch_wins[best_entry[0]] += 1
        for ep, sc, _ in entries:
            epoch_scores[ep].append(sc)

    if not epoch_wins:
        return

    # Majority vote: epoch with most wins; tiebreak by avg score
    best_epoch = max(
        epoch_wins.keys(),
        key=lambda ep: (
            epoch_wins[ep],
            sum(epoch_scores.get(ep, [])) / max(len(epoch_scores.get(ep, [])), 1)
        )
    )

    out_path = os.path.join(sweep_root, f'best_results_{method}.txt')
    tmp_path = out_path + '.tmp'
    updated_utc = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'

    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(f"# EEG-FM-Bench Sweep Model-Level Best Results ({method.upper()})\n")
        f.write(f"# Updated: {updated_utc}\n")
        f.write(f"# Sweep root: {sweep_root}\n")
        f.write(f"#\n")
        f.write(f"# === MAJORITY-VOTE BEST EPOCH: {best_epoch} ===")
        f.write(f"  (wins: {epoch_wins[best_epoch]}/{sum(epoch_wins.values())} downstream tasks)\n")
        avg_score = sum(epoch_scores.get(best_epoch, [])) / max(len(epoch_scores.get(best_epoch, [])), 1)
        f.write(f"# Average score at epoch {best_epoch}: {avg_score:.4f}\n")
        f.write(f"#\n")

        # Vote summary
        f.write(f"# --- Epoch vote counts ---\n")
        for ep in sorted(epoch_wins.keys()):
            ep_avg = sum(epoch_scores.get(ep, [])) / max(len(epoch_scores.get(ep, [])), 1)
            f.write(f"#   epoch_{ep}: {epoch_wins[ep]} wins, avg_score={ep_avg:.4f}\n")
        f.write(f"#\n")

        # Per-dataset detail
        f.write(f"# --- Per-dataset best pretrained epoch ---\n")
        f.write(f"# {'dataset':<20s} {'best_pretrained_epoch':>22s} {'score':>10s} {'ft_epoch':>10s}\n")
        for ds_name in sorted(ds_winner.keys()):
            ep, sc, row = ds_winner[ds_name]
            ft_epoch = row.get('epoch', '')
            f.write(f"# {ds_name:<20s} {ep:>22d} {sc:>10.4f} {ft_epoch:>10s}\n")
        f.write(f"#\n")

        # Full TSV
        f.write("\t".join(['pretrained_epoch'] + MASTER_RESULTS_COLUMNS) + "\n")
        for r in sorted(records, key=lambda x: (x['pretrained_epoch'], x.get('dataset', ''))):
            f.write(str(r['pretrained_epoch']) + "\t")
            f.write("\t".join(r.get(col, '') for col in MASTER_RESULTS_COLUMNS) + "\n")

    os.replace(tmp_path, out_path)
    print(f"  Written: {out_path} (best_epoch={best_epoch})")


def aggregate(sweep_root: str):
    """Main aggregation logic."""
    records = collect_sweep_records(sweep_root)
    if not records:
        print(f"No records found under {sweep_root}")
        return

    n_epochs = len(set(r['pretrained_epoch'] for r in records))
    n_ft = len([r for r in records if r['method'] == 'ft'])
    n_lp = len([r for r in records if r['method'] == 'lp'])
    print(f"Collected {len(records)} records from {n_epochs} epochs (FT: {n_ft}, LP: {n_lp})")

    for method_label in ['ft', 'lp']:
        method_records = [r for r in records if r['method'] == method_label]
        if method_records:
            write_sweep_summary(sweep_root, method_label, method_records)

    write_sweep_summary(sweep_root, 'combined', records)


def main():
    parser = argparse.ArgumentParser(description='Aggregate sweep results with majority-vote best epoch.')
    parser.add_argument('sweep_root', help='Path to sweep root directory (e.g. runs/20260227_093832/)')
    parser.add_argument('--watch', action='store_true',
                        help='Re-aggregate every 60 seconds until interrupted')
    parser.add_argument('--interval', type=int, default=60,
                        help='Watch interval in seconds (default: 60)')
    args = parser.parse_args()

    if not os.path.isdir(args.sweep_root):
        print(f"ERROR: Not a directory: {args.sweep_root}")
        sys.exit(1)

    if args.watch:
        print(f"Watching {args.sweep_root} every {args.interval}s (Ctrl+C to stop)")
        try:
            while True:
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Aggregating...")
                aggregate(args.sweep_root)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        aggregate(args.sweep_root)


if __name__ == '__main__':
    main()
