#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Airplane Boarding Simulation (no external dependencies)
------------------------------------------------------------------
Implements the assignment model (50 rows x 6 seats) and compares 4 boarding methods:
  1) Random
  2) Rows ascending (random seats within a row)
  3) Rows descending (random seats within a row)
  4) Steffen-like method (windows->middle->aisle; odd then even rows; back to front)

- Uses only the Python standard library (random, csv, statistics, argparse).
- Saves CSVs with all runs and a summary including 95% confidence intervals.

Usage examples:
    python boarding_simulation_standalone.py --runs 100 --seed 42 --outdir ./outputs
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict
from typing import List, Tuple, Dict

# ----- Model constants -----
N_ROWS = 50
SEATS_PER_SIDE = 3  # 0=aisle, 1=middle, 2=window
SIDES = ['L', 'R']
BAG_MEAN = 0.5  # minutes
BASE_CROSS_MEAN = 0.25  # minutes
CROSS_PER_BLOCKER = 0.5  # minutes per blocking seated passenger

Passenger = Tuple[int, str, int]  # (row, side, pos)


def all_passengers() -> List[Passenger]:
    return [(row, side, pos)
            for row in range(1, N_ROWS + 1)
            for side in SIDES
            for pos in range(SEATS_PER_SIDE)]


# ---------- Boarding methods ----------
def order_random(passengers: List[Passenger]) -> List[Passenger]:
    seq = passengers.copy()
    random.shuffle(seq)
    return seq


def order_rows_ascending(passengers: List[Passenger]) -> List[Passenger]:
    seq: List[Passenger] = []
    for row in range(1, N_ROWS + 1):
        row_seats = [(r, s, p) for (r, s, p) in passengers if r == row]
        random.shuffle(row_seats)
        seq.extend(row_seats)
    return seq


def order_rows_descending(passengers: List[Passenger]) -> List[Passenger]:
    seq: List[Passenger] = []
    for row in range(N_ROWS, 0, -1):
        row_seats = [(r, s, p) for (r, s, p) in passengers if r == row]
        random.shuffle(row_seats)
        seq.extend(row_seats)
    return seq


def order_steffen(passengers: List[Passenger]) -> List[Passenger]:
    # Steffen-like: windows (2), then middle (1), then aisle (0);
    # odd rows then even rows; back to front within each group.
    by_group: Dict[Tuple[int, int], List[Passenger]] = defaultdict(list)
    for (r, s, p) in passengers:
        by_group[(p, r % 2)].append((r, s, p))
    seq: List[Passenger] = []
    for p in [2, 1, 0]:
        for parity in [1, 0]:  # odd then even
            block = by_group[(p, parity)]
            block.sort(key=lambda t: (-t[0], t[1]))  # back to front, L before R
            seq.extend(block)
    return seq


# ---------- Simulation core ----------
def exp_sample(mean: float) -> float:
    # Python's random.expovariate uses rate lambda (=1/mean)
    return random.expovariate(1.0 / mean) if mean > 0 else 0.0


def simulate_sequence(sequence: List[Passenger]) -> float:
    """
    Returns total time (minutes) until the last passenger is seated.
    Rules per the assignment:
      - Walking time to row is negligible.
      - Baggage time ~ Exp(mean=0.5).
      - Crossing time ~ Exp(mean=0.25 + 0.5 * blockers).
      - Cannot pass anyone stopped at a row <= your row.
      - Pre-boarding sorting time is negligible.
    """
    finish_time: Dict[Passenger, float] = {}
    row_side_to_passengers: Dict[Tuple[int, str], List[Passenger]] = defaultdict(list)
    for p in sequence:
        row_side_to_passengers[(p[0], p[1])].append(p)

    for idx, p in enumerate(sequence):
        r, s, pos = p

        # Aisle blocking: must wait for max finish time among already-entered passengers at rows <= r
        if idx == 0:
            t0 = 0.0
        else:
            t0 = 0.0
            for pp, ft in finish_time.items():
                if pp[0] <= r and ft > t0:
                    t0 = ft

        # Blockers: seatmates on same row/side with smaller pos (closer to aisle), already seated by t0
        blockers = 0
        for mate in row_side_to_passengers[(r, s)]:
            if mate == p:
                continue
            if mate[2] < pos and mate in finish_time and finish_time[mate] <= t0 + 1e-12:
                blockers += 1

        bag = exp_sample(BAG_MEAN)
        cross = exp_sample(BASE_CROSS_MEAN + CROSS_PER_BLOCKER * blockers)
        finish_time[p] = t0 + bag + cross

    return max(finish_time.values()) if finish_time else 0.0


def main():
    parser = argparse.ArgumentParser(description="Airplane boarding simulation (stdlib only)")
    parser.add_argument("--runs", type=int, default=100, help="Repetitions per method (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for CSVs (default: current)")
    args = parser.parse_args()

    random.seed(args.seed)

    passengers = all_passengers()
    methods = {
        "Random": order_random,
        "RowsUp": order_rows_ascending,
        "RowsDown": order_rows_descending,
        "Steffen": order_steffen,
    }

    # Run simulations
    all_rows: List[Tuple[str, int, float]] = []
    for method_name, method_fn in methods.items():
        for run in range(1, args.runs + 1):
            seq = method_fn(passengers)
            total_t = simulate_sequence(seq)
            all_rows.append((method_name, run, total_t))

    # Write detailed results
    import os
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_path = os.path.join(outdir, "boarding_results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "run", "total_minutes"])
        for method_name, run, total_t in all_rows:
            w.writerow([method_name, run, f"{total_t:.6f}"])

    # Compute summary
    from math import sqrt
    summary_rows: List[Tuple[str, int, float, float, float, float, float]] = []
    methods_order = ["Random", "RowsUp", "RowsDown", "Steffen"]
    for method_name in methods_order:
        values = [t for (m, _r, t) in all_rows if m == method_name]
        n = len(values)
        m = statistics.mean(values) if n else float("nan")
        sd = statistics.stdev(values) if n > 1 else float("nan")
        stderr = sd / sqrt(n) if n > 0 else float("nan")
        ci95_low = m - 1.96 * stderr if n > 0 else float("nan")
        ci95_high = m + 1.96 * stderr if n > 0 else float("nan")
        summary_rows.append((method_name, n, m, sd, stderr, ci95_low, ci95_high))

    summary_path = os.path.join(outdir, "boarding_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "count", "mean_minutes", "std", "stderr", "ci95_low", "ci95_high"])
        for row in summary_rows:
            method_name, n, m, sd, stderr, lo, hi = row
            w.writerow([method_name, n, f"{m:.6f}", f"{sd:.6f}", f"{stderr:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    print("Saved:")
    print(" -", results_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
