#!/usr/bin/env python3
import json
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

matplotlib.use('Agg')
OUTPUT_DIR = "swizzle_plots"
BASELINE_SWIZZLE = "0x600"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_outlier_iqr(times):
    if len(times) < 4:
        return [False] * len(times)
    sorted_times = sorted(times)
    n = len(sorted_times)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_times[q1_idx]
    q3 = sorted_times[q3_idx]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [t < lower or t > upper for t in times]


with open("swizzle_perf_raw.json", "r") as f:
    data = json.load(f)

groups = defaultdict(list)
for entry in data:
    key = (entry["m"], entry["n"], entry["k"], entry["swizzle"])
    groups[key].append(entry)


filtered = []
removed = 0

for key, entries in groups.items():
    times = [e["time"] for e in entries]
    outliers = is_outlier_iqr(times)
    for entry, is_out in zip(entries, outliers):
        if is_out:
            removed += 1
        else:
            filtered.append(entry)

print(f"Removed {removed} outliers out of {len(data)} total entries")
print(f"Filtered data: {len(filtered)} entries")

filtered_path = os.path.join(OUTPUT_DIR, "swizzle_perf_filtered.json")
with open(filtered_path, "w") as f:
    json.dump(filtered, f, indent=2)
print(f"Saved filtered data to {filtered_path}")

summary_groups = defaultdict(list)
for entry in filtered:
    key = (entry["m"], entry["n"], entry["k"], entry["swizzle"])
    summary_groups[key].append(entry)

summary = []
for (m, n, k, swizzle), entries in summary_groups.items():
    times = sorted([e["time"] for e in entries])
    n_times = len(times)
    q1_idx = n_times // 4
    q3_idx = 3 * n_times // 4
    summary.append({
        "swizzle": swizzle,
        "m": m,
        "n": n,
        "k": k,
        "median": times[n_times // 2],
        "q1": times[q1_idx],
        "q3": times[q3_idx],
        "min": times[0],
        "max": times[-1],
        "count": n_times
    })

summary_path = os.path.join(OUTPUT_DIR, "swizzle_perf_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to {summary_path}")

baseline_lookup = {}
for s in summary:
    key = (s["m"], s["n"], s["k"])
    if s["swizzle"] == BASELINE_SWIZZLE:
        baseline_lookup[key] = s["median"]

plot_groups = defaultdict(list)
for entry in filtered:
    key = (entry["m"], entry["n"], entry["k"])
    plot_groups[key].append(entry)

median_lookup = {}
for (m, n, k, swizzle), entries in summary_groups.items():
    times = sorted([e["time"] for e in entries])
    median_lookup[(m, n, k, swizzle)] = times[len(times) // 2]

for (m, n, k), entries in plot_groups.items():
    swizzle_data = defaultdict(list)
    for entry in entries:
        swizzle_data[entry["swizzle"]].append(entry["time"])

    swizzles = sorted(swizzle_data.keys(), key=lambda x: int(x, 16))
    data_by_swizzle = [swizzle_data[s] for s in swizzles]

    baseline_median = baseline_lookup.get((m, n, k))
    if baseline_median is None:
        msg = (f"Warning: baseline {BASELINE_SWIZZLE} not found for"
               f"m={m}, n={n}, k={k}, skipping plot")
        print(msg)
        continue

    fig, ax = plt.subplots(figsize=(max(8, len(swizzles) * 0.8), 6))
    bp = ax.boxplot(data_by_swizzle, tick_labels=swizzles, patch_artist=True)

    for i, swizzle in enumerate(swizzles):
        median = median_lookup.get((m, n, k, swizzle), 0)
        color = 'lightgreen' if median < baseline_median else 'pink'
        bp['boxes'][i].set_facecolor(color)

        pct_diff = (median - baseline_median) / baseline_median * 100
        annotation = f"{pct_diff:+.1f}%"
        y_offset = max(swizzle_data[swizzle]) + 50
        ax.annotate(annotation, xy=(i + 1, y_offset), ha='center', fontsize=8,
                    color='green' if pct_diff < 0 else 'red')

    ax.axhline(baseline_median, color='black', linestyle='--', linewidth=1.5,
               label=f'Baseline ({BASELINE_SWIZZLE}): {baseline_median}ms')

    ax.set_xlabel("Swizzle")
    ax.set_ylabel("Time (ms)")
    ax.set_title((f"Matrix Multiplication Performance\n"
                  f"m={m}, n={n}, k={k}, baseline={BASELINE_SWIZZLE}"))
    ax.tick_params(axis='x', rotation=45)

    legend_elements = [Patch(facecolor='lightgreen',
                             label='Faster than baseline'),
                       Patch(facecolor='pink',
                             label='Slower than baseline')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    filename = f"boxplot_{m}x{n}x{k}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved plot to {filepath}")

print(f"\nAll artefacts saved to {OUTPUT_DIR}/")
