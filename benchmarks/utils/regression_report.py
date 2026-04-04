"""
Generate a markdown regression report comparing two benchmark CSV files.

Uses only stdlib — no pandas/numpy required.

Usage:
    python -m benchmarks.utils.regression_report <baseline.csv> <diff.csv>

Writes /tmp/regression-report.txt and prints to stdout.
"""

import csv
import json
import math
import os
import sys

# Thresholds for flagging regressions / improvements.
CODEGEN_THRESHOLD = 0.10   # 10% — Python codegen has more run-to-run noise
SOLVETIME_THRESHOLD = 0.05  # 5%  — C binary, less noise

OUTPUT_PATH = "/tmp/regression-report.txt"


def _read_csv(path):
    """Return a dict mapping problem name → row dict (values as floats where possible)."""
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            data[name] = {
                "codegen_time_s": float(row["codegen_time_s"]),
                "solved": int(row["solved"]),
                "obj": float(row["obj"]),
                "solve_time_s": float(row["solve_time_s"]),
            }
    return data


def _pct(diff_val, base_val):
    if base_val == 0 or math.isnan(base_val) or math.isnan(diff_val):
        return math.nan
    return (diff_val - base_val) / abs(base_val)


def _fmt_pct(p):
    if math.isnan(p):
        return "N/A"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p * 100:.1f}%"


def _read_meta(baseline_csv):
    """Return meta dict from meta.json in the same directory as baseline_csv, or {}."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(baseline_csv)), "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def generate_report(baseline_csv, diff_csv, artifact_url=None):
    baseline = _read_csv(baseline_csv)
    diff = _read_csv(diff_csv)
    meta = _read_meta(baseline_csv)

    all_names = sorted(set(baseline) | set(diff))
    common = [n for n in all_names if n in baseline and n in diff]

    baseline_label = meta.get("baseline_branch", "main")
    if "baseline_sha" in meta:
        baseline_label += f" (`{meta['baseline_sha']}`)"

    diff_label = meta.get("diff_branch", "branch")
    if "diff_sha" in meta:
        diff_label += f" (`{meta['diff_sha']}`)"

    lines = []

    if artifact_url:
        lines.append(f"### [Download benchmark artifacts]({artifact_url})\n")

    lines.append("### Benchmark Regression Report\n")
    lines.append(f"**Baseline:** {baseline_label}")
    lines.append(f"**Branch:** {diff_label}\n")

    # ── Solved count ───────────────────────────────────────────────────────────
    base_solved = sum(1 for v in baseline.values() if v["solved"] == 1)
    diff_solved = sum(1 for v in diff.values() if v["solved"] == 1)
    total = len(all_names)
    lines.append(f"**Baseline:** {base_solved}/{total} problems solved")
    lines.append(f"**Branch:** {diff_solved}/{total} problems solved\n")

    # Correctness changes
    status_changes = []
    for name in all_names:
        b_ok = name in baseline and baseline[name]["solved"] == 1
        d_ok = name in diff and diff[name]["solved"] == 1
        if b_ok and not d_ok:
            status_changes.append(f"- **{name}**: ⚠️ regression — branch did NOT solve (baseline did)")
        elif d_ok and not b_ok:
            status_changes.append(f"- **{name}**: ✅ improvement — branch solved (baseline did NOT)")

    if status_changes:
        lines.append("#### Correctness Changes")
        lines.extend(status_changes)
        lines.append("")

    # ── Codegen time table ─────────────────────────────────────────────────────
    codegen_regressions = []
    codegen_improvements = []
    codegen_rows = []
    for name in common:
        b = baseline[name]["codegen_time_s"]
        d = diff[name]["codegen_time_s"]
        p = _pct(d, b)
        codegen_rows.append((name, b, d, p))
        if not math.isnan(p):
            if p > CODEGEN_THRESHOLD:
                codegen_regressions.append((name, b, d, p))
            elif p < -CODEGEN_THRESHOLD:
                codegen_improvements.append((name, b, d, p))

    lines.append(f"#### Codegen Time (flagged if |Δ| > {CODEGEN_THRESHOLD*100:.0f}%)\n")
    lines.append("| Problem | Baseline (s) | Branch (s) | Change |")
    lines.append("|---------|-------------|------------|--------|")
    for name, b, d, p in codegen_rows:
        flag = " ⚠️" if p > CODEGEN_THRESHOLD else (" ✅" if p < -CODEGEN_THRESHOLD else "")
        lines.append(f"| {name} | {b:.4f} | {d:.4f} | {_fmt_pct(p)}{flag} |")
    lines.append("")

    # ── Solve time table ───────────────────────────────────────────────────────
    solvetime_regressions = []
    solvetime_improvements = []
    solvetime_rows = []
    both_solved = [n for n in common if baseline[n]["solved"] == 1 and diff[n]["solved"] == 1]
    for name in both_solved:
        b = baseline[name]["solve_time_s"]
        d = diff[name]["solve_time_s"]
        p = _pct(d, b)
        solvetime_rows.append((name, b, d, p))
        if not math.isnan(p):
            if p > SOLVETIME_THRESHOLD:
                solvetime_regressions.append((name, b, d, p))
            elif p < -SOLVETIME_THRESHOLD:
                solvetime_improvements.append((name, b, d, p))

    lines.append(f"#### Solve Time — problems solved by both (flagged if |Δ| > {SOLVETIME_THRESHOLD*100:.0f}%)\n")
    lines.append("| Problem | Baseline (s) | Branch (s) | Change |")
    lines.append("|---------|-------------|------------|--------|")
    for name, b, d, p in solvetime_rows:
        flag = " ⚠️" if p > SOLVETIME_THRESHOLD else (" ✅" if p < -SOLVETIME_THRESHOLD else "")
        lines.append(f"| {name} | {b:.6f} | {d:.6f} | {_fmt_pct(p)}{flag} |")
    lines.append("")

    # ── Summary ────────────────────────────────────────────────────────────────
    if codegen_regressions or solvetime_regressions:
        lines.append("#### ⚠️ Regressions")
        for name, b, d, p in codegen_regressions:
            lines.append(f"- **{name}** codegen: {b:.4f}s → {d:.4f}s ({_fmt_pct(p)})")
        for name, b, d, p in solvetime_regressions:
            lines.append(f"- **{name}** solvetime: {b:.6f}s → {d:.6f}s ({_fmt_pct(p)})")
        lines.append("")

    if codegen_improvements or solvetime_improvements:
        lines.append("#### ✅ Improvements")
        for name, b, d, p in codegen_improvements:
            lines.append(f"- **{name}** codegen: {b:.4f}s → {d:.4f}s ({_fmt_pct(p)})")
        for name, b, d, p in solvetime_improvements:
            lines.append(f"- **{name}** solvetime: {b:.6f}s → {d:.6f}s ({_fmt_pct(p)})")
        lines.append("")

    if not status_changes and not codegen_regressions and not solvetime_regressions:
        lines.append("_No regressions detected._\n")

    return "\n".join(lines)


if __name__ == "__main__":
    baseline_csv = sys.argv[1]
    diff_csv = sys.argv[2]
    artifact_url = os.environ.get("ARTIFACT_URL") or None

    report = generate_report(baseline_csv, diff_csv, artifact_url=artifact_url)
    print(report)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to {OUTPUT_PATH}")
