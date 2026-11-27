#!/usr/bin/env python3
"""Plot the six measured CPAEP workloads exported by tb_mac_array_gemm.sv.

Install once:  python3 -m pip install matplotlib
Run:           python3 plot_roofline.py
Other input:   python3 plot_roofline.py results/run2.csv --output-dir plots/run2
On Windows, use `py` instead of `python3` if that is your Python launcher.

Place this script in the project root. First finish the tb_mac_array_gemm
simulation: it writes roofline_results.csv in the simulator's working directory.
By default, the plotter saves roofline.png, roofline.pdf, roofline.svg and
roofline_metrics.csv in a roofline folder next to that input CSV.

One MAC counts as two integer operations. Traffic counts every complete SRAM
bank word read or written; register accesses are outside this SRAM roofline.
No workload timings or arithmetic intensities are hardcoded in this script.
Point labels show AI to three decimals and performance to two decimals; the
metrics CSV retains higher precision. The figure has no main title or footer,
so its caption can be supplied by the report.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


RAW_FIELDS = (
    "schema_version", "workload", "M", "K", "N", "cycles", "compute_cycles",
    "a_read_words", "b_read_words", "c_write_words", "num_macs", "num_banks",
    "bank_word_bits", "peak_bandwidth_bits_per_cycle", "m_reuse", "gemm_rows",
    "gemv_fragment_elements", "output_queue_depth", "status",
)
CONFIG_FIELDS = (
    "num_macs", "num_banks", "bank_word_bits", "peak_bandwidth_bits_per_cycle",
    "m_reuse", "gemm_rows", "gemv_fragment_elements", "output_queue_depth",
)
COUNT_FIELDS = ("a_read_words", "b_read_words", "c_write_words")
DERIVED_FIELDS = (
    "mac_count", "operations", "a_read_bytes", "b_read_bytes", "c_write_bytes",
    "traffic_bytes", "arithmetic_intensity_ops_per_byte", "throughput_ops_per_cycle",
    "cycles_per_mkn", "spatial_util_pct", "temporal_util_pct", "total_util_pct",
    "peak_compute_ops_per_cycle", "peak_bandwidth_bytes_per_cycle",
    "ridge_point_ops_per_byte", "roofline_ops_per_cycle", "roofline_efficiency_pct",
    "achieved_bandwidth_bytes_per_cycle", "bandwidth_util_pct",
)


def load_results(path: Path) -> list[dict]:
    """Validate a completed six-row export and derive all plotting quantities."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("The CSV is empty. Finish a successful simulation first.")
        if len(set(header)) != len(header):
            raise ValueError("The CSV contains duplicate column names.")
        missing = sorted(set(RAW_FIELDS) - set(header))
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(missing))
        rows = list(reader)
    if len(rows) != 6:
        raise ValueError(
            f"Expected six complete workload rows, found {len(rows)}. "
            "Run the entire testbench successfully before plotting."
        )

    results = []
    names = set()
    for row_number, raw in enumerate(rows, start=2):
        if None in raw or any(raw.get(key) is None for key in RAW_FIELDS):
            raise ValueError(f"CSV row {row_number} has an incorrect number of fields.")
        row = {key: raw[key].strip() for key in RAW_FIELDS}
        name = row["workload"]
        if not name or name in names:
            raise ValueError(f"CSV row {row_number} has an empty or duplicate workload name.")
        names.add(name)
        if row["status"] != "PASS":
            raise ValueError(f"{name}: status must be PASS, got {row['status']!r}.")
        for key in RAW_FIELDS:
            if key in ("workload", "status"):
                continue
            try:
                value = int(row[key])
            except ValueError as exc:
                raise ValueError(f"{name}: {key} must be an integer.") from exc
            minimum = 0 if key in COUNT_FIELDS else 1
            if value < minimum:
                raise ValueError(f"{name}: {key} must be at least {minimum}.")
            row[key] = value
        if row["schema_version"] != 1:
            raise ValueError(f"{name}: unsupported CSV schema version {row['schema_version']}.")
        if results and any(row[key] != results[0][key] for key in CONFIG_FIELDS):
            raise ValueError("All six workloads must use the same accelerator configuration.")
        if row["bank_word_bits"] % 8:
            raise ValueError(f"{name}: bank_word_bits must be a multiple of eight.")
        bandwidth_bits = row["peak_bandwidth_bits_per_cycle"]
        if bandwidth_bits > row["num_banks"] * row["bank_word_bits"]:
            raise ValueError(f"{name}: bandwidth exceeds the physical bank widths.")

        macs = row["M"] * row["K"] * row["N"]
        cycles, compute_cycles = row["cycles"], row["compute_cycles"]
        if compute_cycles > cycles:
            raise ValueError(f"{name}: compute_cycles exceeds total cycles.")
        if macs > row["num_macs"] * compute_cycles:
            raise ValueError(f"{name}: useful work exceeds the recorded compute capacity.")
        word_bytes = row["bank_word_bits"] // 8
        a_bytes, b_bytes, c_bytes = (row[key] * word_bytes for key in COUNT_FIELDS)
        traffic = a_bytes + b_bytes + c_bytes
        if traffic <= 0:
            raise ValueError(f"{name}: total SRAM traffic must be positive.")
        if 8 * traffic > bandwidth_bits * cycles:
            raise ValueError(f"{name}: average SRAM traffic exceeds the peak bandwidth.")

        operations = 2 * macs
        peak = 2 * row["num_macs"]
        bandwidth = bandwidth_bits / 8
        intensity = operations / traffic
        performance = operations / cycles
        roof = min(peak, bandwidth * intensity)
        row.update(
            mac_count=macs, operations=operations,
            a_read_bytes=a_bytes, b_read_bytes=b_bytes, c_write_bytes=c_bytes,
            traffic_bytes=traffic,
            arithmetic_intensity_ops_per_byte=intensity,
            throughput_ops_per_cycle=performance,
            cycles_per_mkn=cycles / macs,
            spatial_util_pct=100 * macs / (row["num_macs"] * compute_cycles),
            temporal_util_pct=100 * compute_cycles / cycles,
            total_util_pct=100 * macs / (row["num_macs"] * cycles),
            peak_compute_ops_per_cycle=peak,
            peak_bandwidth_bytes_per_cycle=bandwidth,
            ridge_point_ops_per_byte=peak / bandwidth,
            roofline_ops_per_cycle=roof,
            roofline_efficiency_pct=100 * performance / roof,
            achieved_bandwidth_bytes_per_cycle=traffic / cycles,
            bandwidth_util_pct=100 * traffic / (cycles * bandwidth),
        )
        results.append(row)
    return results


def save_metrics(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS + DERIVED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in row.items()})


def plot_results(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # Works on machines without a graphical display.
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch, Rectangle
        from matplotlib.ticker import LogLocator, StrMethodFormatter, MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required. Install it with: python3 -m pip install matplotlib "
            "(or: py -m pip install matplotlib on Windows)."
        ) from exc

    peak = rows[0]["peak_compute_ops_per_cycle"]
    bandwidth = rows[0]["peak_bandwidth_bytes_per_cycle"]
    knee = peak / bandwidth
    intensities = [row["arithmetic_intensity_ops_per_byte"] for row in rows]
    performances = [row["throughput_ops_per_cycle"] for row in rows]
    x_min = 2 ** math.floor(math.log2(min(min(intensities), knee) / 2))
    x_max = 2 ** math.ceil(math.log2(max(max(intensities), knee) * 2))
    y_min = 2 ** math.floor(math.log2(min(min(performances), bandwidth * x_min) / 2))
    colors = {"GeMM": "#2466a3", "GeMV": "#c76817"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.97)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, peak * 1.7)
    ax.plot([x_min, knee, x_max], [bandwidth * x_min, peak, peak],
            color="#263746", linewidth=2.3, label="Theoretical roofline", zorder=2)
    ax.vlines(knee, y_min, peak, color="#8998a5", linestyles="dashed", linewidth=1)
    ax.text(0.985, peak * 1.12, f"Peak compute: {peak:g} OP/cycle",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=11)
    slope_x = max(x_min * 1.6, min(knee / 6, knee * 0.65))
    ax.annotate(f"Peak SRAM bandwidth:\n{bandwidth:g} bytes/cycle",
                xy=(slope_x, bandwidth * slope_x), xytext=(0.035, 0.53),
                textcoords="axes fraction", fontsize=10.5, ha="left",
                arrowprops={"arrowstyle": "-", "color": "#566675"},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    def point_label(row: dict, *, include_units: bool = True) -> str:
        ai_unit = " OP/byte" if include_units else ""
        perf_unit = " OP/cycle" if include_units else ""
        return (f"{row['workload']}\n"
                f"AI = {row['arithmetic_intensity_ops_per_byte']:.3f}{ai_unit}\n"
                f"P = {row['throughput_ops_per_cycle']:.2f}{perf_unit}")

    gemv_rows = [row for row in rows if row["N"] == 1]
    zoom_gemv = len(gemv_rows) > 1
    for mode in ("GeMM", "GeMV"):
        subset = [row for row in rows if (row["N"] == 1) == (mode == "GeMV")]
        if not subset:
            continue
        ax.scatter([row["arithmetic_intensity_ops_per_byte"] for row in subset],
                   [row["throughput_ops_per_cycle"] for row in subset],
                   s=58, color=colors[mode], edgecolors="white", linewidths=0.7,
                   label=mode, zorder=4)
        if mode == "GeMV" and zoom_gemv:
            point = subset[len(subset) // 2]
            ax.annotate("GeMV workloads\n(see detail)",
                        xy=(point["arithmetic_intensity_ops_per_byte"],
                            point["throughput_ops_per_cycle"]),
                        xytext=(-22, 30), textcoords="offset points",
                        ha="right", fontsize=10, color=colors[mode],
                        arrowprops={"arrowstyle": "-", "color": colors[mode]})
        elif mode == "GeMM":
            # A fixed label column prevents collisions when the measured
            # performances change; leaders attach to the labels' left edges.
            # Higher labels render after lower leaders to protect their text.
            label_rows = sorted(subset, key=lambda row: row["throughput_ops_per_cycle"],
                                reverse=True)
            spacing = min(0.12, 0.27 / max(1, len(label_rows) - 1))
            for index, row in enumerate(label_rows):
                ax.annotate(point_label(row),
                            xy=(row["arithmetic_intensity_ops_per_byte"],
                                row["throughput_ops_per_cycle"]),
                            xytext=(0.975, 0.79 - index * spacing),
                            textcoords="axes fraction", fontsize=9,
                            zorder=3.3 - 0.05 * index,
                            color=colors[mode], ha="right", va="center",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                            arrowprops={"arrowstyle": "-", "color": colors[mode],
                                        "linewidth": 0.7, "relpos": (0, 0.5)})
        else:
            for index, row in enumerate(subset):
                below = row["throughput_ops_per_cycle"] >= 0.8 * peak
                dy = -(17 + index * 15) if below else 12 + index * 13
                ax.annotate(point_label(row),
                            xy=(row["arithmetic_intensity_ops_per_byte"],
                                row["throughput_ops_per_cycle"]),
                            xytext=(8, dy), textcoords="offset points", fontsize=10,
                            color=colors[mode], ha="left",
                            arrowprops={"arrowstyle": "-", "color": colors[mode],
                                        "linewidth": 0.7})

    if zoom_gemv:
        detail = ax.inset_axes([0.58, 0.075, 0.395, 0.35])
        gx = [row["arithmetic_intensity_ops_per_byte"] for row in gemv_rows]
        gy = [row["throughput_ops_per_cycle"] for row in gemv_rows]
        x_pad = max((max(gx) - min(gx)) * 0.8, statistics.mean(gx) * 0.003)
        x_lo, x_hi = max(min(gx) - x_pad, min(gx) / 2), max(gx) + x_pad
        roof_x = [x_lo] + ([knee] if x_lo < knee < x_hi else []) + [x_hi]
        roof_y = [min(peak, bandwidth * value) for value in roof_x]
        y_all = gy + roof_y
        y_pad = max((max(y_all) - min(y_all)) * 0.5, max(y_all) * 0.025)
        y_lo, y_hi = max(0, min(y_all) - y_pad), max(y_all) + y_pad
        detail.set_xlim(x_lo, x_hi)
        detail.set_ylim(y_lo, y_hi)
        detail.plot(roof_x, roof_y, color="#263746", linewidth=1.3)
        detail.scatter(gx, gy, s=34, color=colors["GeMV"], zorder=4)
        # Alternate fixed label bands so the three close operating points
        # remain legible when AI and performance are included in each label.
        ordered_gemv = sorted(gemv_rows,
                              key=lambda row: row["arithmetic_intensity_ops_per_byte"])
        for index, row in enumerate(ordered_gemv):
            fraction = index / max(1, len(ordered_gemv) - 1)
            upper_band = index % 2 == 1
            label_x = 0.03 + 0.94 * fraction
            align = "left" if index == 0 else (
                "right" if index == len(ordered_gemv) - 1 else "center")
            detail.annotate(point_label(row),
                            xy=(row["arithmetic_intensity_ops_per_byte"],
                                row["throughput_ops_per_cycle"]),
                            xytext=(label_x, 0.94 if upper_band else 0.06),
                            textcoords="axes fraction", ha=align,
                            va="top" if upper_band else "bottom",
                            fontsize=8, color=colors["GeMV"],
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                            arrowprops={"arrowstyle": "-", "color": colors["GeMV"],
                                        "linewidth": 0.65, "shrinkA": 3, "shrinkB": 4})
        detail.set_title("GeMV detail (linear axes)", fontsize=9, pad=6)
        detail.set_xlabel("AI [OP/byte]", fontsize=8, labelpad=2)
        detail.set_ylabel("P [OP/cycle]", fontsize=8, labelpad=2)
        detail.xaxis.set_major_locator(MaxNLocator(3))
        detail.yaxis.set_major_locator(MaxNLocator(4))
        detail.tick_params(labelsize=8)
        detail.ticklabel_format(style="plain", useOffset=False)
        detail.grid(alpha=0.18)
        # Main-axes reference/connector lines must not show through the
        # inset tick labels, which are drawn outside its white plotting area.
        for tick_label in detail.get_xticklabels() + detail.get_yticklabels():
            tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.7})

        # Show the exact zoomed range on the main axes. Explicit connections
        # to the inset's left corners preserve the intended visual link.
        connector_color = "#8998a5"
        ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                               fill=False, edgecolor=connector_color,
                               linewidth=0.8, zorder=3))
        for source_y, inset_y in ((y_hi, 1), (y_lo, 0)):
            ax.add_artist(ConnectionPatch(
                xyA=(x_hi, source_y), coordsA=ax.transData,
                xyB=(0, inset_y), coordsB=detail.transAxes,
                axesA=ax, axesB=detail, color=connector_color,
                linewidth=0.9, clip_on=False, zorder=1.5))

    ax.set_xlabel("Arithmetic intensity [OP/byte]", labelpad=10)
    ax.set_ylabel("Achieved performance [OP/cycle]", labelpad=10)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=2, numticks=12))
        axis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.grid(which="major", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"roofline.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("roofline_results.csv"),
                        help="testbench CSV (default: roofline_results.csv)")
    parser.add_argument("--output-dir", type=Path,
                        help="output folder (default: roofline next to the input CSV)")
    args = parser.parse_args(argv)
    try:
        source = args.csv_path.expanduser().resolve()
        rows = load_results(source)
        output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                      else source.parent / "roofline")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_results(rows, output_dir)
        save_metrics(rows, output_dir / "roofline_metrics.csv")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Read six PASS workloads from: {source}")
    print(f"Saved roofline.png, roofline.pdf, roofline.svg and roofline_metrics.csv in:\n  {output_dir}")
    print(f"Mean normalized latency: {statistics.mean(r['cycles_per_mkn'] for r in rows):.9f} cycles/MKN")
    print("Mean utilization: " + ", ".join(
        f"{label} {statistics.mean(r[field] for r in rows):.2f}%"
        for label, field in (("spatial", "spatial_util_pct"),
                             ("temporal", "temporal_util_pct"), ("total", "total_util_pct"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
