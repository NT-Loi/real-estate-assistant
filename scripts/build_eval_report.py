"""Build tables and lightweight plots from saved evaluation outputs.

This script is intentionally dependency-light: it uses only the Python standard
library and writes Markdown, CSV, and SVG files. Input files are produced by:

- scripts/run_evaluation.py
- scripts/evaluate_custom_metrics.py
- scripts/evaluate_ragas.py
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RATE_KEYS = [
    "tool_subset_hit_rate",
    "tool_exact_hit_rate",
    "source_hit_rate",
    "behavior_match_rate",
    "intent_hit_rate",
    "filter_key_hit_rate",
    "has_citation_rate",
    "citation_hallucination_rate",
]

RATE_LABELS = {
    "tool_subset_hit_rate": "Tool subset hit",
    "tool_exact_hit_rate": "Tool exact hit",
    "source_hit_rate": "Expected source hit",
    "behavior_match_rate": "Behavior match",
    "intent_hit_rate": "Intent hit",
    "filter_key_hit_rate": "Filter key hit",
    "has_citation_rate": "Has citation",
    "citation_hallucination_rate": "Citation hallucination",
}


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def safe_metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_bar_svg(
    path: Path,
    title: str,
    items: list[tuple[str, float]],
    width: int = 980,
    row_h: int = 32,
    left: int = 250,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = 70 + row_h * len(items)
    bar_w = width - left - 120
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for idx, (label, value) in enumerate(items):
        value = max(0.0, min(float(value), 1.0))
        y = 62 + idx * row_h
        w = max(1, int(bar_w * value))
        color = "#1f77b4" if "hallucination" not in label.lower() else "#d62728"
        lines.extend([
            f'<text x="24" y="{y + 18}" font-family="Arial, sans-serif" font-size="13">{html.escape(label)}</text>',
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="20" rx="3" fill="#edf2f7"/>',
            f'<rect x="{left}" y="{y}" width="{w}" height="20" rx="3" fill="{color}"/>',
            f'<text x="{left + bar_w + 12}" y="{y + 15}" font-family="Arial, sans-serif" font-size="13">{value * 100:.1f}%</text>',
        ])
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png_plots(
    output_dir: Path,
    aggregate: dict[str, Any],
    by_category: dict[str, Any],
    ragas_agg: dict[str, float],
    ragas_cat: dict[str, dict[str, Any]],
    ragas_keys: list[str],
    tool_rows: list[list[Any]],
    worst_rows: list[list[Any]],
) -> list[str]:
    """Write richer matplotlib/seaborn PNG plots when plotting deps exist."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.9)
    created: list[str] = []

    def save(fig, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(f"plots/{filename}")

    overall_df = pd.DataFrame([
        {"metric": RATE_LABELS.get(key, key), "value": float(aggregate.get(key) or 0.0)}
        for key in RATE_KEYS
        if key in aggregate
    ])
    if not overall_df.empty:
        overall_df["percent"] = overall_df["value"] * 100
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        colors = ["#d62728" if "hallucination" in m.lower() else "#2f6fdd" for m in overall_df["metric"]]
        sns.barplot(data=overall_df, y="metric", x="percent", ax=ax, palette=colors, hue="metric", legend=False)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Rate (%)")
        ax.set_ylabel("")
        ax.set_title("Overall Deterministic Metrics")
        for patch, value in zip(ax.patches, overall_df["percent"]):
            ax.text(min(value + 1.0, 99), patch.get_y() + patch.get_height() / 2, f"{value:.1f}%", va="center")
        save(fig, "overall_rates.png")

    category_rows = []
    for category, stats in sorted(by_category.items()):
        row = {"category": category}
        for key in [
            "tool_subset_hit_rate",
            "source_hit_rate",
            "behavior_match_rate",
            "filter_key_hit_rate",
            "citation_hallucination_rate",
        ]:
            row[RATE_LABELS.get(key, key)] = float(stats.get(key) or 0.0) * 100
        latency = stats.get("latency_seconds") or {}
        row["P50 latency"] = float(latency.get("p50") or 0.0)
        row["P90 latency"] = float(latency.get("p90") or 0.0)
        row["N"] = int(stats.get("total") or 0)
        category_rows.append(row)
    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        metric_cols = [
            RATE_LABELS["tool_subset_hit_rate"],
            RATE_LABELS["source_hit_rate"],
            RATE_LABELS["behavior_match_rate"],
            RATE_LABELS["filter_key_hit_rate"],
            RATE_LABELS["citation_hallucination_rate"],
        ]
        fig, ax = plt.subplots(figsize=(11, max(4.8, 0.55 * len(category_df) + 2)))
        sns.heatmap(
            category_df.set_index("category")[metric_cols],
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Rate (%)"},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Category Metric Heatmap")
        save(fig, "category_metric_heatmap.png")

        latency_df = category_df.melt(
            id_vars=["category"],
            value_vars=["P50 latency", "P90 latency"],
            var_name="metric",
            value_name="seconds",
        )
        fig, ax = plt.subplots(figsize=(11, max(4.8, 0.45 * len(category_df) + 2)))
        sns.barplot(data=latency_df, y="category", x="seconds", hue="metric", ax=ax)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("")
        ax.set_title("Latency By Category")
        save(fig, "category_latency.png")

    if ragas_agg:
        ragas_df = pd.DataFrame([
            {"metric": key, "score": value}
            for key, value in ragas_agg.items()
        ])
        fig, ax = plt.subplots(figsize=(9.5, 4.5))
        sns.barplot(data=ragas_df, y="metric", x="score", ax=ax, color="#1b9e77")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean score")
        ax.set_ylabel("")
        ax.set_title("Ragas Mean Scores")
        for patch, value in zip(ax.patches, ragas_df["score"]):
            ax.text(min(value + 0.015, 0.98), patch.get_y() + patch.get_height() / 2, f"{value:.3f}", va="center")
        save(fig, "ragas_scores.png")

    if ragas_cat and ragas_keys:
        rows = []
        for category, stats in ragas_cat.items():
            row = {"category": category}
            for key in ragas_keys:
                value = stats.get(key)
                row[key] = float(value) if value is not None else 0.0
            rows.append(row)
        ragas_cat_df = pd.DataFrame(rows)
        if not ragas_cat_df.empty:
            fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.55 * len(ragas_cat_df) + 2)))
            sns.heatmap(
                ragas_cat_df.set_index("category")[ragas_keys],
                annot=True,
                fmt=".3f",
                cmap="Greens",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Score"},
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("Ragas Scores By Category")
            save(fig, "ragas_by_category_heatmap.png")

    if tool_rows:
        tool_df = pd.DataFrame(tool_rows, columns=["tool", "expected_count", "actual_count"])
        tool_df = tool_df.melt(id_vars=["tool"], value_vars=["expected_count", "actual_count"], var_name="kind", value_name="count")
        fig, ax = plt.subplots(figsize=(11, max(4.8, 0.45 * len(tool_rows) + 2)))
        sns.barplot(data=tool_df, y="tool", x="count", hue="kind", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title("Expected vs Actual Tool Calls")
        save(fig, "tool_distribution.png")

    if worst_rows:
        counts = Counter(str(row[1]) for row in worst_rows)
        issue_df = pd.DataFrame([{"category": k, "count": v} for k, v in counts.most_common()])
        fig, ax = plt.subplots(figsize=(10, max(4.4, 0.45 * len(issue_df) + 2)))
        sns.barplot(data=issue_df, y="category", x="count", ax=ax, color="#d95f02")
        ax.set_xlabel("Worst-case count")
        ax.set_ylabel("")
        ax.set_title("Worst Cases By Category")
        save(fig, "worst_cases_by_category.png")

    return created


def ragas_metric_keys(records: list[dict[str, Any]]) -> list[str]:
    skip = {"user_input", "retrieved_contexts", "response", "reference"}
    keys: list[str] = []
    for row in records:
        for key, value in row.items():
            if key in skip or key in keys:
                continue
            if safe_metric_value(row, key) is not None:
                keys.append(key)
    return keys


def ragas_aggregate(records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    agg = {}
    for key in metric_keys:
        values = [v for row in records if (v := safe_metric_value(row, key)) is not None]
        avg = mean(values)
        if avg is not None:
            agg[key] = avg
    return agg


def ragas_by_category(rows: list[dict[str, Any]], records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, dict[str, Any]]:
    question_to_category = {str(row.get("question") or ""): str(row.get("category") or "<unknown>") for row in rows}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = question_to_category.get(str(record.get("user_input") or ""), "<unknown>")
        buckets[category].append(record)

    output: dict[str, dict[str, Any]] = {}
    for category, bucket in sorted(buckets.items()):
        item = {"sample_count": len(bucket)}
        item.update(ragas_aggregate(bucket, metric_keys))
        output[category] = item
    return output


def build_report(
    run_dir: Path,
    output_dir: Path,
    results_path: Path,
    custom_path: Path,
    ragas_path: Path,
    summary_path: Path,
    config_path: Path,
) -> Path:
    rows = load_jsonl(results_path)
    custom = load_json(custom_path)
    ragas = load_json(ragas_path)
    summary = load_json(summary_path)
    config = load_json(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    aggregate = custom.get("aggregate") or {}
    by_category = custom.get("by_category") or {}
    ragas_records = ragas.get("records") or []
    ragas_keys = ragas_metric_keys(ragas_records)
    ragas_agg = ragas_aggregate(ragas_records, ragas_keys)
    ragas_cat = ragas_by_category(rows, ragas_records, ragas_keys)

    overall_rows = [
        [RATE_LABELS.get(key, key), pct(aggregate.get(key))]
        for key in RATE_KEYS
        if key in aggregate
    ]
    write_csv(tables_dir / "overall_metrics.csv", ["metric", "value"], overall_rows)

    category_rows = []
    for category, stats in sorted(by_category.items()):
        latency = stats.get("latency_seconds") or {}
        category_rows.append([
            category,
            stats.get("total", 0),
            pct(stats.get("tool_subset_hit_rate")),
            pct(stats.get("source_hit_rate")),
            pct(stats.get("behavior_match_rate")),
            pct(stats.get("filter_key_hit_rate")),
            pct(stats.get("citation_hallucination_rate")),
            num(latency.get("p50"), 2),
            num(latency.get("p90"), 2),
        ])
    write_csv(
        tables_dir / "category_metrics.csv",
        ["category", "n", "tool_subset_hit", "source_hit", "behavior_match", "filter_key_hit", "citation_hallucination", "latency_p50_s", "latency_p90_s"],
        category_rows,
    )

    ragas_rows = [[key, num(value, 4)] for key, value in ragas_agg.items()]
    write_csv(tables_dir / "ragas_metrics.csv", ["metric", "mean"], ragas_rows)

    ragas_category_rows = []
    for category, stats in ragas_cat.items():
        ragas_category_rows.append(
            [category, stats.get("sample_count", 0)] + [num(stats.get(key), 4) for key in ragas_keys]
        )
    write_csv(tables_dir / "ragas_by_category.csv", ["category", "ragas_n"] + ragas_keys, ragas_category_rows)

    expected_tools = custom.get("expected_tool_distribution") or {}
    actual_tools = custom.get("actual_tool_distribution") or {}
    tool_rows = []
    for tool in sorted(set(expected_tools) | set(actual_tools)):
        tool_rows.append([tool, expected_tools.get(tool, 0), actual_tools.get(tool, 0)])
    write_csv(tables_dir / "tool_distribution.csv", ["tool", "expected_count", "actual_count"], tool_rows)

    worst_rows = []
    for item in custom.get("worst_cases") or []:
        metrics = item.get("metrics") or {}
        worst_rows.append([
            item.get("id"),
            item.get("category"),
            item.get("issue_score"),
            item.get("question"),
            ", ".join(item.get("expected_tools") or []),
            ", ".join(item.get("actual_tools") or []),
            metrics.get("elapsed_seconds"),
            metrics.get("citation_hallucination"),
        ])
    write_csv(tables_dir / "worst_cases.csv", ["id", "category", "issue_score", "question", "expected_tools", "actual_tools", "elapsed_seconds", "citation_hallucination"], worst_rows)

    png_plots = write_png_plots(
        output_dir=output_dir,
        aggregate=aggregate,
        by_category=by_category,
        ragas_agg=ragas_agg,
        ragas_cat=ragas_cat,
        ragas_keys=ragas_keys,
        tool_rows=tool_rows,
        worst_rows=worst_rows,
    )

    chart_items = [
        (RATE_LABELS.get(key, key), float(aggregate.get(key) or 0.0))
        for key in RATE_KEYS
        if key in aggregate
    ]
    write_bar_svg(plots_dir / "overall_rates.svg", "Overall Deterministic Metrics", chart_items)

    write_bar_svg(
        plots_dir / "category_source_hit.svg",
        "Expected Source Hit By Category",
        [(category, float(stats.get("source_hit_rate") or 0.0)) for category, stats in sorted(by_category.items())],
        left=310,
    )
    write_bar_svg(
        plots_dir / "category_behavior_match.svg",
        "Behavior Match By Category",
        [(category, float(stats.get("behavior_match_rate") or 0.0)) for category, stats in sorted(by_category.items())],
        left=310,
    )

    ragas_chart = [(key, value) for key, value in ragas_agg.items()]
    if ragas_chart:
        write_bar_svg(plots_dir / "ragas_scores.svg", "Ragas Mean Scores", ragas_chart)

    markdown = []
    markdown.append("# Evaluation Report\n")
    markdown.append("## Run Summary\n")
    markdown.append(md_table(
        ["Field", "Value"],
        [
            ["Run directory", str(run_dir)],
            ["Dataset", config.get("dataset", "N/A")],
            ["Evaluated cases", summary.get("total") or aggregate.get("total") or len(rows)],
            ["Ragas samples", ragas.get("sample_count", len(ragas_records))],
            ["Ragas judge", ragas.get("judge_model", "N/A")],
            ["Ragas metrics", ", ".join(ragas.get("metrics") or [])],
            ["Max iterations", config.get("max_iterations", "N/A")],
        ],
    ))

    markdown.append("\n## Overall Deterministic Metrics\n")
    markdown.append(md_table(["Metric", "Value"], overall_rows))
    markdown.append(f"\n![Overall rates](plots/overall_rates.svg)\n")
    if "plots/overall_rates.png" in png_plots:
        markdown.append(f"\n![Overall rates PNG](plots/overall_rates.png)\n")

    markdown.append("\n## Category Breakdown\n")
    markdown.append(md_table(
        ["Category", "N", "Tool subset", "Source hit", "Behavior", "Filter key", "Citation halluc.", "P50 latency", "P90 latency"],
        category_rows,
    ))
    markdown.append("\n![Source hit by category](plots/category_source_hit.svg)\n")
    markdown.append("\n![Behavior match by category](plots/category_behavior_match.svg)\n")
    if "plots/category_metric_heatmap.png" in png_plots:
        markdown.append("\n![Category metric heatmap](plots/category_metric_heatmap.png)\n")
    if "plots/category_latency.png" in png_plots:
        markdown.append("\n![Category latency](plots/category_latency.png)\n")

    markdown.append("\n## Ragas Metrics\n")
    if ragas_rows:
        markdown.append(md_table(["Metric", "Mean"], ragas_rows))
        markdown.append("\n![Ragas scores](plots/ragas_scores.svg)\n")
        if "plots/ragas_scores.png" in png_plots:
            markdown.append("\n![Ragas scores PNG](plots/ragas_scores.png)\n")
    else:
        markdown.append("No numeric Ragas records were found.\n")

    if ragas_category_rows:
        markdown.append("\n## Ragas By Category\n")
        markdown.append(md_table(["Category", "Ragas N"] + ragas_keys, ragas_category_rows))
        if "plots/ragas_by_category_heatmap.png" in png_plots:
            markdown.append("\n![Ragas by category heatmap](plots/ragas_by_category_heatmap.png)\n")

    markdown.append("\n## Tool Distribution\n")
    markdown.append(md_table(["Tool", "Expected", "Actual"], tool_rows))
    if "plots/tool_distribution.png" in png_plots:
        markdown.append("\n![Tool distribution](plots/tool_distribution.png)\n")

    markdown.append("\n## Main Findings\n")
    findings = []
    if aggregate.get("filter_key_hit_rate", 1.0) < 0.75:
        findings.append(f"- Filter extraction is the weakest deterministic metric ({pct(aggregate.get('filter_key_hit_rate'))}). Review query parser coverage and tool argument normalization.")
    if aggregate.get("citation_hallucination_rate", 0.0) > 0.05:
        findings.append(f"- Citation hallucination exists in {pct(aggregate.get('citation_hallucination_rate'))} of rows. Prefer returning only citations actually used in the final answer and tighten answer packaging.")
    if by_category:
        weakest_source = min(by_category.items(), key=lambda kv: kv[1].get("source_hit_rate", 1.0))
        findings.append(f"- Weakest source-hit category: {weakest_source[0]} ({pct(weakest_source[1].get('source_hit_rate'))}).")
        weakest_behavior = min(by_category.items(), key=lambda kv: kv[1].get("behavior_match_rate", 1.0))
        findings.append(f"- Weakest behavior-match category: {weakest_behavior[0]} ({pct(weakest_behavior[1].get('behavior_match_rate'))}).")
    if ragas_agg:
        weakest_ragas = min(ragas_agg.items(), key=lambda kv: kv[1])
        findings.append(f"- Lowest Ragas metric: {weakest_ragas[0]} ({num(weakest_ragas[1], 3)}). Use the per-category table to locate the failure mode.")
    markdown.extend(findings or ["- No obvious aggregate issue found from the available metrics."])

    markdown.append("\n## Worst Cases\n")
    markdown.append(f"See `{tables_dir.relative_to(output_dir) / 'worst_cases.csv'}` for the full worst-case list.")
    markdown.append(md_table(
        ["ID", "Category", "Issue Score", "Question"],
        [[r[0], r[1], r[2], str(r[3])[:140] + ("..." if len(str(r[3])) > 140 else "")] for r in worst_rows[:10]],
    ))
    if "plots/worst_cases_by_category.png" in png_plots:
        markdown.append("\n![Worst cases by category](plots/worst_cases_by_category.png)\n")

    markdown.append("\n## Artifacts\n")
    markdown.append("- `tables/overall_metrics.csv`\n- `tables/category_metrics.csv`\n- `tables/ragas_metrics.csv`\n- `tables/ragas_by_category.csv`\n- `tables/tool_distribution.csv`\n- `tables/worst_cases.csv`\n- `plots/overall_rates.svg`\n- `plots/category_source_hit.svg`\n- `plots/category_behavior_match.svg`\n- `plots/ragas_scores.svg`\n")
    if png_plots:
        markdown.append("\nPNG plots:\n")
        markdown.extend(f"- `{plot}`" for plot in png_plots)

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Markdown/CSV/SVG evaluation report.")
    parser.add_argument("--run-dir", type=Path, default=Path("logs/eval_runs/baseline_gemini-2.5-flash"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--custom", type=Path, default=None)
    parser.add_argument("--ragas", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "report")
    report_path = build_report(
        run_dir=run_dir,
        output_dir=output_dir,
        results_path=args.results or (run_dir / "results.jsonl"),
        custom_path=args.custom or (run_dir / "custom_metrics.json"),
        ragas_path=args.ragas or (run_dir / "ragas_results.json"),
        summary_path=args.summary or (run_dir / "summary.json"),
        config_path=args.config or (run_dir / "config.json"),
    )
    print(json.dumps({"report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
