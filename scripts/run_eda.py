"""EDA + resampling entry point.

Usage (from project root):
    python scripts/run_eda.py

For each turbine this:
  1. loads the raw 10-minute CSV and profiles it (range, gaps, value ranges);
  2. resamples it to hourly bins, flagging low-coverage hours (`is_gap`) and
     hours that underperform their own empirical power curve
     (`curtailment_suspected`);
  3. writes the processed hourly table to data/processed/turbine_<id>_hourly.parquet;
  4. saves diagnostic plots to outputs/eda/plots/;
  5. writes a combined outputs/eda/eda_summary.md report.

Nothing here touches weather data or the forecasting model -- this is the
data-quality foundation step (see PLAN_AND_ARCHITECTURE.md, section 9, step 1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import io, resample, schema

PLOTS_DIR = schema.EDA_OUTPUT_DIR / "plots"


def profile_raw(turbine_id: int, df_raw: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    span_start, span_end = df_raw["timestamp"].min(), df_raw["timestamp"].max()
    expected_rows = int((span_end - span_start).total_seconds() / 60 / schema.RAW_FREQ_MINUTES) + 1
    return {
        "turbine_id": turbine_id,
        "n_rows": len(df_raw),
        "start": span_start,
        "end": span_end,
        "expected_rows_if_continuous": expected_rows,
        "missing_10min_samples": expected_rows - len(df_raw),
        "missing_pct": 100 * (expected_rows - len(df_raw)) / expected_rows,
        "n_gap_segments": len(gaps),
        "largest_gap_days": gaps["duration_days"].iloc[0] if len(gaps) else 0.0,
        "wind_speed_min": df_raw["wind_speed"].min(),
        "wind_speed_max": df_raw["wind_speed"].max(),
        "power_min": df_raw["power"].min(),
        "power_max": df_raw["power"].max(),
        "temperature_min": df_raw["temperature"].min(),
        "temperature_max": df_raw["temperature"].max(),
        "n_nan": int(df_raw.isna().sum().sum()),
        "n_duplicate_timestamps": int(df_raw["timestamp"].duplicated().sum()),
    }


def plot_monthly_overview(hourly: pd.DataFrame, turbine_id: int) -> None:
    monthly = (
        hourly.loc[~hourly["is_gap"]]
        .set_index("timestamp")[["wind_speed", "power", "temperature"]]
        .resample("MS")
        .mean()
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    monthly["wind_speed"].plot(ax=axes[0], color="tab:blue")
    axes[0].set_ylabel("Wind speed, m/s")
    monthly["power"].plot(ax=axes[1], color="tab:orange")
    axes[1].set_ylabel("Norm. power")
    monthly["temperature"].plot(ax=axes[2], color="tab:green")
    axes[2].set_ylabel("Temp, C")
    axes[0].set_title(f"Turbine {turbine_id}: monthly mean (gap hours excluded)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"turbine_{turbine_id}_monthly_overview.png", dpi=110)
    plt.close(fig)


def plot_power_curve(hourly: pd.DataFrame, turbine_id: int) -> None:
    valid = hourly.loc[~hourly["is_gap"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["wind_speed"], valid["power"], s=3, alpha=0.15, color="tab:blue", label="hourly obs")
    bin_width = 0.5
    wind_bin = (valid["wind_speed"] // bin_width) * bin_width
    curve = valid.groupby(wind_bin)["power"].median()
    ax.plot(curve.index, curve.values, color="black", linewidth=2, label="median power curve")
    suspected = valid.loc[valid["curtailment_suspected"]]
    ax.scatter(
        suspected["wind_speed"], suspected["power"], s=8, color="red", alpha=0.6, label="curtailment_suspected"
    )
    ax.set_xlabel("Wind speed, m/s")
    ax.set_ylabel("Normalized power")
    ax.set_title(f"Turbine {turbine_id}: empirical power curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"turbine_{turbine_id}_power_curve.png", dpi=110)
    plt.close(fig)


def plot_coverage_by_month(hourly: pd.DataFrame, turbine_id: int) -> None:
    coverage = hourly.set_index("timestamp")["is_gap"].resample("MS").apply(lambda s: 100 * (1 - s.mean()))
    labels = [d.strftime("%Y-%m") for d in coverage.index]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(range(len(coverage)), coverage.values, color="tab:purple")
    ax.set_xticks(range(len(coverage)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("% hours with valid data")
    ax.set_title(f"Turbine {turbine_id}: monthly hourly-data coverage")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"turbine_{turbine_id}_coverage_by_month.png", dpi=110)
    plt.close(fig)


def plot_diurnal_seasonal(hourly: pd.DataFrame, turbine_id: int) -> None:
    valid = hourly.loc[~hourly["is_gap"]].copy()
    valid["hour"] = valid["timestamp"].dt.hour
    valid["month"] = valid["timestamp"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    valid.groupby("hour")["power"].mean().plot(ax=axes[0], marker="o")
    axes[0].set_title(f"Turbine {turbine_id}: avg power by hour-of-day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Mean norm. power")

    valid.groupby("month")["power"].mean().plot(ax=axes[1], marker="o", color="tab:orange")
    axes[1].set_title(f"Turbine {turbine_id}: avg power by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean norm. power")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"turbine_{turbine_id}_diurnal_seasonal.png", dpi=110)
    plt.close(fig)


def plot_distributions(hourly: pd.DataFrame, turbine_id: int) -> None:
    valid = hourly.loc[~hourly["is_gap"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(valid["wind_speed"], bins=50, color="tab:blue")
    axes[0].set_title("Wind speed, m/s")
    axes[1].hist(valid["power"], bins=50, color="tab:orange")
    axes[1].set_title("Normalized power")
    axes[2].hist(valid["temperature"], bins=50, color="tab:green")
    axes[2].set_title("Temperature, C")
    fig.suptitle(f"Turbine {turbine_id}: hourly value distributions")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"turbine_{turbine_id}_distributions.png", dpi=110)
    plt.close(fig)


def write_summary_report(profiles: list[dict], hourly_frames: dict[int, pd.DataFrame], gaps_by_turbine: dict[int, pd.DataFrame]) -> None:
    lines = ["# EDA summary: HackAlem VES turbine dataset", ""]
    lines.append("Generated by `scripts/run_eda.py`. Source: `data/raw/*.csv` (10-minute logger data).")
    lines.append("")

    for p in profiles:
        tid = p["turbine_id"]
        hourly = hourly_frames[tid]
        n_hours = len(hourly)
        n_gap_hours = int(hourly["is_gap"].sum())
        n_curtailed = int(hourly["curtailment_suspected"].sum())
        corr = hourly.loc[~hourly["is_gap"], ["wind_speed", "power"]].corr().iloc[0, 1]

        lines += [
            f"## Turbine {tid}",
            "",
            f"- Raw rows: {p['n_rows']:,} | range: {p['start']} -> {p['end']}",
            f"- Raw resolution: 10 min | missing 10-min samples: {p['missing_10min_samples']:,} "
            f"({p['missing_pct']:.2f}% of the continuous-series expectation)",
            f"- Gap segments (non-10-min steps): {p['n_gap_segments']} | largest gap: {p['largest_gap_days']:.1f} days",
            f"- NaNs in raw data: {p['n_nan']} | duplicate timestamps: {p['n_duplicate_timestamps']}",
            f"- Value ranges: wind_speed [{p['wind_speed_min']:.2f}, {p['wind_speed_max']:.2f}] m/s, "
            f"power [{p['power_min']:.2f}, {p['power_max']:.2f}], "
            f"temperature [{p['temperature_min']:.2f}, {p['temperature_max']:.2f}] C",
            "",
            f"- After resampling to hourly: {n_hours:,} hourly bins, of which {n_gap_hours:,} "
            f"({100*n_gap_hours/n_hours:.2f}%) flagged `is_gap` (coverage < "
            f"{schema.HOURLY_COVERAGE_THRESHOLD:.0%} of the 6 expected 10-min samples).",
            f"- Curtailment/anomaly flag (`curtailment_suspected`): {n_curtailed:,} hours "
            f"({100*n_curtailed/n_hours:.2f}% of non-gap hours) where power is far below the turbine's own "
            "empirical power curve for that wind speed.",
            f"- Wind speed vs power correlation (hourly, non-gap): {corr:.3f}",
            "",
            "Top 5 raw data gaps:",
            "",
            "| start | end | duration (days) |",
            "|---|---|---|",
        ]
        top_gaps = gaps_by_turbine[tid].head(5)
        for _, row in top_gaps.iterrows():
            lines.append(f"| {row['gap_start']} | {row['gap_end']} | {row['duration_days']:.2f} |")
        lines.append("")
        lines.append(f"Plots: `outputs/eda/plots/turbine_{tid}_*.png`")
        lines.append("")

    report_path = schema.EDA_OUTPUT_DIR / "eda_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary report written to {report_path}")


def main() -> None:
    schema.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    profiles = []
    hourly_frames = {}
    gaps_by_turbine = {}

    for turbine_id in schema.RAW_FILES:
        print(f"\n=== Turbine {turbine_id} ===")
        df_raw = io.load_turbine_raw(turbine_id)
        gaps = resample.detect_raw_gaps(df_raw)
        gaps_by_turbine[turbine_id] = gaps

        profile = profile_raw(turbine_id, df_raw, gaps)
        profiles.append(profile)
        print(
            f"raw rows={profile['n_rows']:,} range=[{profile['start']} .. {profile['end']}] "
            f"missing_10min={profile['missing_10min_samples']:,} ({profile['missing_pct']:.2f}%) "
            f"gap_segments={profile['n_gap_segments']}"
        )

        hourly = resample.build_processed_hourly(df_raw)
        hourly_frames[turbine_id] = hourly
        n_gap_hours = int(hourly["is_gap"].sum())
        n_curtailed = int(hourly["curtailment_suspected"].sum())
        print(
            f"hourly bins={len(hourly):,} is_gap={n_gap_hours:,} "
            f"curtailment_suspected={n_curtailed:,}"
        )

        out_path = schema.PROCESSED_DATA_DIR / f"turbine_{turbine_id}_hourly.parquet"
        hourly.to_parquet(out_path, index=False)
        print(f"saved -> {out_path}")

        plot_monthly_overview(hourly, turbine_id)
        plot_power_curve(hourly, turbine_id)
        plot_coverage_by_month(hourly, turbine_id)
        plot_diurnal_seasonal(hourly, turbine_id)
        plot_distributions(hourly, turbine_id)
        print(f"plots -> {PLOTS_DIR}")

    write_summary_report(profiles, hourly_frames, gaps_by_turbine)


if __name__ == "__main__":
    main()
