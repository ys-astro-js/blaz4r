#!/usr/bin/env python3
# Gemini GNIRS ITC 최종 S/N 곡선을 그린다.
# S/N 중앙값 정의는 README.md에 정리되어 있다.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

C_KM_S = 299_792.458
EXPECTED_EXPOSURES = (30, 60, 90)

# 사용자가 요청한 명시적 색상. 30/60/90 exposure 곡선이 구분되어야 한다.
CURVE_STYLES = {
    30: {"color": "tab:blue", "linestyle": "-", "label": "30 exp"},
    60: {"color": "tab:orange", "linestyle": "-", "label": "60 exp"},
    90: {"color": "tab:green", "linestyle": "-", "label": "90 exp"},
}


@dataclass(frozen=True)
class SnrRun:
    target: str
    nexp: int
    run_dir: Path
    wavelength_nm: np.ndarray
    snr: np.ndarray
    line_center_nm: float
    fwhm_km_s: float

    @property
    def fwhm_width_nm(self) -> float:
        # 속도폭을 파장폭으로 변환한다.
        return self.line_center_nm * self.fwhm_km_s / C_KM_S

    @property
    def fwhm_lower_nm(self) -> float:
        return self.line_center_nm - 0.5 * self.fwhm_width_nm

    @property
    def fwhm_upper_nm(self) -> float:
        return self.line_center_nm + 0.5 * self.fwhm_width_nm

    def snr_at_line_center(self) -> float:
        # 선 중심에 가장 가까운 파장 격자의 S/N을 쓴다.
        idx = int(np.nanargmin(np.abs(self.wavelength_nm - self.line_center_nm)))
        return float(self.snr[idx])

    def median_snr_within_fwhm(self) -> float:
        # Mg II FWHM 구간 안의 S/N 중앙값을 계산한다.
        mask = (self.wavelength_nm >= self.fwhm_lower_nm) & (self.wavelength_nm <= self.fwhm_upper_nm)
        if not np.any(mask):
            return float("nan")
        return float(np.nanmedian(self.snr[mask]))

    def min_snr_within_fwhm(self) -> float:
        # 보조 진단용 최소 S/N.
        mask = (self.wavelength_nm >= self.fwhm_lower_nm) & (self.wavelength_nm <= self.fwhm_upper_nm)
        if not np.any(mask):
            return float("nan")
        return float(np.nanmin(self.snr[mask]))


def read_ascii_two_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # ITC ASCII의 wavelength와 S/N 두 열을 읽는다.
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        raise ValueError(f"Not enough numeric rows in {path}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def read_payload(path: Path) -> tuple[float, float, int | None]:
    # payload에서 Mg II 중심 파장[nm], FWHM[km/s], 노출 횟수를 읽는다.
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    line_wavelength_um = float(payload["lineWavelength"])
    line_center_nm = line_wavelength_um * 1000.0
    fwhm_km_s = float(payload["lineWidth"])

    nexp = None
    if "numExpA" in payload:
        try:
            nexp = int(float(payload["numExpA"]))
        except ValueError:
            nexp = None
    return line_center_nm, fwhm_km_s, nexp


def nexp_from_dir_name(path: Path) -> int | None:
    m = re.search(r"nexp[_-]?(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1))


def discover_runs(root: Path, wanted_exposures: Iterable[int]) -> dict[str, list[SnrRun]]:
    wanted = set(wanted_exposures)
    by_target: dict[str, list[SnrRun]] = {}

    if not root.exists():
        raise FileNotFoundError(f"ITC output root does not exist: {root}")

    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # 이전 ITC script가 만든 debug folder는 대상으로 보지 않는다.
        if target_dir.name.startswith("itc_debug"):
            continue
        target = target_dir.name

        for run_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            nexp_guess = nexp_from_dir_name(run_dir)
            if nexp_guess is not None and nexp_guess not in wanted:
                continue

            ascii_path = run_dir / "final_s2n_ascii.txt"
            payload_path = run_dir / "payload.json"
            if not ascii_path.exists() or not payload_path.exists():
                continue

            line_center_nm, fwhm_km_s, nexp_payload = read_payload(payload_path)
            nexp = nexp_payload if nexp_payload is not None else nexp_guess
            if nexp is None or nexp not in wanted:
                continue

            wavelength_nm, snr = read_ascii_two_columns(ascii_path)
            run = SnrRun(
                target=target,
                nexp=nexp,
                run_dir=run_dir,
                wavelength_nm=wavelength_nm,
                snr=snr,
                line_center_nm=line_center_nm,
                fwhm_km_s=fwhm_km_s,
            )
            by_target.setdefault(target, []).append(run)

    for target, runs in by_target.items():
        runs.sort(key=lambda r: r.nexp)
    return by_target


def finite_max(values: Iterable[float], default: float = 1.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.nanmax(arr))


def plot_target(target: str, runs: list[SnrRun], output_dir: Path, window_factor: float) -> list[dict[str, str | float | int]]:
    if not runs:
        return []

    # 같은 대상의 노출별 run은 동일한 Mg II 설정을 쓴다.
    line_center = runs[0].line_center_nm
    fwhm_lower = runs[0].fwhm_lower_nm
    fwhm_upper = runs[0].fwhm_upper_nm
    fwhm_width = runs[0].fwhm_width_nm

    x_min = line_center - window_factor * fwhm_width
    x_max = line_center + window_factor * fwhm_width

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)

    # Mg II FWHM 구간을 먼저 그려서 S/N 곡선이 위에 올라오도록 한다.
    ax.axvspan(fwhm_lower, fwhm_upper, color="0.85", alpha=0.55, label="Mg II FWHM range")
    ax.axvline(line_center, color="black", linewidth=1.2, linestyle="--", label="Mg II line center")

    summary_rows: list[dict[str, str | float | int]] = []
    ymax_candidates: list[float] = []

    for run in runs:
        style = CURVE_STYLES.get(run.nexp, {"color": None, "linestyle": "-", "label": f"{run.nexp} exp"})
        mask_plot = (run.wavelength_nm >= x_min) & (run.wavelength_nm <= x_max)
        if not np.any(mask_plot):
            mask_plot = np.ones_like(run.wavelength_nm, dtype=bool)

        ax.plot(
            run.wavelength_nm[mask_plot],
            run.snr[mask_plot],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            label=style["label"],
        )

        med = run.median_snr_within_fwhm()
        min_snr = run.min_snr_within_fwhm()
        center_snr = run.snr_at_line_center()
        ymax_candidates.extend([float(np.nanmax(run.snr[mask_plot])), med])

        # FWHM 내부 S/N 중앙값을 같은 색의 수평 점선으로 표시한다.
        if math.isfinite(med):
            ax.hlines(
                med,
                fwhm_lower,
                fwhm_upper,
                colors=style["color"],
                linestyles=":",
                linewidth=2.2,
            )

        summary_rows.append(
            {
                "target": target,
                "nexp": run.nexp,
                "line_center_nm": line_center,
                "fwhm_km_s": run.fwhm_km_s,
                "fwhm_lower_nm": run.fwhm_lower_nm,
                "fwhm_upper_nm": run.fwhm_upper_nm,
                "snr_at_line_center": center_snr,
                "median_snr_within_fwhm": med,
                "min_snr_within_fwhm": min_snr,
                "final_s2n_ascii": str(run.run_dir / "final_s2n_ascii.txt"),
            }
        )

    ymax = finite_max(ymax_candidates, default=1.0)
    ax.set_ylim(bottom=0, top=max(1.0, ymax * 1.15))
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("S/N")
    ax.set_title(f"{target}: Gemini/GNIRS ITC Final S/N near Mg II")

    # 노출별 FWHM 내부 S/N 중앙값을 그림 안쪽 text box에 적는다.
    med_lines = []
    for row in summary_rows:
        med = row["median_snr_within_fwhm"]
        if isinstance(med, float) and math.isfinite(med):
            med_lines.append(f"{row['nexp']} exp median S/N = {med:.2f}")
        else:
            med_lines.append(f"{row['nexp']} exp median S/N = n/a")
    text = "\n".join(med_lines)
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
    )

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_target = re.sub(r"[^A-Za-z0-9_.+-]+", "_", target)
    png_path = output_dir / f"{safe_target}_snr_mgii.png"
    pdf_path = output_dir / f"{safe_target}_snr_mgii.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    for row in summary_rows:
        row["plot_png"] = str(png_path)
        row["plot_pdf"] = str(pdf_path)
    return summary_rows


def write_summary(rows: list[dict[str, str | float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "nexp",
        "line_center_nm",
        "fwhm_km_s",
        "fwhm_lower_nm",
        "fwhm_upper_nm",
        "snr_at_line_center",
        "median_snr_within_fwhm",
        "min_snr_within_fwhm",
        "final_s2n_ascii",
        "plot_png",
        "plot_pdf",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Gemini GNIRS ITC S/N curves around Mg II.")
    parser.add_argument("--root", default="gnirs_itc_outputs", help="Root folder created by run_gnirs_itc_batch_v5.py")
    parser.add_argument("--outdir", default="gnirs_itc_snr_plots", help="Output folder for PNG/PDF plots")
    parser.add_argument("--summary", default="gnirs_itc_snr_plots/summary_snr_within_fwhm.csv", help="Summary CSV path")
    parser.add_argument("--target", default=None, help="Only plot one target folder name")
    parser.add_argument(
        "--window-factor",
        type=float,
        default=3.0,
        help="Plot half-width in units of Mg II FWHM width. Default: ±3 FWHM widths around line center.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    summary_path = Path(args.summary)

    runs_by_target = discover_runs(root, EXPECTED_EXPOSURES)
    if args.target:
        runs_by_target = {args.target: runs_by_target.get(args.target, [])}

    all_rows: list[dict[str, str | float | int]] = []
    for target, runs in sorted(runs_by_target.items()):
        if not runs:
            print(f"[skip] {target}: no final_s2n_ascii.txt + payload.json for 30/60/90")
            continue
        available = ", ".join(str(r.nexp) for r in runs)
        missing = sorted(set(EXPECTED_EXPOSURES) - {r.nexp for r in runs})
        if missing:
            print(f"[warn] {target}: missing nexp {missing}; plotting available: {available}")
        else:
            print(f"[plot] {target}: nexp {available}")
        rows = plot_target(target, runs, outdir, args.window_factor)
        all_rows.extend(rows)

    write_summary(all_rows, summary_path)
    print(f"[done] wrote {len(all_rows)} summary rows to {summary_path}")
    print(f"[done] plots written to {outdir}")


if __name__ == "__main__":
    main()
