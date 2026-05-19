# 준비된 CSV 입력값으로 Gemini GNIRS ITC를 일괄 실행한다.
# 입력값의 물리 가정과 실행 순서는 README.md에 정리되어 있다.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ITC_URL = "https://itc.gemini.edu/itc/servlets/web/ITCgnirs.html"
DEFAULT_INPUT_CSV = Path("itc-inputs.csv")
DEFAULT_OUTDIR = Path("gnirs_itc_outputs")
DEFAULT_SUMMARY_CSV = Path("gnirs_itc_batch_summary.csv")
SAMPLE_CHOICES = ("confirmed", "candidates", "all")

EXPOSURE_COUNTS = [30, 60, 90]
COADDS = 1
EXPOSURE_TIME_SEC = 120
FRACTION_ON_SOURCE = 1.0
REQUEST_SLEEP_SEC = 1.0
TIMEOUT_SEC = 90
DEFAULT_DITHER_SIZE_ARCSEC = 5.0

# Gemini ITC 입력 화면의 현재 cgs 단위 선택값.
ITC_CGS_LINE_FLUX_UNIT = "ergs_flux"
ITC_CGS_CONTINUUM_UNIT = "ergs_fd_wavelength"

# GNIRS long-slit J/H/K 분광 가능 범위 [micron].
GNIRS_SPECTROSCOPY_WINDOWS_UM = {
    "J": (1.17, 1.37),
    "H": (1.47, 1.80),
    "K": (1.91, 2.49),
}

# 고정 GNIRS 설정. 일부 선택값은 현재 입력 화면에서 다시 확인한다.
ITC_FIXED_VALUES = {
    "Instrument": "GNIRS",
    "Profile": "POINT",
    "Distribution": "ELINE",
    "Recession": "REDSHIFT",
    # CSV 파장은 이미 관측 파장이므로 ITC redshift는 0으로 둔다.
    "z": "0",
    "v": "0.0",
    "PixelScale": "PS_015",       # 0.15 arcsec/pix
    "SlitWidth": "SW_6",          # 0.675 arcsec slit. 가능하면 option text로 재확인한다.
    "Disperser": "D_32",          # 32 l/mm
    "CrossDispersed": "NO",
    "Filter": "spectroscopy",
    "ReadMode": "VERY_FAINT",     # Very Faint Objects 모드
    "WellDepth": "SHALLOW",
    "Coating": "SILVER",
    "IssPort": "SIDE_LOOKING",
    "Type": "PWFS",
    "FieldLens": "OUT",
    "GuideStarType": "NGS",
    "ImageQuality": "PERCENT_85", # IQ85 / Poor 조건
    "CloudCover": "PERCENT_70",   # CC70 / Cirrus 조건
    "WaterVapor": "ANY",
    "SkyBackground": "ANY",
    "Airmass": "1.5",
    "calcMethod": "s2n",
    "numCoaddsA": str(COADDS),
    "expTimeA": str(EXPOSURE_TIME_SEC),
    "fracOnSourceA": f"{FRACTION_ON_SOURCE:g}",
    "analysisMethod": "autoAper",
    # S/N ASCII 파일은 후처리에서 직접 읽는다.
    "PlotLimits": "AUTO",
    # AB magnitude 대신 Jy 값으로 넣어 등급 체계 혼동을 줄인다.
    "psSourceUnits": "JY",
}

REQUIRED_COLUMNS = [
    "name",
    "z",
    "RA J2000",
    "Dec J2000",
    "ITC point source spatially integrated brightness [AB mag]",
    "ITC point source brightness band",
    "ITC spectral distribution line wavelength [micron]",
    "ITC line flux [erg/s/cm^2]",
    "ITC line width FWHM [km/s]",
    "ITC continuum flux density [erg/s/cm^2/A]",
]

@dataclass
class Target:
    name: str
    z: float
    blazar_class: str
    ra: str
    dec: str
    brightness_ab_mag: float
    brightness_band: str
    line_wavelength_um: float
    line_flux_erg_s_cm2: float
    line_width_km_s: float
    continuum_erg_s_cm2_a: float
    notes: str = ""


def finite_float(value: object) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def abmag_to_jy(m_ab: float) -> float:
    # AB 정의: f_nu[Jy] = 3631 * 10^(-0.4 m_AB)
    return 3631.0 * 10.0 ** (-0.4 * m_ab)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_") or "target"


def norm_text(s: str) -> str:
    return (
        s.replace("²", "2")
        .replace("Å", "A")
        .replace("−", "-")
        .replace(" ", "")
        .lower()
    )


def get_select_options(form, name: str) -> list[dict[str, str]]:
    sel = form.find("select", attrs={"name": name})
    if sel is None:
        return []
    return [
        {"value": o.get("value", o.get_text(" ", strip=True)), "text": o.get_text(" ", strip=True)}
        for o in sel.find_all("option")
    ]


def select_option_value(form, name: str, include_all: list[str], fallback: str) -> str:
    # 보이는 이름 또는 HTML value에서 원하는 선택지를 찾는다.
    includes = [norm_text(x) for x in include_all]
    for opt in get_select_options(form, name):
        t = norm_text(f"{opt['text']} {opt['value']}")
        if all(x in t for x in includes):
            return opt["value"]
    return fallback


def select_option_by_text_or_value(form, name: str, wanted_text: str, fallback: str) -> str:
    wanted = norm_text(wanted_text)
    for opt in get_select_options(form, name):
        if wanted in norm_text(opt["text"]) or wanted == norm_text(opt["value"]):
            return opt["value"]
    return fallback


def wavelength_in_band(line_wavelength_um: float, band: str) -> bool:
    # line_wavelength_um이 특정 GNIRS spectroscopy window 안에 있는지 확인한다.
    window = GNIRS_SPECTROSCOPY_WINDOWS_UM.get(band)
    if window is None:
        return False
    lo, hi = window
    return lo <= line_wavelength_um <= hi


def spectroscopy_window_band(line_wavelength_um: float) -> str | None:
    # line_wavelength_um이 들어가는 GNIRS J/H/K spectroscopy window를 찾는다.
    for band in GNIRS_SPECTROSCOPY_WINDOWS_UM:
        if wavelength_in_band(line_wavelength_um, band):
            return band
    return None


def include_row_for_sample(row: pd.Series, sample: str) -> bool:
    if "blazar_class" not in row.index or sample == "all":
        return True
    blazar_class = str(row.get("blazar_class", "")).strip().lower()
    if sample == "confirmed":
        return blazar_class == "y"
    if sample == "candidates":
        return blazar_class == "c"
    raise ValueError(f"Unknown sample selection: {sample}")


def load_targets(path: Path, sample: str = "confirmed") -> list[Target]:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 필요한 열이 없습니다: {missing}")

    targets: list[Target] = []
    skip_reasons: dict[str, int] = {}
    for _, row in df.iterrows():
        if not include_row_for_sample(row, sample):
            skip_reasons[f"outside {sample} sample"] = skip_reasons.get(f"outside {sample} sample", 0) + 1
            continue
        vals = {
            "brightness": finite_float(row["ITC point source spatially integrated brightness [AB mag]"]),
            "line_wave": finite_float(row["ITC spectral distribution line wavelength [micron]"]),
            "line_flux": finite_float(row["ITC line flux [erg/s/cm^2]"]),
            "line_width": finite_float(row["ITC line width FWHM [km/s]"]),
            "continuum": finite_float(row["ITC continuum flux density [erg/s/cm^2/A]"]),
        }
        if any(v is None for v in vals.values()):
            skip_reasons["missing numeric ITC input"] = skip_reasons.get("missing numeric ITC input", 0) + 1
            continue
        band = str(row["ITC point source brightness band"]).strip().upper()
        if band not in {"J", "H", "K"}:
            skip_reasons["invalid brightness band"] = skip_reasons.get("invalid brightness band", 0) + 1
            continue
        line_wave = float(vals["line_wave"])
        if spectroscopy_window_band(line_wave) is None:
            skip_reasons["line wavelength outside GNIRS spectroscopy windows"] = (
                skip_reasons.get("line wavelength outside GNIRS spectroscopy windows", 0) + 1
            )
            continue
        targets.append(
            Target(
                name=str(row["name"]).strip(),
                z=float(row["z"]),
                blazar_class=str(row.get("blazar_class", "")).strip(),
                ra=str(row["RA J2000"]).strip(),
                dec=str(row["Dec J2000"]).strip(),
                brightness_ab_mag=float(vals["brightness"]),
                brightness_band=band,
                line_wavelength_um=line_wave,
                line_flux_erg_s_cm2=float(vals["line_flux"]),
                line_width_km_s=float(vals["line_width"]),
                continuum_erg_s_cm2_a=float(vals["continuum"]),
                notes=str(row.get("notes/cautions", "")),
            )
        )
    for reason, count in sorted(skip_reasons.items()):
        print(f"[skip input] {count} rows: {reason}")
    return targets


def extract_form(session: requests.Session):
    r = session.get(ITC_URL, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    form = soup.find("form")
    if form is None:
        raise RuntimeError("GNIRS ITC 페이지에서 form을 찾지 못했습니다.")
    submit_url = urljoin(ITC_URL, form.get("action") or ITC_URL)
    method = (form.get("method") or "get").lower()
    return submit_url, method, form


def initial_payload(form) -> dict[str, str]:
    payload: dict[str, str] = {}

    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        typ = (inp.get("type") or "text").lower()
        value = inp.get("value", "")
        if typ in {"radio", "checkbox"}:
            if inp.has_attr("checked"):
                payload[name] = value
        elif typ not in {"submit", "button", "image", "reset", "file"}:
            payload.setdefault(name, value)

    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.find("option", selected=True) or sel.find("option")
        if opt is not None:
            payload[name] = opt.get("value", opt.get_text(strip=True))

    for textarea in form.find_all("textarea"):
        name = textarea.get("name")
        if name:
            payload[name] = textarea.get_text()
    return payload


def dump_form_fields(form, path: Path) -> None:
    rows = []
    for tag in form.find_all(["input", "select", "textarea"]):
        row = {
            "tag": tag.name,
            "type": tag.get("type", "") if tag.name == "input" else tag.name,
            "name": tag.get("name", ""),
            "value": tag.get("value", ""),
            "text": tag.get_text(" ", strip=True),
        }
        if tag.name == "select":
            row["options"] = get_select_options(form, tag.get("name", ""))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def dynamic_select_overrides(form) -> dict[str, str]:
    # Gemini ITC 입력 화면에서 단위/설정 선택값을 찾는다.
    return {
        # 선 세기와 연속광은 CSV의 cgs 단위를 유지한다.
        "lineFluxUnits": select_option_value(form, "lineFluxUnits", ["erg", "flux"], ITC_CGS_LINE_FLUX_UNIT),
        "lineContinuumUnits": select_option_value(form, "lineContinuumUnits", ["erg", "fd", "wavelength"], ITC_CGS_CONTINUUM_UNIT),
        "PlotLimits": select_option_by_text_or_value(form, "PlotLimits", "Autoscale", "AUTO"),
        "SlitWidth": select_option_by_text_or_value(form, "SlitWidth", "0.675", "SW_6"),
        "ReadMode": select_option_by_text_or_value(form, "ReadMode", "Very Faint", "VERY_FAINT"),
    }


def configure_payload(target: Target, n_exp: int, base: dict[str, str], select_overrides: dict[str, str], dither_size: float) -> dict[str, str]:
    p = dict(base)
    p.update(ITC_FIXED_VALUES)
    p.update(select_overrides)

    # 점광원 밝기는 Jy 단위로 넣는다.
    p["psSourceNorm"] = f"{abmag_to_jy(target.brightness_ab_mag):.8e}"
    p["WavebandDefinition"] = target.brightness_band

    # Mg II 선과 연속광 입력값은 앞 단계 CSV의 값을 그대로 사용한다.
    p["lineWavelength"] = f"{target.line_wavelength_um:.7f}"
    p["instrumentCentralWavelength"] = f"{target.line_wavelength_um:.7f}"
    p["lineFlux"] = f"{target.line_flux_erg_s_cm2:.8e}"
    p["lineWidth"] = f"{target.line_width_km_s:.3f}"
    p["lineContinuum"] = f"{target.continuum_erg_s_cm2_a:.8e}"

    p["numExpA"] = str(int(n_exp))
    p["numCoaddsA"] = str(int(COADDS))
    p["expTimeA"] = str(int(EXPOSURE_TIME_SEC))
    p["fracOnSourceA"] = f"{FRACTION_ON_SOURCE:g}"

    # slit spectroscopy의 위치 이동 관측을 반영한다.
    p["offset"] = f"{dither_size:g}"

    # Autoscale이어도 Gemini server가 plot range 숫자 필드를 요구한다.
    p["PlotLimits"] = select_overrides.get("PlotLimits", "AUTO")
    width_um = 0.02
    p["plotWavelengthL"] = f"{max(0.80, target.line_wavelength_um - width_um):.6f}"
    p["plotWavelengthU"] = f"{min(2.50, target.line_wavelength_um + width_um):.6f}"

    return p


def validate_payload(payload: dict[str, str], target: Target, n_exp: int, dither_size: float) -> list[str]:
    warnings: list[str] = []
    expected = {
        "lineWavelength": f"{target.line_wavelength_um:.7f}",
        "instrumentCentralWavelength": f"{target.line_wavelength_um:.7f}",
        "lineFlux": f"{target.line_flux_erg_s_cm2:.8e}",
        "lineContinuum": f"{target.continuum_erg_s_cm2_a:.8e}",
        "numExpA": str(int(n_exp)),
        "numCoaddsA": str(int(COADDS)),
        "expTimeA": str(int(EXPOSURE_TIME_SEC)),
        "ImageQuality": "PERCENT_85",
        "CloudCover": "PERCENT_70",
        "WaterVapor": "ANY",
        "SkyBackground": "ANY",
        "Airmass": "1.5",
        "Filter": "spectroscopy",
        "CrossDispersed": "NO",
        "offset": f"{dither_size:g}",
        "lineFluxUnits": ITC_CGS_LINE_FLUX_UNIT,
        "lineContinuumUnits": ITC_CGS_CONTINUUM_UNIT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            warnings.append(f"{key}={payload.get(key)!r}, expected {value!r}")

    if payload.get("PlotLimits") not in {"AUTO", "Autoscale", "auto", "autoscale"}:
        warnings.append(f"PlotLimits={payload.get('PlotLimits')!r}; expected Autoscale/AUTO")

    line_window = spectroscopy_window_band(target.line_wavelength_um)
    if line_window is None:
        warnings.append(
            f"lineWavelength={target.line_wavelength_um:.7f} micron is outside "
            "GNIRS J/H/K spectroscopy windows"
        )
    elif line_window != target.brightness_band:
        warnings.append(
            f"brightness normalization uses {target.brightness_band}-band while "
            f"Mg II falls in the GNIRS {line_window}-band spectroscopy window"
        )

    for key in ["lineWavelength", "lineFlux", "lineWidth", "lineContinuum"]:
        try:
            x = float(payload[key])
        except Exception:
            warnings.append(f"{key} is not numeric: {payload.get(key)!r}")
            continue
        if not math.isfinite(x) or x <= 0:
            warnings.append(f"{key} is non-positive: {payload.get(key)!r}")

    return warnings


def looks_like_numeric_ascii(text: str, min_rows: int = 5) -> bool:
    # 다운로드한 텍스트가 ITC numeric ASCII처럼 보이면 True를 반환한다.
    numeric_rows = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            [float(x) for x in parts[:2]]
        except Exception:
            continue
        numeric_rows += 1
        if numeric_rows >= min_rows:
            return True
    return False


def save_itc_ascii_artifacts(session: requests.Session, html: str, base_url: str, run_dir: Path, save_html_tables: bool = False) -> dict[str, str]:
    # Gemini ITC 결과 페이지의 spectroscopy ASCII 네 종류를 저장한다.
    saved: dict[str, str] = {}
    soup = BeautifulSoup(html, "lxml")

    # server 쪽 diagnostic <pre> block이 실제로 있을 때만 보존한다.
    for i, pre in enumerate(soup.find_all("pre"), 1):
        text = pre.get_text("\n")
        if text.strip():
            path = run_dir / f"pre_{i:02d}.txt"
            path.write_text(text, encoding="utf-8")
            saved[f"pre_{i:02d}"] = str(path)

    if save_html_tables:
        for i, table in enumerate(soup.find_all("table"), 1):
            text = table.get_text("\t", strip=True)
            if text:
                path = run_dir / f"table_{i:02d}.txt"
                path.write_text(text, encoding="utf-8")
                saved[f"table_{i:02d}"] = str(path)

    expected = {
        "SignalData": "signal_spectrum_ascii.txt",
        "BackgroundData": "background_spectrum_ascii.txt",
        "SingleS2NData": "single_exposure_s2n_ascii.txt",
        "FinalS2NData": "final_s2n_ascii.txt",
    }

    from urllib.parse import parse_qs

    for a in soup.find_all("a", href=True):
        href = a["href"]
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if qs.get("type", [""])[0] != "txt":
            continue
        filename_key = qs.get("filename", [""])[0]
        if filename_key not in expected:
            continue

        out_name = expected[filename_key]
        try:
            r = session.get(url, timeout=TIMEOUT_SEC)
            r.raise_for_status()
        except Exception as exc:
            saved[f"{filename_key}_download_error"] = repr(exc)
            continue

        # 숫자 파일처럼 보이면 정식 산출물로 저장하고, 아니면 debug file로 둔다.
        text = r.text
        path = run_dir / out_name
        if looks_like_numeric_ascii(text):
            path.write_text(text, encoding="utf-8")
            saved[filename_key] = str(path)
        else:
            debug_path = run_dir / f"debug_unexpected_{filename_key}.txt"
            debug_path.write_text(text, encoding="utf-8")
            saved[f"{filename_key}_unexpected_content"] = str(debug_path)

    missing = [k for k in expected if k not in saved]
    if missing:
        saved["missing_expected_ascii"] = ",".join(missing)
    return saved

def extract_possible_sn(html: str) -> str:
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    for pat in [
        r"Total\s*S/?N\s*(?:ratio)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"S/?N\s*(?:ratio)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"Signal\s*to\s*Noise\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1)
    return ""


def submit_one(session: requests.Session, target: Target, n_exp: int, outdir: Path, dump_form: bool, dither_size: float, save_html_tables: bool) -> dict[str, str]:
    submit_url, method, form = extract_form(session)
    if dump_form:
        dump_form_fields(form, outdir / "itc_debug" / "form_fields.json")

    select_overrides = dynamic_select_overrides(form)
    payload = configure_payload(target, n_exp, initial_payload(form), select_overrides, dither_size)
    validation_warnings = validate_payload(payload, target, n_exp, dither_size)

    run_dir = outdir / safe_name(target.name) / f"nexp_{int(n_exp):03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if method == "post":
        response = session.post(submit_url, data=payload, timeout=TIMEOUT_SEC)
    else:
        response = session.get(submit_url, params=payload, timeout=TIMEOUT_SEC)
    response.raise_for_status()

    html_path = run_dir / "response.html"
    html_path.write_text(response.text, encoding="utf-8")
    saved_ascii = save_itc_ascii_artifacts(session, response.text, response.url, run_dir, save_html_tables=save_html_tables)

    err = ""
    err_text = " ".join(Path(p).read_text(errors="ignore")[:1000] for k, p in saved_ascii.items() if k.startswith("pre_") and Path(p).exists())
    if "not a valid" in err_text.lower() or "exception" in err_text.lower() or "error" in err_text.lower():
        err = err_text.strip()

    return {
        "name": target.name,
        "z": str(target.z),
        "blazar_class": target.blazar_class,
        "RA J2000": target.ra,
        "Dec J2000": target.dec,
        "n_exposures": str(int(n_exp)),
        "numCoaddsA": str(COADDS),
        "expTimeA_sec": str(EXPOSURE_TIME_SEC),
        "fracOnSourceA": f"{FRACTION_ON_SOURCE:g}",
        "dither_size_arcsec_ABBA": f"{dither_size:g}",
        "brightness_band": target.brightness_band,
        "input_brightness_AB_mag": f"{target.brightness_ab_mag:.6g}",
        "payload_psSourceNorm_Jy": payload["psSourceNorm"],
        "input_line_wavelength_micron": f"{target.line_wavelength_um:.7f}",
        "payload_lineFlux_erg_s_cm2": payload["lineFlux"],
        "payload_lineFluxUnits": payload.get("lineFluxUnits", ""),
        "payload_lineContinuum_erg_s_cm2_A": payload["lineContinuum"],
        "payload_lineContinuumUnits": payload.get("lineContinuumUnits", ""),
        "line_width_km_s": f"{target.line_width_km_s:.3f}",
        "payload_PlotLimits": payload.get("PlotLimits", ""),
        "possible_total_SN_from_html": extract_possible_sn(response.text),
        "payload_json": str(run_dir / "payload.json"),
        "response_html": str(html_path),
        "signal_spectrum_ascii": saved_ascii.get("SignalData", ""),
        "background_spectrum_ascii": saved_ascii.get("BackgroundData", ""),
        "single_exposure_s2n_ascii": saved_ascii.get("SingleS2NData", ""),
        "final_s2n_ascii": saved_ascii.get("FinalS2NData", ""),
        "missing_expected_ascii": saved_ascii.get("missing_expected_ascii", ""),
        "debug_or_optional_files": ";".join(v for k, v in saved_ascii.items() if k not in {"SignalData", "BackgroundData", "SingleS2NData", "FinalS2NData", "missing_expected_ascii"}),
        "validation_warnings": "; ".join(validation_warnings),
        "server_error_text": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-submit Gemini GNIRS ITC calculations.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--dump-form", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", choices=SAMPLE_CHOICES, default="confirmed")
    parser.add_argument("--dither-size", type=float, default=DEFAULT_DITHER_SIZE_ARCSEC)
    parser.add_argument("--save-html-tables", action="store_true", help="Save HTML summary tables as debug files; off by default because these are not ASCII spectra.")
    args = parser.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    summary_csv = Path(args.summary)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(input_csv, sample=args.sample)
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("계산 가능한 target이 없습니다. 입력 CSV의 ITC 값 열을 확인하세요.")

    session = requests.Session()
    session.headers.update({"User-Agent": "BLAZ4R-GNIRS-ITC-batch/0.5"})

    rows: list[dict[str, str]] = []
    for target in targets:
        for n_exp in EXPOSURE_COUNTS:
            print(f"Submitting {target.name} / {n_exp} exposures")
            try:
                rows.append(submit_one(session, target, n_exp, outdir, args.dump_form, args.dither_size, args.save_html_tables))
            except Exception as exc:
                rows.append({
                    "name": target.name,
                    "z": str(target.z),
                    "RA J2000": target.ra,
                    "Dec J2000": target.dec,
                    "n_exposures": str(int(n_exp)),
                    "error": repr(exc),
                })
            time.sleep(REQUEST_SLEEP_SEC)

    fieldnames = sorted({k for r in rows for k in r})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {summary_csv}")
    print(f"Wrote outputs under {outdir}/")


if __name__ == "__main__":
    main()
