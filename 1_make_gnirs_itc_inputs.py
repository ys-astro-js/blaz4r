# Gemini GNIRS Mg II ITC 입력 CSV를 만든다.
# 계산 가정과 근거는 README.md에 정리되어 있다.

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.ned import Ned
from astroquery.ukidss import Ukidss

C_CGS = 2.99792458e10  # 빛의 속도 [cm/s]

# Mg II 대표 정지 파장 [micron].
MGII_REST_UM = 0.279875

# UKIDSS/WFCAM Vega magnitude를 AB magnitude로 바꾸는 보정값.
UKIDSS_AB_OFFSETS = {
    "J": 0.938,
    "H": 1.379,
    "K": 1.900,
}

# UKIDSS/WFCAM 필터 대표 파장 [micron].
BAND_LAMBDA_UM = {
    "J": 1.248,
    "H": 1.631,
    "K": 2.201,
}

# GNIRS long-slit J/H/K 분광 가능 범위 [micron].
GNIRS_SPECTROSCOPY_WINDOWS_UM = {
    "J": (1.17, 1.37),
    "H": (1.47, 1.80),
    "K": (1.91, 2.49),
}

# 실제 분광 측정값이 없을 때 쓰는 ITC용 기본 가정값.
DEFAULT_FWHM_KMS = 4000.0
DEFAULT_EW_REST_A = 30.0
DEFAULT_CONTINUUM_ALPHA_NU = -0.44
SAMPLE_CHOICES = ("confirmed", "candidates", "all")


def is_good_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v) and v > -90000000


def mag_ab_to_fnu_jy(m_ab: float) -> float:
    # AB magnitude를 f_nu[Jy]로 변환한다.
    return 3631.0 * 10 ** (-0.4 * m_ab)


def fnu_jy_to_flambda_cgs_per_a(fnu_jy: float, lambda_um: float) -> float:
    # f_nu[Jy]를 f_lambda[erg/s/cm^2/A]로 변환한다.
    lam_cm = lambda_um * 1e-4
    return fnu_jy * 1e-23 * C_CGS / (lam_cm * lam_cm) * 1e-8


def choose_itc_band(lambda_obs_um: float) -> str | None:
    # Mg II가 들어가는 GNIRS 분광 window를 찾는다.
    for band, (lo, hi) in GNIRS_SPECTROSCOPY_WINDOWS_UM.items():
        if lo <= lambda_obs_um <= hi:
            return band
    return None


def photometric_band_candidates(lambda_obs_um: float) -> list[str]:
    # Mg II 관측 파장에 가까운 측광 band부터 사용한다.
    return sorted(BAND_LAMBDA_UM, key=lambda band: abs(math.log(BAND_LAMBDA_UM[band] / lambda_obs_um)))


def choose_continuum_band(lambda_obs_um: float) -> str:
    return photometric_band_candidates(lambda_obs_um)[0]


def extrapolate_flambda_power_law(
    flambda_ref: float,
    lambda_ref_um: float,
    lambda_target_um: float,
    alpha_nu: float = DEFAULT_CONTINUUM_ALPHA_NU,
) -> float:
    # 필터 중심의 연속광을 Mg II 관측 파장으로 보정한다.
    alpha_lambda = -(alpha_nu + 2.0)
    return flambda_ref * (lambda_target_um / lambda_ref_um) ** alpha_lambda


def include_target_for_sample(row: dict[str, str], sample: str) -> bool:
    blazar_class = row.get("blazar_class", "").strip().lower()
    if sample == "confirmed":
        return blazar_class == "y"
    if sample == "candidates":
        return blazar_class == "c"
    if sample == "all":
        return True
    raise ValueError(f"Unknown sample selection: {sample}")


def get_col(row: Any, names: list[str]) -> Any | None:
    for name in names:
        if name in row.colnames:
            return row[name]
    return None


def best_ukidss_row(table: Table) -> Any | None:
    if table is None or len(table) == 0:
        return None
    if "distance" in table.colnames:
        return table[table["distance"].argmin()]
    return table[0]


def query_ukidss_las(coord: SkyCoord, radius_arcsec: float = 2.0) -> Any | None:
    table = Ukidss.query_region(
        coord,
        radius=radius_arcsec * u.arcsec,
        programme_id="LAS",
        database="UKIDSSDR9PLUS",
    )
    return best_ukidss_row(table)


def get_ukidss_mag(row: Any, band: str) -> tuple[float | None, float | None, str]:
    # 점광원용 aperture magnitude를 우선 사용한다.
    mag_names = [f"{band}AperMag3", f"{band.lower()}AperMag3", f"{band}apermag3"]
    err_names = [f"{band}AperMag3Err", f"{band.lower()}AperMag3Err", f"{band}apermag3err"]
    mag = get_col(row, mag_names)
    err = get_col(row, err_names)
    if mag is None or not is_good_number(mag):
        return None, None, f"UKIDSS {band}AperMag3 없음"
    err_value = float(err) if err is not None and is_good_number(err) else None
    return float(mag), err_value, "UKIDSS LAS AperMag3"


def find_ned_object_name(coord: SkyCoord, radius_arcsec: float = 3.0) -> str | None:
    tbl = Ned.query_region(coord, radius=radius_arcsec * u.arcsec)
    if tbl is None or len(tbl) == 0:
        return None
    # 가장 가까운 NED object를 고른다.
    sep_cols = [c for c in tbl.colnames if "sep" in c.lower() or "distance" in c.lower()]
    if sep_cols:
        row = tbl[tbl[sep_cols[0]].argmin()]
    else:
        row = tbl[0]
    for key in ["Object Name", "Object_Name", "ObjectName"]:
        if key in row.colnames:
            return str(row[key])
    return None


def read_ned_photometry_near_ir(ned_name: str, wanted_band: str) -> dict[str, Any] | None:
    # NED photometry는 자동 판별 실패 가능성이 높아 보조 메모로만 쓴다.
    try:
        phot = Ned.get_table(ned_name, table="photometry")
    except Exception:
        return None
    if phot is None or len(phot) == 0:
        return None

    band_patterns = {
        "J": ["2MASS J", "UKIDSS J", "J ", " J"],
        "H": ["2MASS H", "UKIDSS H", "H ", " H"],
        "K": ["2MASS K", "UKIDSS K", "Ks", "K_s", "K ", " K"],
    }

    text_cols = [c for c in phot.colnames if phot[c].dtype.kind in "OUS"]
    for row in phot:
        joined = " ".join(str(row[c]) for c in text_cols if row[c] is not None)
        if not any(pat.lower() in joined.lower() for pat in band_patterns[wanted_band]):
            continue
        return {"ned_name": ned_name, "raw": joined[:300]}
    return None


@dataclass
class ItcRow:
    name: str
    z: float
    blazar_class: str
    ra: str
    dec: str
    mgii_wavelength_um: float | None = None
    itc_brightness_band: str = ""
    point_source_brightness_ab_mag: float | None = None
    catalog_mag_vega: float | None = None
    catalog_mag_err: float | None = None
    catalog_source: str = ""
    continuum_flux_density_erg_s_cm2_a: float | None = None
    continuum_reference_wavelength_um: float | None = None
    continuum_target_wavelength_um: float | None = None
    continuum_alpha_nu_assumed: float | None = None
    continuum_extrapolation_factor: float | None = None
    line_flux_erg_s_cm2: float | None = None
    line_width_km_s: float | None = None
    ew_rest_a_assumed: float | None = None
    gnirs_spectroscopy_window: str = ""
    note: str = ""


def process_target(row: dict[str, str], *, use_ned_fallback: bool = False) -> ItcRow:
    name = row["name"]
    z = float(row["z"])
    coord = SkyCoord(row["ra"], row["dec"], unit=(u.hourangle, u.deg), frame="icrs")
    lambda_obs_um = MGII_REST_UM * (1.0 + z)
    spectroscopy_window = choose_itc_band(lambda_obs_um)
    band_candidates = photometric_band_candidates(lambda_obs_um)
    preferred_band = band_candidates[0]

    out = ItcRow(
        name=name,
        z=z,
        blazar_class=row.get("blazar_class", "").strip(),
        ra=row["ra"],
        dec=row["dec"],
        mgii_wavelength_um=lambda_obs_um,
    )
    out.gnirs_spectroscopy_window = spectroscopy_window or ""

    notes: list[str] = []
    if spectroscopy_window is None:
        notes.append("Mg II 관측파장이 GNIRS J/H/K spectroscopy window 밖/edge이므로 실제 ITC 실행은 별도 확인 필요")

    try:
        uk_row = query_ukidss_las(coord)
        if uk_row is None:
            notes.append("UKIDSS LAS match 없음")
        else:
            selected: tuple[str, float, float | None, str] | None = None
            missing_band_notes: list[str] = []
            for band in band_candidates:
                mag_vega, mag_err, source_note = get_ukidss_mag(uk_row, band)
                if mag_vega is None:
                    missing_band_notes.append(source_note)
                    continue
                selected = (band, mag_vega, mag_err, source_note)
                break

            if selected is None:
                notes.extend(missing_band_notes[:1])
            else:
                wanted_band, mag_vega, mag_err, source_note = selected

                # UKIDSS Vega magnitude를 AB magnitude로 바꾼다.
                m_ab = mag_vega + UKIDSS_AB_OFFSETS[wanted_band]

                # AB magnitude에서 연속광 기준 밝기값을 만든다.
                fnu_jy = mag_ab_to_fnu_jy(m_ab)
                lambda_ref_um = BAND_LAMBDA_UM[wanted_band]
                flambda_ref = fnu_jy_to_flambda_cgs_per_a(fnu_jy, lambda_ref_um)

                # 필터 중심값을 Mg II 관측 파장으로 보정한다.
                flambda = extrapolate_flambda_power_law(flambda_ref, lambda_ref_um, lambda_obs_um)
                extrapolation_factor = flambda / flambda_ref

                # 연속광과 정지계 등가폭으로 Mg II 선 세기를 추정한다.
                line_flux = flambda * DEFAULT_EW_REST_A * (1.0 + z)

                out.point_source_brightness_ab_mag = m_ab
                out.itc_brightness_band = wanted_band
                out.catalog_mag_vega = mag_vega
                out.catalog_mag_err = mag_err
                out.catalog_source = source_note
                out.continuum_flux_density_erg_s_cm2_a = flambda
                out.continuum_reference_wavelength_um = lambda_ref_um
                out.continuum_target_wavelength_um = lambda_obs_um
                out.continuum_alpha_nu_assumed = DEFAULT_CONTINUUM_ALPHA_NU
                out.continuum_extrapolation_factor = extrapolation_factor
                out.line_flux_erg_s_cm2 = line_flux
                out.line_width_km_s = DEFAULT_FWHM_KMS
                out.ew_rest_a_assumed = DEFAULT_EW_REST_A
                if wanted_band != preferred_band:
                    notes.append(f"preferred {preferred_band}-band 값이 없어 {wanted_band}-band로 continuum 보정")
                if spectroscopy_window and wanted_band != spectroscopy_window:
                    notes.append(f"brightness band {wanted_band}는 Mg II spectroscopy window {spectroscopy_window}와 다름")
                notes.append(
                    f"continuum은 {wanted_band}-band 중심 {lambda_ref_um:.3f}um에서 "
                    f"Mg II {lambda_obs_um:.3f}um로 f_nu∝nu^{DEFAULT_CONTINUUM_ALPHA_NU:g} 보정"
                )
                notes.append("line flux는 실제 Mg II 측정값이 아니라 EW_rest 가정값으로 계산")
    except Exception as exc:
        notes.append(f"UKIDSS query 실패: {exc}")

    if out.point_source_brightness_ab_mag is None and use_ned_fallback:
        try:
            ned_name = find_ned_object_name(coord)
            if ned_name is None:
                notes.append("NED match 없음")
            else:
                ned_phot = None
                ned_band = preferred_band
                for band in band_candidates:
                    ned_phot = read_ned_photometry_near_ir(ned_name, band)
                    ned_band = band
                    if ned_phot is not None:
                        break
                if ned_phot is None:
                    notes.append(f"NED {preferred_band}-band 우선 near-IR 자동 판별 실패")
                else:
                    notes.append(f"NED 후보 {ned_band}-band photometry 있음: {ned_phot['ned_name']}; 수동 확인 필요")
        except Exception as exc:
            notes.append(f"NED query 실패: {exc}")
    elif out.point_source_brightness_ab_mag is None:
        notes.append("NED 후보 조회는 기본 실행에서 생략(--with-ned-fallback 사용 시 확인)")

    if out.point_source_brightness_ab_mag is None:
        out.note = "; ".join(notes) + "; 선택 밴드 관측값이 없으므로 ITC 값 비움"
    else:
        out.note = "; ".join(notes)
    return out


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
            return f"{x:.6e}"
        return f"{x:.6f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Gemini GNIRS ITC inputs for BLAZ4R Mg II observations.")
    parser.add_argument("--input-csv", type=Path, default=Path("blaz4r_table1_targets.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("gnirs_itc_inputs.csv"))
    parser.add_argument("--sample", choices=SAMPLE_CHOICES, default="confirmed")
    parser.add_argument(
        "--with-ned-notes",
        "--with-ned-fallback",
        dest="with_ned_notes",
        action="store_true",
        help="Query NED for candidate near-IR photometry notes when UKIDSS is missing. This never auto-parses magnitudes.",
    )
    args = parser.parse_args()

    rows: list[ItcRow] = []
    total_input_rows = 0
    skipped_by_sample = 0
    with args.input_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            total_input_rows += 1
            if not include_target_for_sample(row, args.sample):
                skipped_by_sample += 1
                continue
            rows.append(process_target(row, use_ned_fallback=args.with_ned_notes))
    print(f"[sample] {args.sample}: selected {len(rows)} of {total_input_rows} rows; skipped {skipped_by_sample}")
    if args.sample == "confirmed" and total_input_rows == 64 and len(rows) == 52:
        print(
            "[sample warning] local input has 52 rows marked blazar_class=y and 12 marked c; "
            "it is not the full current BLAZ4R final confirmed catalog."
        )

    fieldnames = [
        "name",
        "z",
        "blazar_class",
        "RA J2000",
        "Dec J2000",
        "ITC spectral distribution line wavelength [micron]",
        "ITC point source brightness band",
        "ITC point source spatially integrated brightness [AB mag]",
        "catalog magnitude [Vega mag]",
        "catalog magnitude uncertainty [mag]",
        "catalog/source",
        "ITC continuum flux density [erg/s/cm^2/A]",
        "continuum reference wavelength [micron]",
        "continuum target wavelength [micron]",
        "assumed continuum alpha_nu",
        "continuum extrapolation factor",
        "ITC line flux [erg/s/cm^2]",
        "ITC line width FWHM [km/s]",
        "assumed EW_rest [A]",
        "GNIRS spectroscopy window containing Mg II",
        "notes/cautions",
    ]

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "name": r.name,
                "z": fmt(r.z),
                "blazar_class": r.blazar_class,
                "RA J2000": r.ra,
                "Dec J2000": r.dec,
                "ITC spectral distribution line wavelength [micron]": fmt(r.mgii_wavelength_um),
                "ITC point source brightness band": r.itc_brightness_band,
                "ITC point source spatially integrated brightness [AB mag]": fmt(r.point_source_brightness_ab_mag),
                "catalog magnitude [Vega mag]": fmt(r.catalog_mag_vega),
                "catalog magnitude uncertainty [mag]": fmt(r.catalog_mag_err),
                "catalog/source": r.catalog_source,
                "ITC continuum flux density [erg/s/cm^2/A]": fmt(r.continuum_flux_density_erg_s_cm2_a),
                "continuum reference wavelength [micron]": fmt(r.continuum_reference_wavelength_um),
                "continuum target wavelength [micron]": fmt(r.continuum_target_wavelength_um),
                "assumed continuum alpha_nu": fmt(r.continuum_alpha_nu_assumed),
                "continuum extrapolation factor": fmt(r.continuum_extrapolation_factor),
                "ITC line flux [erg/s/cm^2]": fmt(r.line_flux_erg_s_cm2),
                "ITC line width FWHM [km/s]": fmt(r.line_width_km_s),
                "assumed EW_rest [A]": fmt(r.ew_rest_a_assumed),
                "GNIRS spectroscopy window containing Mg II": r.gnirs_spectroscopy_window,
                "notes/cautions": r.note,
            })


if __name__ == "__main__":
    main()
