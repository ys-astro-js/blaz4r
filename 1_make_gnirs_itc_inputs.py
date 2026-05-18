from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.ned import Ned
from astroquery.ukidss import Ukidss

C_CGS = 2.99792458e10  # cm/s
C_KMS = 2.99792458e5   # km/s
MGII_REST_UM = 0.2798

# UKIDSS/WFCAM Vega -> AB offsets. Use these for catalog magnitudes, not ITC input system.
# ITC에는 AB mag 또는 flux density로 넣는다. 여기서는 카탈로그 Vega mag를 AB mag로 변환한다.
UKIDSS_AB_OFFSETS = {
    "J": 0.938,
    "H": 1.379,
    "K": 1.900,
}

# 대표 중심 파장. f_nu -> f_lambda 변환과 continuum 근사에 사용한다.
BAND_LAMBDA_UM = {
    "J": 1.248,
    "H": 1.631,
    "K": 2.201,
}

# 예비 ITC 계산용 가정값. 실제 분광값이 있으면 반드시 교체해야 한다.
DEFAULT_FWHM_KMS = 4000.0
DEFAULT_EW_REST_A = 30.0


def is_good_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v) and v > -90000000


def mag_ab_to_fnu_jy(m_ab: float) -> float:
    return 3631.0 * 10 ** (-0.4 * m_ab)


def fnu_jy_to_flambda_cgs_per_a(fnu_jy: float, lambda_um: float) -> float:
    # f_lambda [erg/s/cm^2/A] = f_nu[Jy] * 1e-23 * c / lambda_cm^2 * 1e-8
    lam_cm = lambda_um * 1e-4
    return fnu_jy * 1e-23 * C_CGS / (lam_cm * lam_cm) * 1e-8


def choose_itc_band(lambda_obs_um: float) -> str | None:
    # Mg II가 어느 근적외선 밴드에 걸리는지에 따라 continuum용 밴드를 고른다.
    if 1.10 <= lambda_obs_um < 1.45:
        return "J"
    if 1.45 <= lambda_obs_um < 1.85:
        return "H"
    if 1.85 <= lambda_obs_um <= 2.40:
        return "K"
    return None


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
    # AperMag3는 점광원에 대해 흔히 쓰는 2 arcsec aperture 계열 값이다.
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
    # 가장 가까운 NED object를 고른다. Separation 컬럼 이름은 NED 응답에 따라 다를 수 있어 방어적으로 처리한다.
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
    # NED photometry table은 항목명이 일정하지 않다. 자동 판별 실패 가능성이 높으므로 보조용으로만 쓴다.
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
    ra: str
    dec: str
    mgii_wavelength_um: float | None = None
    itc_brightness_band: str = ""
    point_source_brightness_ab_mag: float | None = None
    catalog_mag_vega: float | None = None
    catalog_mag_err: float | None = None
    catalog_source: str = ""
    continuum_flux_density_erg_s_cm2_a: float | None = None
    line_flux_erg_s_cm2: float | None = None
    line_width_km_s: float | None = None
    ew_rest_a_assumed: float | None = None
    note: str = ""


def process_target(row: dict[str, str]) -> ItcRow:
    name = row["name"]
    z = float(row["z"])
    coord = SkyCoord(row["ra"], row["dec"], unit=(u.hourangle, u.deg), frame="icrs")
    lambda_obs_um = MGII_REST_UM * (1.0 + z)
    wanted_band = choose_itc_band(lambda_obs_um)

    out = ItcRow(name=name, z=z, ra=row["ra"], dec=row["dec"], mgii_wavelength_um=lambda_obs_um)
    if wanted_band is None:
        out.note = "Mg II 관측파장이 UKIDSS J/H/K 범위 밖이므로 계산하지 않음"
        return out
    out.itc_brightness_band = wanted_band

    notes: list[str] = []
    try:
        uk_row = query_ukidss_las(coord)
        if uk_row is None:
            notes.append("UKIDSS LAS match 없음")
        else:
            mag_vega, mag_err, source_note = get_ukidss_mag(uk_row, wanted_band)
            if mag_vega is None:
                notes.append(source_note)
            else:
                m_ab = mag_vega + UKIDSS_AB_OFFSETS[wanted_band]
                fnu_jy = mag_ab_to_fnu_jy(m_ab)
                flambda = fnu_jy_to_flambda_cgs_per_a(fnu_jy, BAND_LAMBDA_UM[wanted_band])

                # 선 플럭스: F_line = f_lambda_cont * EW_rest * (1+z)
                line_flux = flambda * DEFAULT_EW_REST_A * (1.0 + z)

                out.point_source_brightness_ab_mag = m_ab
                out.catalog_mag_vega = mag_vega
                out.catalog_mag_err = mag_err
                out.catalog_source = source_note
                out.continuum_flux_density_erg_s_cm2_a = flambda
                out.line_flux_erg_s_cm2 = line_flux
                out.line_width_km_s = DEFAULT_FWHM_KMS
                out.ew_rest_a_assumed = DEFAULT_EW_REST_A
                notes.append("line flux는 실제 Mg II 측정값이 아니라 EW_rest 가정값으로 계산")
    except Exception as exc:
        notes.append(f"UKIDSS query 실패: {exc}")

    if out.point_source_brightness_ab_mag is None:
        try:
            ned_name = find_ned_object_name(coord)
            if ned_name is None:
                notes.append("NED match 없음")
            else:
                ned_phot = read_ned_photometry_near_ir(ned_name, wanted_band)
                if ned_phot is None:
                    notes.append(f"NED {wanted_band}-band 자동 판별 실패")
                else:
                    notes.append(f"NED 후보 photometry 있음: {ned_phot['ned_name']}; 수동 확인 필요")
        except Exception as exc:
            notes.append(f"NED query 실패: {exc}")

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
    input_csv = Path("blaz4r_table1_targets.csv")
    output_csv = Path("gnirs_itc_inputs.csv")

    rows: list[ItcRow] = []
    with input_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(process_target(row))

    fieldnames = [
        "name",
        "z",
        "RA J2000",
        "Dec J2000",
        "ITC spectral distribution line wavelength [micron]",
        "ITC point source brightness band",
        "ITC point source spatially integrated brightness [AB mag]",
        "catalog magnitude [Vega mag]",
        "catalog magnitude uncertainty [mag]",
        "catalog/source",
        "ITC continuum flux density [erg/s/cm^2/A]",
        "ITC line flux [erg/s/cm^2]",
        "ITC line width FWHM [km/s]",
        "assumed EW_rest [A]",
        "notes/cautions",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "name": r.name,
                "z": fmt(r.z),
                "RA J2000": r.ra,
                "Dec J2000": r.dec,
                "ITC spectral distribution line wavelength [micron]": fmt(r.mgii_wavelength_um),
                "ITC point source brightness band": r.itc_brightness_band,
                "ITC point source spatially integrated brightness [AB mag]": fmt(r.point_source_brightness_ab_mag),
                "catalog magnitude [Vega mag]": fmt(r.catalog_mag_vega),
                "catalog magnitude uncertainty [mag]": fmt(r.catalog_mag_err),
                "catalog/source": r.catalog_source,
                "ITC continuum flux density [erg/s/cm^2/A]": fmt(r.continuum_flux_density_erg_s_cm2_a),
                "ITC line flux [erg/s/cm^2]": fmt(r.line_flux_erg_s_cm2),
                "ITC line width FWHM [km/s]": fmt(r.line_width_km_s),
                "assumed EW_rest [A]": fmt(r.ew_rest_a_assumed),
                "notes/cautions": r.note,
            })


if __name__ == "__main__":
    main()
