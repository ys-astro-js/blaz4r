from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


make_inputs = load_module("1_make_gnirs_itc_inputs.py", "make_gnirs_itc_inputs_under_test")
run_batch = load_module("2_run_gnirs_itc_batch.py", "run_gnirs_itc_batch_under_test")


class GnirsItcLogicTest(unittest.TestCase):
    def test_choose_itc_band_uses_gnirs_spectroscopy_windows(self) -> None:
        self.assertEqual(make_inputs.choose_itc_band(1.369), "J")
        self.assertIsNone(make_inputs.choose_itc_band(1.399))
        self.assertIsNone(make_inputs.choose_itc_band(1.455))
        self.assertEqual(make_inputs.choose_itc_band(1.471), "H")
        self.assertIsNone(make_inputs.choose_itc_band(1.805))
        self.assertEqual(make_inputs.choose_itc_band(1.920), "K")

    def test_continuum_band_is_not_tied_to_spectroscopy_window(self) -> None:
        self.assertIsNone(make_inputs.choose_itc_band(1.455))
        self.assertEqual(make_inputs.choose_continuum_band(1.455), "H")

    def test_power_law_extrapolates_band_flux_to_mgii_wavelength(self) -> None:
        factor = make_inputs.extrapolate_flambda_power_law(1.0, 1.631, 1.455)

        self.assertAlmostEqual(factor, 1.195, places=3)

    def test_default_sample_selection_excludes_candidates(self) -> None:
        confirmed = {"name": "J000000+000000", "blazar_class": "y"}
        candidate = {"name": "J111111+111111", "blazar_class": "c"}

        self.assertTrue(make_inputs.include_target_for_sample(confirmed, "confirmed"))
        self.assertFalse(make_inputs.include_target_for_sample(candidate, "confirmed"))
        self.assertTrue(make_inputs.include_target_for_sample(candidate, "candidates"))
        self.assertTrue(make_inputs.include_target_for_sample(candidate, "all"))

    def test_loader_accepts_brightness_band_different_from_line_window(self) -> None:
        path = ROOT / "tests" / "_tmp_itc_inputs.csv"
        path.write_text(
            "\n".join(
                [
                    "name,z,RA J2000,Dec J2000,ITC point source spatially integrated brightness [AB mag],ITC point source brightness band,ITC spectral distribution line wavelength [micron],ITC line flux [erg/s/cm^2],ITC line width FWHM [km/s],ITC continuum flux density [erg/s/cm^2/A]",
                    "J000000+000000,4.4,00:00:00,+00:00:00,19.0,J,1.510000,1e-15,4000,5e-18",
                ]
            ),
            encoding="utf-8",
        )
        try:
            targets = run_batch.load_targets(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].brightness_band, "J")
        self.assertEqual(run_batch.spectroscopy_window_band(targets[0].line_wavelength_um), "H")

    def test_loader_excludes_candidate_rows_when_class_is_present(self) -> None:
        path = ROOT / "tests" / "_tmp_itc_inputs_with_class.csv"
        path.write_text(
            "\n".join(
                [
                    "name,blazar_class,z,RA J2000,Dec J2000,ITC point source spatially integrated brightness [AB mag],ITC point source brightness band,ITC spectral distribution line wavelength [micron],ITC line flux [erg/s/cm^2],ITC line width FWHM [km/s],ITC continuum flux density [erg/s/cm^2/A]",
                    "J000000+000000,y,4.4,00:00:00,+00:00:00,19.0,H,1.510000,1e-15,4000,5e-18",
                    "J111111+111111,c,4.4,11:11:11,+11:11:11,19.0,H,1.510000,1e-15,4000,5e-18",
                ]
            ),
            encoding="utf-8",
        )
        try:
            confirmed = run_batch.load_targets(path)
            all_rows = run_batch.load_targets(path, sample="all")
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual([target.name for target in confirmed], ["J000000+000000"])
        self.assertEqual(len(all_rows), 2)

    def test_dynamic_select_overrides_use_current_gemini_cgs_values(self) -> None:
        soup = BeautifulSoup(
            """
            <form>
              <select name="lineFluxUnits">
                <option value="watts_flux">W/m2</option>
                <option value="ergs_flux">erg/s/cm2</option>
              </select>
              <select name="lineContinuumUnits">
                <option value="watts_fd_wavelength">W/m2/um</option>
                <option value="ergs_fd_wavelength">erg/s/cm2/A</option>
              </select>
            </form>
            """,
            "lxml",
        )

        overrides = run_batch.dynamic_select_overrides(soup.find("form"))

        self.assertEqual(overrides["lineFluxUnits"], "ergs_flux")
        self.assertEqual(overrides["lineContinuumUnits"], "ergs_fd_wavelength")

    def test_payload_validation_rejects_old_cgs_unit_values(self) -> None:
        target = run_batch.Target(
            name="J000000+000000",
            z=4.4,
            blazar_class="y",
            ra="00:00:00",
            dec="+00:00:00",
            brightness_ab_mag=19.0,
            brightness_band="H",
            line_wavelength_um=1.51,
            line_flux_erg_s_cm2=1.0e-15,
            line_width_km_s=4000.0,
            continuum_erg_s_cm2_a=5.0e-18,
        )
        payload = run_batch.configure_payload(
            target,
            30,
            {},
            {
                "lineFluxUnits": "cgs_flux",
                "lineContinuumUnits": "cgs_fd_wavelength",
                "PlotLimits": "AUTO",
            },
            5.0,
        )

        warnings = run_batch.validate_payload(payload, target, 30, 5.0)

        self.assertTrue(any("lineFluxUnits" in warning for warning in warnings))
        self.assertTrue(any("lineContinuumUnits" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
