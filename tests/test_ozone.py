import pytest
import numpy as np
from ozone import Ozone, utils

# ---- Optional profiling (pytest-safe) ----
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func


# ---- Configuration ----
AM_DATA_PATH = "/Users/namsonnguyen/repo/data/AM_Data/SON50_w_opacity/"
TEST_FIELD_IDX = 5 # Arbitrary index from field map


# ---- Fixtures ----
@pytest.fixture(scope="module")
def ozone_object():
    """
    Create a fully initialized Ozone object.

    Module scope prevents repeated expensive initialization.
    """
    freq_range = (210, 240)
    return Ozone(AM_DATA_PATH, freq_range=freq_range)

@pytest.fixture(scope="module")
def am_data(ozone_object:Ozone):
    """
    Docstring for am_data
    """

    lognscale = ozone_object.am_model.lognscale_map[TEST_FIELD_IDX]
    logairmass = ozone_object.am_model.logairmass_map[TEST_FIELD_IDX]

    # --- Load AM reference data ---
    am_filename = (
        f"{AM_DATA_PATH}"
        f"MaunaKea_Tb_Spectrum_{logairmass:.3f}_+{lognscale:.3f}.out"
    )

    try:
        am_data = np.load(am_filename)
    except FileNotFoundError as e:
        pytest.fail(f"Missing AM reference file: {am_filename}")
    
    return am_data

# ---- Tests ----
@profile
def test_load_dTb_nscale(ozone_object:Ozone, am_data):
    # --- Extract AM and model Jacobians ---
    freq_mask = ozone_object.am_model.freq_mask
    idx = ozone_object.DTB_NSCALE
    lognscale = ozone_object.am_model.lognscale_map[TEST_FIELD_IDX]
    AM_dTb_pwv = am_data[freq_mask, idx] * utils.log_to_linear_nscale_factor(lognscale)
    model_dTb_pwv = ozone_object.am_model.Tb.nscale.jacobian[TEST_FIELD_IDX, TEST_FIELD_IDX]

    assert AM_dTb_pwv.shape == model_dTb_pwv.shape

    # --- Deviation analysis ---
    deviation_spectrum = ((model_dTb_pwv - AM_dTb_pwv) / AM_dTb_pwv) * 100

    # 99th percentile deviation < 1%
    max_deviation = np.percentile(deviation_spectrum, 99)
    assert max_deviation < 1.0

    # Median absolute deviation < 0.1%
    mad = np.median(np.abs(deviation_spectrum))
    assert mad < 0.1

@profile
def test_load_dTb_airmass(ozone_object:Ozone, am_data):
    # --- Extract AM and model Jacobians ---
    freq_mask = ozone_object.am_model.freq_mask
    idx = ozone_object.DTB_AIRMASS
    logairmass = ozone_object.am_model.logairmass_map[TEST_FIELD_IDX]
    AM_dTb_zenith_angle = am_data[freq_mask, idx] * utils.log_to_linear_airmass_factor(logairmass)
    model_dTb_zenith_angle = ozone_object.am_model.Tb.airmass.jacobian[TEST_FIELD_IDX, TEST_FIELD_IDX]

    assert AM_dTb_zenith_angle.shape == model_dTb_zenith_angle.shape

    # --- Deviation analysis ---
    deviation_spectrum = ((model_dTb_zenith_angle - AM_dTb_zenith_angle) / AM_dTb_zenith_angle) * 100

    # 99th percentile deviation < 1%
    max_deviation = np.percentile(deviation_spectrum, 99)
    assert max_deviation < 1.0

    # Median absolute deviation < 0.1%
    mad = np.median(np.abs(deviation_spectrum))
    assert mad < 0.1