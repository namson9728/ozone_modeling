import pytest
import random
from ozone import Ozone
import numpy as np
from line_profiler import profile

AM_DATA_PATH = '/Users/gkeating/newdata2/'

@pytest.fixture
def ozone_object():
    return Ozone(am_model_data_path=AM_DATA_PATH)

@pytest.fixture
def model_spectrum_params(ozone_object):
    nscale_map = ozone_object.data['Nscale']['map']
    nominal_pwv = ozone_object._extract_nominal_pwv()
    pwv_map = nominal_pwv * np.exp(nscale_map)

    airmass_map = np.exp(ozone_object.data['airmass']['map'])
    zenith_map = ozone_object._airmass_to_zenith(airmass_map)

    # Randomize which AM generated datafile to use for testing
    pwv_idx = random.randint(0, len(pwv_map) - 1)
    zenith_idx = random.randint(0, len(zenith_map) - 1)
    model_spectrum, _, _ = ozone_object(pwv_map[pwv_idx], zenith_map[zenith_idx])

    return [model_spectrum, pwv_idx, zenith_idx]


@pytest.fixture
def pwv_Jacobian_params(ozone_object):
    nscale_map = ozone_object.data['Nscale']['map']
    nominal_pwv = ozone_object._extract_nominal_pwv()
    pwv_map = nominal_pwv * np.exp(nscale_map)
    
    start_pwv=pwv_map[0]
    end_pwv=pwv_map[1]
    zenith_angle=1
    points=100

    return [ozone_object, start_pwv, end_pwv, zenith_angle, points]

@pytest.fixture
def zenith_Jacobian_params(ozone_object):
    airmass_map = ozone_object.data['airmass']['map']
    zenith_map = ozone_object._airmass_to_zenith(np.exp(airmass_map))

    start_zenith=zenith_map[1]
    end_zenith=zenith_map[2]
    pwv=5
    points=100

    return [ozone_object, start_zenith, end_zenith, pwv, points]

@profile
def test_cross_check_with_AM(model_spectrum_params, ozone_object):
    model_spectrum = model_spectrum_params[0]
    pwv_idx = model_spectrum_params[1]
    zenith_idx = model_spectrum_params[2]

    Tb_data = ozone_object.data['Tb_scalar_field'][pwv_idx][zenith_idx].reshape(model_spectrum.shape)
    difference_spectrum = ((model_spectrum - Tb_data) / Tb_data) * 100

    threshold = 1    # 1%
    max_deviation = np.percentile(difference_spectrum, 99)

    assert max_deviation < threshold

    threshold = 0.1
    mad = np.sum(difference_spectrum - np.mean(difference_spectrum))/len(difference_spectrum)
    assert mad < threshold

@profile
def test_with_pwv_Jacobian(pwv_Jacobian_params):
    my_ozone = pwv_Jacobian_params[0]
    start_pwv = pwv_Jacobian_params[1]
    end_pwv = pwv_Jacobian_params[2]
    zenith_angle = pwv_Jacobian_params[3]
    points = pwv_Jacobian_params[4]

    test_pwv_map = np.linspace(start_pwv, end_pwv, points)

    base_spectrum, _, _ = my_ozone(start_pwv, zenith_angle)
    expected_spectrum, _, _ = my_ozone(end_pwv, zenith_angle)

    test_spectrum = base_spectrum
    for pwv in test_pwv_map:
        _, pwv_jacobian, _ = my_ozone(pwv, zenith_angle, return_pwv_jacobian=True, return_model_spectrum=False)
        little_jacobian = pwv_jacobian * ((end_pwv-start_pwv)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.1 # 0.1%
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert max_deviation < max_threshold
    assert mad_deviation < mad_threshold

@profile
def test_with_zenith_Jacobian(zenith_Jacobian_params):
    my_ozone = zenith_Jacobian_params[0]
    start_zenith = zenith_Jacobian_params[1]
    end_zenith = zenith_Jacobian_params[2]
    pwv = zenith_Jacobian_params[3]
    points = zenith_Jacobian_params[4]

    zenith_map = np.linspace(start_zenith, end_zenith, points)

    base_spectrum, _, _ = my_ozone(pwv, start_zenith)
    expected_spectrum, _, _ = my_ozone(pwv, end_zenith)

    test_spectrum = base_spectrum
    for zenith in zenith_map:
        _, _, zenith_jacobian = my_ozone(pwv, zenith, return_zenith_jacobian=True, return_model_spectrum=False)
        little_jacobian = zenith_jacobian * ((end_zenith-start_zenith)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.1 # 0.1%
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert max_deviation < max_threshold
    assert mad_deviation < mad_threshold
