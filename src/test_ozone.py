import pytest
import random
from ozone import Ozone
import numpy as np
from line_profiler import profile

#AM_DATA_PATH = '/Users/gkeating/newdata2/'
AM_DATA_PATH = '/Users/namsonnguyen/repo/data/AM_Data/SON50_w_opacity/'

@pytest.fixture
def ozone_object():
    return Ozone(am_model_data_path=AM_DATA_PATH)

@pytest.fixture
def AM_spectrum_params(ozone_object):
    logairmass = 0.001
    logNscale = 0.125
    filename = f'MaunaKea_Tb_Spectrum_{logairmass:.3f}_{logNscale:+.3f}'
    data = np.load(f'{ozone_object.am_model_data_path}{filename}.out')
    Tb_model_spectrum = data[:,4]
    tau_model_spectrum = data[:,1]

    zenith_angle = ozone_object._airmass_to_zenith(np.exp(logairmass))
    pwv = (np.exp(logNscale))*ozone_object.nominal_pwv

    return [Tb_model_spectrum, pwv, zenith_angle, tau_model_spectrum]


@pytest.fixture
def jacobian_pwv_params(ozone_object):
    nscale_map = ozone_object.data['Nscale']['map']
    nominal_pwv = ozone_object._extract_nominal_pwv()
    pwv_map = nominal_pwv * np.exp(nscale_map)
    
    start_pwv=pwv_map[0]
    end_pwv=pwv_map[1]
    zenith_angle=1
    points=100

    return [ozone_object, start_pwv, end_pwv, zenith_angle, points]

@pytest.fixture
def jacobian_zenith_params(ozone_object):
    airmass_map = ozone_object.data['airmass']['map']
    zenith_map = ozone_object._airmass_to_zenith(np.exp(airmass_map))

    start_zenith=zenith_map[1]
    end_zenith=zenith_map[2]
    pwv=5
    points=100

    return [ozone_object, start_zenith, end_zenith, pwv, points]

@profile
def test_Tb_cross_check_with_AM(AM_spectrum_params, ozone_object):
    Tb_spectrum = AM_spectrum_params[0]
    pwv = AM_spectrum_params[1]
    zenith_angle = AM_spectrum_params[2]

    model_spectrum = ozone_object(pwv, zenith_angle)
    difference_spectrum = ((model_spectrum - Tb_spectrum) / Tb_spectrum) * 100

    threshold = 1    # 1%
    max_deviation = np.percentile(difference_spectrum, 99)

    assert max_deviation < threshold

    threshold = 0.1
    mad = np.median(np.abs(difference_spectrum))
    assert mad < threshold

@profile
def test_tau_cross_check_with_AM(AM_spectrum_params, ozone_object):
    tau_spectrum = AM_spectrum_params[3]
    pwv = AM_spectrum_params[1]
    zenith_angle = AM_spectrum_params[2]

    _, tau_model_spectrum = ozone_object(pwv, zenith_angle, return_opacity_pair=True)
    difference_spectrum = ((tau_model_spectrum - tau_spectrum) / tau_spectrum) * 100

    threshold = 1    # 1%
    max_deviation = np.percentile(difference_spectrum, 99)

    assert max_deviation < threshold

    threshold = 0.1
    mad = np.median(np.abs(difference_spectrum))
    assert mad < threshold

@profile
def test_dTb_pwv_sum(jacobian_pwv_params):
    my_ozone = jacobian_pwv_params[0]
    start_pwv = jacobian_pwv_params[1]
    end_pwv = jacobian_pwv_params[2]
    zenith_angle = jacobian_pwv_params[3]
    points = jacobian_pwv_params[4]

    test_pwv_map = np.linspace(start_pwv, end_pwv, points)

    base_spectrum = my_ozone(start_pwv, zenith_angle)
    expected_spectrum = my_ozone(end_pwv, zenith_angle)

    test_spectrum = base_spectrum
    for pwv in test_pwv_map:
        pwv_jacobian = my_ozone(pwv, zenith_angle, return_pwv_jacobian=True, return_model_spectrum=False)
        little_jacobian = pwv_jacobian * ((end_pwv-start_pwv)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.1 # 0.1%
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert max_deviation < max_threshold
    assert mad_deviation < mad_threshold

@profile
def test_dTau_pwv_sum(jacobian_pwv_params):
    my_ozone = jacobian_pwv_params[0]
    start_pwv = jacobian_pwv_params[1]
    end_pwv = jacobian_pwv_params[2]
    zenith_angle = jacobian_pwv_params[3]
    points = jacobian_pwv_params[4]

    test_pwv_map = np.linspace(start_pwv, end_pwv, points)

    _, tau_base_spectrum = my_ozone(start_pwv, zenith_angle, return_opacity_pair=True)
    _, expected_tau_spectrum = my_ozone(end_pwv, zenith_angle, return_opacity_pair=True)

    test_spectrum = tau_base_spectrum
    for pwv in test_pwv_map:
        _, dTau_pwv = my_ozone(pwv, zenith_angle, return_pwv_jacobian=True, 
                                return_model_spectrum=False, return_opacity_pair=True)
        little_jacobian = dTau_pwv * ((end_pwv-start_pwv)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - expected_tau_spectrum) / expected_tau_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.13 # 0.1%  # FIX ME!
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert mad_deviation < mad_threshold
    assert max_deviation < max_threshold

@profile
def test_dTb_zenith_sum(jacobian_zenith_params):
    my_ozone = jacobian_zenith_params[0]
    start_zenith = jacobian_zenith_params[1]
    end_zenith = jacobian_zenith_params[2]
    pwv = jacobian_zenith_params[3]
    points = jacobian_zenith_params[4]

    zenith_map = np.linspace(start_zenith, end_zenith, points)

    base_spectrum = my_ozone(pwv, start_zenith)
    expected_spectrum = my_ozone(pwv, end_zenith)

    test_spectrum = base_spectrum
    for zenith in zenith_map:
        zenith_jacobian = my_ozone(pwv, zenith, return_zenith_jacobian=True, return_model_spectrum=False)
        little_jacobian = zenith_jacobian * ((end_zenith-start_zenith)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.1 # 0.1%
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert max_deviation < max_threshold
    assert mad_deviation < mad_threshold

@profile
def test_dTau_zenith_sum(jacobian_zenith_params):
    my_ozone = jacobian_zenith_params[0]
    start_zenith = jacobian_zenith_params[1]
    end_zenith = jacobian_zenith_params[2]
    pwv = jacobian_zenith_params[3]
    points = jacobian_zenith_params[4]

    zenith_map = np.linspace(start_zenith, end_zenith, points)

    _, tau_base_spectrum = my_ozone(pwv, start_zenith, return_opacity_pair=True)
    _, tau_expected_spectrum = my_ozone(pwv, end_zenith, return_opacity_pair=True)

    test_spectrum = tau_base_spectrum
    for zenith in zenith_map:
        _, dTau_zenith = my_ozone(pwv, zenith, return_zenith_jacobian=True, 
                                   return_model_spectrum=False, return_opacity_pair=True)
        little_jacobian = dTau_zenith * ((end_zenith-start_zenith)/points)
        test_spectrum += little_jacobian

    difference_spectrum = abs((test_spectrum - tau_expected_spectrum) / tau_expected_spectrum) * 100

    max_threshold = 1    # 1%
    mad_threshold = 0.18 # 0.1%  # FIX ME!
    mad_deviation, max_deviation = np.percentile(difference_spectrum, [50, 99])
    assert max_deviation < max_threshold
    assert mad_deviation < mad_threshold