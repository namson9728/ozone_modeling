import pytest
from ozone import Ozone
import numpy as np

# Set-up data
data_path = '/Users/namsonnguyen/repo/data/AM_Data/MaunaKea_SON50/Nscale21_AirMass11/'
my_ozone = Ozone(am_model_data_path=data_path)

nscale_map = my_ozone.data['Nscale']['map']
airmass_map = my_ozone.data['airmass']['map']
zenith_map = my_ozone._airmass_to_zenith(airmass_map)
nominal_pwv = my_ozone._extract_nominal_pwv()
pwv_map = nominal_pwv * (10**nscale_map)

# Kinda sloppy...will clean-up

def test_cross_check_with_AM(idx=0):
    model_spectrum, pwv_jacobian, zenith_jacobian = my_ozone(pwv_map[idx], zenith_map[idx])
    Tb_data = my_ozone.data['Tb_scalar_field'][idx][idx].reshape(model_spectrum.shape)
    difference_spectrum = ((model_spectrum - Tb_data) / Tb_data) * 100

    threshold = 1    # 1%
    max_deviation = np.max(difference_spectrum)

    assert max_deviation < threshold

def test_with_pwv_Jacobian(start_pwv=pwv_map[0], end_pwv=pwv_map[0]+1, zenith=1, points=50):
    pwv_map = np.linspace(start_pwv, end_pwv, points)

    base_spectrum, base_pwv_jacobian, base_zenith_jacobian = my_ozone(start_pwv, zenith)
    expected_spectrum, expected_pwv_jacobian, expected_zenith_jacobian = my_ozone(end_pwv, zenith)

    test_spectrum = base_spectrum
    for pwv in pwv_map:
        spectrum, pwv_jacobian, zenith_jacobian = my_ozone(pwv, zenith)
        little_jacobian = pwv_jacobian * ((end_pwv-start_pwv)/points)
        test_spectrum += little_jacobian

    difference_spectrum = ((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    threshold = 2    # 2%
    max_deviation = np.max(difference_spectrum)

    assert max_deviation < threshold

def test_with_zenith_Jacobian(start_zenith=zenith_map[0], end_zenith=zenith_map[0]+0.3, pwv=5, points=50):
    zenith_map = np.linspace(start_zenith, end_zenith, points)

    base_spectrum, base_pwv_jacobian, base_zenith_jacobian = my_ozone(pwv, start_zenith)
    expected_spectrum, expected_pwv_jacobian, expected_zenith_jacobian = my_ozone(pwv, end_zenith)

    test_spectrum = base_spectrum
    for zenith in zenith_map:
        spectrum, pwv_jacobian, zenith_jacobian = my_ozone(pwv, zenith)
        little_jacobian = zenith_jacobian * ((end_zenith-start_zenith)/points)
        test_spectrum += little_jacobian

    difference_spectrum = ((test_spectrum - expected_spectrum) / expected_spectrum) * 100

    threshold = 2    # 2%
    max_deviation = np.max(difference_spectrum)

    assert max_deviation < threshold