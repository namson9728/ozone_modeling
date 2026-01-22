from dataclasses import dataclass
from typing import Optional, Any, Sequence
from . import utils
from scipy.interpolate import RegularGridInterpolator, CubicHermiteSpline
import numpy as np

@dataclass
class Interpolation:
    regular: dict
    cubic: dict

@dataclass
class Spectrum:
    jacobian: np.ndarray
    interp_func: Interpolation

@dataclass
class Field:
    log_scalar_field: np.ndarray
    nscale: Spectrum
    airmass: Spectrum

@dataclass
class Model:
    lognscale_map: np.ndarray
    logairmass_map: np.ndarray
    freq_map: np.ndarray
    freq_mask: np.ndarray

    Tb: Field
    Tau: Field

    nominal_pwv: float

class Ozone:
    # Values used to create field maps for AM data generation
    MIN_LOGNSCALE, MAX_LOGNSCALE, LOGNSCALE_POINTS = -1.0, 1.0, 9
    MIN_LOGAIRMASS, MAX_LOGAIRMASS, LOGAIRMASS_POINTS = 0.001, 1.281, 9

    # The idex values for how data fields are stored in the AM file for extraction
    FREQ = 0
    TAU_SCALAR = 1
    DTAU_AIRMASS = 2
    DTAU_NSCALE = 3
    TB_SCALAR = 4
    DTB_AIRMASS = 5
    DTB_NSCALE = 6

    def __init__(self, am_data_path:str, delay_spectrum_data_path=None, freq_range=None):
        self.am_data_path = am_data_path
        self.delay_spectrum_data_path = delay_spectrum_data_path
        self.freq_range = freq_range

        self.am_model = self._load_AM_data(self.am_data_path)
        self.am_model.nominal_pwv = self._extract_nominal_pwv()
        self.freq_range = freq_range if None else (np.min(self.am_model.freq_map), np.max(self.am_model.freq_map))

        # Initialize all Tb interpolators
        self._init_interpolators(self.am_model, self.am_model.Tb)


    def _init_empty_model(self, data_path:str):
        logNscale_map = np.linspace(self.MIN_LOGNSCALE, self.MAX_LOGNSCALE, self.LOGNSCALE_POINTS)
        logairmass_map = np.linspace(self.MIN_LOGAIRMASS, self.MAX_LOGAIRMASS, self.LOGAIRMASS_POINTS)

        # Arbitrary file read to grab frequency data which should be consistent throughout all AM files
        filename = f'MaunaKea_Tb_Spectrum_{logairmass_map[0]:.3f}_{logNscale_map[0]:+.3f}'      
        data = np.load(f'{self.am_data_path}{filename}.out')

        freq_map = data[:, self.FREQ]
        freq_mask = np.ones(freq_map.shape, dtype=bool)
        if self.freq_range is not None:
            freq_mask = (freq_map >= self.freq_range[0]) & (freq_map <= self.freq_range[1])
        freq_points = len(freq_map[freq_mask])

        # Initialize empty field arrays for Tb and Tau
        Tb_scalar_field = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        dTb_nscale = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        dTb_airmass = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        Tb = Field(log_scalar_field=Tb_scalar_field, 
                   nscale=Spectrum(jacobian=dTb_nscale, interp_func=Interpolation(None, None)),
                   airmass=Spectrum(jacobian=dTb_airmass, interp_func=Interpolation(None, None)))

        tau_scalar_field = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        dTau_nscale = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        dTau_airmass = np.zeros((self.LOGNSCALE_POINTS, self.LOGAIRMASS_POINTS, freq_points))
        Tau = Field(log_scalar_field=tau_scalar_field,
                    nscale=Spectrum(dTau_nscale, interp_func=Interpolation(None, None)),
                    airmass=Spectrum(dTau_airmass, interp_func=Interpolation(None, None)))
        
        return {
            "lognscale_map": logNscale_map,
            "logairmass_map": logairmass_map,
            "freq_map": freq_map[freq_mask],
            "freq_mask": freq_mask,
            "Tb": Tb,
            "Tau": Tau,
            "nominal_pwv": None
        }
    
    def _init_interpolators(self, model:Model, field:Field):
        field.nscale.interp_func.regular = utils.init_RegularGridInterpolator(
            model, field.nscale.jacobian
        )
        field.airmass.interp_func.regular = utils.init_RegularGridInterpolator(
            model, field.airmass.jacobian
        )

        nscale_cubic_interp_dict = {}
        for idx in range(len(model.logairmass_map)):
            nscale_cubic_interp_dict[idx] = utils.init_CubicHermiteSpline(
                model.lognscale_map,
                field.log_scalar_field[:, idx, :],
                field.nscale.jacobian[:, idx, :]
            )
        field.nscale.interp_func.cubic = nscale_cubic_interp_dict

        airmass_cubic_interp_dict = {}
        for idx in range(len(model.lognscale_map)):
            airmass_cubic_interp_dict[idx] = utils.init_CubicHermiteSpline(
                model.logairmass_map,
                field.log_scalar_field[idx, :, :],
                field.airmass.jacobian[idx, :, :]
            )
        field.airmass.interp_func.cubic = airmass_cubic_interp_dict

    def _load_AM_data(self, data_path:str):
        model = Model(**self._init_empty_model(data_path))

        freq_mask = model.freq_mask
        tau = model.Tau
        Tb = model.Tb

        # Convert global var to local for optimization
        tau_scalar = self.TAU_SCALAR
        dtau_airmass = self.DTAU_AIRMASS
        dtau_nscale = self.DTAU_NSCALE
        tb_scalar = self.TB_SCALAR
        dtb_airmass = self.DTB_AIRMASS
        dtb_nscale = self.DTB_NSCALE
        for idx, lognscale in enumerate(model.lognscale_map):
            for jdx, logairmass in enumerate(model.logairmass_map):
                filename = f'MaunaKea_Tb_Spectrum_{logairmass:.3f}_{lognscale:+.3f}'
                data = np.load(f'{self.am_data_path}{filename}.out')

                tau.log_scalar_field[idx,jdx] = data[freq_mask,tau_scalar]
                log_dTau_airmass = data[freq_mask,dtau_airmass]
                tau.airmass.jacobian[idx, jdx] = log_dTau_airmass * utils.log_to_linear_airmass_factor(logairmass)
                log_dTau_nscale = data[freq_mask,dtau_nscale]
                tau.nscale.jacobian[idx, jdx] = log_dTau_nscale * utils.log_to_linear_nscale_factor(lognscale)

                Tb.log_scalar_field[idx, jdx] = data[freq_mask,tb_scalar]
                log_dTb_airmass = data[freq_mask,dtb_airmass]
                Tb.airmass.jacobian[idx, jdx] = log_dTb_airmass * utils.log_to_linear_airmass_factor(logairmass)
                log_dTb_nscale = data[freq_mask,dtb_nscale]
                Tb.nscale.jacobian[idx, jdx] = log_dTb_nscale * utils.log_to_linear_nscale_factor(lognscale)

        return model
    
    def _extract_nominal_pwv(self, dataset:str='Tb'):
        """Returns the nominal pwv value extracted from one of the AM .err files.

        Args:
            dataset : str
                choose which dataset to grab the .err file ['Tb', 'delay']
                (default 'Tb')
        """

        # Arbitrary idx (3) for any AM data file to extract .err file
        lognscale = self.am_model.lognscale_map[3]   
        logairmass = self.am_model.logairmass_map[3]
        err_file = f'{logairmass:.3f}_{lognscale:+.3f}'
        
        if dataset == 'Tb':
            err_file = f"{self.am_data_path}MaunaKea_Tb_Spectrum_{err_file}.err"
        elif dataset == 'delay' or dataset == 'Tau':
            err_file = f"{self.delay_spectrum_data_path}MaunaKea_Delay_Spectrum_{err_file}.err"
        else:
            raise ValueError(f"Invalid dataset '{dataset}': must be either 'Tb' or 'delay/Tau'")

        with open(err_file, "r") as file:
            text_data = file.readlines()

        look_for_pwv = False
        for idx in range(len(text_data)):
            if "total" in text_data[idx]: look_for_pwv = True
            if look_for_pwv:
                if "um_pwv" in text_data[idx]: pwv = text_data[idx]
        print(pwv)
        pwv = pwv.partition('(')
        pwv = float(pwv[2].partition(' ')[0])

        # Convert to um to mm
        nominal_pwv = (pwv * 10**-3) / np.exp(lognscale)
        print(nominal_pwv)
        return nominal_pwv
    
    def _bilinear_jacobian_interp(
        self, eval_nscale: float | slice, eval_airmass: float | slice, spectrum:Spectrum, norm_factor=None
    ):
        """Returns the interpolation spectrum using the provided 2DRegularGrid interpolation function.

        A normalization factor can be provided which is multiplied to the entire spectrum.
        """
        jacobian = spectrum.jacobian
        index_dict = {"nscale": [eval_nscale],  "airmass": [eval_airmass]}
        weights_dict = {"nscale": [1], "airmass": [1]}

        for item, eval_val in zip(["nscale", "airmass"], [eval_nscale, eval_airmass]):
            if isinstance(eval_val, slice):
                continue
            axis_map = self.am_model.lognscale_map if item == 'nscale' else self.am_model.logairmass_map
            temp_idx = np.searchsorted(axis_map, eval_val, 'left')
            if eval_val == axis_map[temp_idx]:
                index_dict[item] = [temp_idx] # type: ignore
            else:
                assert (temp_idx >= 1) and (temp_idx < len(axis_map))
                eval_val = index_dict[item][0]
                index_dict[item] = [temp_idx - 1, temp_idx] # type: ignore
                l_val = axis_map[temp_idx - 1]
                r_val = axis_map[temp_idx]
                del_val = r_val - l_val
                weights_dict[item] = [
                    (r_val - eval_val) / del_val, (eval_val - l_val) / del_val
                ]
        nscale_idx = index_dict["nscale"]
        airmass_idx = index_dict["airmass"]
        set_result = True
        if (len(airmass_idx) == 1) and (len(nscale_idx) == 1):
            result = jacobian[nscale_idx[0], airmass_idx[0]]
        else:
            for n_idx, n_weight in zip(nscale_idx, weights_dict["nscale"]):
                for a_idx, a_weight in zip(airmass_idx, weights_dict["airmass"]):
                    if set_result:
                        result = jacobian[n_idx, a_idx] * (a_weight * n_weight)
                        set_result = False
                    else:
                        result += jacobian[n_idx, a_idx] * (a_weight * n_weight)
        if not isinstance(nscale_idx[0], slice):
            result = result[None]
        if not isinstance(airmass_idx[0], slice):
            result = result[:, None]

        return result if norm_factor is None else (result * norm_factor)
    
    def _interp_spectrum(self, eval_airmass, eval_nscale, field:Field,
                         airmass_cubic_interp_dict, nscale_cubic_interp_dict):
        '''Returns the interpolation of the data given an airmass and nscale value.

        Args:
            eval_airmass : int
                The airmass value at which the interpolation will evaluate.
            eval_nscale : int
                The nscale value at which the interpolation will evaluate.
        '''
        # But we _only_ need the nearest two airmasses for the interpolation
        logairmass_map = self.am_model.logairmass_map
        lognscale_map = self.am_model.lognscale_map
        assert (eval_airmass >= logairmass_map[0]) and (eval_airmass <= logairmass_map[-1]), (
            "eval_airmass out of bounds!"
        )
        assert (eval_nscale >= lognscale_map[0]) and (eval_nscale <= lognscale_map[-1]), (
            "nscale out of bounds!"
        )

        # Figure out index and slice info...
        # First do airmass
        right_airmass_idx = np.searchsorted(logairmass_map, eval_airmass[0], 'right')
        if right_airmass_idx == len(logairmass_map):
            right_airmass_idx -= 1
        left_airmass_idx = right_airmass_idx - 1
        airmass_slice = slice(left_airmass_idx, right_airmass_idx + 1)
        # Next do nscale
        right_nscale_idx = np.searchsorted(lognscale_map, eval_nscale[0], 'right')
        if right_nscale_idx == len(lognscale_map):
            right_nscale_idx -= 1
        left_nscale_idx = right_nscale_idx - 1
        nscale_slice = slice(left_nscale_idx, right_nscale_idx + 1)

        first_eval = np.stack(
            (
                nscale_cubic_interp_dict[left_airmass_idx](eval_nscale),
                nscale_cubic_interp_dict[right_airmass_idx](eval_nscale),
            ),
            axis=1,
        )

        # Interpolate for nscale Jacobian at the chosen nscale
        mod_jacobian = self._bilinear_jacobian_interp(
            eval_nscale=eval_nscale[0],
            eval_airmass=airmass_slice,
            spectrum=field.airmass
        )

        final_interp_func = CubicHermiteSpline(
            x=logairmass_map[airmass_slice], y=first_eval, dydx=mod_jacobian, axis=1
        )

        na_interp = final_interp_func(eval_airmass)

        # Now map out airmass -> nscale 
        first_eval = np.stack(
            (
                airmass_cubic_interp_dict[left_nscale_idx](eval_airmass),
                airmass_cubic_interp_dict[right_nscale_idx](eval_airmass),
            ),
            axis=0,
        )

        # Interpolate for nscale Jacobian at the chosen nscale
        mod_jacobian = self._bilinear_jacobian_interp(
            eval_nscale=nscale_slice,
            eval_airmass=eval_airmass[0],
            spectrum=field.nscale
        )
        final_interp_func = CubicHermiteSpline(
            x=lognscale_map[nscale_slice], y=first_eval, dydx=mod_jacobian, axis=0
        )

        an_interp = final_interp_func(eval_nscale)

        return (an_interp + na_interp) * 0.5
    
    
    def __call__(
            self,
            eval_pwv,
            eval_za,
            freq_array=None,
            return_array=['Tb', 'dTb_pwv', 'dTb_za']
    ):
        eval_lognscale = utils.pwv_to_log_nscale(eval_pwv, self.am_model.nominal_pwv)
        eval_logairmass = utils.za_to_log_airmass(eval_za)

        return_args = []
        model_spectrum = None
        if 'Tb' in return_array:
            model_spectrum = self._interp_spectrum(
                eval_nscale=[eval_lognscale],
                eval_airmass=[eval_logairmass],
                field=self.am_model.Tb,
                airmass_cubic_interp_dict=self.am_model.Tb.airmass.interp_func.cubic,
                nscale_cubic_interp_dict=self.am_model.Tb.nscale.interp_func.cubic
            ).flatten()
            if freq_array is not None:
                model_spectrum = np.interp(
                     freq_array.flatten(), self.data['freq']['map'], model_spectrum # type: ignore
                ).reshape(freq_array.shape)
            return_args.append(model_spectrum)

        pwv_jacobian = None
        if 'dTb_pwv' in return_array:
            nscale_to_pwv_normalization_factor = 1 / eval_pwv
            pwv_jacobian = self._bilinear_jacobian_interp(
                eval_nscale=[eval_lognscale],
                eval_airmass=[eval_logairmass],
                spectrum=self.am_model.Tb.nscale,
                norm_factor=nscale_to_pwv_normalization_factor
            ).flatten()
            if freq_array is not None:
                pwv_jacobian = np.interp(
                    freq_arr.flatten(), self.data['freq']['map'], pwv_jacobian # type: ignore
                ).reshape(freq_array.shape)
            return_args.append(pwv_jacobian)

        if 'dTb_za' in return_array:
            airmass_to_zenith_normalization_factor = np.tan(eval_za)
            zenith_jacobian = self._bilinear_jacobian_interp(
                eval_nscale=[eval_lognscale],
                eval_airmass=[eval_logairmass],
                spectrum=self.am_model.Tb.airmass,
                norm_factor=airmass_to_zenith_normalization_factor
            ).flatten()
            if freq_array is not None:
                zenith_jacobian = np.interp(
                    freq_arr.flatten(), self.data['freq']['map'], zenith_jacobian # type: ignore
                ).reshape(freq_array.shape)
            return_args.append(zenith_jacobian)

        return return_args
    
