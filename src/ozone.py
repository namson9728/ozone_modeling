import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicHermiteSpline
class Ozone:
    """Ozone model object that stores the AM generated data.

    Attributes:
        `am_model_data_path` : str
            The path where the AM generated data is stored.
        `data` : dict
            A dictionary containing the AM generated data.
        `nscale` : float
            The nscale value used to generate the AM model.
        `airmass` : float
            The airmass value used to generate the AM model.
        `nominal_pwv` : float
            The normalization factor to convert from nscale to pwv.
        `lognscale_jacobian_interp_func` : RegularGridInterpolator
            The interpolation function for nscale that is used to calculate the PWV Jacobian.
        `logairmass_jacobian_interp_func` : RegularGridInterpolator
            The interpolation function for airmass that is used to calculate the zenith angle Jacobian.
        `CubicHermiteSplineInterp_func` : CubicHermiteSpline
            The interpolation function that is used to calculate the model Tb spectrum at `nscale` and `airmas`

    Typical Usage Example:
        ```
        my_ozone = Ozone(am_model_data_path=data_path)
        ```

    Format of `data` dictionary:
    ```
    {'airmass':{
        'map':airmass_map,
        'jacobian':airmass_jacobian,
        'points':airmass_points
    },
    'Nscale':{
        'map':Nscale_map,
        'jacobian':Nscale_jacobian,
        'points':Nscale_points
    },
    'freq':{
        'map':freq_map,
        'points':freq_points
    },
    'Tb_scalar_field':Tb_scalar_field
    }
    ```
    """
    def __init__(self, am_model_data_path:str, freq_range=None):
        self.am_model_data_path = am_model_data_path
        self.data = self._load_model_data(freq_range=freq_range)
        self.lognscale = None
        self.logairmass = None
        self.nominal_pwv = self._extract_nominal_pwv()

        Nscale_map = self.data['Nscale']['map']
        Tb_scalar_field = self.data['Tb_scalar_field']
        Nscale_jacobian = self.data['Nscale']['jacobian']

        self.lognscale_jacobian_interp_func = RegularGridInterpolator(
            points=(self.data['Nscale']['map'], self.data['airmass']['map'], self.data['freq']['map']), 
            values=self.data['Nscale']['jacobian'], method="linear"
        )

        self.logairmass_jacobian_interp_func = RegularGridInterpolator(
            points=(self.data['Nscale']['map'], self.data['airmass']['map'], self.data['freq']['map']), 
            values=self.data['airmass']['jacobian'], method="linear"
        )

        self.CubicHermiteSplineInterp_func = CubicHermiteSpline(
            x=Nscale_map,
            y=Tb_scalar_field,
            dydx=Nscale_jacobian,
            axis=0,
        )

        self.cubic_interp_dict = {}
        for idx in range(len(self.data['airmass']['map'])):
            self.cubic_interp_dict[idx] = CubicHermiteSpline(
                x=Nscale_map,
                y=Tb_scalar_field[:, idx:idx+1, :],
                dydx=Nscale_jacobian[:, idx:idx+1, :],
                axis=0,
            )

    def _load_model_data(self, freq_range=None):
        """Returns the AM generated data stored in a dictionary.

        The dictionary is in the following format:
        ```
        {'airmass':{
            'map':airmass_map,
            'jacobian':airmass_jacobian,
            'points':airmass_points
        },
        'Nscale':{
            'map':Nscale_map,
            'jacobian':Nscale_jacobian,
            'points':Nscale_points
        },
        'freq':{
            'map':freq_map,
            'points':freq_points
        },
        'Tb_scalar_field':Tb_scalar_field
        }
        ```
        """
        min_logNscale, max_logNnscale, logNscale_points = -1.0, 1.0, 9
        min_logairmass, max_logairmass, logairmass_points = 0.001, 1.281, 9

        logNscale_map = np.linspace(min_logNscale, max_logNnscale, logNscale_points)
        logairmass_map = np.linspace(min_logairmass, max_logairmass, logairmass_points)
        freq_mask = ...
        init_arr = True

        for idx, logNscale in enumerate(logNscale_map):
            for jdx, logairmass in enumerate(logairmass_map):

                filename = f'MaunaKea_Tb_Spectrum_{logairmass:.3f}_{logNscale:+.3f}'
                data = np.load(f'{self.am_model_data_path}{filename}.out')

                if init_arr:
                    freq_map = data[:, 0]
                    if freq_range is not None:
                        freq_mask = (freq_map >= freq_range[0]) & (freq_map <= freq_range[1])
                    freq_map = freq_map[freq_mask]

                    freq_points = len(freq_map)
                    Tb_scalar_field = np.zeros((logNscale_points, logairmass_points, freq_points))
                    Nscale_jacobian = np.zeros((logNscale_points, logairmass_points, freq_points))
                    airmass_jacobian = np.zeros((logNscale_points, logairmass_points, freq_points))
                    init_arr = False

                Tb_scalar_field[idx,jdx] = data[freq_mask, 2]
                airmass_jacobian[idx,jdx] = data[freq_mask,3] / np.sqrt(np.exp(2*logairmass)-1)
                Nscale_jacobian[idx,jdx] = data[freq_mask, 4] * np.exp(logNscale)

        return {'airmass':{
                'map':logairmass_map,
                'jacobian':airmass_jacobian,
                'points':logairmass_points
            },
            'Nscale':{
                'map':logNscale_map,
                'jacobian':Nscale_jacobian,
                'points':logNscale_points
            },
            'freq':{
                'map':freq_map,
                'points':freq_points
            },
            'Tb_scalar_field':Tb_scalar_field
            }
    
    def _2DCubicHermiteSpline(self, eval_airmass, eval_nscale, init_interp_func):
        '''Returns the interpolation of the data given an airmass and nscale value.

        Args:
            eval_airmass : int
                The airmass value at which the interpolation will evaluate.
            eval_nscale : int
                The nscale value at which the interpolation will evaluate.
            init_interp_func : CubicHermiteSpline
                The Cubic Hermite Spline interpolation function initialized upon object creation.
        '''
        # We want to do the whole freq spectrum (for now)
        freq_map = self.data['freq']['map']

        # But we _only_ need the nearest two airmasses for the interpolation
        airmass_map = self.data['airmass']['map']
        first_idx = np.nonzero(airmass_map <= eval_airmass)[0][-1]
        last_idx = np.nonzero(airmass_map >= eval_airmass)[0][0] + 2
        if last_idx > len(airmass_map):
            airmass_slice = slice(first_idx - 1, last_idx - 1)
        else:
            airmass_slice = slice(first_idx, last_idx)

        first_eval = init_interp_func(eval_nscale)[:, airmass_slice, :]

        # Interpolate for nscale Jacobian at the chosen airmass
        x,y,z = np.meshgrid(
            eval_nscale,
            airmass_map[airmass_slice],
            freq_map,
            indexing='ij',
        )

        mod_jacobian = self.logairmass_jacobian_interp_func(
            (x.flatten(),y.flatten(),z.flatten())
        ).reshape(x.shape)

        final_interp_func = CubicHermiteSpline(
            x=airmass_map[airmass_slice], y=first_eval, dydx=mod_jacobian, axis=1
        )

        return final_interp_func(eval_airmass)
    
    def _2DRegularGridInterpolator(self, eval_airmass:int, eval_nscale:int, interp_func, normalization_factor=None):
        """Returns the interpolation spectrum using the provided 2DRegularGrid interpolation function.

        A normalization factor can be provided which is multiplied to the entire spectrum.
        """

        x,y,z = np.meshgrid(eval_nscale, eval_airmass, self.data['freq']['map'], indexing='ij')

        spectrum = interp_func((x.flatten(),y.flatten(),z.flatten())).reshape(x.shape)[0,0]
        
        if normalization_factor is not None:
            return spectrum * normalization_factor

        return spectrum

    def _bilinear_jacobian_interp(
        self, eval_nscale: float | slice, eval_airmass: float | slice, field, norm_factor=None
    ):
        """Returns the interpolation spectrum using the provided 2DRegularGrid interpolation function.

        A normalization factor can be provided which is multiplied to the entire spectrum.
        """
        data = self.data[field]['jacobian']
        index_dict = {"Nscale": [eval_nscale],  "airmass": [eval_airmass]}
        weights_dict = {"Nscale": [1], "airmass": [1]}

        for item, eval_val in zip(["Nscale", "airmass"], [eval_nscale, eval_airmass]):
            if isinstance(eval_val, slice):
                continue
            axis_map = self.data[item]['map']
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
        nscale_idx = index_dict["Nscale"]
        airmass_idx = index_dict["airmass"]
        set_result = True
        if (len(airmass_idx) == 1) and (len(nscale_idx) == 1):
            result = data[nscale_idx[0], airmass_idx[0]]
        else:
            for n_idx, n_weight in zip(nscale_idx, weights_dict["Nscale"]):
                for a_idx, a_weight in zip(airmass_idx, weights_dict["airmass"]):
                    if set_result:
                        result = data[n_idx, a_idx] * (a_weight * n_weight)
                        set_result = False
                    else:
                        result += data[n_idx, a_idx] * (a_weight * n_weight)
        if not isinstance(nscale_idx[0], slice):
            result = result[None]
        if not isinstance(airmass_idx[0], slice):
            result = result[:, None]

        return result if norm_factor is None else (result * norm_factor)

    def _interp_spectrum(self, eval_airmass, eval_nscale):
        '''Returns the interpolation of the data given an airmass and nscale value.

        Args:
            eval_airmass : int
                The airmass value at which the interpolation will evaluate.
            eval_nscale : int
                The nscale value at which the interpolation will evaluate.
        '''
        # But we _only_ need the nearest two airmasses for the interpolation
        airmass_map = self.data['airmass']['map']
        assert (eval_airmass >= airmass_map[0]) and (eval_airmass <= airmass_map[-1]), (
            "eval_airmass out of bounds!"
        )
        right_airmass_idx = np.searchsorted(airmass_map, eval_airmass[0], 'right')
        if right_airmass_idx == len(airmass_map):
            right_airmass_idx -= 1
        left_airmass_idx = right_airmass_idx - 1
        airmass_slice = slice(left_airmass_idx, right_airmass_idx + 1)

        first_eval = np.concatenate(
            (
                self.cubic_interp_dict[left_airmass_idx](eval_nscale),
                self.cubic_interp_dict[right_airmass_idx](eval_nscale),
            ),
            axis=1,
        )

        # Interpolate for nscale Jacobian at the chosen nscale
        mod_jacobian = self._bilinear_jacobian_interp(
            eval_nscale=eval_nscale[0],
            eval_airmass=airmass_slice,
            field="airmass"
        )
        final_interp_func = CubicHermiteSpline(
            x=airmass_map[airmass_slice], y=first_eval, dydx=mod_jacobian, axis=1
        )

        return final_interp_func(eval_airmass)
    
    def _extract_nominal_pwv(self):
        """Returns the nominal pwv value extracted from one of the AM .err files.
        """
        lognscale = self.data['Nscale']['map'][0]
        logairmass = self.data['airmass']['map'][0]
        err_file = f'{logairmass:.3f}_{lognscale:+.3f}'
        err_file = f"{self.am_model_data_path}MaunaKea_Tb_Spectrum_{err_file}.err"

        with open(err_file, "r") as file:
            text_data = file.readlines()

        look_for_pwv = False
        for idx in range(len(text_data)):
            if "total" in text_data[idx]: look_for_pwv = True
            if look_for_pwv:
                if "um_pwv" in text_data[idx]: pwv = text_data[idx] 

        pwv = pwv.partition('(')
        pwv = float(pwv[2].partition(' ')[0])

        return (pwv*10**-3)/np.exp(lognscale)
    
    def _zenith_to_airmass(self, zenith):
        """Returns the provided zenith angle in radians to the equivalent airmass.
        """
        return 1/np.cos(zenith)
    
    def _airmass_to_zenith(self, airmass):
        """Returns the provided airmass into the equivalent zenith angle in radians.
        """
        return np.arccos(1/airmass)

    def __call__(
            self,
            pwv,
            zenith,
            freq_arr=None,
            return_model_spectrum=True,
            return_pwv_jacobian=False,
            return_zenith_jacobian=False,
            toString=False,
        ):
        """Returns the model Tb spectrum, PWV Jacobian, and zenith angle Jacobian given a pwv and zenith angle value.

        The code will convert the pwv and zenith angle and initialize the model's nscale and airmass attributes.
        In addition to returning the spectrum and Jacobians, the call will also store the results in the model.

        Args:
            pwv : float
                The percentile water vapor value at which the AM model will be interpolated.
            zenith : float
                The zenith angle value at which the AM model will be interpolated.
            return_model_spectrum : bool 
                Set this value to `False` to save computational power if the Tb spectrum is not needed.
                (Default `True`)
            return_pwv_jacobian : bool
                Set this value to `True` to have the call return a calculated PWV Jacobian.
                (Default `False`)
            return_zenith_jacobian : bool
                Set this value to `True` to have the call return a calculated zenith angle Jacobian.
                (Default `False`)
            toString : bool
                Set this value to `True` to have the call print out the converted NScale and airmass value.
                (Default `False`)
        
        Returns:
            The Tb model spectrum, PWV Jacobian, and zenith angle Jacobian.

        **Please note** that even if `return_pwv_jacobian` and `return_zenith_jacobian` are set to `False`, the code will still return the variables as `None`.
        """
        self.lognscale = np.log(pwv / self.nominal_pwv)
        self.logairmass = np.log(self._zenith_to_airmass(zenith))

        if toString:
            print(f"PWV -> nscale: {np.exp(self.lognscale):.2f}")
            print(f"zenith -> airmass: {np.exp(self.logairmass):.2f}")

        return_args = []
        model_spectrum = None
        if return_model_spectrum:
            model_spectrum = self._interp_spectrum(
                eval_airmass=[self.logairmass],
                eval_nscale=[self.lognscale],
            ).flatten()
            if freq_arr is not None:
                model_spectrum = np.interp(
                    self.data['freq']['map'], model_spectrum, freq_arr.flatten() # type: ignore
                ).reshape(freq_arr.shape)
            return_args.append(model_spectrum)
        
        pwv_jacobian = None
        if return_pwv_jacobian:
            nscale_to_pwv_normalization_factor = 1 / pwv
            pwv_jacobian = self._bilinear_jacobian_interp(
                eval_nscale=self.lognscale,
                eval_airmass=self.logairmass,
                field="Nscale",
                norm_factor=nscale_to_pwv_normalization_factor
            ).flatten()
            if freq_arr is not None:
                pwv_jacobian = np.interp(
                    self.data['freq']['map'], pwv_jacobian, freq_arr.flatten() # type: ignore
                ).reshape(freq_arr.shape)
            return_args.append(pwv_jacobian)
        zenith_jacobian = None
        if return_zenith_jacobian:
            airmass_to_zenith_normalization_factor = np.tan(zenith)
            zenith_jacobian = self._bilinear_jacobian_interp(
                eval_nscale=self.lognscale,
                eval_airmass=self.logairmass,
                field="airmass",
                norm_factor=airmass_to_zenith_normalization_factor
            ).flatten()
            if freq_arr is not None:
                zenith_jacobian = np.interp(
                    self.data['freq']['map'], zenith_jacobian, freq_arr.flatten() # type: ignore
                ).reshape(freq_arr.shape)
            return_args.append(zenith_jacobian)

        if len(return_args) == 1:
            return return_args[0]
        else:
            return return_args