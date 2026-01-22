import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicHermiteSpline

def zenith_to_airmass(zenith):
        """Returns the provided zenith angle in radians to the equivalent airmass.
        """
        return 1/np.cos(zenith)
    
def airmass_to_zenith(airmass):
    """Returns the provided airmass into the equivalent zenith angle in radians.
    """
    return np.arccos(1/airmass)

def log_to_linear_airmass_factor(logairmass):
    return 1 / (np.sqrt(np.exp(2*logairmass) - 1))

def log_to_linear_nscale_factor(lognscale):
    return np.exp(lognscale)

def pwv_to_log_nscale(pwv, nominal_pwv):
    return np.log(pwv / nominal_pwv)

def za_to_log_airmass(za):
    return np.log(zenith_to_airmass(za))

def init_RegularGridInterpolator(model, Jacobian):
        """Documentation needed
        """
        lognscale_map = model.lognscale_map
        logairmass_map = model.logairmass_map
        freq_map = model.freq_map

        interp_func = RegularGridInterpolator(
            points=(lognscale_map, logairmass_map, freq_map), 
            values=Jacobian, 
            method="linear"
        )
        
        return interp_func

def init_CubicHermiteSpline(field_map, scalar_field, Jacobian):
        """
        """
        interp_func = CubicHermiteSpline(
            x=field_map,
            y=scalar_field,
            dydx=Jacobian,
            axis=0,
        )

        return interp_func

def _bilinear_jacobian_interp(
        self, eval_nscale: float | slice, eval_airmass: float | slice, field, scalar_field: str, norm_factor=None
    ):
        """Returns the interpolation spectrum using the provided 2DRegularGrid interpolation function.

        A normalization factor can be provided which is multiplied to the entire spectrum.
        """
        if not any(scalar_field == s for s in ['dTb', 'dTau', 'dL']):
            raise TypeError("scalar_field must be either \'dTb\' or \'dTau\' or \'dL\'")
        data = self.data[field][scalar_field]
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

def _interp_spectrum(self, eval_airmass, eval_nscale, scalar_field: str, 
                         airmass_cubic_interp_dict, nscale_cubic_interp_dict):
        '''Returns the interpolation of the data given an airmass and nscale value.

        Args:
            eval_airmass : int
                The airmass value at which the interpolation will evaluate.
            eval_nscale : int
                The nscale value at which the interpolation will evaluate.
        '''
        # But we _only_ need the nearest two airmasses for the interpolation
        airmass_map = self.data['airmass']['map']
        nscale_map = self.data['Nscale']['map']
        assert (eval_airmass >= airmass_map[0]) and (eval_airmass <= airmass_map[-1]), (
            "eval_airmass out of bounds!"
        )
        assert (eval_nscale >= nscale_map[0]) and (eval_nscale <= nscale_map[-1]), (
            "nscale out of bounds!"
        )

        # Figure out index and slice info...
        # First do airmass
        right_airmass_idx = np.searchsorted(airmass_map, eval_airmass[0], 'right')
        if right_airmass_idx == len(airmass_map):
            right_airmass_idx -= 1
        left_airmass_idx = right_airmass_idx - 1
        airmass_slice = slice(left_airmass_idx, right_airmass_idx + 1)
        # Next do nscale
        right_nscale_idx = np.searchsorted(nscale_map, eval_nscale[0], 'right')
        if right_nscale_idx == len(nscale_map):
            right_nscale_idx -= 1
        left_nscale_idx = right_nscale_idx - 1
        nscale_slice = slice(left_nscale_idx, right_nscale_idx + 1)

        first_eval = np.concatenate(
            (
                nscale_cubic_interp_dict[left_airmass_idx](eval_nscale),
                nscale_cubic_interp_dict[right_airmass_idx](eval_nscale),
            ),
            axis=1,
        )

        if not any(scalar_field == s for s in ['dTb', 'dTau', 'dL']):
            raise TypeError("scalar_field must be either \'dTb\' or \'dTau\' or \'dL\'")
        # Interpolate for nscale Jacobian at the chosen nscale
        mod_jacobian = self._bilinear_jacobian_interp(
            eval_nscale=eval_nscale[0],
            eval_airmass=airmass_slice,
            scalar_field=scalar_field,
            field="airmass"
        )
        final_interp_func = CubicHermiteSpline(
            x=airmass_map[airmass_slice], y=first_eval, dydx=mod_jacobian, axis=1
        )

        na_interp = final_interp_func(eval_airmass)

        # Now map out airmass -> nscale 
        first_eval = np.concatenate(
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
            scalar_field=scalar_field,
            field="Nscale"
        )
        final_interp_func = CubicHermiteSpline(
            x=nscale_map[nscale_slice], y=first_eval, dydx=mod_jacobian, axis=0
        )

        an_interp = final_interp_func(eval_nscale)

        return (an_interp + na_interp) * 0.5