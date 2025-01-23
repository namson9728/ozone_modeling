import numpy as np

def DD_CubicHermiteSpline(eval_airmass, eval_nscale, df, reverse=False):
    '''Returns the interpolation of the data given an airmass and nscale value

    `reverse` - set True to reverse the order of operation
    '''

    from scipy.interpolate import CubicHermiteSpline
    from scipy.interpolate import RegularGridInterpolator

    Nscale_map = df.nscale.map[::2]
    Tb_scalar_field = df.data['TB_SCALAR_FIELD'][::2,::2]
    Nscale_jacobian = df.data['NSCALE_JACOBIAN'][::2,::2]
    airmass_map = df.airmass.map[::2]
    freq_map = df.data['FREQUENCY']
    airmass_jacobian = df.data['AIRMASS_JACOBIAN'][::2,::2]

    init_interp_func = CubicHermiteSpline(
        x=Nscale_map if reverse else airmass_map,
        y=Tb_scalar_field,
        dydx=Nscale_jacobian if reverse else airmass_jacobian,
        axis=0 if reverse else 1,
    )

    first_eval = init_interp_func(eval_nscale if reverse else eval_airmass)

    # Interpolate for nscale Jacobian at the chosen airmass
    jacob_interp_func = RegularGridInterpolator(
        points=(Nscale_map, airmass_map, freq_map),
        values=airmass_jacobian if reverse else Nscale_jacobian,
        method="linear",
    )

    x,y,z = np.meshgrid(
        eval_nscale if reverse else Nscale_map,
        airmass_map if reverse else eval_airmass,
        freq_map,
        indexing='ij',
    )

    mod_jacobian = jacob_interp_func(
        (x.flatten(),y.flatten(),z.flatten())
    ).reshape(x.shape)

    final_interp_func = CubicHermiteSpline(
        x=airmass_map if reverse else Nscale_map,
        y=first_eval,
        dydx=mod_jacobian,
        axis=1 if reverse else 0,
    )

    return final_interp_func(eval_airmass if reverse else eval_nscale)