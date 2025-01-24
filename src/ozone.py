import numpy as np

class Ozone:
    def __init__(self, am_model_data_path):
        self.am_model_data_path = am_model_data_path
        self.data = self._load_model_data()
        self.nscale = None
        self.airmass = None

    def _load_model_data(self):
        min_nscale, max_nscale, Nscale_points = -1.0, 1.0, 21
        min_airmass, max_airmass, airmass_points = 1.001, 4.001, 11
        freq_points = 240001

        Nscale_map = np.linspace(min_nscale, max_nscale, Nscale_points)
        airmass_map = np.linspace(min_airmass, max_airmass, airmass_points)

        Tb_scalar_field = np.zeros((Nscale_points, airmass_points, freq_points))
        Nscale_jacobian = np.zeros((Nscale_points, airmass_points, freq_points))
        za_jacobian = np.zeros((Nscale_points, airmass_points, freq_points))
        airmass_jacobian = np.zeros((Nscale_points, airmass_points, freq_points))

        for idx, Nscale in enumerate(Nscale_map):
            for jdx, airmass in enumerate(airmass_map):

                filename = f'MaunaKea_Tb_Spectrum_{airmass:.3f}_{Nscale:.2f}'
                data = np.load(f'{self.am_model_data_path}{filename}.out')

                freq_map = data[:,0]
                
                Tb_scalar_field[idx,jdx] = data[:,2]

                Nscale_jacobian[idx,jdx] = data[:,4] * (np.log(10) * (10 ** Nscale))
                za_jacobian[idx, jdx] = data[:,3]
                airmass_jacobian[idx,jdx] = za_jacobian[idx, jdx] / (airmass * np.sqrt((airmass**2) - 1))

        return {'airmass':{
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
    
    def _DD_CubicHermiteSpline(self, eval_airmass, eval_nscale, data_dict, reverse=False):
        '''Returns the interpolation of the data given an airmass and nscale value

        `reverse` - set True to reverse the order of operation
        '''
        from scipy.interpolate import CubicHermiteSpline
        from scipy.interpolate import RegularGridInterpolator

        Nscale_map = data_dict['Nscale']['map'][::2]
        Tb_scalar_field = data_dict['Tb_scalar_field'][::2,::2]
        Nscale_jacobian = data_dict['Nscale']['jacobian'][::2,::2]
        airmass_map = data_dict['airmass']['map'][::2]
        freq_map = data_dict['freq']['map']
        airmass_jacobian = data_dict['airmass']['jacobian'][::2,::2]

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

    def _extract_nominal_pwv(self):
        err_file = "/Users/namsonnguyen/repo/AM_Data/MaunaKea_SON50/Nscale21_AirMass11/MaunaKea_Tb_Spectrum_1.001_-0.05.err"

        with open(err_file, "r") as file:
            text_data = file.readlines()

        look_for_pwv = False
        for idx in range(len(text_data)):
            if "total" in text_data[idx]: look_for_pwv = True
            if look_for_pwv:
                if "um_pwv" in text_data[idx]: pwv = text_data[idx] 

        pwv = pwv.partition('(')
        pwv = float(pwv[2].partition(' ')[0])

        nscale = err_file.partition(".err")
        idx, val = 0, ''
        while nscale[0][-idx] != '_': val = nscale[0][-idx] + val; idx += 1
        nscale = float(val.partition("/")[0])

        return np.round((pwv*10**-3)/(10**nscale), 2)
    
    def _zenith_to_airmass(self, zenith):
        return 1/np.cos(zenith)
    
    def _airmass_to_zenith(self, airmass):
        return 1/np.cos(airmass)

    def __call__(self, pwv, zenith):
        nominal_pwv = self._extract_nominal_pwv()
        self.nscale = pwv / nominal_pwv

        self.airmass = self._zenith_to_airmass(zenith)

        print(f"PWV -> nscale: {self.nscale:.2f}")
        print(f"zenith -> airmass: {self.airmass:.2f}")

        return self._DD_CubicHermiteSpline(
            eval_airmass = [self.airmass],
            eval_nscale = [self.nscale],
            data_dict = self.data,
            reverse=False
        )