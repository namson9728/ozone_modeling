import numpy as np

class Ozone:
    def __init__(self, am_model_data_path):
        self.am_model_data_path = am_model_data_path
        self.data = self._load_model_data()
        self.nscale = None
        self.airmass = None
        self.nominal_pwv = self._extract_nominal_pwv()

        from scipy.interpolate import RegularGridInterpolator
        from scipy.interpolate import CubicHermiteSpline

        Nscale_map = self.data['Nscale']['map'][::2]
        Tb_scalar_field = self.data['Tb_scalar_field'][::2,::2]
        Nscale_jacobian = self.data['Nscale']['jacobian'][::2,::2]

        self.NscaleRegularGridInterp_func = RegularGridInterpolator(
            points=(self.data['Nscale']['map'], self.data['airmass']['map'], self.data['freq']['map']), 
            values=self.data['Nscale']['jacobian'], method="linear"
        )

        self.AirmassRegularGridInterp_func = RegularGridInterpolator(
            points=(self.data['Nscale']['map'], self.data['airmass']['map'], self.data['freq']['map']), 
            values=self.data['airmass']['jacobian'], method="linear"
        )

        self.CubicHermiteSplineInterp_func = CubicHermiteSpline(
            x=Nscale_map,
            y=Tb_scalar_field,
            dydx=Nscale_jacobian,
            axis=0,
        )

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
    
    def _2DCubicHermiteSpline(self, eval_airmass, eval_nscale, init_interp_func, data_dict):
        '''Returns the interpolation of the data given an airmass and nscale value

        `reverse` - set True to reverse the order of operation
        '''
        from scipy.interpolate import RegularGridInterpolator
        from scipy.interpolate import CubicHermiteSpline

        airmass_map = self.data['airmass']['map'][::2]
        Nscale_map = self.data['Nscale']['map'][::2]
        freq_map = self.data['freq']['map']
        airmass_jacobian = self.data['airmass']['jacobian'][::2,::2]

        first_eval = init_interp_func(eval_nscale)

        # Interpolate for nscale Jacobian at the chosen airmass
        jacob_interp_func = RegularGridInterpolator(
            points=(Nscale_map, airmass_map, freq_map),
            values=airmass_jacobian,
            method="linear",
        )

        x,y,z = np.meshgrid(
            eval_nscale,
            airmass_map,
            freq_map,
            indexing='ij',
        )

        mod_jacobian = jacob_interp_func(
            (x.flatten(),y.flatten(),z.flatten())
        ).reshape(x.shape)

        final_interp_func = CubicHermiteSpline(
            x=airmass_map,
            y=first_eval,
            dydx=mod_jacobian,
            axis=1,
        )

        return final_interp_func(eval_airmass)
    
    def _2DRegularGridInterpolator(self, eval_airmass, eval_nscale, interp_func, normalization_factor=None):

        x,y,z = np.meshgrid(eval_nscale, eval_airmass, self.data['freq']['map'], indexing='ij')

        spectrum = interp_func((x.flatten(),y.flatten(),z.flatten())).reshape(x.shape)[0,0]
        
        if normalization_factor is not None:
            return spectrum * normalization_factor

        return spectrum

    def _extract_nominal_pwv(self):
        err_file = f'{self.data['airmass']['map'][0]:.3f}_{self.data['Nscale']['map'][0]:.2f}'
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

        nscale = err_file.partition(".err")
        idx, val = 0, ''
        while nscale[0][-idx] != '_': val = nscale[0][-idx] + val; idx += 1
        nscale = float(val.partition("/")[0])

        return np.round((pwv*10**-3)/(10**nscale), 2)
    
    def _zenith_to_airmass(self, zenith):
        return 1/np.cos(zenith)
    
    def _airmass_to_zenith(self, airmass):
        return np.arccos(1/airmass)

    def __call__(self, pwv, zenith, return_model_spectrum=True, return_pwv_jacobian=False, return_zenith_jacobian=False, toString=False):
        self.nscale = np.log10(pwv / self.nominal_pwv)
        self.airmass = self._zenith_to_airmass(zenith)

        if toString:
            print(f"PWV -> nscale: {self.nscale:.2f}")
            print(f"zenith -> airmass: {self.airmass:.2f}")

        model_spectrum = None
        if return_model_spectrum:
            model_spectrum = self._2DCubicHermiteSpline(
                eval_airmass=[self.airmass],
                eval_nscale=[self.nscale],
                data_dict=self.data,
                init_interp_func=self.CubicHermiteSplineInterp_func
        )
        pwv_jacobian = None
        if return_pwv_jacobian:
            nscale_to_pwv_normalization_factor = 1 / (np.log(10)*pwv)
            pwv_jacobian = self._2DRegularGridInterpolator(
            eval_airmass=self.airmass,
            eval_nscale=self.nscale,
            interp_func=self.NscaleRegularGridInterp_func,
            normalization_factor=nscale_to_pwv_normalization_factor
        )
        zenith_jacobian = None
        if return_zenith_jacobian:
            airmass_to_zenith_normalization_factor = (1/np.cos(zenith))*np.tan(zenith)
            zenith_jacobian = self._2DRegularGridInterpolator(
            eval_airmass=self.airmass,
            eval_nscale=self.nscale,
            interp_func=self.AirmassRegularGridInterp_func,
            normalization_factor=airmass_to_zenith_normalization_factor
        )

        return model_spectrum, pwv_jacobian, zenith_jacobian