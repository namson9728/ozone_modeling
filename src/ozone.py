import numpy as np

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
        `NscaleRegularGridInterp_func` : RegularGridInterpolator
            The interpolation function for nscale that is used to calculate the PWV Jacobian.
        `AirmassRegularGridInterp_func` : RegularGridInterpolator
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
    def __init__(self, am_model_data_path:str):
        self.am_model_data_path = am_model_data_path
        self.data = self._load_model_data()
        self.nscale = None
        self.airmass = None
        self.nominal_pwv = self._extract_nominal_pwv()

        from scipy.interpolate import RegularGridInterpolator
        from scipy.interpolate import CubicHermiteSpline

        Nscale_map = self.data['Nscale']['map']
        Tb_scalar_field = self.data['Tb_scalar_field']
        Nscale_jacobian = self.data['Nscale']['jacobian']

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
        freq_points = 240001

        logNscale_map = np.linspace(min_logNscale, max_logNnscale, logNscale_points)
        logairmass_map = np.linspace(min_logairmass, max_logairmass, logairmass_points)

        Tb_scalar_field = np.zeros((logNscale_points, logairmass_points, freq_points))
        Nscale_jacobian = np.zeros((logNscale_points, logairmass_points, freq_points))
        airmass_jacobian = np.zeros((logNscale_points, logairmass_points, freq_points))

        for idx, logNscale in enumerate(logNscale_map):
            for jdx, logairmass in enumerate(logairmass_map):

                filename = f'MaunaKea_Tb_Spectrum_{logairmass:.3f}_{logNscale:+.3f}'
                data = np.load(f'{self.am_model_data_path}{filename}.out')

                freq_map = data[:,0]
                Tb_scalar_field[idx,jdx] = data[:,2]
                airmass_jacobian[idx,jdx] = data[:,3] / np.sqrt(np.exp(2*logairmass)-1)
                Nscale_jacobian[idx,jdx] = data[:,4] * np.exp(logNscale)

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
        from scipy.interpolate import RegularGridInterpolator
        from scipy.interpolate import CubicHermiteSpline

        airmass_map = self.data['airmass']['map']
        Nscale_map = self.data['Nscale']['map']
        freq_map = self.data['freq']['map']
        airmass_jacobian = self.data['airmass']['jacobian']

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
    
    def _2DRegularGridInterpolator(self, eval_airmass:int, eval_nscale:int, interp_func, normalization_factor=None):
        """Returns the interpolation spectrum using the provided 2DRegularGrid interpolation function.

        A normalization factor can be provided which is multiplied to the entire spectrum.
        """

        x,y,z = np.meshgrid(eval_nscale, eval_airmass, self.data['freq']['map'], indexing='ij')

        spectrum = interp_func((x.flatten(),y.flatten(),z.flatten())).reshape(x.shape)[0,0]
        
        if normalization_factor is not None:
            return spectrum * normalization_factor

        return spectrum

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

        model_spectrum = None
        if return_model_spectrum:
            model_spectrum = self._2DCubicHermiteSpline(
                eval_airmass=[self.logairmass],
                eval_nscale=[self.lognscale],
                init_interp_func=self.CubicHermiteSplineInterp_func
        )
        pwv_jacobian = None
        if return_pwv_jacobian:
            nscale_to_pwv_normalization_factor = 1 / pwv
            pwv_jacobian = self._2DRegularGridInterpolator(
            eval_airmass=self.logairmass,
            eval_nscale=self.lognscale,
            interp_func=self.NscaleRegularGridInterp_func,
            normalization_factor=nscale_to_pwv_normalization_factor
        )
        zenith_jacobian = None
        if return_zenith_jacobian:
            airmass_to_zenith_normalization_factor = np.tan(zenith)
            zenith_jacobian = self._2DRegularGridInterpolator(
            eval_airmass=self.logairmass,
            eval_nscale=self.lognscale,
            interp_func=self.AirmassRegularGridInterp_func,
            normalization_factor=airmass_to_zenith_normalization_factor
        )

        return model_spectrum, pwv_jacobian, zenith_jacobian