import numpy as np

class Ozone:
    def __init__(self, nscale, airmass, frequency):
        self.nscale = nscale
        self.airmass = airmass
        self.frequency = frequency
        self._data = None

    @property
    def nscale(self):
        return self._nscale

    @nscale.setter
    def nscale(self, nscale):
        if type(nscale) != tuple:
            raise ValueError("nscale must be a tuple containing the min, max, and num_points nscale values respectively")
        elif len(nscale) != 3:
            raise ValueError(f"Unexpected nscale tuple length. Expected length 3 but got length {len(nscale)}")
        else:
            self._nscale = Nscale(nscale[0], nscale[1], nscale[2])

    @property
    def airmass(self):
        return self._airmass

    @airmass.setter
    def airmass(self, airmass):
        if type(airmass) != tuple:
            raise ValueError("airmass must be a tuple containing the min, max, and num_points airmass values respectively")
        elif len(airmass) != 3:
            raise ValueError(f"Unexpected airmass tuple length. Expected length 3 but got length {len(airmass)}")
        else:
            self._airmass= Airmass(airmass[0], airmass[1], airmass[2])

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, frequency):
        self._frequency = Frequency(frequency)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if self._data is None:
            self._data = {}
            self._data['RAW'] = data
            shape = (self._nscale.points, self._airmass.points, self._frequency.points)
            Tb_scalar_field = np.zeros(shape)
            nscale_jacobian = np.zeros(shape)
            za_jacobian = np.zeros(shape)
            airmass_jacobian = np.zeros(shape)

            for idx, nscale in enumerate(self._nscale.map):
                for jdx, airmass in enumerate(self._airmass.map):
                    freq_map = self._data['RAW'][idx,jdx,:,0]
                    Tb_scalar_field[idx,jdx] = self._data['RAW'][idx,jdx,:,2]
                    nscale_jacobian[idx,jdx] = self._data['RAW'][idx,jdx,:,4] * (np.log(10) * (10 ** nscale))
                    za_jacobian[idx, jdx] = self._data['RAW'][idx,jdx,:,3]
                    airmass_jacobian[idx,jdx] = za_jacobian[idx,jdx] / (airmass * np.sqrt((airmass**2) - 1))

        self._data['NSCALE_JACOBIAN'] = nscale_jacobian
        self._data['ZENITH_JACOBIAN'] = za_jacobian
        self._data['AIRMASS_JACOBIAN'] = airmass_jacobian
        self._data['TB_SCALAR_FIELD'] = Tb_scalar_field
        self._data['FREQUENCY'] = freq_map

    def __call__(self, zenith, pwv):
        return "Will return the atmospheric model at specified argument"

    def __str__(self):
        return f"Ozone Object in development!"
    
class Nscale():
    def __init__(self, min_val, max_val, points):
        self.min = min_val
        self.max = max_val
        self.points = points
        self._map = None

    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, min_val):
        self._min = min_val

    @property
    def max(self):
        return self._max
    
    @max.setter
    def max(self, max_val):
        self._max = max_val

    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = points
    
    @property
    def map(self):
        if self._map is None:
            self._map = np.linspace(self._min, self._max, self._points)
        return self._map

    @property
    def jacobian(self):
        return self._map

class Airmass(Ozone):
    def __init__(self, min_val, max_val, points):
        self.min = min_val
        self.max = max_val
        self.points = points
        self._map = None

    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, min_val):
        self._min = min_val

    @property
    def max(self):
        return self._max
    
    @max.setter
    def max(self, max_val):
        self._max = max_val

    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = points

    @property
    def map(self):
        if self._map is None:
            self._map = np.linspace(self._min, self._max, self._points)
        return self._map

class Frequency(Ozone):
    def __init__(self, points):
        self.points = points

    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = points