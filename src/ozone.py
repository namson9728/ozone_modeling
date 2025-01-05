import numpy as np

class Nscale:
    def __init__(self, min_val, max_val, points):
        self._min = min_val
        self._max = max_val
        self._points = points

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

class Airmass:
    def __init__(self, min_val, max_val, points):
        self._min = min_val
        self._max = max_val
        self._points = points

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


class Ozone:
    def __init__(self, nscale, airmass):
        self.nscale = nscale
        self.airmass = airmass

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

    def __str__(self):
        return f"Ozone Object in development!"