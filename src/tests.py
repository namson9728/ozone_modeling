import numpy as np
import unittest
from src.ozone import Ozone

class TestOzone(unittest.TestCase):

    def setUp(self):
        self.n_min, self.n_max, self.n_points, self.n_map = 1, 3, 3, [1, 2, 3]
        self.a_min, self.a_max, self.a_points, self.a_map = 8, 10, 3, [8, 9, 10]

        self.test_object = Ozone(
            nscale=(self.n_min, self.n_max, self.n_points), 
            airmass=(self.a_min, self.a_max, self.a_points), frequency=100
        )

    def test_nscale_min(self):
        expected = self.n_min
        actual = self.test_object.nscale.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_nscale_max(self):
        expected = self.n_max
        actual = self.test_object.nscale.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_nscale_points(self):
        expected = self.n_points
        actual = self.test_object.nscale.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_min(self):
        expected = self.a_min
        actual = self.test_object.airmass.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_max(self):
        expected = self.a_max
        actual = self.test_object.airmass.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_points(self):
        expected = self.a_points
        actual = self.test_object.airmass.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_nscale_min(self):
        expected = 100
        self.test_object.nscale.min = expected
        actual = self.test_object.nscale.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_nscale_max(self):
        expected = 1000
        self.test_object.nscale.max = expected
        actual = self.test_object.nscale.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_nscale_points(self):
        expected = 10000
        self.test_object.nscale.points = expected
        actual = self.test_object.nscale.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_airmass_min(self):
        expected = 100
        self.test_object.airmass.min = expected
        actual = self.test_object.airmass.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_airmass_max(self):
        expected = 1000
        self.test_object.airmass.max = expected
        actual = self.test_object.airmass.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_airmass_points(self):
        expected = 10000
        self.test_object.airmass.points = expected
        actual = self.test_object.airmass.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_frequency_points(self):
        expected = 100
        actual = self.test_object.frequency.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_set_frequency_points(self):
        expected = 1000
        self.test_object.frequency.points = expected
        actual = self.test_object.frequency.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_get_nscale_map(self):
        expected = self.n_map
        actual = self.test_object.nscale.map
        unittest.TestCase.assertIsNone(
            np.testing.assert_array_equal(expected, actual), None
        )
    
    def test_get_airmass_map(self):
        expected = self.a_map
        actual = self.test_object.airmass.map
        unittest.TestCase.assertIsNone(
            np.testing.assert_array_equal(expected, actual), None
        )


if __name__ == '__main__':
    unittest.main()