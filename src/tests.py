import unittest
from src.ozone import Ozone

class TestOzone(unittest.TestCase):

    def setUp(self):
        self.test_object = Ozone(nscale=(0,1,2), airmass=(3,4,5))

    def test_nscale_min(self):
        expected = 0
        actual = self.test_object.nscale.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_nscale_max(self):
        expected = 1
        actual = self.test_object.nscale.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_nscale_points(self):
        expected = 2
        actual = self.test_object.nscale.points
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_min(self):
        expected = 3
        actual = self.test_object.airmass.min
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_max(self):
        expected = 4
        actual = self.test_object.airmass.max
        self.assertEqual(
            actual, expected, f"Expected {expected} but got {actual}"
        )

    def test_airmass_points(self):
        expected = 5
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

if __name__ == '__main__':
    unittest.main()