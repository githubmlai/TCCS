from __future__ import division  # force floating point division
import unittest

import numpy as np

import model_generation as mg


class TestFileOperation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calc_von_karman_energy_spectrum_scale_factor(self):
        scale_factor = mg.calc_von_karman_energy_spectrum_scale_factor(dimension=1,
                                                                       hurst_related_exponent=-0.25)
        self.assertAlmostEqual(scale_factor, 1.3110, places=4)

        scale_factor = mg.calc_von_karman_energy_spectrum_scale_factor(dimension=1,
                                                                       hurst_related_exponent=0.25)
        self.assertAlmostEqual(scale_factor, 0.5991, places=4)

        scale_factor = mg.calc_von_karman_energy_spectrum_scale_factor(dimension=1,
                                                                       hurst_related_exponent=0.5)
        self.assertAlmostEqual(scale_factor, 1, places=4)

        scale_factor = mg.calc_von_karman_energy_spectrum_scale_factor(dimension=1,
                                                                       hurst_related_exponent=0.75)
        self.assertAlmostEqual(scale_factor, 1.3110, places=4)

        scale_factor = mg.calc_von_karman_energy_spectrum_scale_factor(dimension=1,
                                                                       hurst_related_exponent=1)
        self.assertAlmostEqual(scale_factor, np.pi / 2, places=4)


if __name__ == '__main__':
    unittest.main()
