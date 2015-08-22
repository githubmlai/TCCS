from __future__ import division  # force floating point division
import os
import sys

import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
from scipy.stats import kstest

sys.path.append('/home/mlai/geo/mlai/')
from utility import plot_helper as ph


def calc_log_abs_fft_noise(fft_well_log,
                           list_wave_number,
                           hurst_related_exponent,
                           characteristic_dist):
    """This value should be almost constant

    Assumes fft_well_log = sqrt{von_karman_energy_spectrum} * G  where G is Fourier Transform of Gaussian white noise
    with zero mean and some variance, sigma_x^2.  We note that G is a complex variable.

    Taking the magnitude of the above equation, squaring, taking log, and rearranging terms we have

       log|fft_well_log| - 1/2 * log(|von_karman_energy_spectrum|) = log |G|

    The right hand side, calculated by this function, should be approximately constant.


    """

    von_karman_energy_spectrum = calc_von_karman_energy_spectrum(list_wave_number,
                                                                 hurst_related_exponent,
                                                                 characteristic_dist)
    return np.log(np.abs(fft_well_log)) - 0.5 * np.log(np.abs(von_karman_energy_spectrum))

def calc_von_karman_energy_spectrum(list_wave_number,
                                    hurst_related_exponent,
                                    characteristic_dist,
                                    sigma_squared=1.0,
                                    dimension=1.0):
    """Returns evaluation of Eq (7), E_{H,b}^{(s)}, in 'Fractal Heterogeneities
       and attenuation (2010)'."""

    von_karman_energy_spectrum_scale_factor = calc_von_karman_energy_spectrum_scale_factor(dimension,
                                                                                           hurst_related_exponent)

    decay = np.power(2 * characteristic_dist, dimension) / np.power(1.0 + np.square(2.0 * np.pi
                                                                                    * characteristic_dist
                                                                                    * list_wave_number),
                                                                    hurst_related_exponent + dimension / 2.0)

    return sigma_squared * von_karman_energy_spectrum_scale_factor * decay


def calc_von_karman_energy_spectrum_scale_factor(dimension,
                                                 hurst_related_exponent):
    """Returns evaluation of Eq (8), C_{H}^{(s)}, in 'Fractal Heterogeneities
       and attenuation (2010)'."""

    gamma_ratio = (scipy.special.gamma(hurst_related_exponent + dimension / 2.0) /
                   scipy.special.gamma(hurst_related_exponent))

            # not sure if gamma_ratio has absolute value or not!
    if np.isfinite(gamma_ratio):

            #TODO: what to do when gamma ratio 0 ???  (6/23/15)
        if gamma_ratio == 0:
            return 1e-5
        else:
            return np.power(scipy.pi, dimension / 2.0) * np.abs(gamma_ratio)
    else:
        print "dimension=%g, hurst_related_exponent=%g, gamma_ratio=%g" % (dimension,
                                                                           hurst_related_exponent,
                                                                           gamma_ratio)
        raise ValueError("gamma ratio not finite!")

def make_rayleigh_rand_var_of_abs_fft_noise(squared_energy_of_window,
                                            variance_of_noise_in_time_domain):
    rayleigh_scale_parameter_abs_fft_noise = \
        calc_rayleigh_scale_parameter_abs_fft_noise(squared_energy_of_window,
                                                    variance_of_noise_in_time_domain)
    return rayleigh(loc=0.0, scale=rayleigh_scale_parameter_abs_fft_noise)


def calc_likelihood_abs_fft_well_log(fft_well_log,
                                     hurst_related_exponent,
                                     characteristic_dist,
                                     list_wave_number,
                                     squared_energy_of_window,
                                     variance_of_noise_in_time_domain):
    """Finds probability of observing magnitude of fft of well log given parameters.

    Assumes fft_well_log = sqrt{von_karman_energy_spectrum} * G  where G is Fourier Transform of Gaussian white noise
    with zero mean and some variance, sigma_x^2.  We note that G is a complex variable.

    Taking the magnitude of the above equation and rearranging terms we have

       |G| = |fft_well_log| / | sqrt{von_karman_energy_spectrum} |

    The distribution of the magnitude of G is a Rayleigh distribution with scale parameter
    sigma' = sqrt{E_w * sigma_x^2 / 2} where E_w is the squared energy of the window used to obtain the given well log
    segment and sigma_x^2 is the variance of the noise G is the time domain.
    See Eq (43) of "DFT of Noise" by Mark A. Richard in 'doc' folder.

    Hence we have

       Prob( |G| ) = |G|/ sigma'^2 exp(-|G|^2/(2 sigma'^2))

    Each value in the fft_well_log is independent of all others and hence we can simply multiply the individual
    probabilities of each value.  Taking the log of the likelihood and ignoring constants we have

    """

    von_karman_energy_spectrum = calc_von_karman_energy_spectrum(list_wave_number,
                                                                 hurst_related_exponent,
                                                                 characteristic_dist)
    rayleigh_rand_var_instance = calc_rayleigh_rand_var_instances_from_fft_well_log(fft_well_log,
                                                                                    von_karman_energy_spectrum)

    rayleigh_scale_parameter = calc_rayleigh_scale_parameter_abs_fft_noise(squared_energy_of_window,
                                                                           variance_of_noise_in_time_domain)
    list_prob = rayleigh.logpdf(rayleigh_rand_var_instance, loc=0, scale=rayleigh_scale_parameter)
    # list_prob = (-0.5 * np.log(von_karman_energy_spectrum)
    #              - np.square(np.abs(fft_well_log))/ von_karman_energy_spectrum
    #              * 1/(2 * np.square(rayleigh_scale_parameter_abs_fft_noise)))
    output = np.sum(list_prob)
    if np.isnan(output) or np.isinf(output):
        output = -10000000
    return output

def calc_ks_test_stat(fft_well_log,
                      list_wave_number,
                      hurst_related_exponent,
                      characteristic_dist,
                      squared_energy_of_window,
                      variance_of_noise_in_time_domain):
    von_karman_energy_spectrum = calc_von_karman_energy_spectrum(list_wave_number,
                                                                 hurst_related_exponent,
                                                                 characteristic_dist)
    rayleigh_scale_parameter_abs_fft_noise = \
        calc_rayleigh_scale_parameter_abs_fft_noise(squared_energy_of_window,
                                                    variance_of_noise_in_time_domain)
    rayleigh_rand_var_instance = calc_rayleigh_rand_var_instances_from_fft_well_log(fft_well_log,
                                                                                    von_karman_energy_spectrum)

    test_stat_p_value = kstest(rayleigh_rand_var_instance, 'rayleigh', args=(0, rayleigh_scale_parameter_abs_fft_noise))
    return test_stat_p_value[0]


def calc_rayleigh_scale_parameter_abs_fft_noise(squared_energy_of_window,
                                                variance_of_noise_in_time_domain):
    return np.sqrt(squared_energy_of_window * variance_of_noise_in_time_domain / 2)


def calc_rayleigh_rand_var_instances_from_fft_well_log(fft_well_log,
                                                       von_karman_energy_spectrum):
    return np.abs(fft_well_log) / abs(np.sqrt(von_karman_energy_spectrum))


def calc_fft_well_log_von_karman_energy_spectrum_dist(fft_well_log,
                                                      list_wave_number,
                                                      hurst_related_exponent,
                                                      characteristic_dist,
                                                      squared_energy_of_window,
                                                      variance_of_noise_in_time_domain):
    # von_karman_energy_spectrum = calc_von_karman_energy_spectrum(list_wave_number,
    #                                                              hurst_related_exponent,
    #                                                              characteristic_dist)
    #
    # difference = np.abs(fft_well_log) - np.sqrt(von_karman_energy_spectrum)

    dimension = 1
    sigma = 1
    von_karman_energy_spectrum_scale_factor = calc_von_karman_energy_spectrum_scale_factor(dimension,
                                                                                           hurst_related_exponent)
    intercept = np.log(np.abs(sigma * np.sqrt(2 * characteristic_dist * von_karman_energy_spectrum_scale_factor)))
    slope = hurst_related_exponent/2 + 1/4
    approx_log_abs_fft_well_log = intercept - slope * np.log(1 + np.square(2 * np.pi * list_wave_number
                                                                           * characteristic_dist))
    difference = np.log(np.abs(fft_well_log)) - approx_log_abs_fft_well_log
    return np.linalg.norm(difference)


def plot_sample_and_theoretical_pdf_of_abs_fft_noise(fft_well_log,
                                                     list_wave_number,
                                                     list_hurst_related_exponent,
                                                     list_characteristic_distance,
                                                     squared_energy_of_window,
                                                     variance_of_noise_in_time_domain,
                                                     output_directory,
                                                     output_file_base_name,
                                                     figure_output_settings):

    figure = plt.figure()
        # sample pdf (histogram)
    for hurst_related_exponent_local, characteristic_dist_local in zip(list_hurst_related_exponent,
                                                                       list_characteristic_distance):
        von_karman_energy_spectrum = calc_von_karman_energy_spectrum(list_wave_number,
                                                                     hurst_related_exponent_local,
                                                                     characteristic_dist_local)
        rayleigh_rand_var_instance = \
            calc_rayleigh_rand_var_instances_from_fft_well_log(fft_well_log,
                                                               von_karman_energy_spectrum)
        label_str = "(%.3f,%.3f)" % (hurst_related_exponent_local, characteristic_dist_local)
        plt.hist(rayleigh_rand_var_instance, bins=100, normed=True, label=label_str, histtype='step')
        plt.legend(loc='upper right')

        # theoretical pdf
    random_variable = make_rayleigh_rand_var_of_abs_fft_noise(squared_energy_of_window,
                                                              variance_of_noise_in_time_domain)
    support = np.linspace(0, max(np.abs(fft_well_log)))
    plt.plot(support, random_variable.pdf(support))

    output_file = output_directory + os.path.sep + output_file_base_name
    ph.saveFigureToFile(figure, output_file, **figure_output_settings)


if __name__ == '__main__':
    pass
