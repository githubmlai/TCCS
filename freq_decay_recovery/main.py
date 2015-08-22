'''
Created on Mar 5, 2015

@author: mlai
'''
#########
#
#  TODO
#
#  URGENT, IMPORTANT
#
#  NOT URGENT, IMPORTANT
#  =collect mean variance error on different H,b combinations when using brute force (6/24/15)
#
#  URGENT, NOT IMPORTANT
#
#  NOT URGENT, NOT IMPORTANT
#
#

from __future__ import division  # force floating point division
import os
from time import strftime
from ConfigParser import SafeConfigParser
import logging

import numpy as np





# get utility package
import sys

# add relative path of "utility" to current path, put at beginning

# sys.path.append(os.path.join(os.path.getcwd(),/home/mlai/geo/mlai/')
sys.path.append('/home/mlai/geo/mlai')
from utility import plot_helper
from utility import math_operation as math_op
from utility import file_operation
from utility import miscellaneous_helper as misc
import matplotlib.pyplot as plt
import model_generation as mg
from scipy import optimize
import abc
import warnings


class LikelihoodMeasure(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calc_inverse_likelihood(self,
                                hurst_chardist_opt_variables,
                                *params):
        """
        :param hurst_chardist_opt_variables:  tuple (hurst_related_exponent, characteristic distance)
        :param *params: fft_well_log, list_wave_number
        :return:  input arguments that occur with high probability have a small return value whereas input arguments
                  with low probability have a large return value
        """
        return

class MaxSemblanceLogAbsFftNoise(LikelihoodMeasure):
    def calc_inverse_likelihood(self,
                                hurst_chardist_opt_variables,
                                *params):
        hurst_related_exponent, characteristic_dist = hurst_chardist_opt_variables
        fft_well_log, list_wave_number = params

        log_abs_fft_noise = mg.calc_log_abs_fft_noise(fft_well_log,
                                                      list_wave_number,
                                                      hurst_related_exponent,
                                                      characteristic_dist)

        num_fft_well_log = fft_well_log.size
                    # If noise is zero, log_abs_fft_noise = 0 vector,
                    # We prevent division of zero by adding non-zero constant to log_abs_fft_noise.
        log_abs_fft_noise = log_abs_fft_noise + 1
        sum_log_abs_fft_noise = np.sum(log_abs_fft_noise)
        semblance = sum_log_abs_fft_noise / (np.sqrt(sum_log_abs_fft_noise) * np.sqrt(num_fft_well_log))
        return np.abs(semblance - 1)

class MinVarianceLogAbsFftNoise(LikelihoodMeasure):
    def calc_inverse_likelihood(self,
                                hurst_chardist_opt_variables,
                                *params):
        hurst_related_exponent, characteristic_dist = hurst_chardist_opt_variables
        fft_well_log, list_wave_number = params

        log_abs_fft_noise = mg.calc_log_abs_fft_noise(fft_well_log,
                                                      list_wave_number,
                                                      hurst_related_exponent,
                                                      characteristic_dist)
        return np.var(log_abs_fft_noise)


class MaxMedanBalanceLogAbsFftNoise(LikelihoodMeasure):
    def calc_inverse_likelihood(self,
                                hurst_chardist_opt_variables,
                                *params):
        hurst_related_exponent, characteristic_dist = hurst_chardist_opt_variables
        fft_well_log, list_wave_number = params

        log_abs_fft_noise = mg.calc_log_abs_fft_noise(fft_well_log,
                                                      list_wave_number,
                                                      hurst_related_exponent,
                                                      characteristic_dist)
        half_num_value = np.floor(log_abs_fft_noise.size / 2)
        first_half_median = np.median(log_abs_fft_noise[:half_num_value])
        second_half_median = np.median(log_abs_fft_noise[half_num_value + 1:])
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            out = np.abs(first_half_median - second_half_median)
            if len(w):
                # do something with the first warning
                print w[0].message
                print "hurst_related_exponent = %g, characteristic_dist=%g" % (hurst_related_exponent,
                                                                               characteristic_dist)
                print "first_half_median = %g, second_half_median=%g" % (first_half_median, second_half_median)
                raise
        return out

# def exhaustive_search(list_hurst_related_exponent,
#                       list_characteristic_dist,
#                       fft_well_log,
#                       list_wave_number,
#                       squared_energy_of_window,
#                       variance_gaussian_noise_time_domain,
#                       likelihood_measure,
#                       do_parallel=0):
#
#     size_list_hurst_related_exponent = list_hurst_related_exponent.size
#     size_list_characteristic_dist = list_characteristic_dist.size
#
#     if do_parallel:
#         num_cores = multiprocessing.cpu_count()
#         print 'num_cores = %g' % num_cores
#
#         list_hurst_characteristic_combination = list(itertools.product(list_hurst_related_exponent,
#                                                                        list_characteristic_dist))
#
#         output_as_vector = np.zeros(size_list_hurst_related_exponent * size_list_characteristic_dist)
#         output_as_vector = \
#             Parallel(n_jobs=3, verbose=5)(delayed(mg.calc_likelihood_abs_fft_well_log)(fft_well_log,
#                                                                                        hurst_characteristic_combination[
#                                                                                            0],
#                                                                                        hurst_characteristic_combination[
#                                                                                            1],
#                                                                                        list_wave_number_true,
#                                                                                        num_value_well_log)
#                                           for hurst_characteristic_combination in list_hurst_characteristic_combination)
#
#         return np.reshape(output_as_vector, [size_list_hurst_related_exponent,
#                                              size_list_characteristic_dist])
#     else:
#         output = np.zeros([size_list_hurst_related_exponent,
#                            size_list_characteristic_dist])
#         for idx_hurst_related_exponent in np.arange(size_list_hurst_related_exponent):
#             hurst_related_exponent_local = list_hurst_related_exponent[idx_hurst_related_exponent]
#             for idx_characteristic_dist in np.arange(size_list_characteristic_dist):
#                 characteristic_dist_local = list_characteristic_dist[idx_characteristic_dist]
#                 output[idx_hurst_related_exponent, idx_characteristic_dist] = \
#                     likelihood_measure.calc_likelihood(hurst_related_exponent_local,
#                                                        characteristic_dist_local,
#                                                        fft_well_log,
#                                                        list_wave_number,
#                                                        squared_energy_of_window,
#                                                        variance_gaussian_noise_time_domain)
#         return output


if __name__ == '__main__':

    # ###############################################
    #       Choose test to run
    test_to_run = 'likelihood'
    # ###############################################
    for case in math_op.switch(test_to_run):
        if case('likelihood'):

            # ##################################################################
            #                USER CONFIG: choose run parameters
            # ##################################################################
            do_parallel = 0
            num_value_well_log = 2 ** 12
            no_noise = 0  # if true, no noise in observed well log e.g well log is exactly von karman spectrum

            hurst_related_exponent_true = 0.5  # (-0.25,  -0.25,  0.25,   0.5, 0.75)
            characteristic_dist_true = 5.0  # [m]   # (10.0,  5.00, 10.00, 5.00, 3)
            mean_gaussian_noise_time_domain = 0.0
            variance_gaussian_noise_time_domain_true = 1.0e-10

            hurst_hurst_related_exponent_guess_start = max(hurst_related_exponent_true - 0.45, -0.49)
            hurst_hurst_related_exponent_guess_stop = hurst_related_exponent_true + 0.45
            hurst_hurst_related_exponent_guess_step = 0.01

            characteristic_dist_guess_start = max(characteristic_dist_true - 2, 10e-3)
            characteristic_dist_guess_stop = characteristic_dist_true + 2
            characteristic_dist_guess_step = 0.05

            likelihood_measure = MaxMedanBalanceLogAbsFftNoise()  # MinVarianceLogAbsFftNoise()
                                                                  # MaxMedanBalanceLogAbsFftNoise
                                                                  # MaxSemblanceLogAbsFftNoise

            # -->name of config file
            logging_level = logging.DEBUG
            config_file = "user_settings.cfg"
            print_to_stdout = 1

            ####################################
            # read in config file
            parser = SafeConfigParser()
            list_successfully_read_file = parser.read(config_file)
            assert list_successfully_read_file, "parser failed to read file = %s" % config_file
            figure_output_settings = plot_helper.getFigureOutputSettings(parser)

            # setup output directory
            output_directory_base = parser.get('web', 'output_directory_base')
            base_dir = misc.get_base_directory_of_calling_script()
            output_directory = output_directory_base + os.path.sep + base_dir + os.sep + strftime('%m%d%Y_%H%M%S')
            file_operation.makeDirectoryIfNotExist(output_directory)

            # setup logging
            unique_to_function_call_logger = misc.createUniqueToFunctionCallLogger()

            # -->log to file
            log_file = output_directory + \
                       os.path.sep + \
                       unique_to_function_call_logger.name + \
                       ".log"
            file_console_handler = logging.FileHandler(log_file)
            misc.setupHandlerAndAddToLogger(file_console_handler,
                                            unique_to_function_call_logger,
                                            logging_level)

            # -->log to stdout
            if print_to_stdout:
                stdout_console_handler = logging.StreamHandler(sys.stdout)
                misc.setupHandlerAndAddToLogger(stdout_console_handler,
                                                unique_to_function_call_logger,
                                                logging_level)
            list_hurst_related_exponent_guess = np.arange(hurst_hurst_related_exponent_guess_start,
                                                          hurst_hurst_related_exponent_guess_stop,
                                                          hurst_hurst_related_exponent_guess_step)
            list_characteristic_dist_guess = np.arange(characteristic_dist_guess_start,
                                                       characteristic_dist_guess_stop,
                                                       characteristic_dist_guess_step)
            unique_to_function_call_logger.info("----> list_hurst_related_exponent_guess")
            unique_to_function_call_logger.info(str(list_hurst_related_exponent_guess))
            unique_to_function_call_logger.info("----> list_characteristic_dist_guess")
            unique_to_function_call_logger.info(str(list_characteristic_dist_guess))

                                    # generate fft well log
            half_num_value_well_log_plus_one = np.floor(num_value_well_log) / 2 + 1
            list_wave_number_true = (np.arange(half_num_value_well_log_plus_one)
                                     / num_value_well_log)
                                    # --> generate noise
            gaussian_noise = np.random.normal(mean_gaussian_noise_time_domain,
                                              variance_gaussian_noise_time_domain_true,
                                              num_value_well_log)
            fft_gaussian_noise = np.fft.fft(gaussian_noise)
            fft_gaussian_noise_truncate = fft_gaussian_noise[0:half_num_value_well_log_plus_one]
            if no_noise:
                fft_gaussian_noise_truncate = 1

            von_karman_energy_spectrum_true = mg.calc_von_karman_energy_spectrum(list_wave_number_true,
                                                                                 hurst_related_exponent_true,
                                                                                 characteristic_dist_true)

            fft_well_log_true = np.multiply(np.sqrt(von_karman_energy_spectrum_true), fft_gaussian_noise_truncate)

                                    # calculate the true likelihood
            params = (fft_well_log_true, list_wave_number_true)
            inverse_likelihood_true = likelihood_measure.calc_inverse_likelihood(
                (hurst_related_exponent_true, characteristic_dist_true),
                *params)
            unique_to_function_call_logger.info("----> true values")
            unique_to_function_call_logger.info('b_true = %g, H_true = %g, inverse_likelihood_true = %g,' %
                                                (characteristic_dist_true,
                                                 hurst_related_exponent_true,
                                                 inverse_likelihood_true))

                                    # brute force (exhaustive) search
            do_brute_force_search = 1
            if do_brute_force_search:
                rranges = (slice(hurst_hurst_related_exponent_guess_start,
                                 hurst_hurst_related_exponent_guess_stop,
                                 hurst_hurst_related_exponent_guess_step),
                           slice(characteristic_dist_guess_start,
                                 characteristic_dist_guess_stop,
                                 characteristic_dist_guess_step))

                num_test = 3
                test_result = np.zeros((num_test, 2))
                for test_index in range(num_test):
                    brute_opt_result = optimize.brute(likelihood_measure.calc_inverse_likelihood,
                                                      rranges,
                                                      args=params,
                                                      full_output=True,
                                                      finish=optimize.fmin) #finish=optimize.fmin
                    opt_variable = brute_opt_result[0]
                    exhaustive_search_result_str = 'b_opt = %g, H_opt = %g, inverse_likelihood_opt = %g,' % (opt_variable[1],
                                                                                                             opt_variable[0],
                                                                                                             brute_opt_result[1])
                    test_result[test_index:] = [opt_variable[0],opt_variable[1]]
                    unique_to_function_call_logger.info(exhaustive_search_result_str)

                print test_result
                                # plot exhaustive search results
                figure, ax = plt.subplots(1)
                likelihood_by_hurst_characteristic = 1 / brute_opt_result[3]
                                # the third return argument of brute search is the search grid,
                                # so likelihood_by_hurst_characteristic should be the value *inside* those bounds.
                                # Therefore, remove the last value from likelihood_by_hurst_characteristic
                likelihood_by_hurst_characteristic = likelihood_by_hurst_characteristic[:-1, :-1]
                p = ax.pcolormesh(brute_opt_result[2][1],
                                  brute_opt_result[2][0],
                                  likelihood_by_hurst_characteristic)
                figure.colorbar(p)
                plt.xlabel('b (characteristic_distance)')
                plt.xticks(rotation=90)
                plt.ylabel('H (hurst_related_exponent)')
                ax.set_title(exhaustive_search_result_str)
                output_file_base_name = "true_v_opt_b_H"
                output_file = output_directory + os.path.sep + output_file_base_name
                plot_helper.saveFigureToFile(figure, output_file, **figure_output_settings)

                            # constrained optimization
            do_box_constrained_optimization = 0
            if do_box_constrained_optimization:
                hurst_hurst_related_exponent_min = -0.49
                hurst_hurst_related_exponent_max = 1.0
                characteristic_dist_min = 1.0
                characteristic_dist_max = 12.0

                initial_guess = (0.5 * (hurst_hurst_related_exponent_min + hurst_hurst_related_exponent_max),
                                 0.5 * (characteristic_dist_min + characteristic_dist_max))
                minimize_result = optimize.minimize(likelihood_measure.calc_inverse_likelihood,
                                                    initial_guess,
                                                    args=params,
                                                    method="TNC",
                                                    bounds=((hurst_hurst_related_exponent_min,
                                                             hurst_hurst_related_exponent_max),
                                                            (characteristic_dist_min,
                                                             characteristic_dist_max)))
                print minimize_result

            # directory is loaded from webbrowser
            display_all_dir_and_jpg_php = parser.get('web', 'display_all_dir_and_jpg_php')
            plot_helper.copyDisplayAllDirAndJpgPhpAsIndexPhp(output_directory, display_all_dir_and_jpg_php)

            ###################################################################



            break
        if case():  # default, could also just omit condition or 'if True'
            print "unknown test_to_run = %s \n" % test_to_run
            # No need to break here, it'll stop anyway
