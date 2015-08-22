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
#  --make Eric's bad example (3/26/15) 
#  --refactor median balance (3/12/15)
#  --refactor package setup (3/12/15)
#
#  URGENT, NOT IMPORTANT
#
#  NOT URGENT, NOT IMPORTANT
#
#

from __future__ import division  #force floating point division
import numpy as np
import scipy.signal as signal
import os
import rsf.api
from time import strftime
from ConfigParser import SafeConfigParser
from algo import median_balance as mb 
import matplotlib.pyplot as plt
import collections
import logging
import json
            #get utility package
import sys 
sys.path.append('/home/mlai/geo/mlai/')
from utility import plot_helper 
from utility import math_operation as math_op
from utility import file_operation
from utility import miscellaneous_helper as misc
from utility import data_generator as data_gen
import semblance_optimization as semb
import logloglinear as lll

def get_ricker_data(list_of_power, 
                    num_time_sample, 
                    num_point_ricker_wavelet, 
                    width_parameter_ricker_wavelet, 
                    domain_time_sample, 
                    noise_level):
    num_trace = list_of_power.size
    true_data_by_trace_time = data_gen.createCubedGaussianWhiteNoiseConvolvedWithRickerWavelet(
                                                num_trace,
                                                num_time_sample,
                                                num_point_ricker_wavelet,
                                                width_parameter_ricker_wavelet)
            
    data_by_trace_time = np.zeros((num_trace,num_time_sample))
            
    for idx_power in range(list_of_power.size):
        attenuation_by_time_sample =  np.power(domain_time_sample,
                                                            -list_of_power[idx_power])
        additive_noise =  np.random.randn(num_time_sample) * noise_level

        data_by_trace_time[idx_power,:] =  \
            attenuation_by_time_sample * true_data_by_trace_time[idx_power,:] + additive_noise
    return data_by_trace_time

def compareMedianBalanceVsRegressionGivenGroundTruth(
                list_of_power, 
                num_time_sample=2000,
                origin_time_sample = 0.01,
                delta_time_sample = 0.002,  
                num_point_ricker_wavelet=100, 
                width_parameter_ricker_wavelet=10, 
                noise_level=1,
                initial_exponent_power=2, 
                max_iteration=60, 
                delta_exponent_power_tolerance=1e-12,
                output_directory='/tmp'):
    
    output = collections.namedtuple('compareMedianBalanceVsRegression', 
                                          ['list_median_balance_recovered_power', 
                                           'list_regression_recovered_power',
                                           'data_by_trace_time'
                                           'domain_time_sample']) 
    output.domain_time_sample = math_op.calculateLinearlySpacedValues(origin_time_sample, 
                                                               delta_time_sample, 
                                                               num_time_sample)
    output.data_by_trace_time = get_ricker_data(list_of_power, 
                                         num_time_sample, 
                                         num_point_ricker_wavelet, 
                                         width_parameter_ricker_wavelet, 
                                         output.domain_time_sample, 
                                         noise_level)
    output.list_median_balance_recovered_power = []
    output.list_regression_recovered_power = []
    for power, trace in zip(list_of_power, output.data_by_trace_time):    
                        #median_balance
        normalized_trace = np.squeeze(math_op.divideEachRowByItsMaxAbsValue(np.array([trace])))                       
        median_result = mb.recoverDomainWeightedGainViaMedianBalancing( normalized_trace, 
                        output.domain_time_sample,
                        initial_exponent_power,
                        max_iteration,
                        delta_exponent_power_tolerance,
                        output_directory,
                        print_to_stdout = 0,
                        logging_level=logging.CRITICAL)
        output.list_median_balance_recovered_power.append(median_result.domain_exponent_power)
                        #regression
        output_logloglinear = lll.findDomainPowerViaLogLogLinearRegresstion(
                                                                    np.abs(trace),
                                                                    output.domain_time_sample)
        output.list_regression_recovered_power.append(-output_logloglinear.slope)
    
    l2_median_balance_powers_error = np.linalg.norm(output.list_median_balance_recovered_power - list_of_power)
    l2_regression_powers_error = np.linalg.norm(output.list_regression_recovered_power - list_of_power)
    print 'l2_median_balance_powers_error= %g' % l2_median_balance_powers_error
    print 'l2_regression_powers_error = %g' % l2_regression_powers_error
    print 'l2_median_balance_powers_error / l2_regression_powers_error = %g' % (l2_median_balance_powers_error/l2_regression_powers_error)


    
    return output



if __name__ == '__main__':


    ##########choose test##########################    
    test_to_run = 'median_balance'
    #################################################3
    for case in math_op.switch(test_to_run):
        if case('compare_medianbal_vs_regression'):
            list_of_power = np.array([2]*10)
            output = compareMedianBalanceVsRegressionGivenGroundTruth(
                                 list_of_power, 
                                 num_time_sample = 2000,
                                 origin_time_sample = 1,
                                 delta_time_sample = 0.002, 
                                 width_parameter_ricker_wavelet = 10, 
                                 noise_level = 0)
            powers_col_by_col = np.transpose(np.vstack((output.list_median_balance_recovered_power,output.list_regression_recovered_power)))
            plt.figure()
            plt.hist(powers_col_by_col,
                     histtype='bar',
                     label=['median balance', 'regression'])
            plt.legend()
            figure, axes_array = plt.subplots(2,1)
            axes = axes_array[0]
            axes.plot( output.domain_time_sample,output.data_by_trace_time[0,:] * output.domain_time_sample**list_of_power[0])
            axes = axes_array[1]
            plt.xscale('log')
            plt.yscale('log')
            axes.scatter( output.domain_time_sample,output.data_by_trace_time[0,:] )

            plt.show()
            break
        if case('median_balance'):
                      
            ###################################################################
            #                USER CONFIG: choose run parameters
            ###################################################################                
                            #--> input data_type
            input_data_type = 'rsf'
            num_trace = 3
            num_time_sample = 2000
            delta_time_sample = 0.002
            origin_time_sample = delta_time_sample
            origin_trace = -2
            delta_trace = 0.05            
                            #--> powers to attenuate synthetic data with
            list_of_power = math_op.calculateLinearlySpacedValues(0.25, 0.5, num_trace)    
                            #-->location of config file
            config_file = "/home/mlai/geo/mlai/median_balance/user_settings.cfg"
                            #-->change these values for code speedup
            logging_level = logging.DEBUG
            print_to_stdout = 0
                            #-->view input as family of traces or individual traces
            treat_traces_individually = 1

            ###################################################################                        
            domain_time_sample = math_op.calculateLinearlySpacedValues(
                                                        origin_time_sample,
                                                        delta_time_sample,
                                                        num_time_sample)
            domain_time_sample_with_end_point = math_op.calculateLinearlySpacedValues(
                                                        origin_time_sample,
                                                        delta_time_sample,
                                                        num_time_sample + 1)
            list_offset = math_op.calculateLinearlySpacedValues(origin_trace, delta_trace, num_trace)
            list_offset_with_end_point = math_op.calculateLinearlySpacedValues(origin_trace, delta_trace, num_trace + 1)
                            
                            #TODO get directory of script to build path (3/5/15)
                            #read in config file
            parser = SafeConfigParser()
            list_successfully_read_file = parser.read(config_file)
            assert list_successfully_read_file, "parser failed to read file = %s" % config_file
            figure_output_settings = plot_helper.getFigureOutputSettings(parser)
            max_iteration = parser.getint('median_balance_algo','max_iteration')
            delta_exponent_power_tolerance = parser.getfloat('median_balance_algo','delta_exponent_power_tolerance')               
            initial_exponent_power = parser.getfloat('median_balance_algo','initial_exponent_power')  
        
                            #setup output directory
            ouput_directory_base = parser.get('web','output_directory_base')                  
            output_directory = ouput_directory_base + os.path.sep + input_data_type + os.sep + strftime('%m%d%Y_%H%M%S')
            file_operation.makeDirectoryIfNotExist(output_directory)
        
                            #setup input data
            for case in math_op.switch(input_data_type):
                if case('rsf'):
                            #read in rsf data file as numpy array                   
                    have_ungained_signal = 0     
                    data_file = "/home/mlai/geo/mlai/timeFrequencyCorrection/data.rsf"                
                    rsf_input = rsf.api.Input(data_file)
                    num_time_sample = rsf_input.int("n1")
                    num_trace = rsf_input.int("n2")   
                    origin_time_sample = rsf_input.float("o1")   
                    origin_trace = rsf_input.float("o2")   
                    delta_time_sample = rsf_input.float("d1")   
                    delta_trace = rsf_input.float("d2")   
                    unit_time = rsf_input.string("unit1")
                    unit_trace = rsf_input.string("unit2")
                    
                            #hack to use only portion of data
                    #num_trace = 15
        
                    domain_time_sample = math_op.calculateLinearlySpacedValues(
                                                        origin_time_sample,
                                                        delta_time_sample,
                                                        num_time_sample)
                    domain_time_sample_with_end_point = math_op.calculateLinearlySpacedValues(
                                                        origin_time_sample,
                                                        delta_time_sample,
                                                        num_time_sample + 1)
                    list_offset = math_op.calculateLinearlySpacedValues(origin_trace, delta_trace, num_trace)
                    list_offset_with_end_point = math_op.calculateLinearlySpacedValues(origin_trace, delta_trace, num_trace + 1)
                    
                    data_by_trace_time = np.zeros((num_trace,num_time_sample),'f')
                    rsf_input.read(data_by_trace_time)
                    
                    list_offset = math_op.calculateLinearlySpacedValues(origin_trace, delta_trace, num_trace)
                    break
                if case('constant'):
                        #create synthetic data that decays exponentially
                    have_ungained_signal = 1    
                    true_data_by_trace_time = np.ones((num_trace,num_time_sample))
                    data_by_trace_time = np.zeros((num_trace,num_time_sample))
                    for idx_power in range(list_of_power.size):
                        data_by_trace_time[idx_power,:] =  np.power(domain_time_sample,
                                                                    -list_of_power[idx_power])
                    break
                if case('noise_ricker_convolved'):
                    have_ungained_signal = 1    
                    num_point_ricker_wavelet = 100   
                    width_parameter_ricker_wavelet = 10            
                    
                    
                    true_data_by_trace_time = data_gen.createCubedGaussianWhiteNoiseConvolvedWithRickerWavelet(
                                                        num_trace,
                                                        num_time_sample,
                                                        num_point_ricker_wavelet,
                                                        width_parameter_ricker_wavelet)
                    
                    data_by_trace_time = np.zeros((num_trace,num_time_sample))
                    
                    for idx_power in range(list_of_power.size):
                        attenuation_by_time_sample =  np.power(domain_time_sample,
                                                                    -list_of_power[idx_power])
                        data_by_trace_time[idx_power,:] =  \
                            attenuation_by_time_sample * true_data_by_trace_time[idx_power,:]
                    break
                if case('white_noise'):
                    have_ungained_signal = 1    
                    true_data_by_trace_time = np.random.randn(num_trace,num_time_sample)
                    data_by_trace_time = np.zeros((num_trace,num_time_sample))
                    for idx_power in range(list_of_power.size):
                        attenuation_by_time_sample =  np.power(domain_time_sample,
                                                                    -list_of_power[idx_power])
                        data_by_trace_time[idx_power,:] =  \
                            attenuation_by_time_sample * true_data_by_trace_time[idx_power,:]
                    break
                if case(): # default, could also just omit condition or 'if True'
                    print "unknown input_data_type = %s \n" % input_data_type     
            
            processed_data_by_trace_time = np.zeros((num_trace,num_time_sample),'f')
            if (treat_traces_individually):
                list_power_by_traceindex = np.zeros(num_trace)
                list_iterationcount_by_traceindex = np.zeros(num_trace)
                list_traceindex = range(num_trace)
                for trace_index in list_traceindex:
                    trace = data_by_trace_time[[trace_index],:]
                                
                                #normalize values
                    normalized_trace = math_op.divideEachRowByItsMaxAbsValue(trace)                      
                    
                                #setup output directory
                    local_output_directory = output_directory + os.sep + "data" + str(trace_index)
                    file_operation.makeDirectoryIfNotExist(local_output_directory) 
                               
                    median_balance_output = \
                      mb.recoverDomainWeightedGainViaMedianBalancing( normalized_trace, 
                                                                      domain_time_sample,
                                                                      initial_exponent_power,
                                                                      max_iteration,
                                                                      delta_exponent_power_tolerance,
                                                                      local_output_directory,
                                                                      print_to_stdout = 1,
                                                                      logging_level=logging.DEBUG)
                    weighted_trace = math_op.weightSignalByDomainExponentiated(
                                                        domain_time_sample,
                                                        np.squeeze(trace),
                                                        median_balance_output.domain_exponent_power)
                    processed_data_by_trace_time[trace_index,:] = weighted_trace
                    list_power_by_traceindex[trace_index] = median_balance_output.domain_exponent_power
                    list_iterationcount_by_traceindex[trace_index] = median_balance_output.iteration_count
                    
                                        #do plotting              
                    trace_description = "trace index = %d" % trace_index
                    weighted_trace_description = "power = %g corrected"  % median_balance_output.domain_exponent_power 
    
                                    #plot true vs recovered
                                    # --> if don't know true, just plot original data
                    if (have_ungained_signal):
                        true_data = true_data_by_trace_time[trace_index,:]
                        trace_description = trace_description + ", true power = %g" % list_of_power[trace_index]
                                    #call routine to get statistics on true signal
                        initial_domain_exponent_power_true = 0
                        max_iteration_true = 1
                        delta_exponent_power_tolerance_true = 1e-5
                        mb.recoverDomainWeightedGainViaMedianBalancing(
                                                    true_data, 
                                                    domain_time_sample,
                                                    initial_domain_exponent_power_true,
                                                    max_iteration_true,
                                                    delta_exponent_power_tolerance_true,
                                                    local_output_directory)
                    else:               
                        true_data = trace
                        
                    value_by_truerecovered_range = np.vstack((true_data,weighted_trace))
                    output_file_base_name = "%dtruetrace" % trace_index 
                    plot_helper.plotSignalMagnitudePhase(domain_time_sample,
                                             value_by_truerecovered_range,
                                             [trace_description,weighted_trace_description],
                                             local_output_directory,
                                             output_file_base_name,
                                             figure_output_settings)
                    
                    output_file_base_name = "%dtrueerror" % trace_index
                    plot_helper.plotDifferenceBetweenTwoSignal(domain_time_sample,
                                             value_by_truerecovered_range,
                                             [trace_description,weighted_trace_description],
                                             local_output_directory,
                                             output_file_base_name,
                                             figure_output_settings) 
                     
                    output_file_base_name = "%dsemblancevsiteration" % trace_index
                    plot_helper.plotSemblanceVsIterationCount(median_balance_output.iteration_information.iteration_count,
                                  median_balance_output.iteration_information.semblance,
                                  local_output_directory,
                                  output_file_base_name,
                                  figure_output_settings)
                    display_all_dir_and_jpg_php=parser.get('web','display_all_dir_and_jpg_php')
                    plot_helper.copyDisplayAllDirAndJpgPhpAsIndexPhp(local_output_directory,display_all_dir_and_jpg_php)
                                    
                                        #maximize semblance
                    domain_power_min=0.01
                    domain_power_max=15
                    num_domain_power=1000
                    list_domain_power = np.linspace(domain_power_min,domain_power_max,num_domain_power)
                    
                    list_semblance =  semb.findListSemblanceForSignalGainedWithListOfDomainPower(
                                                                    np.squeeze(trace),
                                                                    domain_time_sample,
                                                                    list_domain_power)
                    output_file_base_name = "%dsemblancevsdomainpower" % trace_index
                    plot_helper.plotSemblanceVsDomainPower(list_domain_power,
                                  list_semblance,
                                  local_output_directory,
                                  output_file_base_name,
                                  figure_output_settings)
                    
                                        #log log linear fit
                    output_logloglinear = lll.findDomainPowerViaLogLogLinearRegresstion(
                                                                        np.abs(trace),
                                                                        domain_time_sample)                    
                    unique_to_function_call_logger = misc.createUniqueToFunctionCallLogger()         
                    log_file = local_output_directory + \
                               os.path.sep + \
                               unique_to_function_call_logger.name + \
                               "logloglinear.log"            
                    file_console_handler = logging.FileHandler(log_file)
                    misc.setupHandlerAndAddToLogger(file_console_handler, unique_to_function_call_logger)                    
                    unique_to_function_call_logger.info("slope=%g,intercept=%g,r_value=%g,p_value=%g,std_err=%g" % 
                                                        (output_logloglinear.slope,
                                                         output_logloglinear.intercept,
                                                         output_logloglinear.r_value,
                                                         output_logloglinear.p_value,
                                                         output_logloglinear.std_err))
                    #END FOR
                    
                plot_helper.plotPowerByTraceIndex(list_traceindex,
                          list_power_by_traceindex, 
                          output_directory,
                          'power_by_trace_index',
                          figure_output_settings)
                
                plot_helper.plotIterationCountByTraceIndex(list_traceindex,
                          list_iterationcount_by_traceindex, 
                          output_directory,
                          'iterationcount_by_trace_index',
                          figure_output_settings)         
            else:
                                            #treat traces as familiy of traces
                                            # --> setup output directory
                local_output_directory = output_directory + os.sep + "family" 
                file_operation.makeDirectoryIfNotExist(local_output_directory) 
                           
                median_balance_output = \
                  mb.recoverDomainWeightedGainViaMedianBalancing(data_by_trace_time,
                                                                 domain_time_sample,
                                                                 initial_exponent_power,
                                                                 max_iteration,
                                                                 delta_exponent_power_tolerance,
                                                                 local_output_directory,
                                                                 print_to_stdout=print_to_stdout,
                                                                 logging_level=logging_level)
                weighted_trace = math_op.weightSignalByDomainExponentiated(
                                                    domain_time_sample,
                                                    data_by_trace_time,
                                                    median_balance_output.domain_exponent_power)
                processed_data_by_trace_time = weighted_trace
           
                 
                            #plot entire shot record
            output_file_base_name = 'shot_record_gained'
            plot_helper.plotSeismicShotRecord(domain_time_sample_with_end_point,
                                       list_offset_with_end_point,
                                       processed_data_by_trace_time,
                                       output_directory,
                                       output_file_base_name,
                                       figure_output_settings)
            output_file_base_name = 'shot_record_original'
            plot_helper.plotSeismicShotRecord(domain_time_sample_with_end_point,
                                       list_offset_with_end_point,
                                       data_by_trace_time,
                                       output_directory,
                                       output_file_base_name,
                                       figure_output_settings)
            display_all_dir_and_jpg_php=parser.get('web','display_all_dir_and_jpg_php')
            plot_helper.copyDisplayAllDirAndJpgPhpAsIndexPhp(output_directory,display_all_dir_and_jpg_php)         
            
            processed_data_by_trace_time_file_name = output_directory + os.path.sep + "processed_data_by_trace_time"           
            np.save(processed_data_by_trace_time_file_name,processed_data_by_trace_time)
            break
        if case('low_pass_filter'):
            fir_filter_length = 61
            filter_cutoff = 0.3
            filter_window = 'hamming'
            num_trace = 6
            num_subplot_col = 2
            data_by_trace_time = data_by_trace_time[:,:num_trace]
            filter_numerator_coefficient = signal.firwin(fir_filter_length, 
                                                     cutoff = filter_cutoff, 
                                                     window = filter_window)
            data_by_trace_time = signal.lfilter(filter_numerator_coefficient, 
                   1, 
                   data_by_trace_time, 
                   axis=0)
            plot_helper.plotGroupsOfColumnDataInSubplots(data_by_trace_time, 
                                     num_trace, 
                                     num_subplot_col)
     
            fft_data_by_time_offset = np.abs(
                                     np.fft.fftshift(
                                                   np.fft.fft(data_by_trace_time,axis=0),
                                                   axes=(0,))
                                     )
            plot_helper.plotGroupsOfColumnDataInSubplots(fft_data_by_time_offset, 
                                     num_trace, 
                                     num_subplot_col)    
            break
        if case('plot_domain_exp_weight_and_transform'):
            list_of_power = np.linspace(0,1,5)
            for my_power in list_of_power:
                plot_helper.plotSignalMagnitudePhase(domain_time_sample,np.power(domain_time_sample,my_power))
            break
        if case(): # default, could also just omit condition or 'if True'
            print "unknown test_to_run = %s \n" % test_to_run
        # No need to break here, it'll stop anyway
    
    
