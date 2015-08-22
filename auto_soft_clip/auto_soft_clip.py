__author__ = 'mlai'
import numpy as np
import sys
sys.path.append('/home/mlai/geo/mlai/')
from utility import plot_helper
from utility import math_operation as math_op
from utility import file_operation as file_op
from utility import miscellaneous_helper as misc
from utility import data_generator as data_gen

def soft_clip(input, scalar_parameter):
    return ( scalar_parameter * input ) / np.sqrt(1 + scalar_parameter ** 2 * input ** 2)



if __name__ == '__main__':
        # load in data
    rsf_file_name = "/home/mlai/geo/mlai/timeFrequencyCorrection/data.rsf"
    output = file_op.readInRsfFileAsNumpyArrayAndHeader(rsf_file_name)
        # apply soft clip
    print "here"
        # measure entropy