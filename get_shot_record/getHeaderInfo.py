from __future__ import division  #force floating point division
import numpy as np
import math
import collections
import logging
import sys
import os
import time
sys.path.append('/home/mlai/geo/mlai/')
from utility import plot_helper 
from utility import math_operation as math_op
from utility import miscellaneous_helper as misc   
from utility import file_operation as file_op
                                     
if __name__ == '__main__':
    list_available_shot_number = ['02','03','06','08','10','20','24','25','27','30','32','33','34','35']  

    for shot_number in list_available_shot_number:
        shot_name = 'wz.' + shot_number
        rsf_file_name = shot_name + '.H'
        header_dictionary = file_op.convertRsfHeaderIntoDictionary(rsf_file_name)
        file_op.saveObject(header_dictionary, shot_name + "_header_dict")        


               
         
 
 
