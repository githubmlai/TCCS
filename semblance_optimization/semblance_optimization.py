from __future__ import division  #force floating point division
import numpy as np
import sys 
sys.path.append('/home/mlai/geo/mlai/')
from utility import plot_helper 
from utility import math_operation as math_op
from utility import file_operation
from utility import miscellaneous_helper as misc
    

    
def findListSemblanceForSignalGainedWithListOfDomainPower(
                                    signal_value,
                                    signal_domain,
                                    list_domain_power):
    num_domain_power = np.size(list_domain_power)
    list_semblance = np.zeros(num_domain_power)
    
    for idx_domain_power in range(num_domain_power):
        domain_power = list_domain_power[idx_domain_power]
        weighted_signal = math_op.weightSignalByDomainExponentiated(signal_domain,signal_value,domain_power)
        list_semblance[idx_domain_power] = math_op.calculateSemblance(weighted_signal)

    return list_semblance
    
    
    
if __name__ == '__main__':
    pass