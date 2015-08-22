from __future__ import division  #force floating point division
from scipy import stats
import numpy as np
import collections

def findDomainPowerViaLogLogLinearRegresstion(list_signal, list_domain):
    output = collections.namedtuple('logloglinearregression', 
                                          ['slope', 
                                           'intercept',
                                           'r_value',
                                           'p_value'
                                           'std_err'])     
    log_list_signal = np.log(list_signal)
    log_list_domain = np.log(list_domain)
    output.slope, output.intercept, output.r_value, output.p_value, output.std_err = \
                            stats.linregress(log_list_domain,log_list_signal)
#     output.slope = slope
#     output.intercept = intercept
#     output.r_value = r_value
#     output.p_value = p_value
#     output.std_err = std_err       
                     
    return output

               
if __name__ == '__main__':
    pass