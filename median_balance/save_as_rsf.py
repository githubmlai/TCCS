import numpy, m8r
inp = numpy.load('processed_data_by_trace_time.npy')
out = m8r.File(inp)
out.sfin()
!sfcp /home/mlai/datapath/tmp5HwQV4 processed_data_by_trace_time.rsf
