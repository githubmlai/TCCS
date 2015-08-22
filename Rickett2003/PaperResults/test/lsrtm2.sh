#!/bin/bash

# *****************************************************************
# This script is used as the wrapper for sfmpilsrtm
# Author:  Junzhe Sun
# E-mail:  junzhesun@utexas.edu
# *****************************************************************

in=$RANDOM.rsf
<&0 sfcp > $in
out=$RANDOM.rsf
ibrun tacc_affinity sfmpilsrtmgmres input=$in output=$out $@
<$out sfcp