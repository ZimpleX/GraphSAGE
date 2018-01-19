"""
The macros defined by me to help understand the graphSAGE code.
"""


BREAK_PT = False
PROFILE = False
SAMPLE_SIZE = [25,20]
avg_time_samplesize = []

###################################
# Profiling results when          #
# varying the layer 2 sample size #
###################################
# graph: ppi; feats: 50
# [25,10]: 0.17441
# [25,15]: 0.26314
# [25,20]: 0.38354
# [25,25]: 0.46530
# [25,30]: 0.46187
# [25,35]: 0.61510
# [25,40]: 0.72461

# graph ppi; feats: 100
# [25,10]: 0.27799
# [25,15]: 0.37732
# [25,20]: 0.48268
# [25,25]: 0.63442
# [25,30]: 0.75621
# [25,35]: 0.81510
# [25,40]: 1.05584

feat_vec_size = 320

################################################
# Profiling results when                       #
# varying feature vector length of each vertex #
################################################
# graph: ppi; sample size: [25,10] 
# 10: 0.10678
# 20: 0.12790
# 40: 0.16285
# 80: 0.23150
# 160: 0.34751
# 320: 0.59007

# graph: ppi; sample size: [25,20]
# 10: 0.24203
# 20: 0.22715
# 40: 0.28137
# 80: 0.43023
# 160: 0.69069
# 320: 1.21002
