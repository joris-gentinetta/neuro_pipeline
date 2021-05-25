import numpy as np
# path = r'E:\anxiety_ephys\211\circus\phase_files/2021-03-13_mBWfus011_after_arena_ephys.npy'
# a = np.load(path)
# b= a[:, 1000:1100]
# pta = [28, 29, 27, 12, 20, 21, 11, 7, 1, 8, 10, 15, 18, 23, 26, 31, 2, 6, 9, 14, 17, 22, 25, 30, 19, 24, 0, 13, 16, 5,
#        4, 3, 32, 45, 43, 56, 35, 40, 60, 52, 33, 38, 41, 46, 49, 54, 57, 62, 34, 39, 42, 47, 50, 55, 58, 63, 44, 36, 59,
#        61, 37, 48, 53, 51]
#
# # pad[x] denotes the pad corresponding to data['amplifier_data'][x]
# pad = [pta[channel] + 1 for channel in range(32)]
# reordered = np.empty(b.shape)
# for i in range(32):
#     reordered[i] = b[pad.index(i+1)]
# pass

masked = np.ma.masked_array([1,2], mask=[False, True])
boolean = masked < 3
summe = boolean.sum()
pass