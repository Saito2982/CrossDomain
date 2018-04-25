#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

set_data = sys.argv[1]
setting = sys.argv[2]

svd = np.load("./result/movie.review/SVD/genre"+set_data+"_set"+setting+"/recom.npy")
#nmf = np.load("./result/nmf/c_pre.npy")
#pmf = np.load("./result/pmf/c_pre.npy")
#pv_d1 = np.load("./result/value_EE/c_pre.npy")
pv_ml3 = np.load("./result/movie.review/ML3_liner/genre"+set_data+"_set"+setting+"/recom.npy")
w=0.4

plt.figure(1)
left = np.array([1, 2, 3, 4, 5, 6, 7])
plt.xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2],("~3", "4~6", "7~10", "11~20", "21~30", "31~40", "41~"))
#p1 = plt.plot(left, pmf,  linestyle="solid")
p2 = plt.bar(left, svd, width=0.4,  linestyle="dashed")
#p3 = plt.plot(left, nmf, linestyle="dashdot")
p4 = plt.bar(left+w, pv_ml3, width=0.4,  linestyle="dotted")
plt.legend((p2[0],p4[0]),("SVD", "ML"), loc='upper right')
plt.ylabel("The number of recommended items")
plt.xlabel("number of ratings")
plt.ylim(ymax = svd.max()+100, ymin = 0)
plt.savefig("./fig/recom3.svg")


svd = np.load("./result/movie.review/SVD/genre"+set_data+"_set"+setting+"/recom2.npy")
#nmf = np.load("./result/nmf/c_pre.npy")
#pmf = np.load("./result/pmf/c_pre.npy")
#pv_d1 = np.load("./result/value_EE/c_pre.npy")
pv_ml3 = np.load("./result/movie.review/ML3_liner/genre"+set_data+"_set"+setting+"/recom2.npy")
w=0.4

plt.figure(2)
left = np.array([1, 2, 3, 4, 5, 6, 7])
plt.xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2],("~3", "4~6", "7~10", "11~20", "21~30", "31~40", "41~"))
#p1 = plt.plot(left, pmf,  linestyle="solid")
p2 = plt.bar(left, svd, width=0.4,  linestyle="dashed")
#p3 = plt.plot(left, nmf, linestyle="dashdot")
p4 = plt.bar(left+w, pv_ml3, width=0.4,  linestyle="dotted")
plt.legend((p2[0],p4[0]),("SVD", "ML"), loc='upper right')
plt.ylabel("The number of recommended items")
plt.xlabel("number of ratings")
plt.ylim(ymax = svd.max()+100, ymin = 0)
plt.savefig("./fig/recom4.svg")