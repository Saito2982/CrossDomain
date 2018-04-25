#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

set_data = sys.argv[1]
setting = sys.argv[2]

svd = np.load("./result/movie.review/SVD/genre"+set_data+"_set"+setting+"/c_pre.npy")
#nmf = np.load("./result/nmf/c_pre.npy")
#pmf = np.load("./result/pmf/c_pre.npy")
#pv_d1 = np.load("./result/value_EE/c_pre.npy")
pv_ml3 = np.load("./result/movie.review/ML3_liner/genre"+set_data+"_set"+setting+"/c_pre.npy")

plt.figure(1)
left = np.array([1, 2, 3, 4, 5, 6])
plt.xticks([1,2,3,4,5,6],("~3", "4~6", "7~10", "11~20", "21~30", "31~40"))
#p1 = plt.plot(left, pmf,  linestyle="solid")
p2 = plt.plot(left, svd,  linestyle="dashed")
#p3 = plt.plot(left, nmf, linestyle="dashdot")
p4 = plt.plot(left, pv_ml3,  linestyle="dotted")
plt.legend((p2[0],p4[0]),("SVD", "ML"), loc=2)
plt.ylabel("precision")
plt.xlabel("number of ratings")
plt.ylim(ymax = svd.max()+0.1, ymin = pv_ml3.min()-0.1)
plt.savefig("./fig/pre_users.svg")

svd = np.load("./result/movie.review/SVD/genre"+set_data+"_set"+setting+"/c_rec.npy")
#pmf = np.load("./result/pmf/c_rec.npy")
#nmf = np.load("./result/nmf/c_rec.npy")
#pv_d1 = np.load("./result/value_EE/c_rec.npy")
pv_ml3 = np.load("./result/movie.review/ML3_liner/genre"+set_data+"_set"+setting+"/c_rec.npy")

plt.figure(2)
left = np.array([1, 2, 3, 4, 5, 6])
plt.xticks([1,2,3,4,5,6],("~3", "4~6", "7~10", "11~20", "21~30", "31~40"))
#p1 = plt.plot(left, pmf,  linestyle="solid")
p2 = plt.plot(left, svd,  linestyle="dashed")
#p3 = plt.plot(left, nmf, linestyle="dashdot")
p4 = plt.plot(left, pv_ml3,  linestyle="dotted")
plt.legend((p2[0],p4[0]),("SVD","ML"))
plt.ylabel("recall")
plt.xlabel("number of ratings")
plt.ylim(ymax = svd.max()+0.1, ymin = pv_ml3.min()-0.1)
plt.savefig("./fig/rec_users.svg")

svd = np.load("./result/movie.review/SVD/genre"+set_data+"_set"+setting+"/c_dcg.npy")
#nmf = np.load("./result/nmf/c_dcg.npy")
#pmf = np.load("./result/pmf/c_dcg.npy")
#pv_d1 = np.load("./result/value_EE/c_dcg.npy")
pv_ml3 = np.load("./result/movie.review/ML3_liner/genre"+set_data+"_set"+setting+"/c_dcg.npy")

plt.figure(3)
left = np.array([1, 2, 3, 4, 5, 6])
plt.xticks([1,2,3,4,5,6],("~3", "4~6", "7~10", "11~20", "21~30", "31~40"))
#p1 = plt.plot(left, pmf,  linestyle="solid")
p2 = plt.plot(left, svd,  linestyle="dashed")
#p3 = plt.plot(left, nmf, linestyle="dashdot")
p4 = plt.plot(left, pv_ml3,  linestyle="dotted")
plt.legend((p2[0],p4[0]),("SVD" "ML"))
plt.ylabel("nDCG")
plt.xlabel("number of ratings")
plt.ylim(ymax = svd.max()+0.1, ymin = pv_ml3.min()-0.1)
plt.savefig("./fig/dcg_users.svg")
