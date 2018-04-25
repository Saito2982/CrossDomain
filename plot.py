#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

method = sys.argv[1]

if method == "value_EE":
  title = "PV R=(E E)"
elif method == "value_2EE":
  title = "PV R=(2E E)"
elif method == "value_EO":
  title = "PV R=(E O)"
elif method == "SVD":
  title = "SVD"
elif method == "pmf":
  title = "PMF"
elif method == "machine1":
  title = "PV ML_reg"
elif method == "machine2":
  title = "PV ML_arg"
elif method == "ML3_liner":
  title = "ML3_liner"

x10 = np.load("./result/movie.review/" + method + "/P10.npy")
x03 = np.load("./result/movie.review/" + method + "/P03.npy")
x06 = np.load("./result/movie.review/" + method + "/P06.npy")
x00 = np.load("./result/movie.review/" + method + "/P00.npy")

left0 = np.array([0.0, 0.3, 0.6, 0.9])
height0 = np.array([x00.mean(), x03.mean(), x06.mean(), x10.mean()])
labels = np.array(["0.0", "3.33...", "6.66...", "1.0"])

plt.figure(1)
plt.bar(left0, height0, tick_label=labels, align="center", width=0.1)
plt.title(title)
plt.xlabel("precision")
plt.ylabel("number of users")
plt.ylim(ymax = 10000, ymin = 0)
plt.savefig("./fig/" + method + "_Pat3.png")

x10 = np.load("./result/movie.review/" + method + "/R10.npy")
x08 = np.load("./result/movie.review/" + method + "/R08.npy")
x06 = np.load("./result/movie.review/" + method + "/R06.npy")
x04 = np.load("./result/movie.review/" + method + "/R04.npy")
x02 = np.load("./result/movie.review/" + method + "/R02.npy")

left1 = np.array([0., 0.2, 0.4, 0.6, 0.8])
height1 = np.array([x02.mean(), x04.mean(), x06.mean(), x08.mean(), x10.mean()])

plt.figure(2)
plt.bar(left1, height1, width=0.2)
plt.title(title)
plt.xlabel("Recall")
plt.ylabel("number of users")
plt.ylim(ymax = 9000, ymin = 0)
plt.savefig("./fig/" + method + "_Rat3.png")

x10 = np.load("./result/movie.review/" + method + "/D10.npy")
x08 = np.load("./result/movie.review/" + method + "/D08.npy")
x06 = np.load("./result/movie.review/" + method + "/D06.npy")
x04 = np.load("./result/movie.review/" + method + "/D04.npy")
x02 = np.load("./result/movie.review/" + method + "/D02.npy")

left = np.array([0., 0.2, 0.4, 0.6, 0.8])
height = np.array([x02.mean(), x04.mean(), x06.mean(), x08.mean(), x10.mean()])

plt.figure(3)
plt.bar(left, height, width=0.2)
plt.title(title)
plt.xlabel("nDCG")
plt.ylabel("number of users")
plt.ylim(ymax = 20000, ymin = 0)
plt.savefig("./fig/" + method + "_nDCGat3.png")


