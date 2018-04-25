#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

setting = sys.argv[1]

if setting == "4":
  d1 = np.loadtxt("./genre1-7/data/d12/data.csv",delimiter=",").astype(np.int64)
  d2 = np.loadtxt("./genre1-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  d3 = np.loadtxt("./genre1-10/data/d12/data.csv",delimiter=",").astype(np.int64)
  d4 = np.loadtxt("./genre7-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  d5 = np.loadtxt("./genre7-10/data/d12/data.csv",delimiter=",").astype(np.int64)
if setting == "5":
  d1 = np.loadtxt("./genre1-7/data/d21/data.csv",delimiter=",").astype(np.int64)
  d2 = np.loadtxt("./genre1-9/data/d21/data.csv",delimiter=",").astype(np.int64)
  d3 = np.loadtxt("./genre1-10/data/d21/data.csv",delimiter=",").astype(np.int64)
  d4 = np.loadtxt("./genre7-9/data/d21/data.csv",delimiter=",").astype(np.int64)
  d5 = np.loadtxt("./genre7-10/data/d21/data.csv",delimiter=",").astype(np.int64)

c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
count = np.zeros((5,5))

for i in d1:
  if i[2] == 1:
    c1 = c1 + 1
  elif i[2] == 2:
    c2 = c2 + 1
  elif i[2] == 3:
    c3 = c3 + 1
  elif i[2] == 4:
    c4 = c4 + 1
  else:
    c5 = c5 + 1
count[0] = np.array([c1, c2, c3, c4, c5])
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
for i in d2:
  if i[2] == 1:
    c1 = c1 + 1
  elif i[2] == 2:
    c2 = c2 + 1
  elif i[2] == 3:
    c3 = c3 + 1
  elif i[2] == 4:
    c4 = c4 + 1
  else:
    c5 = c5 + 1
count[1] = np.array([c1, c2, c3, c4, c5])
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
for i in d3:
  if i[2] == 1:
    c1 = c1 + 1
  elif i[2] == 2:
    c2 = c2 + 1
  elif i[2] == 3:
    c3 = c3 + 1
  elif i[2] == 4:
    c4 = c4 + 1
  else:
    c5 = c5 + 1
count[2] = np.array([c1, c2, c3, c4, c5])
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
for i in d4:
  if i[2] == 1:
    c1 = c1 + 1
  elif i[2] == 2:
    c2 = c2 + 1
  elif i[2] == 3:
    c3 = c3 + 1
  elif i[2] == 4:
    c4 = c4 + 1
  else:
    c5 = c5 + 1
count[3] = np.array([c1, c2, c3, c4, c5])
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
for i in d5:
  if i[2] == 1:
    c1 = c1 + 1
  elif i[2] == 2:
    c2 = c2 + 1
  elif i[2] == 3:
    c3 = c3 + 1
  elif i[2] == 4:
    c4 = c4 + 1
  else:
    c5 = c5 + 1
count[4] = np.array([c1, c2, c3, c4, c5])

w=0.15
left = np.array([1,2,3,4,5])
labels = np.array(["1", "2", "3", "4", "5"])
plt.xticks([1.375,2.375,3.375,4.375,5.375],("1","2","3","4","5"))

plt.figure(1)
#plt.bar(left, count[0,:], tick_label=labels, align="center")
p1 = plt.bar(left, count[0,:], width=0.15,  linestyle="dashed")
p2 = plt.bar(left+w, count[1,:], width=0.15,  linestyle="dashed")
p3 = plt.bar(left+w*2, count[2,:], width=0.15,  linestyle="dashed")
p4 = plt.bar(left+w*3, count[3,:], width=0.15,  linestyle="dashed")
p5 = plt.bar(left+w*4, count[4,:], width=0.15,  linestyle="dashed")
plt.legend((p1,p2,p3,p4,p5),("genre1-7","genre1-9","genre1-10","genre7-9","genre7-10"), loc='upper left')
plt.xlabel("rating")
plt.ylabel("number of ratings")
plt.savefig("./fig/count_evalutions.svg")