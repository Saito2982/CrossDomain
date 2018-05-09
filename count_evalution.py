import sys
import numpy as np
import collections
import matplotlib.pyplot as plt

setting = sys.argv[1]
w=0.15

c = np.zeros((5,7))
c2 = np.zeros((5,7))


if setting == "4":
  t1 = np.loadtxt("./genre1-7/data/d12/data.csv",delimiter=",").astype(np.int64)
  t2 = np.loadtxt("./genre1-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  t3 = np.loadtxt("./genre1-10/data/d12/data.csv",delimiter=",").astype(np.int64)
  t4 = np.loadtxt("./genre7-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  t5 = np.loadtxt("./genre7-10/data/d12/data.csv",delimiter=",").astype(np.int64)
elif setting == "5":
  t1 = np.loadtxt("./genre1-7/data/d21/data.csv",delimiter=",").astype(np.int64)
  t2 = np.loadtxt("./genre1-9/data/d21/data.csv",delimiter=",").astype(np.int64)
  t3 = np.loadtxt("./genre1-10/data/d21/data.csv",delimiter=",").astype(np.int64)
  t4 = np.loadtxt("./genre7-9/data/d21/data.csv",delimiter=",").astype(np.int64)
  t5 = np.loadtxt("./genre7-10/data/d21/data.csv",delimiter=",").astype(np.int64)
elif setting == "1":
  t1 = np.loadtxt("./genre1-7/data/d11/data.csv",delimiter=",").astype(np.int64)
  t2 = np.loadtxt("./genre1-9/data/d11/data.csv",delimiter=",").astype(np.int64)
  t3 = np.loadtxt("./genre1-10/data/d11/data.csv",delimiter=",").astype(np.int64)
  t4 = np.loadtxt("./genre7-9/data/d11/data.csv",delimiter=",").astype(np.int64)
  t5 = np.loadtxt("./genre7-10/data/dd11/data.csv",delimiter=",").astype(np.int64)
elif setting == "2":
  t1 = np.loadtxt("./genre1-7/data/d12/data.csv",delimiter=",").astype(np.int64)
  t2 = np.loadtxt("./genre1-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  t3 = np.loadtxt("./genre1-10/data/d12/data.csv",delimiter=",").astype(np.int64)
  t4 = np.loadtxt("./genre7-9/data/d12/data.csv",delimiter=",").astype(np.int64)
  t5 = np.loadtxt("./genre7-10/data/d12/data.csv",delimiter=",").astype(np.int64)
elif setting == "3":
  t1 = np.loadtxt("./genre1-7/data/d1/data.csv",delimiter=",").astype(np.int64)
  t2 = np.loadtxt("./genre1-9/data/d1/data.csv",delimiter=",").astype(np.int64)
  t3 = np.loadtxt("./genre1-10/data/d1/data.csv",delimiter=",").astype(np.int64)
  t4 = np.loadtxt("./genre7-9/data/d1/data.csv",delimiter=",").astype(np.int64)
  t5 = np.loadtxt("./genre7-10/data/d1/data.csv",delimiter=",").astype(np.int64)



count_dict1 = collections.Counter(t1[:,1])
count_dict2 = collections.Counter(t2[:,1])
count_dict3 = collections.Counter(t3[:,1])
count_dict4 = collections.Counter(t4[:,1])
count_dict5 = collections.Counter(t5[:,1])
count2_dict1 = collections.Counter(t1[:,0])
count2_dict2 = collections.Counter(t2[:,0])
count2_dict3 = collections.Counter(t3[:,0])
count2_dict4 = collections.Counter(t4[:,0])
count2_dict5 = collections.Counter(t5[:,0])

for k ,v in count_dict1.items():
  if v <= 5:
    c[0,0] += 1
  elif v <= 10:
  	c[0,1] += 1
  elif v <= 20:
  	c[0,2] += 1
  elif v <= 40:
  	c[0,3] += 1
  elif v <= 60:
  	c[0,4] += 1
  elif v <= 100:
  	c[0,5] += 1
  else:
  	c[0,6] +=1

for k ,v in count_dict2.items():
  if v <= 5:
    c[1,0] += 1
  elif v <= 10:
  	c[1,1] += 1
  elif v <= 20:
  	c[1,2] += 1
  elif v <= 40:
  	c[1,3] += 1
  elif v <= 60:
  	c[1,4] += 1
  elif v <= 100:
  	c[1,5] += 1
  else:
  	c[1,6] +=1

for k ,v in count_dict3.items():
  if v <= 5:
    c[2,0] += 1
  elif v <= 10:
    c[2,1] += 1
  elif v <= 20:
    c[2,2] += 1
  elif v <= 40:
    c[2,3] += 1
  elif v <= 60:
    c[2,4] += 1
  elif v <= 100:
    c[2,5] += 1
  else:
    c[2,6] +=1

for k ,v in count_dict4.items():
  if v <= 5:
    c[3,0] += 1
  elif v <= 10:
    c[3,1] += 1
  elif v <= 20:
    c[3,2] += 1
  elif v <= 40:
    c[3,3] += 1
  elif v <= 60:
    c[3,4] += 1
  elif v <= 100:
    c[3,5] += 1
  else:
    c[3,6] +=1

for k ,v in count_dict5.items():
  if v <= 5:
    c[4,0] += 1
  elif v <= 10:
    c[4,1] += 1
  elif v <= 20:
    c[4,2] += 1
  elif v <= 40:
    c[4,3] += 1
  elif v <= 60:
    c[4,4] += 1
  elif v <= 100:
    c[4,5] += 1
  else:
    c[4,6] +=1

#-------------------------------------------------------------

for k ,v in count2_dict1.items():
  if v <= 5:
    c2[0,0] += 1
  elif v <= 10:
    c2[0,1] += 1
  elif v <= 20:
    c2[0,2] += 1
  elif v <= 40:
    c2[0,3] += 1
  elif v <= 60:
    c2[0,4] += 1
  elif v <= 100:
    c2[0,5] += 1
  else:
    c2[0,6] +=1

for k ,v in count2_dict2.items():
  if v <= 5:
    c2[1,0] += 1
  elif v <= 10:
    c2[1,1] += 1
  elif v <= 20:
    c2[1,2] += 1
  elif v <= 40:
    c2[1,3] += 1
  elif v <= 60:
    c2[1,4] += 1
  elif v <= 100:
    c2[1,5] += 1
  else:
    c2[1,6] +=1

for k ,v in count2_dict3.items():
  if v <= 5:
    c2[2,0] += 1
  elif v <= 10:
    c2[2,1] += 1
  elif v <= 20:
    c2[2,2] += 1
  elif v <= 40:
    c2[2,3] += 1
  elif v <= 60:
    c2[2,4] += 1
  elif v <= 100:
    c2[2,5] += 1
  else:
    c2[2,6] +=1

for k ,v in count2_dict4.items():
  if v <= 5:
    c2[3,0] += 1
  elif v <= 10:
    c2[3,1] += 1
  elif v <= 20:
    c2[3,2] += 1
  elif v <= 40:
    c2[3,3] += 1
  elif v <= 60:
    c2[3,4] += 1
  elif v <= 100:
    c2[3,5] += 1
  else:
    c2[3,6] +=1

for k ,v in count2_dict5.items():
  if v <= 5:
    c2[4,0] += 1
  elif v <= 10:
    c2[4,1] += 1
  elif v <= 20:
    c2[4,2] += 1
  elif v <= 40:
    c2[4,3] += 1
  elif v <= 60:
    c2[4,4] += 1
  elif v <= 100:
    c2[4,5] += 1
  else:
    c2[4,6] +=1

plt.figure(1)
left = np.array([1, 2, 3, 4, 5, 6, 7])
plt.xticks([1.375,2.375,3.375,4.375,5.375,6.375,7.375],("~5", "6~10", "11~20", "21~40", "41~60", "61~100", "101~"))
p1 = plt.bar(left, c[0,:], width=0.15,  linestyle="dashed")
p2 = plt.bar(left+w, c[1,:], width=0.15,  linestyle="dashed")
p3 = plt.bar(left+w*2, c[2,:], width=0.15,  linestyle="dashed")
p4 = plt.bar(left+w*3, c[3,:], width=0.15,  linestyle="dashed")
p5 = plt.bar(left+w*4, c[4,:], width=0.15,  linestyle="dashed")
plt.legend((p1,p2,p3,p4,p5),("genre1-7","genre1-9","genre1-10","genre7-9","genre7-10"), loc='upper right')
plt.ylabel("The number of items")
plt.xlabel("number of ratings")
plt.ylim(ymax = 750, ymin = 0)
plt.savefig("./fig/count_item.svg")

plt.figure(2)
left = np.array([1, 2, 3, 4, 5, 6, 7])
plt.xticks([1.375,2.375,3.375,4.375,5.375,6.375,7.375],("~5", "6~10", "11~20", "21~40", "41~60", "61~100", "101~"))
p1 = plt.bar(left, c2[0,:], width=0.15,  linestyle="dashed")
p2 = plt.bar(left+w, c2[1,:], width=0.15,  linestyle="dashed")
p3 = plt.bar(left+w*2, c2[2,:], width=0.15,  linestyle="dashed")
p4 = plt.bar(left+w*3, c2[3,:], width=0.15,  linestyle="dashed")
p5 = plt.bar(left+w*4, c2[4,:], width=0.15,  linestyle="dashed")
plt.legend((p1,p2,p3,p4,p5),("genre1-7","genre1-9","genre1-10","genre7-9","genre7-10"), loc='upper right')
plt.ylabel("The number of users")
plt.xlabel("number of ratings")
plt.ylim(ymax = 1250, ymin = 0)
plt.savefig("./fig/count_user.svg")