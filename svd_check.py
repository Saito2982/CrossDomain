import sys
import nimfa
import numpy as np
import scipy.sparse as sp
import pandas as pd
import gc
import os
import mysql.connector
import random
import itertools
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold
from multiprocessing import Pool

# import module
import machine_learning as ml
import evaluation as ev
import parsonal_value as pv
#np.set_printoptions(threshold=np.inf)
# Default values
CPU = 1
dataset = "movie.review"
eta0 = 0.45
repeate = 10
sepalate = 1
attribute = 5

#=======================================================================================================================
# Name : makeMatrix
# Argument : ALL ... All data from numpy
#    Purpose ... porpese index
# Role :  make user-item matrix from evaluation format
#=======================================================================================================================

def makeMatrix(data, index, user_list, item_list):
  # lil matrix is a sparse matrix format (eliminated zero values)
  matrix = sp.lil_matrix((len(user_list), len(item_list)))
  # translate numpy into Dataframe
  data = pd.DataFrame(assign_index(data, index))
  for line in data.itertuples():
    # line[1] ... userID
    # line[2] ... itemID
    # line[3] ... rating value
    matrix[line[1], line[2]] = line[3]
  return matrix.tocsr()

#=======================================================================================================================
# Name : assign_index
# Argument : ALL ... All data from numpy
#            Purpose ... purpose index
# Role :  assign separated index data into numpy format
#=======================================================================================================================

def assign_index(ALL, Purpose):
  # attribute + 3 equals dataset format; user_ID, item_ID, time, attributes
  # Assigned ... all data in numpy format
  Assigned = np.zeros((len(Purpose), attribute + 3)).astype(np.int64)

  for i, j in enumerate(Purpose):
    Assigned[i] = ALL[j]
  return Assigned

#=======================================================================================================================
# Name : users_in_testdata
# Argument : n ... top-N recommendation count
#            test_matrix ... matrix witch elements include only in the test data
#            user_list ... user's ID
# Role :  make users list (the number of evaluations in test data is more than n)
#=======================================================================================================================

def users_in_testdata(n, test_matrix, user_list):
  test_user_list = np.zeros(len(user_list)).astype(np.int64)
  test_matrix = test_matrix.todense()
  for i,t in enumerate(test_matrix):
    if(t[t.nonzero()].size >= n):
      test_user_list[i] = 1
  return test_user_list

def calculate(method):
  global dataset
  precision_all = []
  recall_all = []
  ndcg_all = []
  a=0.0
  b=0.0
  c=0.0
  
  user_Mu = np.loadtxt("./genre1-10/data/d22/user.csv",delimiter=",").astype(np.int64)
  user_Mv = np.loadtxt("./genre1-10/data/d11/user.csv",delimiter=",").astype(np.int64)
  test_user = np.loadtxt("./genre1-10/data/d21/user.csv",delimiter=",").astype(np.int64)

  item_Mu = np.loadtxt("./genre1-10/data/d22/item.csv",delimiter=",").astype(np.int64)
  item_Mv = np.loadtxt("./genre1-10/data/d11/item.csv",delimiter=",").astype(np.int64)
  test_item = np.loadtxt("./genre1-10/data/d21/item.csv",delimiter=",").astype(np.int64)

  data_Mu = np.loadtxt("./genre1-10/data/d22/data.csv",delimiter=",").astype(np.int64)
  data_Mv = np.loadtxt("./genre1-10/data/d11/data.csv",delimiter=",").astype(np.int64)
  test_data = np.loadtxt("./genre1-10/data/d21/data_s.csv",delimiter=",").astype(np.int64)

  train_index = np.loadtxt("./genre1-10/data/d22/index.csv",delimiter=",").astype(np.int64)
  train_index2 = np.loadtxt("./genre1-10/data/d11/index.csv",delimiter=",").astype(np.int64)
  train_index3 = np.loadtxt("./genre1-10/data/d21/index_s.csv",delimiter=",").astype(np.int64)

  u = np.loadtxt("./u.csv",delimiter=",")
  s = np.loadtxt("./s.csv",delimiter=",")
  vt = np.loadtxt("./vt.csv",delimiter=",")
  s_diag_matrix = np.diag(s)
  seq = (0,1,2,3,4)
  C_list = list(itertools.permutations(seq))
  vt_new = np.zeros(vt.shape)
  
  for i in range(120):
    for j in range(5):
      vt_new[j,:] = vt[C_list[i][j],:]
  
  
    for i in range(1):
      pred = np.dot(np.dot(u, s_diag_matrix), vt_new)

      test_matrix = makeMatrix(test_data, train_index3, test_user, test_item)
      test_users = users_in_testdata(3, test_matrix, test_user)

      # calculating precision, recall, and nDCG using "pred"
      pre, rec = ev.precision(3, pred, np.array(test_matrix.todense()), test_user, test_item)
      dcg = ev.nDCG(3, pred, np.array(test_matrix.todense()), test_user, test_item)

      # print result which shows users' average, into standard output
      print("Process ID : " + str(os.getpid()))
      print("Repeat : " + str(i + 1))
      #print("K-fold crossvalidation : " + str(j + 1) + "/" + str(sepalate))
      print("Dataset : " + dataset)
      print("Mthod : " + method)
      print("Precision : " + str(np.mean(pre[test_users.nonzero()])))
      precision_all.append(np.mean(pre[test_users.nonzero()]))
      print("Recall : " + str(np.mean(rec[test_users.nonzero()])))
      recall_all.append(np.mean(rec[test_users.nonzero()]))
      print("nDCG : " + str(np.mean(dcg[test_users.nonzero()])))
      ndcg_all.append(np.mean(dcg[test_users.nonzero()]))
      print("=================================================================================================")
      
      gc.collect()
      np.savetxt("precision_all.csv", precision_all, delimiter=",")
      np.savetxt("recall_all.csv", recall_all, delimiter=",")
      np.savetxt("ndcg_all.csv", ndcg_all, delimiter=",")


if __name__ == "__main__":

  # Pool : the number of CPU.
  p = Pool(CPU)
  '''
  methods = ["SVD", "NMF", "RMrate_liner", "D1_liner", "D2_liner", "D3_liner", "D4_liner", "D5_liner", "RMrate_square",
             "D1_square", "D2_square", "D3_square", "D4_square", "D5_square", "ML1_liner", "ML2_liner", "ML3_liner",
             "ML1_square", "ML2_square", "ML3_square"]
  '''
  #p.map(calculate, methods)
  # PMF takes considerable memory so that it is separated.

  methods = ["SVD"]
  p.map(calculate,methods)

  print("Program completed...")