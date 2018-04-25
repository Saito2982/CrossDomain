import sys
import nimfa
import numpy as np
import scipy.sparse as sp
import pandas as pd
import gc
import os
import mysql.connector
import random
import collections
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
sepalate = 3
attribute = 5

def learning(method, train_matrix, train_index, data, user_list, item_list):
  if method == "SVD":
    u, s, vt = svds(train_matrix, k=attribute)
    s_diag_matrix = np.diag(s)
    return np.dot(np.dot(u, s_diag_matrix), vt)
  elif method == "ML3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u, R), v.T)

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
  a=0.0
  b=0.0
  c=0.0
  c_pre = np.array([0.,0.,0.,0.,0.,0.,0.])
  c_rec = np.array([0.,0.,0.,0.,0.,0.,0.])
  c_dcg = np.array([0.,0.,0.,0.,0.,0.,0.])

  set_data = sys.argv[1]
  setting = sys.argv[2]

  if setting == "1":
    user_list = np.loadtxt("./genre"+set_data+"/data/d11/user.csv",delimiter=",").astype(np.int64)  #U1

    item_list = np.loadtxt("./genre"+set_data+"/data/d11/item.csv",delimiter=",").astype(np.int64)  #genre

    data = np.loadtxt("./genre"+set_data+"/data/d11/data.csv",delimiter=",").astype(np.int64)
  elif setting == "2":
    user_list = np.loadtxt("./genre"+set_data+"/data/d22/user.csv",delimiter=",").astype(np.int64)  #U1

    item_list = np.loadtxt("./genre"+set_data+"/data/d22/item.csv",delimiter=",").astype(np.int64)  #genre

    data = np.loadtxt("./genre"+set_data+"/data/d22/data.csv",delimiter=",").astype(np.int64)
  elif setting == "3":
    user_list = np.loadtxt("./genre"+set_data+"/data/d1/user.csv",delimiter=",").astype(np.int64)  #U1

    item_list = np.loadtxt("./genre"+set_data+"/data/d1/item.csv",delimiter=",").astype(np.int64)  #genre

    data = np.loadtxt("./genre"+set_data+"/data/d1/data.csv",delimiter=",").astype(np.int64)



  precision = np.zeros((repeate, sepalate, len(user_list)))
  recall = np.zeros((repeate, sepalate, len(user_list)))
  nDCG = np.zeros((repeate, sepalate, len(user_list)))

  # repeated K cross-validation
  for i in range(repeate):

    # kf : model cross validation from sikit-learn
    # shuffle sepalating train and test data and random state change per repeat time
    kf = KFold(n_splits=sepalate, random_state=i, shuffle=True)

    j = 0
    for train_index, test_index in kf.split(data):
      # make train and test matrix
      train_matrix = makeMatrix(data, train_index, user_list, item_list)
      test_matrix = makeMatrix(data, test_index, user_list, item_list)
      test_users = users_in_testdata(3, test_matrix, user_list)
      # learning model and calculate predicted user-item matrix
      pred = learning(method, train_matrix, train_index, data, user_list, item_list)
      # np.save("./pred_temp.npy", pred)

      count_dict = collections.Counter(data[:,1])

      # calculating precision, recall, and nDCG using "pred"
      pre, rec, new_c_pre, new_c_rec, recom, recom2 = ev.precision(3, pred, np.array(test_matrix.todense()), user_list, item_list, count_dict)
      dcg ,new_c_dcg = ev.nDCG(3, pred, np.array(test_matrix.todense()), user_list, item_list)
      # save users' values for each criterion
      precision[i,j,:] = pre
      recall[i,j,:]    = rec
      nDCG[i,j,:]      = dcg
      c_pre = c_pre + new_c_pre
      c_rec = c_rec + new_c_rec
      c_dcg = c_dcg + new_c_dcg

      # print result which shows users' average, into standard output
      print("Process ID : " + str(os.getpid()))
      print("Repeat : " + str(i + 1))
      print("K-fold crossvalidation : " + str(j + 1) + "/" + str(sepalate))
      print("Dataset : " + dataset)
      print("Mthod : " + method)
      print("Precision : " + str(np.mean(pre[test_users.nonzero()])))
      print("Recall : " + str(np.mean(rec[test_users.nonzero()])))
      print("nDCG : " + str(np.mean(dcg[test_users.nonzero()])))
      print("=================================================================================================")
      a += np.mean(pre[test_users.nonzero()])
      b += np.mean(rec[test_users.nonzero()])
      c += np.mean(dcg[test_users.nonzero()])

      # gavege collection, numpy_ndarray not used hereafter
      del pred
      del test_matrix
      del train_matrix
      gc.collect()
      j = j + 1

  
  c_pre = c_pre / 30
  c_rec = c_rec / 30
  c_dcg = c_dcg / 30
  print(c_pre)
  np.save("result/" + dataset + "/" + method + "/Precision.npy", precision)
  np.save("result/" + dataset + "/" + method + "/Recall.npy", recall)
  np.save("result/" + dataset + "/" + method + "/nDCG.npy", nDCG)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/recom.npy", recom)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/recom2.npy", recom2)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_pre.npy", c_pre)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_rec.npy", c_rec)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_dcg.npy", c_dcg)

  print("Precision AVE : " + str(a / 30))
  print("Recall AVE : " + str(b / 30))
  print("nDCG AVE : " + str(c / 30))


if __name__ == "__main__":

  # Pool : the number of CPU.
  p = Pool(CPU)

  methods = ["SVD","ML3_liner"]
  p.map(calculate,methods)

  print("Program completed...")