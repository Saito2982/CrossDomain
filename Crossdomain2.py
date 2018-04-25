import sys
import nimfa
import numpy as np
import scipy.sparse as sp
import pandas as pd
import gc
import os
import mysql.connector
import random
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

def learning(method, train_matrix, train_index, data, user_list, item_list):
  if method == "SVD":
    u, s, vt = svds(train_matrix, k=attribute)
    np.savetxt("u.csv", u, delimiter=",")
    np.savetxt("s.csv", s, delimiter=",")
    np.savetxt("vt.csv", vt, delimiter=",")
    s_diag_matrix = np.diag(s)
    return u
  elif method == "PMF":
    pmf = nimfa.Pmf(train_matrix.toarray(), seed="random_vcol", rank=attribute, max_iter=50, rel_error=1e-5)
    pmf_fit = pmf()
    return np.array(pmf_fit.fitted())
  elif method == "NMF":
    nmf = nimfa.Nmf(train_matrix, seed="random_vcol", rank=attribute, max_iter=100, rel_error=1e-5, update='euclidean')
    nmf_fit = nmf()
    return nmf_fit.fitted().toarray()
  elif method == "RMrate_liner":
    u, v = pv.rmrate_standard(train_index, data, user_list, item_list, attribute)
    return u
  elif method == "D1_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -np.identity(attribute)]
    return u
  elif method == "D2_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[2 * np.identity(attribute), -np.identity(attribute)]
    return u
  elif method == "D3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -2 * np.identity(attribute)]
    return u
  elif method == "D4_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), np.zeros((attribute, attribute))]
    return u
  elif method == "D5_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.zeros((attribute, attribute)), -np.identity(attribute)]
    return u
  elif method == "ML1_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml1(train_matrix, eta0, u, v, attribute)
    return u
  elif method == "ML2_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml2(train_matrix, eta0, u, v, attribute)
    return u
  elif method == "ML3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return u
  elif method == "R2_RMrate":
    u, v = pv.rmrate_square_standard(train_index, data, user_list, item_list, attribute)
    return u
  elif method == "D1_sqaure":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -np.identity(attribute)]
    return u
  elif method == "D2_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[2 * np.identity(attribute), -np.identity(attribute)]
    return u
  elif method == "D3_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -2 * np.identity(attribute)]
    return u
  elif method == "D4_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), np.zeros((attribute, attribute))]
    return u
  elif method == "D5_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.zeros((attribute, attribute)), -np.identity(attribute)]
    return u
  elif method == "ML1_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml1(train_matrix, eta0, u, v, attribute)
    return u
  elif method == "ML2_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml2(train_matrix, eta0, u, v, attribute)
    return u
  elif method == "ML3_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return u

def learning2(method, train_matrix, train_index, data, user_list, item_list, u2):
  if method == "SVD":
    u, s, vt = svds(train_matrix, k=attribute)
    s_diag_matrix = np.diag(s)
    return np.dot(np.dot(u2, s_diag_matrix), vt)
  elif method == "PMF":
    pmf = nimfa.Pmf(train_matrix.toarray(), seed="random_vcol", rank=attribute, max_iter=50, rel_error=1e-5)
    pmf_fit = pmf()
    return np.array(pmf_fit.fitted())
  elif method == "NMF":
    nmf = nimfa.Nmf(train_matrix, seed="random_vcol", rank=attribute, max_iter=100, rel_error=1e-5, update='euclidean')
    nmf_fit = nmf()
    return nmf_fit.fitted().toarray()
  elif method == "RMrate_liner":
    u, v = pv.rmrate_standard(train_index, data, user_list, item_list, attribute)
    return np.dot(u2, v.T)
  elif method == "D1_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D2_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[2 * np.identity(attribute), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -2 * np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D4_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), np.zeros((attribute, attribute))]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D5_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.zeros((attribute, attribute)), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML1_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml1(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML2_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml2(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)
  elif method == "R2_RMrate":
    u, v = pv.rmrate_square_standard(train_index, data, user_list, item_list, attribute)
    return np.dot(u2, v.T)
  elif method == "D1_sqaure":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D2_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[2 * np.identity(attribute), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D3_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), -2 * np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D4_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.identity(attribute), np.zeros((attribute, attribute))]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "D5_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = np.c_[np.zeros((attribute, attribute)), -np.identity(attribute)]
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML1_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml1(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML2_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml2(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)
  elif method == "ML3_square":
    u, v = pv.rmrate_square(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return np.dot(np.dot(u2, R), v.T)

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
  '''
  #setting 4
  user_Mu = np.loadtxt("./genre7-9/data/d11/user.csv",delimiter=",").astype(np.int64)
  user_Mv = np.loadtxt("./genre7-9/data/d22/user.csv",delimiter=",").astype(np.int64)
  test_user = np.loadtxt("./genre7-9/data/d12/user.csv",delimiter=",").astype(np.int64)

  item_Mu = np.loadtxt("./genre7-9/data/d11/item.csv",delimiter=",").astype(np.int64)
  item_Mv = np.loadtxt("./genre7-9/data/d22/item.csv",delimiter=",").astype(np.int64)
  test_item = np.loadtxt("./genre7-9/data/d12/item.csv",delimiter=",").astype(np.int64)

  data_Mu = np.loadtxt("./genre7-9/data/d11/data.csv",delimiter=",").astype(np.int64)
  data_Mv = np.loadtxt("./genre7-9/data/d22/data.csv",delimiter=",").astype(np.int64)
  test_data = np.loadtxt("./genre7-9/data/d12/data.csv",delimiter=",").astype(np.int64)

  train_index = np.loadtxt("./genre7-9/data/d11/index.csv",delimiter=",").astype(np.int64)
  train_index2 = np.loadtxt("./genre7-9/data/d22/index.csv",delimiter=",").astype(np.int64)
  train_index3 = np.loadtxt("./genre7-9/data/d12/index.csv",delimiter=",").astype(np.int64)
  '''
  
  #setting 5
  user_Mu = np.loadtxt("./genre7-9/data/d22/user.csv",delimiter=",").astype(np.int64)
  user_Mv = np.loadtxt("./genre7-9/data/d11/user.csv",delimiter=",").astype(np.int64)
  test_user = np.loadtxt("./genre7-9/data/d21/user.csv",delimiter=",").astype(np.int64)

  item_Mu = np.loadtxt("./genre7-9/data/d22/item.csv",delimiter=",").astype(np.int64)
  item_Mv = np.loadtxt("./genre7-9/data/d11/item.csv",delimiter=",").astype(np.int64)
  test_item = np.loadtxt("./genre7-9/data/d21/item.csv",delimiter=",").astype(np.int64)

  data_Mu = np.loadtxt("./genre7-9/data/d22/data.csv",delimiter=",").astype(np.int64)
  data_Mv = np.loadtxt("./genre7-9/data/d11/data.csv",delimiter=",").astype(np.int64)
  test_data = np.loadtxt("./genre7-9/data/d21/data.csv",delimiter=",").astype(np.int64)

  train_index = np.loadtxt("./genre7-9/data/d22/index.csv",delimiter=",").astype(np.int64)
  train_index2 = np.loadtxt("./genre7-9/data/d11/index.csv",delimiter=",").astype(np.int64)
  train_index3 = np.loadtxt("./genre7-9/data/d21/index.csv",delimiter=",").astype(np.int64)
  

  for i in range(repeate):

    Mu_matrix = makeMatrix(data_Mu, train_index, user_Mu, item_Mu)
    u = learning(method, Mu_matrix, train_index, data_Mu, user_Mu, item_Mu)

    Mv_matrix = makeMatrix(data_Mv, train_index2, user_Mv, item_Mv)
    pred = learning2(method, Mv_matrix, train_index2, data_Mv, user_Mv, item_Mv, u)

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
    print("Recall : " + str(np.mean(rec[test_users.nonzero()])))
    print("nDCG : " + str(np.mean(dcg[test_users.nonzero()])))
    print("=================================================================================================")

    a += np.mean(pre[test_users.nonzero()])
    b += np.mean(rec[test_users.nonzero()])
    c += np.mean(dcg[test_users.nonzero()])
    # gavege collection, numpy_ndarray not used hereafter
    #del pred
    #del test_matrix
    #del train_matrix
    gc.collect()
    #np.save("result/" + dataset + "/" + method + "/Precision.npy", precision)
    #np.save("result/" + dataset + "/" + method + "/Recall.npy", recall)
    #np.save("result/" + dataset + "/" + method + "/nDCG.npy", nDCG)
  print("Precision AVE : " + str(a / 10))
  print("Recall AVE : " + str(b / 10))
  print("nDCG AVE : " + str(c / 10))

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

  methods = ["SVD","ML3_liner"]
  p.map(calculate,methods)

  print("Program completed...")