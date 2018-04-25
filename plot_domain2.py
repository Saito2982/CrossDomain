import sys
import nimfa
import numpy as np
import scipy.sparse as sp
import pandas as pd
import gc
import os
import math
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

def makeMatrix(data, train_matrix, test_matrix, train_index,  test_index, user_list, item_list):

  train_data = pd.DataFrame(assign_index(data, train_index))
  test_data = pd.DataFrame(assign_index(data, train_index))

  for line in train_data.itertuples():
    train_matrix[line[2]][line[1]] = line[3]
  for line in test_data.itertuples():
    test_matrix[line[2]][line[1]] = line[3]

  return train_matrix.as_matrix().astype(np.float64), test_matrix.as_matrix().astype(np.float64)

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

def nDCG(n, pred, test, item_list, lt_n):
  nDCG = np.zeros(len(pred) - lt_n)
  D02 = 0
  D04 = 0
  D06 = 0
  D08 = 0
  D10 = 0
  l = 0
  count = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c = np.array([0,0,0,0,0,0,0])
  for i,p in enumerate(pred):
    DCG = 0.
    iDCG = 0.
    t = test[i]
    ground_truth = t.nonzero()
    p_test = p[ground_truth]
    ranking_p_arg = np.argsort(p_test)[::-1]
    test_item = item_list[ground_truth]
    truth = t[ground_truth]
    ranking_t = np.sort(truth)[::-1]
    if len(ranking_p_arg) >= 3:
      for j in range(n):
        for k in range(len(test_item)):
          if test_item[k] == test_item[ranking_p_arg[j]]:
            if j == 0:
              DCG = truth[k]
            else:
              DCG = DCG + (truth[k] / math.log(j + 1, 2))
        if j == 0:
          iDCG = ranking_t[j]
        else:
          iDCG = iDCG + (ranking_t[j] / math.log(j + 1, 2)) 
      nDCG[l] = DCG / iDCG
      if nDCG[l] <= 0.2:
        D02 = D02 + 1
      elif nDCG[l] <= 0.4:
        D04 = D04 + 1
      elif nDCG[l] <= 0.6:
        D06 = D06 + 1
      elif nDCG[l] <= 0.8:
        D08 = D08 + 1
      else:
        D10 = D10 + 1
      
      if len(ranking_p_arg) <= 5:
        count[0] = count[0] + nDCG[l]
        count_c[0] = count_c[0] + 1
      elif len(ranking_p_arg) <= 10:
        count[1] = count[1] + nDCG[l]
        count_c[1] = count_c[1] + 1
      elif len(ranking_p_arg) <= 20:
        count[2] = count[2] + nDCG[l]
        count_c[2] = count_c[2] + 1
      elif len(ranking_p_arg) <= 40:
        count[3] = count[3] + nDCG[l]
        count_c[3] = count_c[3] + 1
      elif len(ranking_p_arg) <= 80:
        count[4] = count[4] + nDCG[l]
        count_c[4] = count_c[4] + 1
      elif len(ranking_p_arg) <= 160:
        count[5] = count[5] + nDCG[l]
        count_c[5] = count_c[5] + 1
      else:
        count[6] = count[6] + nDCG[l]
        count_c[6] = count_c[6] + 1

      l = l + 1

  count = count / count_c
  return np.mean(nDCG), np.std(nDCG), np.max(nDCG), np.min(nDCG), D02, D04, D06, D08, D10, count, count_c

def precision(n, pred, test, item_list, lt_n):
  precision = np.zeros(len(pred) - lt_n)
  recall = np.zeros(len(pred) - lt_n)
  p00 = 0
  p033 = 0
  p066 = 0
  p100 = 0
  r02 = 0
  r04 = 0
  r06 = 0
  r08 = 0
  r10 = 0
  l = 0
  count_pre = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c_pre = np.array([0,0,0,0,0,0,0])
  count_rec = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c_rec = np.array([0,0,0,0,0,0,0])

  for i, p in enumerate(pred):
    tp = 0
    fp = 0
    truth_all = 0
    t = test[i]
    ground_truth = t.nonzero()
    ranking_p_arg = np.argsort(p[ground_truth])[::-1]
    if len(ranking_p_arg) >= n:
      test_item = item_list[ground_truth]
      truth = t[ground_truth]
      for j in range(n):
        for k in range(len(test_item)):
          if test_item[k] == test_item[ranking_p_arg[j]]:
            if truth[k] >= 4.:
              tp = tp + 1.0
            else:
              fp = fp + 1.0

      for j in range(len(truth)):
        if truth[j] >= 4.0:
          truth_all = truth_all + 1

      precision[l] = tp / (tp + fp)
      if truth_all > 0:
        recall[l] = tp / truth_all

      if precision[l] == 0:
        p00 = p00 + 1
      elif precision[l] < 0.4:
        p033 = p033 + 1
      elif precision[l] < 0.7:
        p066 = p066 + 1
      else:
        p100 = p100 + 1

      if recall[l] <= 0.2:
        r02 = r02 + 1
      elif recall[l] <= 0.4:
        r04 = r04 + 1
      elif recall[l] <= 0.6:
        r06 = r06 + 1
      elif recall[l] <= 0.8:
        r08 = r08 + 1
      else:
        r10 = r10 + 1

      if len(ranking_p_arg) <= 5:
        count_pre[0] = count_pre[0] + precision[l]
        count_c_pre[0] = count_c_pre[0] + 1
      elif len(ranking_p_arg) <= 10:
        count_pre[1] = count_pre[1] + precision[l]
        count_c_pre[1] = count_c_pre[1] + 1
      elif len(ranking_p_arg) <= 20:
        count_pre[2] = count_pre[2] + precision[l]
        count_c_pre[2] = count_c_pre[2] + 1
      elif len(ranking_p_arg) <= 40:
        count_pre[3] = count_pre[3] + precision[l]
        count_c_pre[3] = count_c_pre[3] + 1
      elif len(ranking_p_arg) <= 80:
        count_pre[4] = count_pre[4] + precision[l]
        count_c_pre[4] = count_c_pre[4] + 1
      elif len(ranking_p_arg) <= 160:
        count_pre[5] = count_pre[5] + precision[l]
        count_c_pre[5] = count_c_pre[5] + 1
      else:
        count_pre[6] = count_pre[6] + precision[l]
        count_c_pre[6] = count_c_pre[6] + 1

      if len(ranking_p_arg) <= 5:
        count_rec[0] = count_rec[0] + recall[l]
        count_c_rec[0] = count_c_rec[0] + 1
      elif len(ranking_p_arg) <= 10:
        count_rec[1] = count_rec[1] + recall[l]
        count_c_rec[1] = count_c_rec[1] + 1
      elif len(ranking_p_arg) <= 20:
        count_rec[2] = count_rec[2] + recall[l]
        count_c_rec[2] = count_c_rec[2] + 1
      elif len(ranking_p_arg) <= 40:
        count_rec[3] = count_rec[3] + recall[l]
        count_c_rec[3] = count_c_rec[3] + 1
      elif len(ranking_p_arg) <= 80:
        count_rec[4] = count_rec[4] + recall[l]
        count_c_rec[4] = count_c_rec[4] + 1
      elif len(ranking_p_arg) <= 160:
        count_rec[5] = count_rec[5] + recall[l]
        count_c_rec[5] = count_c_rec[5] + 1
      else:
        count_rec[6] = count_rec[6] + recall[l]
        count_c_rec[6] = count_c_rec[6] + 1



      l = l + 1
  count_pre = count_pre / count_c_pre
  count_rec = count_rec / count_c_rec
      
  return precision.mean(), precision.std(), precision.max(), precision.min(), recall.mean(), recall.std(), recall.max(), recall.min(), p00, p033, p066, p100, r02, r04, r06, r08, r10, count_pre, count_c_pre, count_rec, count_c_rec

def search_lt_n(n, test_data):
  lt_n = 0
  for t in test_data:
    if t[t.nonzero()].shape[0] < n:
      lt_n = lt_n + 1

  return lt_n

def calculate(method):
  eta0 = 0.45
  Pat3_ave = np.zeros((10, 3))
  Pat3_std = np.zeros((10, 3))
  Pat3_max = np.zeros((10, 3))
  Pat3_min = np.zeros((10, 3))
  Rat3_ave = np.zeros((10,3))
  Rat3_std = np.zeros((10,3))
  Rat3_max = np.zeros((10,3))
  Rat3_min = np.zeros((10,3))
  nDCGat3_ave = np.zeros((10,3))
  nDCGat3_std = np.zeros((10,3))
  nDCGat3_max = np.zeros((10,3))
  nDCGat3_min = np.zeros((10,3))
  P0 = np.zeros((10,3)).astype(np.int64)
  P03 = np.zeros((10,3)).astype(np.int64)
  P06 = np.zeros((10,3)).astype(np.int64)
  P10 = np.zeros((10,3)).astype(np.int64)
  D02 = np.zeros((10,3)).astype(np.int64)
  D04 = np.zeros((10,3)).astype(np.int64)
  D06 = np.zeros((10,3)).astype(np.int64)
  D08 = np.zeros((10,3)).astype(np.int64)
  D10 = np.zeros((10,3)).astype(np.int64)
  R02 = np.zeros((10,3)).astype(np.int64)
  R04 = np.zeros((10,3)).astype(np.int64)
  R06 = np.zeros((10,3)).astype(np.int64)
  R08 = np.zeros((10,3)).astype(np.int64)
  R10 = np.zeros((10,3)).astype(np.int64)
  lt_3 = np.zeros((10,3)).astype(np.int64)

  c_pre = np.array([0.,0.,0.,0.,0.,0.,0.])
  c_rec = np.array([0.,0.,0.,0.,0.,0.,0.])
  c_dcg = np.array([0.,0.,0.,0.,0.,0.,0.])
  pre_count = np.array([0,0,0,0,0,0,0])
  rec_count = np.array([0,0,0,0,0,0,0])
  dcg_count = np.array([0,0,0,0,0,0,0])
  
  user_list = np.loadtxt("./genre1-10/data/d11/user.csv",delimiter=",").astype(np.int64)  #U1
  
  item_list = np.loadtxt("./genre1-10/data/d11/item.csv",delimiter=",").astype(np.int64)  #genre1

  data = np.loadtxt("./genre1-10/data/d11/data.csv",delimiter=",").astype(np.int64)

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
      train = pd.DataFrame(index=user_list, columns=item_list).fillna(0)
      test = pd.DataFrame(index=user_list, columns=item_list).fillna(0)
      # make train and test matrix
      train_matrix, test_matrix = makeMatrix(data, train, test, train_index, test_index, user_list, item_list)
      #test_users = users_in_testdata(3, test_matrix, user_list)
      # learning model and calculate predicted user-item matrix
      pred = learning(method, train_matrix, train_index, data, user_list, item_list)

      # calculating precision, recall, and nDCG using "pred"
      lt_3[i,j] = search_lt_n(3, test_matrix)
      pre,rec,Pat3_ave[i,j], Pat3_std[i,j], Pat3_max[i,j], Pat3_min[i,j], Rat3_ave[i,j], Rat3_std[i,j], Rat3_max[i,j], Rat3_min[i,j], P0[i,j], P03[i,j], P06[i,j], P10[i,j], R02[i,j], R04[i,j], R06[i,j], R08[i,j], R10[i,j], new_c_pre, new_pre_count, new_c_rec, new_rec_count = precision(3, pred, test_matrix, item_list, lt_3[i,j])
      dcg, nDCGat3_ave[i,j], nDCGat3_std[i,j], nDCGat3_max[i,j], nDCGat3_min[i,j], D02[i,j], D04[i,j], D06[i,j], D08[i,j], D10[i,j], new_c_dcg, new_dcg_count = nDCG(3, pred, test_matrix, item_list, lt_3[i,j])
      # save users' values for each criterion
      precision[i,j,:] = pre
      recall[i,j,:]    = rec
      nDCG[i,j,:]      = dcg

      c_pre = c_pre + new_c_pre
      c_rec = c_rec + new_c_rec
      c_dcg = c_dcg + new_c_dcg
      pre_count = pre_count + new_pre_count
      rec_count = rec_count + new_rec_count
      dcg_count = dcg_count + new_dcg_count

      print("count:" + str(i) + ", precision=" + str(np.mean(pre[test_users.nonzero()])) + ", recall=" + str(np.mean(rec[test_users.nonzero()])) +", nDCG=" + str(np.mean(dcg[test_users.nonzero()])))

      # gavege collection, numpy_ndarray not used hereafter
      del pred
      del test_matrix
      del train_matrix
    gc.collect()
    j = j + 1

  c_pre = c_pre / 30
  c_rec = c_rec / 30
  c_dcg = c_dcg / 30
  pre_count = pre_count / 30
  rec_count = rec_count / 30
  dcg_count = dcg_count / 30
  print(c_pre)

  np.save("result/movie.review/" + method + "/Pat3_ave.npy", Pat3_ave)
  np.save("result/movie.review/" + method + "/Pat3_std.npy", Pat3_std)
  np.save("result/movie.review/" + method + "/Pat3_max.npy", Pat3_max)
  np.save("result/movie.review/" + method + "/Pat3_min.npy", Pat3_min)
  np.save("result/movie.review/" + method + "/Rat3_ave.npy", Rat3_ave)
  np.save("result/movie.review/" + method + "/Rat3_std.npy", Rat3_std)
  np.save("result/movie.review/" + method + "/Rat3_max.npy", Rat3_max)
  np.save("result/movie.review/" + method + "/Rat3_min.npy", Rat3_min)
  np.save("result/movie.review/" + method + "/nDCGat3_ave.npy", nDCGat3_ave)
  np.save("result/movie.review/" + method + "/nDCGat3_std.npy", nDCGat3_std)
  np.save("result/movie.review/" + method + "/nDCGat3_max.npy", nDCGat3_max)
  np.save("result/movie.review/" + method + "/nDCGat3_min.npy", nDCGat3_min)
  np.save("result/movie.review/" + method + "/P00.npy", P0)
  np.save("result/movie.review/" + method + "/P03.npy", P03)
  np.save("result/movie.review/" + method + "/P06.npy", P06)
  np.save("result/movie.review/" + method + "/P10.npy", P10)
  np.save("result/movie.review/" + method + "/D02.npy", D02)
  np.save("result/movie.review/" + method + "/D04.npy", D04)
  np.save("result/movie.review/" + method + "/D06.npy", D06)
  np.save("result/movie.review/" + method + "/D08.npy", D08)
  np.save("result/movie.review/" + method + "/D10.npy", D10)
  np.save("result/movie.review/" + method + "/R02.npy", R02)
  np.save("result/movie.review/" + method + "/R04.npy", R04)
  np.save("result/movie.review/" + method + "/R06.npy", R06)
  np.save("result/movie.review/" + method + "/R08.npy", R08)
  np.save("result/movie.review/" + method + "/R10.npy", R10)
  np.save("result/movie.review/" + method + "/lt_3.npy", lt_3)
  np.save("result/movie.review/" + method + "/c_pre.npy", c_pre) 
  np.save("result/movie.review/" + method + "/c_rec.npy", c_rec) 
  np.save("result/movie.review/" + method + "/c_dcg.npy", c_dcg) 
  np.save("result/movie.review/" + method + "/pre_count.npy", pre_count) 
  np.save("result/movie.review/" + method + "/rec_count.npy", rec_count) 
  np.save("result/movie.review/" + method + "/dcg_count.npy", dcg_count)

if __name__ == "__main__":

  # Pool : the number of CPU.
  p = Pool(CPU)
  '''
  methods = ["SVD", "NMF", "RMrate_liner", "D1_liner", "D2_liner", "D3_liner", "D4_liner", "D5_liner", "RMrate_square",
             "D1_square", "D2_square", "D3_square", "D4_square", "D5_square", "ML1_liner", "ML2_liner", "ML3_liner",
             "ML1_square", "ML2_square", "ML3_square"]
  '''
  methods = ["SVD", "ML3_liner"]
  p.map(calculate,methods)

  print("Program completed...")