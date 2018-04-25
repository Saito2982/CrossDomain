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
  elif method == "ML3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
    R = ml.pv_ml3(train_matrix, eta0, u, v, attribute)
    return u

def learning2(method, train_matrix, train_index, data, user_list, item_list, u2):
  if method == "SVD":
    u, s, vt = svds(train_matrix, k=attribute)
    s_diag_matrix = np.diag(s)
    return np.dot(np.dot(u2, s_diag_matrix), vt)
  elif method == "ML3_liner":
    u, v = pv.rmrate(train_index, data, user_list, item_list, attribute)
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

def nDCG(n, pred, test, user_list, item_list, count_dict):
  # initialization
  nDCG = np.zeros(len(user_list))
  D02 = 0
  D04 = 0
  D06 = 0
  D08 = 0
  D10 = 0
  l = 0
  count = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c = np.array([0,0,0,0,0,0,0])

  for i,p in enumerate(pred):
    # user : i
    # predicted score : p

    # initialize DCG and iDCG
    DCG = 0.
    iDCG = 0.

    # extract test data for user i
    t = test[i]

    # ground_truth : non zero list for test data
    ground_truth = t.nonzero()

    # predicted score corresponding to test data
    p_test = p[ground_truth]

    # ranking of predicted score
    ranking_p_arg = np.argsort(p_test)[::-1]

    # item ID of test data
    test_item = item_list[ground_truth]

    # test data rating
    truth = t[ground_truth]

    # ranking of test data's ratings
    ranking_t = np.sort(truth)[::-1]

    # the number of evaluation in test data more than n
    if len(ranking_p_arg) >= n:


      # j : recommendation result of top-N
      # k : item ID in test data
      for j in range(n):
        for k in range(len(test_item)):

          # calculate DCG
          if k == ranking_p_arg[j]:
            if j == 0:
              DCG = truth[k]
            else:
              DCG = DCG + (truth[k] / math.log(j + 1, 2))

        # calculate iDCG
        if j == 0:
          iDCG = ranking_t[j]
        else:
          iDCG = iDCG + (ranking_t[j] / math.log(j + 1, 2))

      # calc111ulate nDCG
      nDCG[i] = DCG / iDCG

      if nDCG[i] <= 0.2:
        D02 = D02 + 1
      elif nDCG[i] <= 0.4:
        D04 = D04 + 1
      elif nDCG[i] <= 0.6:
        D06 = D06 + 1
      elif nDCG[i] <= 0.8:
        D08 = D08 + 1
      else:
        D10 = D10 + 1
      
      if len(ranking_p_arg) <= 3:
        count[0] = count[0] + nDCG[i]
        count_c[0] = count_c[0] + 1
      elif len(ranking_p_arg) <= 6:
        count[1] = count[1] + nDCG[i]
        count_c[1] = count_c[1] + 1
      elif len(ranking_p_arg) <= 10:
        count[2] = count[2] + nDCG[i]
        count_c[2] = count_c[2] + 1
      elif len(ranking_p_arg) <= 20:
        count[3] = count[3] + nDCG[i]
        count_c[3] = count_c[3] + 1
      elif len(ranking_p_arg) <= 30:
        count[4] = count[4] + nDCG[i]
        count_c[4] = count_c[4] + 1
      elif len(ranking_p_arg) <= 40:
        count[5] = count[5] + nDCG[i]
        count_c[5] = count_c[5] + 1
      else:
        count[6] = count[6] + nDCG[i]
        count_c[6] = count_c[6] + 1

  count = count / count_c
  return nDCG, np.mean(nDCG), np.std(nDCG), np.max(nDCG), np.min(nDCG), D02, D04, D06, D08, D10, count, count_c

def precision(n, pred, test, user_list, item_list, count_dict):
  # initialization
  precision = np.zeros(len(user_list))
  recall = np.zeros(len(user_list))
  p00 = 0
  p033 = 0
  p066 = 0
  p100 = 0
  r02 = 0
  r04 = 0
  r06 = 0
  r08 = 0
  r10 = 0
  count_pre = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c_pre = np.array([0,0,0,0,0,0,0])
  count_rec = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c_rec = np.array([0,0,0,0,0,0,0])
  count_recom = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_recom2 = np.array([0.,0.,0.,0.,0.,0.,0.])
  x = np.array( [] )

  for i, p in enumerate(pred):
    #initialization
    tp = 0
    fp = 0
    truth_all = 0
    t = test[i]
    ground_truth = t.nonzero()
    ground_truth2 = p.nonzero()
    ranking_p_arg = np.argsort(p[ground_truth])[::-1]
    ranking_p_arg2 = np.argsort(p[ground_truth2])[::-1]
    test_item = item_list[ground_truth]

    if len(ranking_p_arg2) >= 3:
      print(i,item_list[ranking_p_arg2[0:3]])
      if i == 0:
        x = np.append( x, ranking_p_arg2 )
      else:
        for v in range(n):
          if count_dict[item_list[ranking_p_arg2[v]]] <= 3:
            count_recom[0] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[0] += 1
          elif count_dict[item_list[ranking_p_arg2[v]]] <= 6:
            count_recom[1] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[1] += 1
          elif count_dict[item_list[ranking_p_arg2[v]]] <= 10:
            count_recom[2] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[2] += 1
          elif count_dict[item_list[ranking_p_arg2[v]]] <= 20:
            count_recom[3] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[3] += 1
          elif count_dict[item_list[ranking_p_arg2[v]]] <= 30:
            count_recom[4] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[4] += 1
          elif count_dict[item_list[ranking_p_arg2[v]]] <= 40:
            count_recom[5] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[5] += 1
          else:
            count_recom[6] += 1
            if ranking_p_arg2[v] not in x:
              x = np.append( x, ranking_p_arg2[v] )
              count_recom2[6] += 1

    if len(ranking_p_arg) >= 3:
      # true ratings
      truth = t[ground_truth]

      for j in range(n):
        for k in range(len(test_item)):
          if k == ranking_p_arg[j]:
            # good impression for items
            if truth[k] >= 4.:
              tp = tp + 1.0
            # bad impression
            else:
              fp = fp + 1.0

      # all items having good impression
      for j in range(len(truth)):
        if truth[j] >= 4.0:
          truth_all += 1

      # calculate precision
      precision[i] = tp / (tp + fp)

      # calculate recall
      if truth_all > 0:
        recall[i] = tp / truth_all

      if precision[i] == 0:
        p00 = p00 + 1
      elif precision[i] < 0.4:
        p033 = p033 + 1
      elif precision[i] < 0.7:
        p066 = p066 + 1
      else:
        p100 = p100 + 1

      if recall[i] <= 0.2:
        r02 = r02 + 1
      elif recall[i] <= 0.4:
        r04 = r04 + 1
      elif recall[i] <= 0.6:
        r06 = r06 + 1
      elif recall[i] <= 0.8:
        r08 = r08 + 1
      else:
        r10 = r10 + 1

      if len(ranking_p_arg) <= 3:
        count_pre[0] = count_pre[0] + precision[i]
        count_c_pre[0] = count_c_pre[0] + 1
      elif len(ranking_p_arg) <= 6:
        count_pre[1] = count_pre[1] + precision[i]
        count_c_pre[1] = count_c_pre[1] + 1
      elif len(ranking_p_arg) <= 10:
        count_pre[2] = count_pre[2] + precision[i]
        count_c_pre[2] = count_c_pre[2] + 1
      elif len(ranking_p_arg) <= 20:
        count_pre[3] = count_pre[3] + precision[i]
        count_c_pre[3] = count_c_pre[3] + 1
      elif len(ranking_p_arg) <= 30:
        count_pre[4] = count_pre[4] + precision[i]
        count_c_pre[4] = count_c_pre[4] + 1
      elif len(ranking_p_arg) <= 40:
        count_pre[5] = count_pre[5] + precision[i]
        count_c_pre[5] = count_c_pre[5] + 1
      else:
        count_pre[6] = count_pre[6] + precision[i]
        count_c_pre[6] = count_c_pre[6] + 1

      if len(ranking_p_arg) <= 3:
        count_rec[0] = count_rec[0] + recall[i]
        count_c_rec[0] = count_c_rec[0] + 1
      elif len(ranking_p_arg) <= 6:
        count_rec[1] = count_rec[1] + recall[i]
        count_c_rec[1] = count_c_rec[1] + 1
      elif len(ranking_p_arg) <= 10:
        count_rec[2] = count_rec[2] + recall[i]
        count_c_rec[2] = count_c_rec[2] + 1
      elif len(ranking_p_arg) <= 20:
        count_rec[3] = count_rec[3] + recall[i]
        count_c_rec[3] = count_c_rec[3] + 1
      elif len(ranking_p_arg) <= 30:
        count_rec[4] = count_rec[4] + recall[i]
        count_c_rec[4] = count_c_rec[4] + 1
      elif len(ranking_p_arg) <= 40:
        count_rec[5] = count_rec[5] + recall[i]
        count_c_rec[5] = count_c_rec[5] + 1
      else:
        count_rec[6] = count_rec[6] + recall[i]
        count_c_rec[6] = count_c_rec[6] + 1

  count_pre = count_pre / count_c_pre
  count_rec = count_rec / count_c_rec
      
  return precision,recall,precision.mean(), precision.std(), precision.max(), precision.min(), recall.mean(), recall.std(), recall.max(), recall.min(), p00, p033, p066, p100, r02, r04, r06, r08, r10, count_pre, count_c_pre, count_rec, count_c_rec, count_recom , count_recom2

def search_lt_n(n, test_data):
  lt_n = 0
  for t in test_data:
    if t[t.nonzero()].shape[0] < n:
      lt_n = lt_n + 1

  return lt_n

def calculate(method):
  a=0.0 
  b=0.0
  c=0.0
  eta0 = 0.45

  set_data = sys.argv[1]
  setting = sys.argv[2]

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

  if setting == '4':
    #setting 4
    user_Mu = np.loadtxt("./genre"+ set_data +"/data/d11/user.csv",delimiter=",").astype(np.int64)
    user_Mv = np.loadtxt("./genre"+ set_data +"/data/d22/user.csv",delimiter=",").astype(np.int64)
    test_user = np.loadtxt("./genre"+ set_data +"/data/d12/user.csv",delimiter=",").astype(np.int64)

    item_Mu = np.loadtxt("./genre"+ set_data +"/data/d11/item.csv",delimiter=",").astype(np.int64)
    item_Mv = np.loadtxt("./genre"+ set_data +"/data/d22/item.csv",delimiter=",").astype(np.int64)
    test_item = np.loadtxt("./genre"+ set_data +"/data/d12/item.csv",delimiter=",").astype(np.int64)

    data_Mu = np.loadtxt("./genre"+ set_data +"/data/d11/data.csv",delimiter=",").astype(np.int64)
    data_Mv = np.loadtxt("./genre"+ set_data +"/data/d22/data.csv",delimiter=",").astype(np.int64)
    test_data = np.loadtxt("./genre"+ set_data +"/data/d12/data.csv",delimiter=",").astype(np.int64)

    train_index = np.loadtxt("./genre"+ set_data +"/data/d11/index.csv",delimiter=",").astype(np.int64)
    train_index2 = np.loadtxt("./genre"+ set_data +"/data/d22/index.csv",delimiter=",").astype(np.int64)
    train_index3 = np.loadtxt("./genre"+ set_data +"/data/d12/index.csv",delimiter=",").astype(np.int64)

  elif setting == '5':
    #setting 5
    user_Mu = np.loadtxt("./genre"+ set_data +"/data/d22/user.csv",delimiter=",").astype(np.int64)
    user_Mv = np.loadtxt("./genre"+ set_data +"/data/d11/user.csv",delimiter=",").astype(np.int64)
    test_user = np.loadtxt("./genre"+ set_data +"/data/d21/user.csv",delimiter=",").astype(np.int64)

    item_Mu = np.loadtxt("./genre"+ set_data +"/data/d22/item.csv",delimiter=",").astype(np.int64)
    item_Mv = np.loadtxt("./genre"+ set_data +"/data/d11/item.csv",delimiter=",").astype(np.int64)
    test_item = np.loadtxt("./genre"+ set_data +"/data/d21/item.csv",delimiter=",").astype(np.int64)

    data_Mu = np.loadtxt("./genre"+ set_data +"/data/d22/data.csv",delimiter=",").astype(np.int64)
    data_Mv = np.loadtxt("./genre"+ set_data +"/data/d11/data.csv",delimiter=",").astype(np.int64)
    test_data = np.loadtxt("./genre"+ set_data +"/data/d21/data.csv",delimiter=",").astype(np.int64)

    train_index = np.loadtxt("./genre"+ set_data +"/data/d22/index.csv",delimiter=",").astype(np.int64)
    train_index2 = np.loadtxt("./genre"+ set_data +"/data/d11/index.csv",delimiter=",").astype(np.int64)
    train_index3 = np.loadtxt("./genre"+ set_data +"/data/d21/index.csv",delimiter=",").astype(np.int64)
  else:
    print("Select setting.")
    sys.exit()

  for i in range(repeate):
    j = 0
    Mu_matrix = makeMatrix(data_Mu, train_index, user_Mu, item_Mu)
    u = learning(method, Mu_matrix, train_index, data_Mu, user_Mu, item_Mu)

    Mv_matrix = makeMatrix(data_Mv, train_index2, user_Mv, item_Mv)
    pred = learning2(method, Mv_matrix, train_index2, data_Mv, user_Mv, item_Mv, u)
    np.savetxt("./result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/pred_temp.csv", pred, delimiter=",")

    test_matrix = makeMatrix(test_data, train_index3, test_user, test_item)
    test_users = users_in_testdata(3, test_matrix, test_user)

    # calculating precision, recall, and nDCG using "pred"
    lt_3[i,j] = search_lt_n(3, test_matrix)
    count_dict = collections.Counter(test_data[:,1])
    pre,rec,Pat3_ave[i,j], Pat3_std[i,j], Pat3_max[i,j], Pat3_min[i,j], Rat3_ave[i,j], Rat3_std[i,j], Rat3_max[i,j], Rat3_min[i,j], P0[i,j], P03[i,j], P06[i,j], P10[i,j], R02[i,j], R04[i,j], R06[i,j], R08[i,j], R10[i,j], new_c_pre, new_pre_count, new_c_rec, new_rec_count, recom, recom2 = precision(3, pred, np.array(test_matrix.todense()), test_user, test_item,count_dict)
    dcg, nDCGat3_ave[i,j], nDCGat3_std[i,j], nDCGat3_max[i,j], nDCGat3_min[i,j], D02[i,j], D04[i,j], D06[i,j], D08[i,j], D10[i,j], new_c_dcg, new_dcg_count = nDCG(3, pred, np.array(test_matrix.todense()), test_user, test_item,count_dict)

    c_pre = c_pre + new_c_pre
    c_rec = c_rec + new_c_rec
    c_dcg = c_dcg + new_c_dcg
    pre_count = pre_count + new_pre_count
    rec_count = rec_count + new_rec_count
    dcg_count = dcg_count + new_dcg_count

    print("count:" + str(i) + ", precision=" + str(np.mean(pre[test_users.nonzero()])) + ", recall=" + str(np.mean(rec[test_users.nonzero()])) +", nDCG=" + str(np.mean(dcg[test_users.nonzero()])))
    a += np.mean(pre[test_users.nonzero()])
    b += np.mean(rec[test_users.nonzero()])
    c += np.mean(dcg[test_users.nonzero()])
    #del pred
    #del train_matrix
    #del test_matrix
    gc.collect()
    j = j + 1

  c_pre = c_pre / 10
  c_rec = c_rec / 10
  c_dcg = c_dcg / 10
  pre_count = pre_count / 10
  rec_count = rec_count / 10
  dcg_count = dcg_count / 10
  print(c_pre)
  print("Precision AVE : " + str(a / 10))
  print("Recall AVE : " + str(b / 10))
  print("nDCG AVE : " + str(c / 10))

  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Pat3_ave.npy", Pat3_ave)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Pat3_std.npy", Pat3_std)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Pat3_max.npy", Pat3_max)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Pat3_min.npy", Pat3_min)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Rat3_ave.npy", Rat3_ave)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Rat3_std.npy", Rat3_std)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Rat3_max.npy", Rat3_max)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/Rat3_min.npy", Rat3_min)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/nDCGat3_ave.npy", nDCGat3_ave)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/nDCGat3_std.npy", nDCGat3_std)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/nDCGat3_max.npy", nDCGat3_max)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/nDCGat3_min.npy", nDCGat3_min)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/P00.npy", P0)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/P03.npy", P03)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/P06.npy", P06)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/P10.npy", P10)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/D02.npy", D02)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/D04.npy", D04)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/D06.npy", D06)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/D08.npy", D08)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/D10.npy", D10)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/R02.npy", R02)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/R04.npy", R04)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/R06.npy", R06)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/R08.npy", R08)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/R10.npy", R10)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/lt_3.npy", lt_3)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_pre.npy", c_pre)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_rec.npy", c_rec)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/c_dcg.npy", c_dcg)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/pre_count.npy", pre_count)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/rec_count.npy", rec_count)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/dcg_count.npy", dcg_count)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/recom.npy", recom)
  np.save("result/movie.review/" + method + "/genre"+ set_data +"_set"+ setting +"/recom2.npy", recom2)

if __name__ == "__main__":

  # Pool : the number of CPU.
  p = Pool(CPU)
  '''
  methods = ["SVD", "NMF", "RMrate_liner", "D1_liner", "D2_liner", "D3_liner", "D4_liner", "D5_liner", "RMrate_square",
             "D1_square", "D2_square", "D3_square", "D4_square", "D5_square", "ML1_liner", "ML2_liner", "ML3_liner",
             "ML1_square", "ML2_square", "ML3_square"]
  '''
  methods = ["SVD","ML3_liner"]
  p.map(calculate,methods)

  print("Program completed...")