import numpy as np
import math

#=======================================================================================================================
#  Modules
#=======================================================================================================================

#=======================================================================================================================
# nDCG
# argument : n ... how many recommend item (top-N)
#            pred ... predicted matrix
#            test ... matrix generated by test data
#            user / item _list ... user' or item' ID
# return : user's nDCG in ndarray
# role : calculate nDCG for each users
#=======================================================================================================================

def nDCG(n, pred, test, user_list, item_list):
  count = np.array([0.,0.,0.,0.,0.,0.,0.])
  count_c = np.array([0,0,0,0,0,0,0])
  # initialization
  nDCG = np.zeros(len(user_list))

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

  return nDCG, count

#=======================================================================================================================
# precision
# argument : n ... how many recommend item (top-N)
#            pred ... predicted matrix
#            test ... matrix generated by test data
#            user / item _list ... user' or item' ID
# return : user's precision and recall in ndarray
# role : calculate precision and recall for each users
#=======================================================================================================================

def precision(n, pred, test, user_list, item_list, count_dict):
  # initialization
  precision = np.zeros(len(user_list))
  recall = np.zeros(len(user_list))
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
    ranking_p_arg2 = np.argsort(p[ground_truth2])[::-1]
    ranking_p_arg = np.argsort(p[ground_truth])[::-1]
    test_item = item_list[ground_truth]

    print(i,item_list[ranking_p_arg2[0:2]])

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

      if len(ranking_p_arg2) >= 3:
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

      if truth_all > 0:
        recall[i] = tp / truth_all
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

  return precision, recall, count_pre, count_rec, count_recom, count_recom2
