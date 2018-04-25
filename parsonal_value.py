import pandas as pd
import numpy as np
import math

#=======================================================================================================================
# Modules
#=======================================================================================================================

#=======================================================================================================================
# rmrate : liner RMRate (item model sepalated negative and positive)
# rmrate_standard : liner RMRate (item model not sepalated)
# rmrate_square : squared RMRate (item model sepalated negative and positive)
# rmrate_square_standard : squared RMRate (item model not sepalated)
#
# argument train_index ... train data index
#          all_data ... evaluation data
#          user / item _list ... user' / item' ID list
#          attribute ... the number of attribute with dataset
# return   user_matrix, item_matrix ... this matrices represent user-RMRate matrix and item-RMRate matrix
# role     generate matrices to model user and item employing personal vlues
#=======================================================================================================================

def rmrate(train_index, all_data, user_list, item_list, attribute):
  # make training data
  train_data = pd.DataFrame(assign_index(all_data, train_index, attribute))
  # make user's or item's evaluation average list
  # xxxx_count : the number of evaluation having users / items
  user_ave, item_ave, user_count, item_count = make_ave(train_data, user_list, item_list)
  # make user / item polarity matrix, witch columns correspond to user / item and rows correspond to attributes
  # elements of matrices are counts matching polarity between ratings and attribute evaluations
  user_matrix, item_matrix = make_polarity_matrix(
    train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute)

  # calculate user RMRate matrix
  # this cords calculate (counts matching polarity / (counts matching polarity + not matching))
  for i in range(len(user_list)):
    for j in range(attribute):
      if user_count[i] != 0:
        # liner RMRate
        user_matrix[i, j] = user_matrix[i, j] / user_count[i]
      else:
        user_matrix[i, j] = 0

  # calculate item RMRate matrix
  for i in range(len(item_list)):
    for j in range(attribute*2):
      if item_count[i] != 0:
        # liner RMRate
        item_matrix[i, j] = item_matrix[i, j] / item_count[i]
      else:
        item_matrix[i, j] = 0

  return user_matrix, item_matrix

# if you want to know program detail, you reference to rmrate in line 5-49
def rmrate_square(train_index, all_data, user_list, item_list, attribute):
  train_data = pd.DataFrame(assign_index(all_data, train_index, attribute))
  user_ave, item_ave, user_count, item_count = make_ave(train_data, user_list, item_list)
  user_matrix, item_matrix = make_polarity_matrix(
    train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute)

  for i in range(len(user_list)):
    for j in range(attribute):
      if user_count[i] != 0:
        # to create squared RMRate
        user_matrix[i, j] = math.pow(user_matrix[i, j] / user_count[i], 2)
      else:
        user_matrix[i, j] = 0

  for i in range(len(item_list)):
    for j in range(attribute* 2):
      if item_count[i] != 0:
        # to create squared RMRate
        item_matrix[i, j] = math.pow(item_matrix[i, j] / item_count[i], 2)
      else:
        item_matrix[i, j] = 0

  return user_matrix, item_matrix

# if you want to know program detail, you reference to rmrate in line 5-49
def rmrate_standard(train_index, all_data, user_list, item_list, attribute):
  train_data = pd.DataFrame(assign_index(all_data, train_index, attribute))
  user_ave, item_ave, user_count, item_count = make_ave(train_data, user_list, item_list)
  user_matrix, item_matrix = make_polarity_matrix_standard(
    train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute)

  for i in range(len(user_list)):
    for j in range(attribute):
      if user_count[i] != 0:
        user_matrix[i, j] = user_matrix[i, j] / user_count[i]
      else:
        user_matrix[i, j] = 0

  for i in range(len(item_list)):
    for j in range(attribute):
      if item_count[i] != 0:
        item_matrix[i, j] = item_matrix[i, j] / item_count[i]
      else:
        item_matrix[i, j] = 0

  return user_matrix, item_matrix

# if you want to know program detail, you reference to rmrate in line 5-49
def rmrate_square_standard(train_index, all_data, user_list, item_list, attribute):
  train_data = pd.DataFrame(assign_index(all_data, train_index, attribute))
  user_ave, item_ave, user_count, item_count = make_ave(train_data, user_list, item_list)
  user_matrix, item_matrix = make_polarity_matrix_standard(
    train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute)

  for i in range(len(user_list)):
    for j in range(attribute):
      if user_count[i] != 0:
        user_matrix[i, j] = math.pow(user_matrix[i, j] / user_count[i], 2)
      else:
        user_matrix[i, j] = 0

  for i in range(len(item_list)):
    for j in range(attribute):
      if item_count[i] != 0:
        item_matrix[i, j] = math.pow(item_matrix[i, j] / item_count[i], 2)
      else:
        item_matrix[i, j] = 0

  return user_matrix, item_matrix

#=======================================================================================================================
# Methods
# followed programs are not necessary in main program, only use in personal_value.py
#=======================================================================================================================

#=======================================================================================================================
# assign_index
# role : change index list to evaluation data list
# this program equal to main programs "assign_index", so you reference to main.py if you want to know details.
#=======================================================================================================================

def assign_index(ALL, Purpose, attribute):
  Assigned = np.zeros((len(Purpose), attribute + 3)).astype(np.int64)

  for i, j in enumerate(Purpose):
    Assigned[i] = ALL[j]

  return Assigned

#=======================================================================================================================
# make_ave
# argument : train_data ... training data (DataFrame; pandas)
#            user / item _list ... user' or item' ID list
# return : user / item _ave ... average of user / item ratings
#          user / item _count ... the number of evaluation having users / items
# role : make user's or item's evaluation average list
#=======================================================================================================================

def make_ave(train_data, user_list, item_list):

  # initialize numpy list
  user_ave = np.zeros(len(user_list))
  user_count = np.zeros(len(user_list)).astype(np.int64)
  item_ave = np.zeros(len(item_list))
  item_count = np.zeros(len(item_list)).astype(np.int64)


  for line in train_data.itertuples():

    for i,user in enumerate(user_list):
      # line[1] : user ID
      if line[1] == user:
        # line[3] : comprehensive evaluation
        user_ave[i] = user_ave[i] + line[3]
        user_count[i] = user_count[i] + 1

    for i,item in enumerate(item_list):
      # line[2] : item ID
      if line[2] == item:
        item_ave[i] = item_ave[i] + line[3]
        item_count[i] = item_count[i] + 1

  # calculate average
  user_ave = user_ave / user_count
  item_ave = item_ave / item_count

  return user_ave, item_ave, user_count, item_count

#=======================================================================================================================
# make_polarity_matrix
# argument : train_data ... training data (DataFrame; pandas)
#            user / item _list ... user' or item' ID list
#            user / item _ave ... average of user / item ratings
#            user / item _count ... the number of evaluation having users / items
#            attribute ... the number of attributes in dataset
# return : user_matrix and item_matrix ... matrices with counts matching polarity
# role : make matrix witch elements are counts matching polarity between ratings and attribute evaluations
#        in addition, RMRate is divided in positive and negative
#=======================================================================================================================

def make_polarity_matrix(train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute):
  # initialize matrices
  user_matrix = np.zeros((len(user_list), attribute))
  item_matrix = np.zeros((len(item_list), attribute * 2))

  for line in train_data.itertuples():

    for i,user in enumerate(user_list):
      # line[1] : user ID
      if line[1] == user:
        # ignore users who do not have rating in training data
        if user_count[i] == 0:
          s_pol = 3
        # rating is more than average of comprehensive rating
        # this imply users give item good impression
        elif line[3] >= user_ave[i]:
          s_pol = 1
        # users give item bad impression
        else:
          s_pol = 0

        # for user i's attribute
        for j in range(attribute):
          # users give item' attribute good impression
          # line[j + 4] : attribute ratings
          if line[j + 1] >= user_ave[i]:
            a_pol = 1
          # users give item' attribute bad impression
          else:
            a_pol = 0

          # polarity matching
          if a_pol == s_pol:
            user_matrix[i, j] = user_matrix[i ,j] + 1

    for i, item in enumerate(item_list):
      # line[2] : item ID
      if line[2] == item:
        # ignore users who do not have rating in training data
        if item_count[i] == 0:
          s_pol = 2
        # rating is more than average of comprehensive rating
        # this imply item is had good impression given by users
        elif line[3] >= item_ave[i]:
          s_pol = 1
        # item is had bad impression given by users
        else:
          s_pol = 0

        # for item i's attribute
        for j in range(attribute):
          # good impression
          if line[j + 4] >= item_ave[i]:
            a_pol = 1
          # bad impression
          else:
            a_pol = 0

          # positive matching polarity
          #   match good impression between item' comprehensive rating and attribute rating
          if a_pol == 1:
            if s_pol == 1:
              item_matrix[i, j] = item_matrix[i, j] + 1.
          # negative matching polarity
          #   match bad impression between item' comprehensive rating and attribute rating
          elif a_pol == 0:
            if s_pol == 0:
              item_matrix[i, j + attribute] = item_matrix[i, j + attribute] + 1.

  return user_matrix, item_matrix

#=======================================================================================================================
# make_polarity_matrix
# role : make matrix witch elements are counts matching polarity between ratings and attribute evaluations
#        in addition, RMRate is ""not divided""
# details are shown in line 184-265
#=======================================================================================================================

def make_polarity_matrix_standard\
                (train_data, user_list, user_ave, user_count, item_list, item_ave, item_count, attribute):
  user_matrix = np.zeros((len(user_list), attribute))
  item_matrix = np.zeros((len(item_list), attribute))

  for line in train_data.itertuples():
    for i, user in enumerate(user_list):
      if line[1] == user:
        if user_count[i] == 0:
          s_pol = 2
        elif line[3] >= user_ave[i]:
          s_pol = 1
        else:
          s_pol = 0

        for j in range(attribute):
          if line[j + 4] >= user_ave[i]:
            a_pol = 1
          else:
            a_pol = 0


          if a_pol == s_pol:
            user_matrix[i, j] = user_matrix[i, j] + 1

    for i, item in enumerate(item_list):
      if line[2] == item:
        if item_count[i] == 0:
          s_pol = 2
        elif line[3] >= item_ave[i]:
          s_pol = 1
        else:
          s_pol = 0

        for j in range(attribute):
          if line[j + 4] >= item_ave[i]:
            a_pol = 1
          else:
            a_pol = 0

          if a_pol == s_pol:
            item_matrix[i, j] = item_matrix[i, j] + 1.

  return user_matrix, item_matrix
