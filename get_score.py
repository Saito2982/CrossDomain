#!/usr/bin/python3

import numpy as np
import get_sql

def nonR(dataset):
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix.csv",delimiter=',')

  return np.argsort(user.dot(item.T))[:,::-1]

def R_EE(dataset):
  attributes = get_sql.count_attributes(dataset)
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix_extend.csv",delimiter=',')
  R    = np.c_[np.identity(attributes), -np.identity(attributes)]

  score = user.dot(R)
  score = np.argsort(score.dot(item.T))[:,::-1]

  return score

def R_2EE(dataset):
  attributes = get_sql.count_attributes(dataset)
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix_extend.csv",delimiter=',')
  R    = np.c_[2*np.identity(attributes), -np.identity(attributes)]

  score = user.dot(R)
  score = np.argsort(score.dot(item.T))[:,::-1]

  return score

def R_E2E(dataset):
  attributes = get_sql.count_attributes(dataset)
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix_extend.csv",delimiter=',')
  R    = np.c_[np.identity(attributes), -2*np.identity(attributes)]

  score = user.dot(R)
  score = np.argsort(score.dot(item.T))[:,::-1]

  return score

def R_OE(dataset):
  attributes = get_sql.count_attributes(dataset)
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix_extend.csv",delimiter=',')
  R    = np.c_[np.zeros([attributes,attributes]), -np.identity(attributes)]

  score = user.dot(R)
  score = np.argsort(score.dot(item.T))[:,::-1]

  return score

def R_EO(dataset):
  attributes = get_sql.count_attributes(dataset)
  user = np.loadtxt("./data/" + dataset + "/user_matrix.csv",delimiter=',')
  item = np.loadtxt("./data/" + dataset + "/item_matrix_extend.csv",delimiter=',')
  R    = np.c_[np.identity(attributes), np.zeros([attributes,attributes])]

  score = user.dot(R)
  score = np.argsort(score.dot(item.T))[:,::-1]

  return score

