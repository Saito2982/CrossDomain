from urllib.request import urlopen
from bs4 import BeautifulSoup
import pymysql
import configparser
import time
import sys
import re
import mysql.connector
import numpy as np

import collections
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold
from multiprocessing import Pool

#np.set_printoptions(threshold=np.inf)
# Default values
CPU = 1
dataset = "movie.review"
eta0 = 0.45
repeate = 10
sepalate = 1
attribute = 5

config = configparser.ConfigParser()
config.read('config.ini')
setting = config['setting']
conn = pymysql.connect(
    host=setting['host'],
    #unix_socket=setting['socket'],
    user=setting['user'],
    password=setting['passwd'],
    db=setting['db'],
    charset='utf8'
)
cur = conn.cursor()
cur.execute("USE movie")

def calculate():
  genre1 = sys.argv[1]
  genre2 = sys.argv[2]
  setting = sys.argv[3]

  #user_id昇順で取得
  cur.execute("SELECT distinct(user_id) FROM (movie.info_genre inner join movie.review on review.movie_id = info_genre.movie_id) where (genre_id=" + genre1 + " or genre_id=" + genre2 + ") ORDER BY user_id;")
  all_user = cur.fetchall()

  #movie_id昇順で取得
  cur.execute("SELECT distinct(review.movie_id) FROM (movie.info_genre inner join movie.review on review.movie_id = info_genre.movie_id) where (genre_id=" + genre1 + " or genre_id=" + genre2 + ") ORDER BY review.movie_id;")
  all_movie = list(cur.fetchall())

  #random.sample(all_user, int(len(all_user)/2))
  #cur.execute("")
  print(all_user[:][0])
  #dataすべて取得
  cur.execute("SELECT genre_id,user_id,review.movie_id,total,story,actor,staging,movie,music FROM (movie.info_genre inner join movie.review on review.movie_id = info_genre.movie_id) where (genre_id=" + genre1 + " or genre_id=" + genre2 + ") ORDER BY user_id;")
  #user_id0から再定義



if __name__ == "__main__":

  # Pool : the number of CPU.
  #p = Pool(CPU)
  #methods = ["SVD","ML3_liner"]
  #p.map(calculate,methods)
  calculate()

  print("Program completed...")