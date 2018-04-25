import numpy as np
import itertools
import random

user_list_all = np.loadtxt("./genre1-7/data/genre1-7_user.csv",delimiter=",").astype(np.int64)
u = np.random.permutation(user_list_all)
u1, u2 = np.split(u, [int(u.size * 0.5)])
np.savetxt("./genre1-7/data/u1.csv", u1, delimiter=",", fmt = "%d")
np.savetxt("./genre1-7/data/u2.csv", u2, delimiter=",", fmt = "%d")