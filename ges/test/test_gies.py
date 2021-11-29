import numpy as np
import sempler
import ges
from ges import utils
from ges.scores.gauss_int_l0_pen import GaussIntL0Pen

A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape)
sample = sempler.LGANM(W, (1, 2), (1, 2))

data1 = sample.sample(n=5000)
data2 = sample.sample(n=5000, do_interventions={4: (1, 2)})
data3 = sample.sample(n=5000, do_interventions={3: (1, 2)})
data = [data1, data2]
interv = [[], [4]]


print(utils.replace_unprotected(A, interv))
print(ges.fit_bic(data, interv, debug=0))

# score = GaussIntL0Pen(data, interv)
# print(score.full_score(A))


A = np.array([[0, 1], [0, 0]])
W = A * np.random.uniform(1, 2, A.shape)
sample = sempler.LGANM(W, (1, 2), (1, 2))
data1 = sample.sample(n=5000)
data2 = sample.sample(n=5000, do_interventions={0: (15, 4)})
data = [data1, data2]
interv = [[], [0]]

