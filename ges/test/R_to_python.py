import numpy as np
import sempler
import ges
from ges.scores.gauss_int_l0_pen import GaussIntL0Pen
import ges.utils as utils
np.set_printoptions(suppress=True)

np.random.seed(12)
true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
W = true_A * np.random.uniform(0, 2, size=true_A.shape)
scm = sempler.LGANM(W, (0, 0), (1, 1))
n = 10000
n1 = 2
n2 = 2
obs_data = scm.sample(n=n)
interv_data_1 = scm.sample(n=n1, do_interventions={1: (0, 3)})
interv_data_2 = scm.sample(n=n2, do_interventions={2: (0, 3)})

data = [obs_data]
#interv = [[], [1], [2]]
interv = [[]]
score = GaussIntL0Pen(data, interv)
A_gies, score_gies_change = ges.fit_bic(data, interv)
A_gies_dag = utils.pdag_to_dag(A_gies)
global_score = score.full_score(A_gies_dag)
print(score._mle_full(A_gies_dag))
print(global_score)
print(score.full_score(np.zeros_like(A_gies)))
print(score_gies_change)
#datacsv = np.concatenate((obs_data, interv_data_1, interv_data_2))
datacsv = np.concatenate((obs_data))
np.savetxt("data", obs_data, delimiter=",")


print(score.full_score(true_A))
print(score.part_sample_cov[0])
