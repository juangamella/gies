import numpy as np
import ges
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
import itertools
from ges.scores.exp_gauss_int_l0_pen import ExpGaussIntL0Pen
np.set_printoptions(suppress=True)
k = 3
while True:
    A = dag_avg_deg(k, 2, random_state=780)
    nxA = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
    nxA_undir = nx.DiGraph.to_undirected(nxA)
    if nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir):
        break
W = A * np.random.uniform(1, 2, size=A.shape)
scm = sempler.LGANM(W, (0, 15), (0, 0.2))
n = 200000
obs_data = scm.sample(n=n)
obs_data2 = scm.sample(n=n)
interv_data = scm.sample(n=n, do_interventions={1: (2, 10)})
# interv_data2 = scm.sample(n=n, do_interventions={2: (2, 10)})
data = [obs_data, interv_data]
A0 = np.zeros_like(A)

score_change = []
pdags = []
full_score_pdag = []
full_score_empty = []
list_interv = []

for L in range(k+1):
    for subset in itertools.combinations(range(k), L):
        list_interv.append(list(subset))

    sets = []
for subset in itertools.product(list_interv, repeat=len(data)-1):
    sets.append([[]] + list(subset))


for set in sets:
    P, P_score_change = ges.exp_fit_bic(data, set)
    score_change.append(P_score_change)
    pdags.append(P)
    score = ExpGaussIntL0Pen(data, set)
    full_score_pdag.append(score.full_score(ges.utils.pdag_to_dag(P)))
    full_score_empty.append(score.full_score(A0))
# for i in score_change:
#     print(i)
# print("\n")
# for i in full_score_pdag:
#     print(i)
# print("\n")

print(max(score_change))
max_elts = np.isclose(score_change, max(score_change), rtol=1e-10)
for i, set in enumerate(sets):
    print(set, score_change[i])
max_change = [set for set, elt in zip(sets, max_elts) if elt == True]
print(max_change)
print(max(full_score_pdag))
max_elts = np.isclose(full_score_pdag, max(full_score_pdag))
print([set for set, elt in zip(sets, max_elts) if elt == True])

print([[], [1]] in max_change)
print(A)

count_markov_eqv = 0
markov_eqv = []
true_P = ges.utils.dag_to_cpdag(A)
A_copy = A.copy()
for i in [1]:
    A_copy[:, i] = 0
true_P_int = ges.utils.dag_to_cpdag(A_copy)
for ind, P in enumerate(pdags):

    # if True:
    if np.all(ges.utils.dag_to_cpdag(ges.utils.pdag_to_dag(P)) == true_P):
        P_int = P.copy()
        for i in sets[ind]:
            P_int[:, i] = 0
        if np.all(ges.utils.dag_to_cpdag(ges.utils.pdag_to_dag(P_int)) == true_P_int):
            count_markov_eqv += 1
            print(sets[ind])
            print(pdags[ind])
true_index = sets.index([[], [1]])
elts = np.isclose(score_change, score_change[true_index], rtol=1e-10)
change = [set for set, elt in zip(sets, elts) if elt == True]
pdags_change = [p for p, elt in zip(pdags, elts) if elt == True]
# for i, set in enumerate(change):
#     print(set, "\n", pdags_change[i])
#     print(ges.exp_fit_bic(data, set))
#     print(ExpGaussIntL0Pen(data, set).sample_cov)
#     print(ExpGaussIntL0Pen(data, set).part_sample_cov)
for i, set in enumerate(sets):
    print(set, "\n", pdags[i])
    print(ges.exp_fit_bic(data, set, debug=3))
    print(ExpGaussIntL0Pen(data, set).sample_cov)
    print(ExpGaussIntL0Pen(data, set).part_sample_cov)
print(len(change))
print(count_markov_eqv)
print(np.all(ges.utils.replace_unprotected(A, [[], [1]]) == pdags[true_index]))

