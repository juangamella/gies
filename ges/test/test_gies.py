import unittest
import numpy as np
import sempler

import ges
from ges.scores.gauss_int_l0_pen import GaussIntL0Pen


class TestGies(unittest.TestCase):
    np.random.seed(12)
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    W = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(W, (0, 0), (0.3, 0.4))
    p = len(W)
    n = 10000
    n1 = 10000
    n2 = 10000
    obs_data = scm.sample(n=n)
    interv_data_1 = scm.sample(n=n1, do_interventions={1: (1, 3)})
    interv_data_2 = scm.sample(n=n2, do_interventions={2: (2, 3)})

    data = [obs_data, interv_data_1, interv_data_2]
    interv = [[], [1], [2]]
    score = GaussIntL0Pen(data, interv)
    A_gies, score_gies_change = ges.fit_bic(data, interv)

    def test_fullscore_true_vs_empty(self):
            print("Score of true vs empty graph")
            true_score = self.score.full_score(self.true_A)
            self.assertIsInstance(true_score, float)
            # Compute score of unconnected graph
            score = self.score.full_score(np.zeros((self.p, self.p)))
            self.assertIsInstance(score, float)
            print("True DAG vs empty:", true_score, score)
            self.assertGreater(true_score, score)

    def test_score_decomposability_obs(self):
        # As a black-box test, make sure the score functions
        # preserve decomposability
        print("Decomposability of observational score")
        full_score = self.score.full_score(self.true_A)
        acc = 0
        for (j, pa) in self.factorization:
            local_score = self.score.local_score(j, pa)
            print("  ", j, pa, local_score)
            acc += local_score
        print("Full vs. acc:", full_score, acc)
        self.assertAlmostEqual(full_score, acc, places=2)

    def test_fullscore_gies_vs_gies_score_changes(self):
        print("The fullscore of the GIES PDAG vs the score returned by GIES")
        A_gies_dag = ges.utils.pdag_to_dag(self.A_gies)
        score_full_gies = self.score.full_score(A_gies_dag)
        self.assertIsInstance(score_full_gies, float)
        # TODO: ...
        score = self.score.full_score(np.zeros((self.p, self.p))) + self.score_gies_change
        self.assertIsInstance(score, float)
        print("GIES DAG vs empty + GIES changes:",score_full_gies , score)
        self.assertAlmostEqual(score_full_gies, score, places=2)

    def test_fullscore_gies_vs_true_score(self):
        print("The fullscore of the GIES PDAG vs the true score")
        A_gies_dag = ges.utils.pdag_to_dag(self.A_gies)
        score_full_gies = self.score.full_score(A_gies_dag)
        self.assertIsInstance(score_full_gies, float)
        # the true score
        true_score = self.score.full_score(self.true_A)
        self.assertIsInstance(true_score, float)
        print("GIES DAG vs empty + GIES changes:",score_full_gies , true_score)
        self.assertAlmostEqual(score_full_gies, true_score, places=2)

    def test_fullscore_all_dags(self):
        cpdag_A = ges.utils.replace_unprotected(self.true_A)
        indexes = list(range(len(cpdag_A)))
        dags = ges.utils.pdag_to_all_dags(cpdag_A, indexes)
        score_dags = list(np.zeros(len(dags)))
        for index, dag in enumerate(dags):
            score_dags[index] = self.score.full_score(dag)
            if index > 1:
                self.assertAlmostEqual(score_dags[index-1], score_dags[index], places=2)
        print(score_dags)


