import gies
import gies.scores
import sempler
import numpy as np

# Generate observational data from a Gaussian SCM using sempler
A = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
)
W = A * np.random.uniform(1, 2, A.shape)  # sample weights
scm = sempler.LGANM(W, (1, 2), (1, 2))
data = [scm.sample(n=5000), scm.sample(n=5000, do_interventions={2: (0, 5)})]

# Define the score class
interventions = [[], [2]]
score_class = gies.scores.GaussIntL0Pen(data, interventions)

# Run GIES with the gaussian BIC score
estimate, score = gies.fit(score_class)

print(estimate, score)

# Output
# [[0. 1. 1. 1. 1.]
#  [1. 0. 1. 1. 1.]
#  [0. 0. 0. 1. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]] 51314.58872363333
