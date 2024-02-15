# Greedy Interventional Equivalence Search (GIES) Algorithm for Causal Discovery

[![PyPI version](https://badge.fury.io/py/gies.svg)](https://badge.fury.io/py/gies)
[![Downloads](https://static.pepy.tech/badge/gies)](https://pepy.tech/project/gies)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This is a python implementation of the GIES algorithm from the paper [*"Characterization and Greedy Learning of Interventional
Markov Equivalence Classes of Directed Acyclic Graphs"*](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) by Alain Hauser and Peter Bühlmann.

The implementation is an extension of the [Python implementation](https://github.com/juangamella/ges) of the GES algorithm, and is the work of Olga Kolotuhina and Juan L. Gamella.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install gies
```

The _only_ dependency outside the Python Standard Library is `numpy>=1.15.0`. See [`requirements.txt`](requirements.txt) for more details.

## When you should use this implementation

To the best of our knowledge, the only other public implementation of GIES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from Python through an easy-to-use wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox), but given its scope, this library contains many additional dependencies (including PyTorch) and still requires you to have `R`.

Thus, **this implementation might be for you if** you want a dependency-light implementation in Python (the only dependency outside the Python Standard Library is numpy).

## Running the algorithm

### Using the Gaussian BIC score: `gies.fit_bic`

GIES comes ready to use with the [Gaussian BIC score](https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case), i.e. the l0-penalized Gaussian likelihood score. This is the variant which is commonly found in the literature, and the one which was implemented in the original paper. It is made available under the function `gies.fit_bic`.

```python
gies.fit_bic(data, I, A0 = None, phases = ['forward', 'backward', 'turning'], iterate = True, debug = 0)
```

**Parameters**

- **data** (np.array): the matrix containing the observations of each variable (each column corresponds to a variable).
- **I** (list of lists of ints): the family of intervention targets, with each list being the targets in the corresponding environment.
- **A0** (np.array, optional): the initial CPDAG on which GIES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GIES procedure are run, and in which order. Defaults to `['forward', 'backward', 'turning']`.
- **iterate** (boolean, default=True): Indicates whether the algorithm should repeat the given phases more than once, until the score is not improved.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG.
- **total_score** (float): the score of the estimate.

**Example**

Here [sempler](https://github.com/juangamella/sempler) is used to generate an observational sample from a Gaussian SCM, but this is not a dependency.

```python
import gies
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

# Run GIES with the gaussian BIC score
interventions = [[], [2]]
estimate, score = gies.fit_bic(data, interventions)

print(estimate, score)

# Output
# [[0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]] -29209.33670496673
```

### Using a custom score: `gies.fit`

While [Hauser and Bühlmann (2012)](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdff) chose the BIC score, any score-equivalent and locally decomposable function is adequate. To run with another score of your choice, you can use

```python
gies.fit(score_class, A0 = None, phases = ['forward', 'backward', 'turning'], iterate = True, debug = 0)
```

where `score_class` is an instance of the class which implements your score. It should inherit from `gies.scores.DecomposableScore`, or define a `local_score` function and a few attributes (see [decomposable_score.py](gies/scores/decomposable_score.py) for more details).

**Parameters**

- **score_class** (gies.scores.DecomposableScore): an instance of a class implementing a locally decomposable score, which inherits from `gies.scores.DecomposableScore`. See [decomposable_score.py](gies/scores/decomposable_score.py) for more details.
- **A0** (np.array, optional): the initial CPDAG on which GIES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GIES procedure are run, and in which order. Defaults to `['forward', 'backward', 'turning']`.
- **iterate** (boolean, default=True): Indicates whether the algorithm should repeat the given phases more than once, until the score is not improved.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG.
- **total_score** (float): the score of the estimate.

**Example**

Running GIES on a custom defined score (in this case the same Gaussian BIC score implemented in `gies.scores.GaussIntL0Pen`).

```python
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
# [[0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]] -30310.806829328416
```

## Code Structure

All the modules can be found inside the `gies/` directory. These include:

  - `gies.main` which is the main module with the calls to start GIES, and contains the implementation of the insert, delete and turn operators.
  - `gies.utils` contains auxiliary functions and the logic to transform a PDAG into a CPDAG, used after each application of an operator.
  - `gies.scores` contains the modules with the score classes:
      - `gies.scores.decomposable_score` contains the base class for decomposable score classes (see that module for more details).
      - `gies.scores.gauss_int_l0_pen` contains an implementation of the cached Gaussian BIC score.
  - `gies.test` contains the modules with the unit tests and tests comparing against the algorithm's R implementation in the 'pcalg' package.   

## Tests

All components come with unit tests to match, and some property-based tests. The output of the overall procedure has been checked against that of the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) implementation over tens of thousands of random graphs. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There are, however, additional dependencies to run the tests. You can find these in [`requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/requirements_tests.txt) and [`R_requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/R_requirements_tests.txt).

**Test modules**

They are in the sub package `gies.test`, in the directory `ges/test/`:

   - `test_decomposable_score.py`: tests for the decomposable score base class.
   - `test_int_score.py`: tests for the Gaussian bic score.
   - `test_pdag_to_cpdag.py`: tests the conversion from PDAG to CPDAG, which is applied after each application of an operator.
   - `test_utils.py`: tests the other auxiliary functions.
   - `test_vs_R`: compares the output of the algorithm to that of the R implementation in the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) package.

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
