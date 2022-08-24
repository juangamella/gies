# Greedy Interventional Equivalence Search (GIES) Algorithm for Causal Discovery

This is a python implementation of the GIES algorithm from the paper [*"Characterization and Greedy Learning of Interventional
Markov Equivalence Classes of Directed Acyclic Graphs"*](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) by Alain Hauser and Peter Bühlmann.

The implementation is an extension of the [Python implementation](https://github.com/juangamella/ges) GES algorithm.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install gies
```

The _only_ dependency outside the Python Standard Library is `numpy>=1.15.0`. See [`requirements.txt`](https://github.com/juangamella/ges/blob/master/requirements.txt) for more details.

## When you should (and shouldn't) use this implementation

To the best of my knowledge, the only other public implementation of GIES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from Python through an easy-to-use wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox), but given its scope, this library contains many additional dependencies (including PyTorch) and still requires you to have `R`.

Thus, **this implementation might be for you if**:

- you want a dependency-light implementation (the only dependency outside the Python Standard Library is numpy), or
- you want to rewrite parts of GES for your own research, but you'd rather do it in Python. The code has been written with an emphasis on readability, and everything is thoroughly documented and referenced back to the [GES](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf)/[GIES](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) papers.

**You should not use this implementation if:**

- you have no interest in modifying GES itself, *and*
- you care about speed, as the `pcalg` implementation is highly optimized and is **very** fast.

## Running the algorithm

### Using the Gaussian BIC score: `gies.fit_bic`

GIES comes ready to use with the [Gaussian BIC score](https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case), i.e. the l0-penalized Gaussian likelihood score. This is the variant which is commonly found in the literature, and the one which was implemented in the original paper. It is made available under the function `gies.fit_bic`.

```python
gies.fit_bic(data, I, A0 = None, phases = ['forward', 'backward', 'turning'], debug = 0)
```

**Parameters**

- **data** (np.array): the matrix containing the observations of each variable (each column corresponds to a variable).
- **I** (list of lists of ints): the family of intervention targets, with each list being the targets in the corresponding environment.
- **A0** (np.array, optional): the initial CPDAG on which GES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GIES procedure are run, and in which order. Defaults to `['forward', 'backward', 'turning']`.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG.
- **total_score** (float): the score of the estimate.

**Example**

Here [sempler](https://github.com/juangamella/sempler) is used to generate an observational sample from a Gaussian SCM, but this is not a dependency.

```python
import ges
import sempler
import numpy as np

# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

# Run GES with the Gaussian BIC score
estimate, score = ges.fit_bic(data)

print(estimate, score)

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 21511.315220683457
```

### Using a custom score: `gies.fit`

While [Hauser and Bühlmann (2012)](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdff) chose the BIC score, any score-equivalent and locally decomposable function is adequate. To run with another score of your choice, you can use

```python
gies.fit(score_class, A0 = None, phases = ['forward', 'backward', 'turning'], debug = 0)
```

where `score_class` is an instance of the class which implements your score. It should inherit from `gies.scores.DecomposableScore`, or define a `local_score` function and a few attributes (see [decomposable_score.py](https://github.com/juangamella/gies/blob/master/gies/scores/decomposable_score.py) for more details).

**Parameters**

- **score_class** (ges.scores.DecomposableScore): an instance of a class implementing a locally decomposable score, which inherits from `ges.scores.DecomposableScore`. See [decomposable_score.py](https://github.com/juangamella/ges/blob/master/ges/scores/decomposable_score.py) for more details.
- **A0** (np.array, optional): the initial CPDAG on which GES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GES procedure are run, and in which order. Defaults to `['forward', 'backward', 'turning']`.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG.
- **total_score** (float): the score of the estimate.

**Example**

Running GIES on a custom defined score (in this case the same Gaussian BIC score implemented in `ges.scores.GaussObsL0Pen`).

```python
import ges
import ges.scores
import sempler
import numpy as np

# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

# Define the score class
score_class = ges.scores.GaussIntL0Pen(data)

# Run GIES with the Gaussian BIC score
estimate, score = gies.fit(score_class)

print(estimate, score)

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 24002.112921580803
```

## Code Structure

All the modules can be found inside the `gies/` directory. These include:

  - `gies.main` which is the main module with the calls to start GIES, and contains the implementation of the insert, delete and turn operators.
  - `gies.utils` contains auxiliary functions and the logic to transform a PDAG into a CPDAG, used after each application of an operator.
  - `gies.scores` contains the modules with the score classes:
      - `ges.scores.decomposable_score` contains the base class for decomposable score classes (see that module for more details).
      - `ges.scores.gauss_obs_l0_pen` contains an implementation of the cached Gaussian BIC score, as used in the original GES paper.
   - `gies.test` contains the modules with the unit tests and tests comparing against the algorithm's implementation in the 'pcalg' package.   

## Tests

All components come with unit tests to match, and some property-based tests. The output of the overall procedure has been checked against that of the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) implementation over tens of thousands of random graphs. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There are, however, additional dependencies to run the tests. You can find these in [`requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/requirements_tests.txt) and [`R_requirements_tests.txt`](https://github.com/juangamella/ges/blob/master/R_requirements_tests.txt).

**Test modules**

They are in the sub package `ges.test`, in the directory `ges/test/`:

   - `test_decomposable_score.py`: tests for the decomposable score base class.
   - `test_gauss_bic.py`: tests for the Gaussian bic score.
   - `test_operators.py`: tests for the insert, delete and turn operators.
   - `test_pdag_to_cpdag.py`: tests the conversion from PDAG to CPDAG, which is applied after each application of an operator.
   - `test_utils.py`: tests the other auxiliary functions.
   - `ges.test.test_vs_pcalg`: compares the output of the algorithm vs. that of `pcalg` for randomly generated graphs.

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
