import numpy as np

"""
Random Search Hyperparameter Tuning
─────────────────────────────────────
Samples `n_trials` random configurations uniformly within the
provided bounds and returns the one with the lowest objective value.

Why Random Search over GWO here?
  - GWO needs many wolves × iterations to converge → expensive for DL
  - Random Search with 10-15 trials gives good coverage of a 3D space
  - Each trial is already expensive (mini model fit), so fewer is better
  - Random Search is embarrassingly parallel if needed later

Usage:
    rs = RandomSearch(n_trials=12, verbose=True)
    best_pos, best_score = rs.optimize(objective_fn, lb, ub)

objective_fn: callable(1D np.array) → scalar (lower is better)
lb, ub: lists/arrays of same length as search space dimension
"""
"""
Random Search doesn't just sample randomly and guess. It:

Samples a random config (e.g. lr=0.001, dropout=0.3, scale=1.2)
Builds a real model with those params
Trains it for 2 epochs on your actual data subset (the 512 images loaded before the search)
Records the val_loss from those 2 epochs
Repeats 12 times, keeping whichever config gave the lowest val_loss

So it runs 12 mini training experiments, each 2 epochs long, on real data. The "random" part is only in how configs are chosen — evaluation is always empirical.
"""

class RandomSearch:
    """
    Random Search optimizer for continuous hyperparameter spaces.

    Replaces Grey Wolf Optimizer (GWO) for Combination 2.
    Simple, robust, and effective for low-dimensional spaces (2–5 params).
    """

    def __init__(self, n_trials=12, verbose=False, seed=None):
        """
        Parameters
        ----------
        n_trials : int
            Number of random configurations to evaluate (10–15 recommended).
        verbose : bool
            Print each trial's result if True.
        seed : int or None
            Random seed for reproducibility.
        """
        self.n_trials = max(1, int(n_trials))
        self.verbose  = bool(verbose)
        if seed is not None:
            np.random.seed(seed)

    def optimize(self, obj_fn, lb, ub):
        """
        Run random search over the parameter space.

        Parameters
        ----------
        obj_fn : callable
            Objective function; takes 1D np.array, returns scalar (minimize).
        lb : array-like
            Lower bounds for each dimension.
        ub : array-like
            Upper bounds for each dimension.

        Returns
        -------
        best_pos   : np.ndarray — best parameter vector found
        best_score : float      — corresponding objective value
        """
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        dim = lb.size

        best_pos   = None
        best_score = float("inf")

        for trial in range(self.n_trials):
            # Sample uniformly in [lb, ub]
            pos   = np.random.uniform(lb, ub, size=dim)
            score = float(obj_fn(pos))

            if self.verbose:
                preview = ", ".join(f"{v:.4f}" for v in pos)
                status  = " ← best" if score < best_score else ""
                print(f"[RS] Trial {trial + 1:>2}/{self.n_trials} | "
                      f"score={score:.6f} | params=[{preview}]{status}")

            if score < best_score:
                best_score = score
                best_pos   = pos.copy()

        if self.verbose:
            print(f"[RS] Done. Best score={best_score:.6f} | "
                  f"params=[{', '.join(f'{v:.4f}' for v in best_pos)}]")

        return best_pos, best_score