# Exact decoder (Milestone 1); probabilistic/noisy decoding added in Milestone 2.
from __future__ import annotations

import itertools

import numpy as np

from simulate import run_clean

_LOG_EPS = 1e-12


class DecodeTable:
    """Precomputed exact-decode lookup for one pooling matrix and a maximum
    of k simultaneous faults.

    Building the table also certifies whether the design is k-separable:
    every fault combination of size 0..k must produce a distinct pooled-test
    outcome pattern, or decoding would be ambiguous. This is a weaker (and
    much cheaper to satisfy) requirement than k-disjunctness, since it
    doesn't need to support the simple "eliminate items seen in a passing
    test" rule - it just needs each outcome to map back to one answer, which
    this table looks up directly.
    """

    def __init__(self, matrix: np.ndarray, k: int):
        self.matrix = matrix
        self.k = k
        n = matrix.shape[1]
        table: dict[bytes, frozenset[int]] = {}
        is_separable = True
        for size in range(k + 1):
            for combo in itertools.combinations(range(n), size):
                key = run_clean(matrix, combo).tobytes()
                if key in table:
                    is_separable = False
                else:
                    table[key] = frozenset(combo)
        self._table = table
        self.is_separable = is_separable

    def decode(self, outcomes: np.ndarray) -> set[int]:
        key = np.asarray(outcomes, dtype=bool).tobytes()
        faulty = self._table.get(key)
        if faulty is None:
            raise ValueError(
                f"pooled-test outcome doesn't match any fault combination "
                f"of size <= {self.k} - more simultaneous faults than this "
                "design tolerates, or noisy results (see Milestone 2)"
            )
        return set(faulty)


class ProbabilisticDecoder:
    """Hand-rolled Bayesian decoder for noisy pooled-test outcomes (Milestone 2).

    Rather than looking up an exact match (DecodeTable), this treats every
    candidate fault combination of size 0..k as a hypothesis, scores each one
    by Bayes' rule using:

    - a prior over combinations, built from a per-item independent fault
      probability (prior_fault_rate) - a combination of size s has prior
      mass prior_fault_rate**s * (1 - prior_fault_rate)**(n - s);
    - a likelihood of the observed outcome pattern under a binary symmetric
      noise channel: each pooled test's true (noiseless) outcome, as computed
      by simulate.run_clean, is independently flipped with probability
      flake_rate, so likelihood is entirely determined by Hamming distance
      between the observed pattern and the combo's noiseless outcome.

    Combination posteriors are normalized (log-sum-exp) into a probability
    distribution over the enumerated hypothesis space, then marginalized per
    item: posterior(i) = sum of posterior mass over every combo containing i.
    This stays within the same combinatorial hypothesis space DecodeTable
    already enumerates (all combos of size <= k), so it's tractable for PoC
    batch sizes without a general probabilistic-programming framework.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        k: int,
        flake_rate: float,
        prior_fault_rate: float | None = None,
    ):
        if not 0.0 <= flake_rate <= 1.0:
            raise ValueError("flake_rate must be in [0, 1]")
        n = matrix.shape[1]
        if prior_fault_rate is None:
            prior_fault_rate = 1.0 / n
        if not 0.0 < prior_fault_rate < 1.0:
            raise ValueError("prior_fault_rate must be in (0, 1)")

        self.matrix = matrix
        self.k = k
        self.flake_rate = flake_rate
        self.prior_fault_rate = prior_fault_rate
        self.n = n

        # Clip flake_rate away from the exact 0/1 boundary so a single
        # contradicted row doesn't zero out a combo's likelihood outright -
        # real observations always carry a little uncertainty about the
        # channel itself. At flake_rate=0 this recovers hard-decode behavior
        # in every practical sense (the clip is far below float precision
        # that matters for ranking).
        flake_c = min(max(flake_rate, _LOG_EPS), 1 - _LOG_EPS)
        self._log_flake = np.log(flake_c)
        self._log_1m_flake = np.log(1 - flake_c)

        log_p = np.log(prior_fault_rate)
        log_1mp = np.log(1 - prior_fault_rate)

        combos: list[tuple[int, ...]] = []
        expected_rows: list[np.ndarray] = []
        log_priors: list[float] = []
        for size in range(k + 1):
            for combo in itertools.combinations(range(n), size):
                combos.append(combo)
                expected_rows.append(run_clean(matrix, combo))
                log_priors.append(size * log_p + (n - size) * log_1mp)

        self._combos = combos
        self._expected = np.array(expected_rows, dtype=bool)  # (num_combos, T)
        self._log_priors = np.array(log_priors)  # (num_combos,)

    def combo_posterior(self, outcomes: np.ndarray) -> list[tuple[frozenset[int], float]]:
        """Posterior probability of every enumerated combo, given observed
        outcomes. Returns (combo, probability) pairs, sorted highest first."""
        outcomes = np.asarray(outcomes, dtype=bool)
        mismatches = (self._expected != outcomes[None, :]).sum(axis=1)
        t = outcomes.shape[0]
        log_lik = (t - mismatches) * self._log_1m_flake + mismatches * self._log_flake
        log_unnorm = self._log_priors + log_lik
        m = log_unnorm.max()
        weights = np.exp(log_unnorm - m)
        weights /= weights.sum()
        pairs = [
            (frozenset(combo), float(w)) for combo, w in zip(self._combos, weights)
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    def posterior(self, outcomes: np.ndarray) -> dict[int, float]:
        """Marginal posterior probability that each item is genuinely
        faulty, given observed (possibly noisy) outcomes."""
        item_post = np.zeros(self.n)
        for combo, weight in self.combo_posterior(outcomes):
            for i in combo:
                item_post[i] += weight
        return {i: float(item_post[i]) for i in range(self.n)}

    def map_estimate(self, outcomes: np.ndarray) -> set[int]:
        """The single highest-posterior fault combination (maximum a
        posteriori estimate) - the best single guess at the true fault set."""
        best_combo, _ = self.combo_posterior(outcomes)[0]
        return set(best_combo)

    def decode(self, outcomes: np.ndarray, threshold: float = 0.5) -> set[int]:
        """Items whose marginal posterior exceeds threshold."""
        return {i for i, p in self.posterior(outcomes).items() if p > threshold}
