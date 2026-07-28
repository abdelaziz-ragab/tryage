import itertools

import pytest

from decode import DecodeTable
from design import binary_encoding_matrix, build_pooling_matrix
from simulate import run_clean

BATCH_SIZES = (8, 16, 32)
FAULT_TOLERANCES = (1, 2, 3)


def _assert_exhaustive_round_trip(matrix, table, n, k):
    """Every combination of 0..k faulty items among n must decode exactly."""
    for size in range(k + 1):
        for combo in itertools.combinations(range(n), size):
            faulty = set(combo)
            outcomes = run_clean(matrix, faulty)
            decoded = table.decode(outcomes)
            assert decoded == faulty, (
                f"n={n} k={k} faulty={faulty} decoded={decoded}"
            )


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("k", FAULT_TOLERANCES)
def test_design_decode_round_trip_exhaustive(n, k):
    matrix = build_pooling_matrix(n, k, seed=0)
    assert matrix.shape[1] == n
    table = DecodeTable(matrix, k)
    assert table.is_separable
    _assert_exhaustive_round_trip(matrix, table, n, k)


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_binary_encoding_is_an_antichain(n):
    # Sanity check on the k=1 construction directly: no item's code may be a
    # bit-subset of another's, or decoding becomes ambiguous (see design.py).
    matrix = binary_encoding_matrix(n)
    codes = [frozenset(matrix[:, i].nonzero()[0].tolist()) for i in range(n)]
    for a, b in itertools.combinations(codes, 2):
        assert not a.issubset(b)
        assert not b.issubset(a)


def test_build_pooling_matrix_rejects_invalid_input():
    with pytest.raises(ValueError):
        build_pooling_matrix(0, 1)
    with pytest.raises(ValueError):
        build_pooling_matrix(8, 0)


def test_benchmark_table_vs_brute_force():
    """Log T (pooled tests needed) against N (brute-force tests) for every
    required (N, k) pair.

    Savings grow with N relative to k: k=1 saves substantially at every
    batch size tested here; k=2 saves once N is large enough (16, 32) but
    not yet at N=8; k=3 doesn't save at all in this range (confirmed by
    extending the search well past T=N) - the technique only pays off once N
    is large enough relative to k. That's a genuine result, not a bug:
    build_pooling_matrix falls back to the identity matrix (T=N) whenever no
    smaller design was found, which is never worse than brute force. See
    Milestone 3 (adaptive fallback) for how higher fault counts get handled
    without paying for a huge upfront non-adaptive design.
    """
    rows = []
    for n in BATCH_SIZES:
        for k in FAULT_TOLERANCES:
            matrix = build_pooling_matrix(n, k, seed=0)
            t = matrix.shape[0]
            assert DecodeTable(matrix, k).is_separable
            assert t <= n, f"n={n} k={k} used t={t} groups - worse than brute force"
            rows.append((n, k, t))

    print(f"\n{'N':>4} {'k':>3} {'T (groups)':>11} {'vs brute force':>16}")
    for n, k, t in rows:
        print(f"{n:>4} {k:>3} {t:>11} {t / n:>15.0%}")

    savings = {(n, k): t for n, k, t in rows}
    for n in BATCH_SIZES:
        assert savings[(n, 1)] < n * 0.85, f"expected clear savings at k=1, n={n}"
    for n in (16, 32):
        assert savings[(n, 2)] < n, f"expected some savings at k=2, n={n}"
