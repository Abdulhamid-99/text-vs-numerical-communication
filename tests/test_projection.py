"""GaussianProjection — Johnson-Lindenstrauss roundtrip + determinism."""
import numpy as np
import pytest

from budgetfair.comm.projection import GaussianProjection


def test_matrix_shape_and_dtype():
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    m = p.matrix
    assert m.shape == (128, 4096)
    assert m.dtype == np.float32


def test_same_seed_gives_same_matrix():
    p1 = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    p2 = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    np.testing.assert_array_equal(p1.matrix, p2.matrix)


def test_different_seeds_give_different_matrices():
    p1 = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    p2 = GaussianProjection(source_dim=4096, target_dim=128, seed=43)
    assert not np.array_equal(p1.matrix, p2.matrix)


def test_project_output_shape():
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=0)
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    out = p.project(v)
    assert out.shape == (128,)
    assert out.dtype == np.float32


def test_project_rejects_wrong_input_shape():
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=0)
    with pytest.raises(ValueError):
        p.project(np.zeros(2048, dtype=np.float32))


def test_target_dim_must_not_exceed_source_dim():
    with pytest.raises(ValueError):
        GaussianProjection(source_dim=128, target_dim=4096, seed=0)


def test_jl_distance_preservation():
    """JL lemma: pairwise distances should be preserved within an epsilon factor.

    For 50 random pairs at source_dim=4096, target_dim=128, the median ratio of
    projected distance to original distance should be ~1.0 (this is the
    norm-preservation property our scaling enforces). Loose tolerance because
    target_dim=128 is below the strict JL bound for 50 pairs.
    """
    rng = np.random.RandomState(1)
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=42)

    n_pairs = 50
    ratios = []
    for _ in range(n_pairs):
        a = rng.randn(4096).astype(np.float32)
        b = rng.randn(4096).astype(np.float32)
        d_orig = np.linalg.norm(a - b)
        d_proj = np.linalg.norm(p.project(a) - p.project(b))
        ratios.append(d_proj / d_orig)

    median = float(np.median(ratios))
    # Should be close to 1 (norm-preserving). Allow generous slack since
    # individual pairs vary and target_dim=128 is small.
    assert 0.8 < median < 1.2, f"median distance ratio {median} far from 1.0"


def test_jl_inner_product_preservation():
    """Cosine similarity is approximately preserved by random projection."""
    rng = np.random.RandomState(2)
    p = GaussianProjection(source_dim=4096, target_dim=512, seed=42)

    a = rng.randn(4096).astype(np.float32)
    b = rng.randn(4096).astype(np.float32)
    cos_orig = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    a_p = p.project(a)
    b_p = p.project(b)
    cos_proj = float(np.dot(a_p, b_p) / (np.linalg.norm(a_p) * np.linalg.norm(b_p)))
    # Two random Gaussian vectors are approximately orthogonal in both spaces.
    # Both cosines should be small in magnitude, and close to each other.
    assert abs(cos_orig - cos_proj) < 0.1


def test_zero_vector_projects_to_zero():
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    v = np.zeros(4096, dtype=np.float32)
    out = p.project(v)
    np.testing.assert_array_equal(out, np.zeros(128, dtype=np.float32))


def test_linearity():
    """Projection is a linear map: P(a*v + b*w) = a*P(v) + b*P(w)."""
    p = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    rng = np.random.RandomState(3)
    v = rng.randn(4096).astype(np.float32)
    w = rng.randn(4096).astype(np.float32)
    a, b = 2.5, -1.7
    lhs = p.project(a * v + b * w)
    rhs = a * p.project(v) + b * p.project(w)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-5)
