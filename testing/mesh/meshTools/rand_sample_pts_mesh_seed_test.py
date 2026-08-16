"""
Reproducibility of the random surface samplers.

Regression tests for gattia/pymskt#54: ``Mesh.rand_pts_around_surface`` had two
independent random draws that a caller could not reach, so identical inputs produced
different point clouds on every call.

* the base surface points went to ``pcu.sample_mesh_random`` / ``sample_mesh_poisson_disk``
  with no ``random_seed``, and pcu's ``random_seed=0`` default means "seed from the current
  time", not "seed 0";
* the offsets used ``np.random.default_rng()`` with no argument, which seeds itself from OS
  entropy and ignores ``np.random.seed()``.

Seeding only one of the two leaves the result random, so every test here checks the whole
call rather than a single draw.
"""

import numpy as np
import pyvista as pv
import pytest

from pymskt.mesh import Mesh
from pymskt.mesh.meshTools import pcu_random_seed, rand_sample_pts_mesh


@pytest.fixture
def sphere(tmp_path):
    path = str(tmp_path / "sphere.vtk")
    pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24).triangulate().save(path)
    return path


def around(path, seed, method="random", distribution="normal", n_pts=500):
    return Mesh(path).rand_pts_around_surface(
        n_pts=n_pts, surface_method=method, distribution=distribution, sigma=0.01, seed=seed
    )


class TestRandPtsAroundSurface:
    def test_the_same_seed_gives_the_same_points(self, sphere):
        assert np.array_equal(around(sphere, 42), around(sphere, 42))

    def test_different_seeds_give_different_points(self, sphere):
        assert not np.array_equal(around(sphere, 42), around(sphere, 7))

    def test_seed_zero_is_reproducible(self, sphere):
        """
        The one that would slip through. pcu reads ``random_seed=0`` as "use the clock", so
        passing a user's seed straight through would make ``seed=0`` silently mean
        "unseeded" -- and 0 is the first seed most people try.
        """
        assert np.array_equal(around(sphere, 0), around(sphere, 0))

    def test_the_default_is_still_unseeded(self, sphere):
        """
        Backwards compatibility: ``seed=None`` must behave exactly as before, including
        being immune to ``np.random.seed()``, which never reached either draw.
        """
        np.random.seed(0)
        first = around(sphere, None)
        np.random.seed(0)
        second = around(sphere, None)
        assert not np.array_equal(first, second)

    def test_the_laplace_distribution_is_seeded_too(self, sphere):
        assert np.array_equal(
            around(sphere, 5, distribution="laplace"),
            around(sphere, 5, distribution="laplace"),
        )


class TestRandSamplePtsMesh:
    """
    The surface-point draw on its own. Both methods are covered here; ``bluenoise`` is not
    covered end-to-end through ``rand_pts_around_surface`` because that path is broken for
    an unrelated reason -- ``sample_mesh_poisson_disk`` returns approximately, not exactly,
    ``num_samples`` points, so adding the offsets raises a broadcast error. That is a
    separate defect and is not addressed here.
    """

    @pytest.mark.parametrize("method", ["random", "bluenoise"])
    def test_the_same_seed_gives_the_same_points(self, sphere, method):
        mesh = Mesh(sphere)
        first = rand_sample_pts_mesh(mesh, n_pts=400, method=method, seed=11)
        second = rand_sample_pts_mesh(mesh, n_pts=400, method=method, seed=11)
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("method", ["random", "bluenoise"])
    def test_different_seeds_give_different_points(self, sphere, method):
        mesh = Mesh(sphere)
        first = rand_sample_pts_mesh(mesh, n_pts=400, method=method, seed=11)
        second = rand_sample_pts_mesh(mesh, n_pts=400, method=method, seed=12)
        assert first.shape != second.shape or not np.array_equal(first, second)


class TestPcuRandomSeed:
    def test_none_maps_to_pcus_unseeded_sentinel(self):
        assert pcu_random_seed(None) == 0

    def test_zero_does_not_map_to_zero(self):
        """Otherwise ``seed=0`` would mean "seed from the clock" inside pcu."""
        assert pcu_random_seed(0) != 0

    def test_it_is_deterministic(self):
        assert pcu_random_seed(3) == pcu_random_seed(3)

    def test_distinct_seeds_map_to_distinct_values(self):
        assert len({pcu_random_seed(i) for i in range(25)}) == 25

    def test_the_result_is_a_positive_32_bit_int(self):
        for seed in (0, 1, 12345, 2**31):
            value = pcu_random_seed(seed)
            assert isinstance(value, int)
            assert 0 < value < 2**31
