"""Tests for quantum walks module."""

import numpy as np

from quantumium.quantum_walks.graphs import (
    line_graph,
    cycle_graph,
    complete_graph,
    hypercube_graph,
    grid_graph,
)
from quantumium.quantum_walks.discrete import (
    discrete_walk,
    grover_walk,
    hadamard_walk_line,
)
from quantumium.quantum_walks.continuous import (
    continuous_walk,
)


# ============================================================
# Graph Tests
# ============================================================


class TestLineGraph:
    def test_shape(self):
        A = line_graph(5)
        assert A.shape == (5, 5)

    def test_symmetric(self):
        A = line_graph(4)
        np.testing.assert_allclose(A, A.T)

    def test_connections(self):
        A = line_graph(4)
        assert A[0, 1] == 1.0
        assert A[0, 2] == 0.0
        assert A[2, 3] == 1.0

    def test_boundary_degree(self):
        A = line_graph(5)
        assert np.sum(A[0]) == 1.0
        assert np.sum(A[4]) == 1.0
        assert np.sum(A[2]) == 2.0

    def test_minimum_size(self):
        try:
            line_graph(1)
            assert False
        except ValueError:
            pass


class TestCycleGraph:
    def test_shape(self):
        A = cycle_graph(6)
        assert A.shape == (6, 6)

    def test_all_degree_2(self):
        A = cycle_graph(5)
        for i in range(5):
            assert np.sum(A[i]) == 2.0

    def test_wraparound(self):
        A = cycle_graph(4)
        assert A[0, 3] == 1.0
        assert A[3, 0] == 1.0

    def test_minimum_size(self):
        try:
            cycle_graph(2)
            assert False
        except ValueError:
            pass


class TestCompleteGraph:
    def test_shape(self):
        A = complete_graph(4)
        assert A.shape == (4, 4)

    def test_no_self_loops(self):
        A = complete_graph(5)
        for i in range(5):
            assert A[i, i] == 0.0

    def test_all_connected(self):
        A = complete_graph(4)
        for i in range(4):
            assert np.sum(A[i]) == 3.0


class TestHypercubeGraph:
    def test_dim1(self):
        A = hypercube_graph(1)
        assert A.shape == (2, 2)
        assert A[0, 1] == 1.0

    def test_dim2(self):
        A = hypercube_graph(2)
        assert A.shape == (4, 4)
        for i in range(4):
            assert np.sum(A[i]) == 2.0

    def test_dim3(self):
        A = hypercube_graph(3)
        assert A.shape == (8, 8)
        for i in range(8):
            assert np.sum(A[i]) == 3.0

    def test_symmetric(self):
        A = hypercube_graph(3)
        np.testing.assert_allclose(A, A.T)


class TestGridGraph:
    def test_shape(self):
        A = grid_graph(3, 4)
        assert A.shape == (12, 12)

    def test_corner_degree(self):
        A = grid_graph(3, 3)
        assert np.sum(A[0]) == 2.0

    def test_edge_degree(self):
        A = grid_graph(3, 3)
        assert np.sum(A[1]) == 3.0

    def test_center_degree(self):
        A = grid_graph(3, 3)
        assert np.sum(A[4]) == 4.0

    def test_symmetric(self):
        A = grid_graph(4, 3)
        np.testing.assert_allclose(A, A.T)


# ============================================================
# Discrete Walk Tests
# ============================================================


class TestDiscreteWalk:
    def test_probability_normalization(self):
        """Probabilities must sum to 1."""
        A = cycle_graph(8)
        result = discrete_walk(A, num_steps=10)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_starts_at_initial(self):
        """At step 0, all probability at initial vertex."""
        A = cycle_graph(6)
        result = discrete_walk(A, num_steps=0, initial_vertex=3)
        assert abs(result.position_probabilities[3] - 1.0) < 1e-10

    def test_spreads_over_time(self):
        """Walk should spread probability across vertices."""
        A = cycle_graph(10)
        result = discrete_walk(A, num_steps=5, coin="hadamard")
        non_zero = np.sum(result.position_probabilities > 1e-10)
        assert non_zero > 1

    def test_complete_graph(self):
        """Walk on complete graph."""
        A = complete_graph(4)
        result = discrete_walk(A, num_steps=10, coin="grover")
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_custom_coin_state(self):
        """Custom initial coin state."""
        A = cycle_graph(6)
        coin_state = np.array([0, 1], dtype=complex)
        result = discrete_walk(A, num_steps=3, initial_coin_state=coin_state)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_dft_coin(self):
        """DFT coin should preserve normalization."""
        A = complete_graph(4)
        result = discrete_walk(A, num_steps=5, coin="dft")
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_result_fields(self):
        A = cycle_graph(5)
        result = discrete_walk(A, num_steps=3)
        assert result.num_vertices == 5
        assert result.num_steps == 3
        assert result.coin_dimension == 2
        assert len(result.full_state) == 10


class TestHadamardWalkLine:
    def test_normalization(self):
        result = hadamard_walk_line(num_positions=21, num_steps=10)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_asymmetric_distribution(self):
        """Hadamard walk with |0> coin is asymmetric — biased left."""
        result = hadamard_walk_line(
            num_positions=41,
            num_steps=15,
            initial_coin_state=np.array([1, 0], dtype=complex),
        )
        center = 20
        left_prob = np.sum(result.position_probabilities[:center])
        right_prob = np.sum(result.position_probabilities[center + 1 :])
        assert left_prob != right_prob  # asymmetric

    def test_symmetric_coin_gives_symmetric_walk(self):
        """Balanced coin (|0>+i|1>)/sqrt(2) gives symmetric distribution."""
        result = hadamard_walk_line(
            num_positions=41,
            num_steps=10,
            initial_coin_state=np.array([1, 1j], dtype=complex) / np.sqrt(2),
        )
        center = 20
        probs = result.position_probabilities
        for offset in range(1, 11):
            assert abs(probs[center - offset] - probs[center + offset]) < 1e-10

    def test_default_center_start(self):
        result = hadamard_walk_line(num_positions=21, num_steps=0)
        assert abs(result.position_probabilities[10] - 1.0) < 1e-10

    def test_ballistic_spread(self):
        """Quantum walk spreads ballistically (linearly with steps), not diffusively."""
        result_short = hadamard_walk_line(num_positions=41, num_steps=5)
        result_long = hadamard_walk_line(num_positions=41, num_steps=15)
        center = 20
        var_short = sum(
            (i - center) ** 2 * p for i, p in enumerate(result_short.position_probabilities)
        )
        var_long = sum(
            (i - center) ** 2 * p for i, p in enumerate(result_long.position_probabilities)
        )
        assert var_long > 3 * var_short


class TestGroverWalk:
    def test_normalization(self):
        A = cycle_graph(8)
        result = grover_walk(A, num_steps=10)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_complete_graph_mixing(self):
        """Grover walk on complete graph has interesting mixing behavior."""
        A = complete_graph(4)
        result = grover_walk(A, num_steps=20)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10


# ============================================================
# Continuous Walk Tests
# ============================================================


class TestContinuousWalk:
    def test_probability_normalization(self):
        A = cycle_graph(6)
        result = continuous_walk(A, time=1.0)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_zero_time(self):
        """At t=0, state unchanged."""
        A = cycle_graph(5)
        result = continuous_walk(A, time=0.0, initial_vertex=2)
        assert abs(result.position_probabilities[2] - 1.0) < 1e-10

    def test_spreads_over_time(self):
        """Walk should spread probability."""
        A = cycle_graph(8)
        result = continuous_walk(A, time=2.0)
        non_zero = np.sum(result.position_probabilities > 1e-10)
        assert non_zero > 1

    def test_cycle_symmetry(self):
        """On cycle graph starting at 0, vertices equidistant from start
        should have equal probability."""
        n = 8
        A = cycle_graph(n)
        result = continuous_walk(A, time=1.5, initial_vertex=0)
        probs = result.position_probabilities
        assert abs(probs[1] - probs[n - 1]) < 1e-10
        assert abs(probs[2] - probs[n - 2]) < 1e-10

    def test_complete_graph_periodicity(self):
        """CTQW on K_n returns to start at t = 2*pi/n."""
        n = 4
        A = complete_graph(n)
        T = 2 * np.pi / n
        result = continuous_walk(A, time=T, initial_vertex=0)
        assert result.position_probabilities[0] > 0.5

    def test_laplacian_hamiltonian(self):
        """Walk with Laplacian should also preserve normalization."""
        A = line_graph(5)
        result = continuous_walk(A, time=1.0, use_laplacian=True)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10
        assert result.hamiltonian == "laplacian"

    def test_gamma_scaling(self):
        """Higher gamma = faster evolution."""
        A = line_graph(5)
        r1 = continuous_walk(A, time=1.0, gamma=0.5, initial_vertex=0)
        r2 = continuous_walk(A, time=1.0, gamma=2.0, initial_vertex=0)
        var1 = sum(i**2 * p for i, p in enumerate(r1.position_probabilities))
        var2 = sum(i**2 * p for i, p in enumerate(r2.position_probabilities))
        assert var2 > var1

    def test_custom_initial_state(self):
        """Superposition initial state."""
        A = line_graph(4)
        state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        result = continuous_walk(A, time=0.5, initial_state=state)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10

    def test_hypercube_walk(self):
        A = hypercube_graph(3)
        result = continuous_walk(A, time=1.0)
        assert abs(np.sum(result.position_probabilities) - 1.0) < 1e-10
        assert result.num_vertices == 8

    def test_result_fields(self):
        A = cycle_graph(5)
        result = continuous_walk(A, time=0.5)
        assert result.num_vertices == 5
        assert result.time == 0.5
        assert len(result.state) == 5
        assert result.hamiltonian == "adjacency"
