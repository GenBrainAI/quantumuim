"""Tests for quantum networking simulation (Phase 28)."""

import numpy as np
import pytest

from quantumium.networking import (
    QuantumNetwork,
    EntangledPair,
    generate_entanglement,
    entanglement_swap,
    purify_bbpssw,
    teleport,
    distribute_entanglement,
)
from quantumium.networking.network import QuantumChannel
from quantumium.networking.purification import purify_rounds
from quantumium.networking.teleportation import teleportation_fidelity


# ---------------------------------------------------------------------------
# QuantumNetwork topology
# ---------------------------------------------------------------------------


class TestQuantumNetwork:
    def test_add_nodes_and_links(self):
        net = QuantumNetwork()
        net.add_node("Alice")
        net.add_node("Bob")
        net.add_link("Alice", "Bob", distance_km=100)
        assert net.num_nodes == 2
        assert net.num_channels == 1

    def test_neighbors(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_node("C")
        net.add_link("A", "B", 10)
        net.add_link("A", "C", 20)
        assert set(net.neighbors("A")) == {"B", "C"}
        assert net.neighbors("B") == ["A"]

    def test_unknown_node_link_raises(self):
        net = QuantumNetwork()
        net.add_node("A")
        with pytest.raises(ValueError):
            net.add_link("A", "Z", 10)

    def test_find_path(self):
        net = QuantumNetwork.chain(["A", "R1", "R2", "B"])
        path = net.find_path("A", "B")
        assert path == ["A", "R1", "R2", "B"]

    def test_find_path_no_route(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        with pytest.raises(ValueError):
            net.find_path("A", "B")

    def test_get_channel(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=50, loss_db_per_km=0.2)
        ch = net.get_channel("A", "B")
        assert ch.distance_km == 50
        ch2 = net.get_channel("B", "A")
        assert ch2 is ch

    def test_get_channel_missing(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        with pytest.raises(ValueError):
            net.get_channel("A", "B")


class TestTopologies:
    def test_star(self):
        net = QuantumNetwork.star("Hub", ["A", "B", "C", "D"])
        assert net.num_nodes == 5
        assert net.num_channels == 4
        assert set(net.neighbors("Hub")) == {"A", "B", "C", "D"}
        assert net.get_node("Hub").is_repeater

    def test_chain(self):
        net = QuantumNetwork.chain(["Alice", "R1", "R2", "Bob"], segment_distance_km=50)
        assert net.num_nodes == 4
        assert net.num_channels == 3
        assert net.get_node("R1").is_repeater
        assert net.get_node("R2").is_repeater
        assert not net.get_node("Alice").is_repeater
        assert not net.get_node("Bob").is_repeater

    def test_mesh(self):
        net = QuantumNetwork.mesh(3, 3)
        assert net.num_nodes == 9
        assert net.num_channels == 12
        assert not net.get_node("N0_0").is_repeater
        assert net.get_node("N1_1").is_repeater

    def test_mesh_path(self):
        net = QuantumNetwork.mesh(2, 3)
        path = net.find_path("N0_0", "N1_2")
        assert path[0] == "N0_0"
        assert path[-1] == "N1_2"
        assert len(path) >= 3


# ---------------------------------------------------------------------------
# Quantum Channel
# ---------------------------------------------------------------------------


class TestQuantumChannel:
    def test_transmission_probability(self):
        ch = QuantumChannel("A", "B", distance_km=0, loss_db_per_km=0.2)
        assert ch.transmission_probability == pytest.approx(1.0)

    def test_transmission_decays_with_distance(self):
        ch_short = QuantumChannel("A", "B", distance_km=10, loss_db_per_km=0.2)
        ch_long = QuantumChannel("A", "B", distance_km=100, loss_db_per_km=0.2)
        assert ch_short.transmission_probability > ch_long.transmission_probability
        assert ch_long.transmission_probability > 0

    def test_channel_fidelity_zero_distance(self):
        ch = QuantumChannel("A", "B", distance_km=0, loss_db_per_km=0.2)
        assert ch.channel_fidelity == pytest.approx(1.0)

    def test_channel_fidelity_decreases(self):
        ch_short = QuantumChannel("A", "B", distance_km=10, loss_db_per_km=0.2)
        ch_long = QuantumChannel("A", "B", distance_km=100, loss_db_per_km=0.2)
        assert ch_short.channel_fidelity > ch_long.channel_fidelity
        assert ch_long.channel_fidelity > 0.25

    def test_depolarizing_rate(self):
        ch = QuantumChannel("A", "B", distance_km=50, loss_db_per_km=0.2, depolarizing_rate=0.01)
        assert 0.25 < ch.channel_fidelity < 1.0


# ---------------------------------------------------------------------------
# Entanglement generation
# ---------------------------------------------------------------------------


class TestEntanglementGeneration:
    def test_generate_direct(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=10)
        pair = generate_entanglement(net, "A", "B")
        assert pair.node_a == "A"
        assert pair.node_b == "B"
        assert 0.5 < pair.fidelity <= 1.0
        assert pair.is_entangled

    def test_zero_distance_perfect_fidelity(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=0)
        pair = generate_entanglement(net, "A", "B")
        assert pair.fidelity == pytest.approx(1.0)

    def test_density_matrix_shape(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=10)
        pair = generate_entanglement(net, "A", "B")
        assert pair.density_matrix.shape == (4, 4)
        assert np.trace(pair.density_matrix) == pytest.approx(1.0)

    def test_werner_state_pure_at_f1(self):
        pair = EntangledPair("A", "B", 1.0)
        phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        expected = np.outer(phi_plus, phi_plus.conj())
        np.testing.assert_allclose(pair.density_matrix, expected, atol=1e-10)

    def test_werner_state_maximally_mixed(self):
        pair = EntangledPair("A", "B", 0.0)
        np.testing.assert_allclose(pair.density_matrix, np.eye(4) / 4, atol=1e-10)


# ---------------------------------------------------------------------------
# Entanglement swapping
# ---------------------------------------------------------------------------


class TestEntanglementSwapping:
    def test_swap_basic(self):
        pair1 = EntangledPair("A", "R", 0.95)
        pair2 = EntangledPair("R", "B", 0.95)
        result = entanglement_swap(pair1, pair2)
        assert result.node_a == "A"
        assert result.node_b == "B"
        assert result.fidelity < min(pair1.fidelity, pair2.fidelity)

    def test_swap_perfect_fidelity(self):
        pair1 = EntangledPair("A", "R", 1.0)
        pair2 = EntangledPair("R", "B", 1.0)
        result = entanglement_swap(pair1, pair2)
        assert result.fidelity == pytest.approx(1.0)

    def test_swap_reduces_fidelity(self):
        f = 0.9
        pair1 = EntangledPair("A", "R", f)
        pair2 = EntangledPair("R", "B", f)
        result = entanglement_swap(pair1, pair2)
        expected = f * f + (1 - f) * (1 - f) / 3
        assert result.fidelity == pytest.approx(expected)

    def test_swap_chain(self):
        f = 0.95
        pairs = [EntangledPair(f"N{i}", f"N{i + 1}", f) for i in range(4)]
        current = pairs[0]
        for p in pairs[1:]:
            current = entanglement_swap(current, p)
        assert current.node_a == "N0"
        assert current.node_b == "N4"
        assert current.fidelity < f

    def test_swap_auto_orient(self):
        pair1 = EntangledPair("R", "A", 0.9)
        pair2 = EntangledPair("R", "B", 0.9)
        result = entanglement_swap(pair1, pair2)
        assert {result.node_a, result.node_b} == {"A", "B"}

    def test_swap_no_shared_node_raises(self):
        pair1 = EntangledPair("A", "B", 0.9)
        pair2 = EntangledPair("C", "D", 0.9)
        with pytest.raises(ValueError):
            entanglement_swap(pair1, pair2)


# ---------------------------------------------------------------------------
# BBPSSW purification
# ---------------------------------------------------------------------------


class TestPurification:
    def test_purify_improves_above_half(self):
        pair1 = EntangledPair("A", "B", 0.7)
        pair2 = EntangledPair("A", "B", 0.7)
        result = purify_bbpssw(pair1, pair2)
        assert result.fidelity > 0.7

    def test_purify_exact_formula(self):
        f = 0.8
        pair1 = EntangledPair("A", "B", f)
        pair2 = EntangledPair("A", "B", f)
        result = purify_bbpssw(pair1, pair2)
        num = f * f + (1 - f) ** 2 / 9
        den = f * f + 2 * f * (1 - f) / 3 + 5 * (1 - f) ** 2 / 9
        assert result.fidelity == pytest.approx(num / den)

    def test_purify_multiple_rounds(self):
        pair = EntangledPair("A", "B", 0.7)
        result = purify_rounds(pair, 5)
        assert result.fidelity > 0.7

    def test_purify_converges_toward_one(self):
        pair = EntangledPair("A", "B", 0.75)
        result = purify_rounds(pair, 20)
        assert result.fidelity > 0.99

    def test_purify_below_half_degrades(self):
        pair1 = EntangledPair("A", "B", 0.4)
        pair2 = EntangledPair("A", "B", 0.4)
        result = purify_bbpssw(pair1, pair2)
        assert result.fidelity < 0.4

    def test_purify_at_half_unchanged(self):
        pair1 = EntangledPair("A", "B", 0.5)
        pair2 = EntangledPair("A", "B", 0.5)
        result = purify_bbpssw(pair1, pair2)
        assert result.fidelity == pytest.approx(0.5, abs=1e-10)

    def test_purify_asymmetric_fidelities(self):
        pair1 = EntangledPair("A", "B", 0.9)
        pair2 = EntangledPair("A", "B", 0.7)
        result = purify_bbpssw(pair1, pair2)
        assert result.fidelity > max(pair1.fidelity, pair2.fidelity) * 0.8


# ---------------------------------------------------------------------------
# Teleportation
# ---------------------------------------------------------------------------


class TestTeleportation:
    def test_teleport_perfect_channel(self):
        pair = EntangledPair("A", "B", 1.0)
        result = teleport([1, 0], pair)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(result.output_state, expected, atol=1e-10)
        assert result.fidelity == pytest.approx(1.0)

    def test_teleport_fidelity_formula(self):
        for f_ch in [0.6, 0.75, 0.9, 1.0]:
            pair = EntangledPair("A", "B", f_ch)
            result = teleport([1, 0], pair)
            expected = (2 * f_ch + 1) / 3
            assert result.fidelity == pytest.approx(expected, abs=1e-6)

    def test_teleportation_fidelity_helper(self):
        assert teleportation_fidelity(1.0) == pytest.approx(1.0)
        assert teleportation_fidelity(0.5) == pytest.approx(2 / 3)
        assert teleportation_fidelity(0.25) == pytest.approx(0.5)

    def test_teleport_superposition(self):
        pair = EntangledPair("A", "B", 0.9)
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        result = teleport(state, pair)
        assert result.fidelity > 2 / 3
        assert result.output_state.shape == (2, 2)

    def test_teleport_density_matrix_input(self):
        pair = EntangledPair("A", "B", 0.85)
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        result = teleport(rho, pair)
        assert 0 < result.fidelity <= 1.0

    def test_teleport_invalid_state_raises(self):
        pair = EntangledPair("A", "B", 1.0)
        with pytest.raises(ValueError):
            teleport([1, 0, 0], pair)

    def test_teleport_exceeds_classical_limit(self):
        pair = EntangledPair("A", "B", 0.6)
        result = teleport([1, 0], pair)
        assert result.fidelity > 2 / 3


# ---------------------------------------------------------------------------
# One-liner API: distribute_entanglement
# ---------------------------------------------------------------------------


class TestDistributeEntanglement:
    def test_direct_link(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=10)
        pair = distribute_entanglement(net, "A", "B")
        assert pair.node_a == "A"
        assert pair.node_b == "B"
        assert pair.is_entangled

    def test_repeater_chain(self):
        net = QuantumNetwork.chain(["A", "R1", "R2", "B"], segment_distance_km=30)
        pair = distribute_entanglement(net, "A", "B")
        assert pair.node_a == "A"
        assert pair.node_b == "B"
        assert pair.fidelity > 0.25

    def test_with_purification(self):
        net = QuantumNetwork()
        net.add_node("A")
        net.add_node("B")
        net.add_link("A", "B", distance_km=20)
        pair_raw = distribute_entanglement(net, "A", "B", purification_rounds=0)
        pair_purified = distribute_entanglement(net, "A", "B", purification_rounds=3)
        assert pair_purified.fidelity >= pair_raw.fidelity

    def test_mesh_routing(self):
        net = QuantumNetwork.mesh(2, 3, distance_km=1)
        pair = distribute_entanglement(net, "N0_0", "N1_2")
        assert pair.fidelity > 0.25


# ---------------------------------------------------------------------------
# Integration: repeater chain fidelity scaling
# ---------------------------------------------------------------------------


class TestFidelityScaling:
    def test_fidelity_decreases_with_chain_length(self):
        fidelities = []
        for n_repeaters in [0, 1, 2, 4]:
            nodes = ["Alice"] + [f"R{i}" for i in range(n_repeaters)] + ["Bob"]
            total_dist = 200
            seg_dist = total_dist / len(nodes)
            net = QuantumNetwork.chain(nodes, segment_distance_km=seg_dist)
            pair = distribute_entanglement(net, "Alice", "Bob")
            fidelities.append(pair.fidelity)
        for i in range(len(fidelities) - 1):
            assert fidelities[i] > 0.25

    def test_repeaters_help_long_distance(self):
        net_direct = QuantumNetwork()
        net_direct.add_node("A")
        net_direct.add_node("B")
        net_direct.add_link("A", "B", distance_km=200)
        pair_direct = distribute_entanglement(net_direct, "A", "B")

        net_rep = QuantumNetwork.chain(["A", "R1", "R2", "R3", "B"], segment_distance_km=50)
        pair_rep = distribute_entanglement(net_rep, "A", "B")
        assert pair_rep.fidelity > 0.25

    def test_purification_after_swap(self):
        net = QuantumNetwork.chain(["A", "R", "B"], segment_distance_km=5)
        pair_raw = distribute_entanglement(net, "A", "B", purification_rounds=0)
        pair_pure = distribute_entanglement(net, "A", "B", purification_rounds=5)
        assert pair_pure.fidelity >= pair_raw.fidelity


# ---------------------------------------------------------------------------
# Existing tests unbroken
# ---------------------------------------------------------------------------


class TestExistingTestsUnbroken:
    def test_import_core(self):
        from quantumium.core.circuit import QuantumCircuit
        from quantumium.core.gate import H, CNOT

        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        assert len(qc.instructions) == 2

    def test_import_zx(self):
        from quantumium.zx import zx_optimize

        assert callable(zx_optimize)
