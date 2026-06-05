"""Tests for quantum random number generation."""

import pytest

from quantumium.applications.qrng import (
    random_bits,
    random_int,
    random_float,
    random_bytes,
    random_choice,
    random_shuffle,
    entropy_test,
)


class TestRandomBits:
    def test_correct_length(self):
        bits = random_bits(10)
        assert len(bits) == 10

    def test_only_01(self):
        bits = random_bits(16)
        assert all(c in "01" for c in bits)

    def test_large_batch(self):
        bits = random_bits(20)
        assert len(bits) == 20

    def test_not_all_same(self):
        bits = random_bits(32)
        assert "0" in bits and "1" in bits


class TestRandomInt:
    def test_in_range(self):
        for _ in range(5):
            val = random_int(0, 10)
            assert 0 <= val < 10

    def test_single_value(self):
        val = random_int(5, 6)
        assert val == 5

    def test_negative_range(self):
        for _ in range(5):
            val = random_int(-5, 5)
            assert -5 <= val < 5

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="must be less"):
            random_int(10, 5)

    def test_distribution(self):
        counts = {i: 0 for i in range(4)}
        for _ in range(40):
            val = random_int(0, 4)
            counts[val] += 1
        for v in counts.values():
            assert v > 1


class TestRandomFloat:
    def test_in_range(self):
        for _ in range(5):
            val = random_float(precision=16)
            assert 0.0 <= val < 1.0

    def test_precision(self):
        val = random_float(precision=8)
        assert 0.0 <= val < 1.0


class TestRandomBytes:
    def test_correct_length(self):
        data = random_bytes(4)
        assert len(data) == 4
        assert isinstance(data, bytes)

    def test_byte_values(self):
        data = random_bytes(8)
        assert all(0 <= b <= 255 for b in data)


class TestRandomChoice:
    def test_returns_item(self):
        items = ["a", "b", "c", "d"]
        for _ in range(5):
            choice = random_choice(items)
            assert choice in items

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            random_choice([])

    def test_single_item(self):
        assert random_choice([42]) == 42


class TestRandomShuffle:
    def test_same_elements(self):
        original = [1, 2, 3, 4, 5]
        shuffled = random_shuffle(original)
        assert sorted(shuffled) == sorted(original)

    def test_does_not_modify_original(self):
        original = [1, 2, 3]
        _ = random_shuffle(original)
        assert original == [1, 2, 3]

    def test_not_always_same_order(self):
        original = list(range(8))
        results = set()
        for _ in range(5):
            shuffled = random_shuffle(original)
            results.add(tuple(shuffled))
        assert len(results) > 1


class TestEntropyTest:
    def test_basic_stats(self):
        result = entropy_test(100)
        assert result["num_bits"] == 100
        assert 0.2 < result["ones_ratio"] < 0.8
        assert result["runs_count"] > 10
        assert result["min_entropy_estimate"] > 0.3
        assert "bits_preview" in result

    def test_preview_length(self):
        result = entropy_test(80)
        assert result["bits_preview"].endswith("...")
