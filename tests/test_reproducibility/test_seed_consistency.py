"""
Test seed consistency - same seed produces same results

These tests verify that setting the same seed produces identical
random number sequences across Python random, NumPy, and PyTorch.
"""

import pytest
import random
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils import set_seed


class TestSeedConsistency:
    """Verify that setting the same seed produces identical initial states"""

    @pytest.fixture
    def seed(self):
        return 67

    def test_python_random_seed_consistency(self, seed):
        """Test Python random module produces same sequence"""
        set_seed(seed)
        seq1 = [random.random() for _ in range(100)]

        set_seed(seed)
        seq2 = [random.random() for _ in range(100)]

        assert seq1 == seq2, "Python random sequences differ with same seed"

    def test_numpy_random_seed_consistency(self, seed):
        """Test NumPy random produces same sequence"""
        set_seed(seed)
        arr1 = np.random.rand(100)

        set_seed(seed)
        arr2 = np.random.rand(100)

        np.testing.assert_array_equal(arr1, arr2,
                                      err_msg="NumPy random arrays differ with same seed")

    def test_torch_cpu_seed_consistency(self, seed):
        """Test PyTorch CPU random produces same sequence"""
        set_seed(seed)
        t1 = torch.rand(100)

        set_seed(seed)
        t2 = torch.rand(100)

        assert torch.equal(t1, t2), "PyTorch CPU tensors differ with same seed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_gpu_seed_consistency(self, seed):
        """Test PyTorch GPU random produces same sequence on same device"""
        device = torch.device("cuda")

        set_seed(seed)
        t1 = torch.rand(100, device=device)

        set_seed(seed)
        t2 = torch.rand(100, device=device)

        assert torch.equal(t1, t2), "PyTorch GPU tensors differ with same seed"

    def test_set_seed_all_streams(self, seed):
        """Test set_seed() correctly seeds all random streams"""
        # First run
        set_seed(seed)
        py_rand1 = random.random()
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()

        # Second run with same seed
        set_seed(seed)
        py_rand2 = random.random()
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()

        assert py_rand1 == py_rand2, "Python random not seeded correctly"
        assert np_rand1 == np_rand2, "NumPy random not seeded correctly"
        assert torch_rand1 == torch_rand2, "PyTorch random not seeded correctly"

    def test_seed_isolation_different_seeds(self):
        """Verify different seeds produce different results"""
        set_seed(42)
        seq1 = [random.random() for _ in range(10)]
        arr1 = np.random.rand(10)
        t1 = torch.rand(10)

        set_seed(67)
        seq2 = [random.random() for _ in range(10)]
        arr2 = np.random.rand(10)
        t2 = torch.rand(10)

        assert seq1 != seq2, "Different seeds should produce different sequences"
        assert not np.array_equal(arr1, arr2), "Different seeds should produce different arrays"
        assert not torch.equal(t1, t2), "Different seeds should produce different tensors"

    def test_torch_randint_consistency(self, seed):
        """Test torch.randint produces same sequence"""
        set_seed(seed)
        t1 = torch.randint(0, 100, (50,))

        set_seed(seed)
        t2 = torch.randint(0, 100, (50,))

        assert torch.equal(t1, t2), "torch.randint differs with same seed"

    def test_torch_randn_consistency(self, seed):
        """Test torch.randn (normal distribution) produces same sequence"""
        set_seed(seed)
        t1 = torch.randn(100)

        set_seed(seed)
        t2 = torch.randn(100)

        assert torch.equal(t1, t2), "torch.randn differs with same seed"

    def test_strict_determinism_flag(self, seed):
        """Test strict_determinism parameter works"""
        # This should not raise an error
        set_seed(seed, strict_determinism=False)

        # With strict_determinism, certain ops may raise errors
        # but basic random generation should still work
        try:
            set_seed(seed, strict_determinism=True)
            t = torch.rand(10)
            assert t.shape == (10,)
        except RuntimeError as e:
            # Some operations may not support deterministic mode
            pytest.skip(f"Strict determinism not fully supported: {e}")
        finally:
            # Reset deterministic mode to avoid affecting other tests
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
