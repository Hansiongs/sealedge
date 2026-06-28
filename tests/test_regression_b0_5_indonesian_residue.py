"""Regression tests for B0.5: Indonesian-language residue removed.

Before B0.5:
- reporting.py used 'Kasus 1/2/3' for edge classification
- candidate.py had 'Kasus 3 default' in a docstring

Both fixed (now English). This test prevents re-introduction.
"""
import inspect


class TestNoIndonesianInUserFacing:
    def test_reporting_no_indonesian(self):
        """'Kasus' must not appear in reporting.py source."""
        import quant_lib.research.reporting as r
        src = inspect.getsource(r)
        assert "Kasus" not in src, "'Kasus' found in reporting.py"

    def test_candidate_no_indonesian(self):
        """'Kasus' must not appear in candidate.py source."""
        import quant_lib.research.candidate as c
        src = inspect.getsource(c)
        assert "Kasus" not in src

    def test_edge_classification_uses_english_labels(self):
        """Edge classification taxonomy must use English labels."""
        import quant_lib.research.reporting as r
        src = inspect.getsource(r)
        assert "CONCENTRATED" in src or "concentrated" in src
        assert "BROAD_WEAK" in src or "broad_weak" in src
        assert "RANDOM" in src or "random" in src
