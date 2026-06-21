import importlib.util
import pathlib
import tempfile
import unittest

_HERE = pathlib.Path(__file__).resolve()
_MOD = _HERE.parents[1] / "analysis" / "paper_v2" / "evidence_audit.py"
_spec = importlib.util.spec_from_file_location("ea_under_test", _MOD)
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

_GOOD_METRICS = (
    "\\newcommand{\\VTwoMinimalFVEAdaptiveFinal}{0.996}\n"
    "\\newcommand{\\VTwoNandaFVEAdaptive}{0.989}\n"
    "\\newcommand{\\VTwoMinimalSeedsCleanFinal}{5}\n"
)
_GOOD_MAIN = (
    "\\begin{tabular}{lrrr}\n\\toprule\nModel & Test acc & Adaptive FVE & Seeds $\\geq 0.95$ \\\\\n"
    "\\midrule\nNanda 1-layer & 1.000 $\\pm$ 0.000 & 0.989 $\\pm$ 0.001 & 5/5 \\\\\n"
    "TRM minimal & 1.000 $\\pm$ 0.000 & 0.996 $\\pm$ 0.002 & 5/5 \\\\\n\\bottomrule\n\\end{tabular}\n"
)
_BAD_MAIN = _GOOD_MAIN.replace("1.000 $\\pm$ 0.000 & 0.996", "0.200 $\\pm$ 0.400 & 0.000")
_CORR = {"trm_minimal": {"mean_fve_adaptive": 0.996, "n_seeds_adaptive_ge_0.95": 5, "n_seeds": 5}}
_CORR_N = {"nanda_a_mlp": {"mean_fve_adaptive": 0.9888, "n_seeds_adaptive_ge_0.95": 5, "n_seeds": 5}}


def _paper(tmp, main_tex):
    p = pathlib.Path(tmp)
    (p / "tables").mkdir()
    (p / "metrics_v2.tex").write_text(_GOOD_METRICS)
    (p / "tables" / "main_results.tex").write_text(main_tex)
    return p


class TestNumericAudit(unittest.TestCase):
    def test_good_tables_pass(self):
        with tempfile.TemporaryDirectory() as t:
            errs = ea.audit_numeric(_paper(t, _GOOD_MAIN), _CORR, _CORR_N)
            self.assertEqual(errs, [])

    def test_corrupted_tables_fail(self):
        with tempfile.TemporaryDirectory() as t:
            errs = ea.audit_numeric(_paper(t, _BAD_MAIN), _CORR, _CORR_N)
            self.assertTrue(errs)
            self.assertTrue(any("0.000" in e or "FVE" in e or "acc" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
