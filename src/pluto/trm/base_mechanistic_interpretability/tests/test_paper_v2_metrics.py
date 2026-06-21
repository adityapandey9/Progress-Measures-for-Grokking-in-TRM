import importlib.util
import pathlib
import tempfile
import unittest

_HERE = pathlib.Path(__file__).resolve()
_MOD_PATH = _HERE.parents[1] / "analysis" / "write_paper_v2_metrics.py"
_spec = importlib.util.spec_from_file_location("wpm_under_test", _MOD_PATH)
wpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wpm)


def _metrics_with_bug_shape():
    """Mimics the real aggregate: final_* corrupted, history_metrics.final correct,
    final_metrics empty for seeds 1-4, corrected_fve present."""
    def seed(name, acc, fve):
        return {
            "seed": name,
            "final_metrics": {} if name != "seed_0" else {"test_accuracy": 0.0, "logit_trig_fve_adaptive": 0.0},
            "history_metrics": {"final": {"test_accuracy": acc, "logit_trig_fve_adaptive": 0.0}},
        }
    minimal_seeds = [seed("seed_0", 1.0, 0), seed("seed_1", 1.0, 0), seed("seed_2", 1.0, 0),
                     seed("seed_3", 1.0, 0), seed("seed_4", 1.0, 0)]
    nanda_seeds = [seed(f"seed_{i}", 1.0, 0) for i in range(5)]
    return {
        "models": {
            "nanda_a_mlp": {
                "seeds": nanda_seeds,
                "final_test_accuracy": {"mean": 1.0, "std": 0.0, "n": 5},
                "final_logit_trig_fve_adaptive": {"mean": 0.0, "std": 0.0, "n": 5},
            },
            "trm_minimal": {
                "seeds": minimal_seeds,
                "final_test_accuracy": {"mean": 0.2, "std": 0.4, "n": 5},
                "final_logit_trig_fve_adaptive": {"mean": 0.0, "std": 0.0, "n": 5},
            },
        }
    }


def _corrected():
    def block(mean, vals):
        return {"mean_fve_adaptive": mean, "n_seeds_adaptive_ge_0.95": 5, "n_seeds": 5,
                "seeds": [{"seed": f"seed_{i}", "fve_adaptive": v} for i, v in enumerate(vals)]}
    return {
        "trm_minimal": block(0.996, [0.9952, 0.9952, 0.9944, 0.9985, 0.9967]),
        "nanda_a_mlp": block(0.9888, [0.9964, 0.9565, 0.996, 0.9978, 0.9971]),
    }


class TestMainResults(unittest.TestCase):
    def test_main_results_uses_correct_sources(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "main_results.tex"
            wpm.write_main_results(_metrics_with_bug_shape(), out, _corrected())
            text = out.read_text()
            self.assertNotIn("0.200", text)          # corrupted accuracy must be gone
            self.assertIn("1.000 $\\pm$ 0.000", text) # correct accuracy
            self.assertIn("0.996", text)              # minimal adaptive FVE (corrected)
            self.assertIn("0.989", text)              # nanda adaptive FVE (corrected, rounds to 0.989)
            self.assertIn("5/5", text)                # clean seeds
            # No FVE cell may read 0.000.
            for line in text.splitlines():
                if "TRM minimal" in line or "Nanda 1-layer" in line:
                    self.assertNotIn("0.000 $\\pm$ 0.000", line)

    def test_nanda_row_from_corrected_when_aggregate_missing(self):
        metrics = {"models": {"trm_minimal": _metrics_with_bug_shape()["models"]["trm_minimal"]}}
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "main_results.tex"
            wpm.write_main_results(metrics, out, _corrected())
            text = out.read_text()
            self.assertIn("Nanda 1-layer", text)
            self.assertIn("0.989", text)


class TestPerSeedTable(unittest.TestCase):
    def test_per_seed_uses_history_and_corrected(self):
        metrics = _metrics_with_bug_shape()
        # add history grok/best so the writer has all fields it reads
        for s in metrics["models"]["trm_minimal"]["seeds"]:
            s["history_metrics"]["grokking"] = {"step": 12000}
            s["history_metrics"]["best_fve"] = {"logit_trig_fve_faithful": 0.99}
            s["seed_diagnosis"] = {"grokking_step": 12000, "likely_causes": ["stable_circuit"]}
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "per_seed_metrics.tex"
            wpm.write_per_seed_table(metrics, out, _corrected())
            text = out.read_text()
            # 5 data rows, none with Final acc 0.000 or Final FVE 0.000
            self.assertEqual(text.count(r"\\"), 6)  # header rule row + 5 data rows
            self.assertNotIn("0.000 & 0.000", text)
            self.assertIn("0.995", text)  # seed_0 corrected fve_adaptive 0.9952 -> 0.995


if __name__ == "__main__":
    unittest.main()
