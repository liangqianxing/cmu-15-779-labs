import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matmul_tiling", ROOT / "26_matmul_tiling.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class MatmulTilingTests(unittest.TestCase):
    def test_exhaustive_small_shapes(self):
        for m in range(5):
            for k in range(5):
                for n in range(5):
                    for tile in (1, 2, 3, 4):
                        with self.subTest(m=m, k=k, n=n, tile=tile):
                            case = lab.run_case(m, k, n, tile)
                            self.assertTrue(case["matches_torch_exactly"])
                            self.assertTrue(case["every_output_written_once"])

    def test_square_counts(self):
        for tile in (1, 2, 4, 8):
            case = lab.run_case(8, 8, 8, tile)
            self.assertEqual(case["naive"]["global_input_loads"], 1024)
            self.assertEqual(case["tiled"]["global_input_loads"], 1024 // tile)
            self.assertEqual(case["tiled"]["multiply_adds"], 512)
            self.assertEqual(case["tiled"]["output_stores"], 64)

    def test_partial_tiles(self):
        case = lab.run_case(3, 5, 7, 4)
        self.assertEqual(case["tiled"]["a_loads"], 30)
        self.assertEqual(case["tiled"]["b_loads"], 35)
        self.assertEqual(case["tiled"]["zero_fills"], 63)
        self.assertEqual(case["tiled"]["multiply_adds"], 256)
        self.assertEqual(case["useful_multiply_adds"], 105)
        self.assertEqual(case["tiled"]["block_barrier_phases"], 8)

    def test_zero_inner_dimension(self):
        case = lab.run_case(3, 0, 5, 4)
        self.assertEqual(case["output_preview"], [[0.0] * 5 for _ in range(3)])
        self.assertEqual(case["tiled"]["global_input_loads"], 0)
        self.assertEqual(case["tiled"]["output_stores"], 15)
        self.assertIsNone(case["logical_load_reduction_factor"])

    def test_random_inputs(self):
        generator = torch.Generator().manual_seed(26)
        for m, k, n, tile in ((3, 5, 7, 4), (7, 3, 5, 2), (17, 9, 13, 8)):
            a = torch.randn(m, k, generator=generator, dtype=torch.float64)
            b = torch.randn(k, n, generator=generator, dtype=torch.float64)
            naive, _ = lab.naive_matmul(a.tolist(), b.tolist(), m, k, n)
            tiled, _ = lab.tiled_matmul(a.tolist(), b.tolist(), m, k, n, tile)
            for actual in (naive, tiled):
                torch.testing.assert_close(
                    torch.tensor(actual, dtype=torch.float64), a @ b,
                    atol=1e-12, rtol=1e-12,
                )

    def test_invalid_inputs(self):
        for case in ((-1, 2, 3, 2), (2, 65, 3, 2), (2, 3, 4, 0),
                     (2, 3, 4, 17), (2.5, 3, 4, 2)):
            with self.subTest(case=case), self.assertRaises(ValueError):
                lab.run_case(*case)

    def test_cli_rejects_invalid_inputs_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for arguments in (("--m", "-1"), ("--n", "65"), ("--tile", "0")):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "26_matmul_tiling.py"),
                     *arguments, "--output", str(output)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
