from __future__ import annotations

import unittest

from aic_gym_gz.live_benchmark import _summary_from_timing


class LiveBenchmarkTest(unittest.TestCase):
    def test_summary_from_timing_uses_trace_fields(self) -> None:
        trace = {
            "num_steps": 4,
            "timing": {
                "ready_to_first_sane_state_latency_s": 1.5,
                "mean_step_latency_s": 0.1,
                "simulated_seconds_per_wall_second": 3.0,
                "samples_per_second": 10.0,
                "total_wall_s": 1.9,
                "simulated_seconds": 5.7,
            },
        }
        summary = _summary_from_timing(trace)
        self.assertEqual(summary["num_steps"], 4)
        self.assertEqual(summary["mean_step_latency_s"], 0.1)
        self.assertEqual(summary["samples_per_second"], 10.0)


if __name__ == "__main__":
    unittest.main()
