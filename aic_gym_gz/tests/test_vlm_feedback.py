from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aic_gym_gz.vlm_feedback import collect_final_frame_samples


class VlmFeedbackFrameSamplingTest(unittest.TestCase):
    def test_collects_evenly_spaced_frames_per_angle_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_dir = root / "videos"
            video_dir.mkdir()
            output_dir = root / "frames"
            video_path = video_dir / "camera_left.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 24),
            )
            try:
                for index in range(20):
                    frame = np.full((24, 32, 3), index * 10, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            samples = collect_final_frame_samples(
                video_dir=video_dir,
                output_dir=output_dir,
                final_observation={"score_geometry": {}},
                final_info={},
                frames_per_angle=10,
            )

            self.assertEqual(len(samples), 10)
            self.assertTrue(all(sample.stream_name == "camera_left" for sample in samples))
            self.assertTrue(all(sample.sample_count_for_stream == 10 for sample in samples))
            self.assertEqual([sample.sample_index for sample in samples], list(range(10)))
            self.assertEqual(samples[0].frame_index, 0)
            self.assertEqual(samples[-1].frame_index, 19)
            self.assertIn("left wrist camera", samples[0].viewpoint_description)
            timestamp_deltas = np.diff([sample.timestamp_s for sample in samples])
            self.assertLess(float(timestamp_deltas.max() - timestamp_deltas.min()), 0.12)
            self.assertTrue(all(Path(sample.diagnostic_image_path).exists() for sample in samples))


if __name__ == "__main__":
    unittest.main()
