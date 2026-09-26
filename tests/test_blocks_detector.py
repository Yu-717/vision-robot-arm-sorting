import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('pipeline', 'tasks', 'tools', 'spatial', 'vision'):
    sys.path.insert(0, str(ROOT / 'src' / sub))
import blocks_detector as B  # noqa: E402
import run as R  # noqa: E402


class DecodeTest(unittest.TestCase):
    def test_letterbox_boxes_come_back_to_image_pixels_and_nms_keeps_one(self):
        out = np.zeros((1, 8, 8400), np.float32)
        # two overlapping 'hazardous' boxes and one 'other', in 640 x 640 letterbox pixels (image 640 x 480: pad 80 on top)
        for i, (cx, cy, w, h, cls, score) in enumerate([(320, 320, 100, 100, 2, 0.9), (322, 318, 100, 100, 2, 0.7),
                                                        (100, 200, 50, 60, 3, 0.8)]):
            out[0, :4, i] = (cx, cy, w, h)
            out[0, 4 + cls, i] = score
        dets = B.decode(out, 1.0, (0, 80), (480, 640))
        self.assertEqual([d['category'] for d in dets], ['hazardous', 'other'])
        self.assertEqual(dets[0]['bbox'], [270, 190, 370, 290])
        self.assertEqual(dets[1]['bbox'], [75, 90, 125, 150])


class CategorizeTest(unittest.TestCase):
    def cube(self, colour=None, votes=None, views=3):
        return dict(colour=colour, sticker=None, detector=dict(votes=votes or {}, views=views) if votes else None)

    def test_colour_decides_and_the_detector_is_recorded(self):
        self.assertEqual(R.categorize(self.cube('red', {'recyclable': 2.5}))[0], 'hazardous')
        self.assertIn('disagreed', R.categorize(self.cube('red', {'recyclable': 2.5}))[1])

    def test_detector_names_a_sticker_top_block_when_it_is_clear(self):
        self.assertEqual(R.categorize(self.cube(None, {'kitchen': 2.4, 'other': 0.5}))[0], 'kitchen')
        self.assertIsNone(R.categorize(self.cube(None, {'kitchen': 0.9, 'other': 0.8}))[0])   # split vote
        self.assertIsNone(R.categorize(self.cube(None, {'kitchen': 0.5}))[0])                 # too weak


if __name__ == '__main__':
    unittest.main()
