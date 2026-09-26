import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ('pipeline', 'tasks', 'tools', 'spatial', 'vision'):
    sys.path.insert(0, str(ROOT / 'src' / sub))
import console_grasp as G  # noqa: E402
import scramble as S  # noqa: E402


class CandidatesTest(unittest.TestCase):
    def test_the_bins_are_not_offered_as_places_to_scatter_to(self):
        """A block started on its own bin is a block the round never sorts."""
        self.assertFalse(set(S.candidates()) & S.bin_markers())

    def test_every_candidate_has_a_reachable_grasp(self):
        for marker in S.candidates():
            bx, by = G.map_to_base(*G.marker_centre(marker))
            self.assertIsNotNone(G.plan_approach(bx, by, None), marker)


class DrawTest(unittest.TestCase):
    def apart(self, markers):
        centres = [[v * 100 for v in G.marker_centre(m)] for m in markers]
        return min(math.dist(a, b) for i, a in enumerate(centres) for b in centres[i + 1:])

    def test_drawn_markers_are_never_neighbours(self):
        for seed in range(25):
            markers = S.draw(4, random.Random(seed))
            self.assertEqual(len(set(markers)), 4)
            self.assertGreaterEqual(self.apart(markers), S.MIN_APART_CM)

    def test_the_draw_is_repeatable_for_a_seed(self):
        self.assertEqual(S.draw(4, random.Random(3)), S.draw(4, random.Random(3)))

    def test_a_pool_that_cannot_seat_them_is_an_error_not_a_tighter_layout(self):
        with self.assertRaises(ValueError):
            S.draw(3, random.Random(0), pool=['40', '41'])        # two markers, three blocks
        with self.assertRaises(ValueError):
            S.draw(2, random.Random(0), pool=['40', '41'])        # neighbours, 6 cm apart


class MoveTest(unittest.TestCase):
    class Fake:
        def __init__(self):
            self.calls = []

    def setUp(self):
        self.saved = S.G.hover, S.G.grasp, S.G.put, S.G.log, S.G.PLAN
        self.calls = []
        S.G.hover = lambda sess: self.calls.append('hover')
        S.G.grasp = lambda sess: self.calls.append('grasp')
        S.G.put = lambda where, sess: self.calls.append(('put', where))
        S.G.log = lambda *a, **k: None
        S.G.PLAN = ROOT / 'data' / 'runs' / 'console_grasp' / 'plan_test.json'
        S.G.PLAN.parent.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: S.G.PLAN.unlink(missing_ok=True))

    def tearDown(self):
        S.G.hover, S.G.grasp, S.G.put, S.G.log, S.G.PLAN = self.saved

    def test_a_move_runs_the_verified_stages_in_order(self):
        cube = dict(map_cm=[5.0, 8.0], yaw_deg=0.0, category='kitchen')
        out = S.move(cube, '38', lambda c: dict(plan='x'), None)
        self.assertEqual(self.calls, ['hover', 'grasp', ('put', '38')])
        self.assertEqual(out['marker'], '38')
        self.assertEqual(out['frm'], [5.0, 8.0])

    def test_an_unreachable_block_stops_before_any_motion(self):
        with self.assertRaises(G.ConsoleError):
            S.move(dict(map_cm=[0, 30], yaw_deg=0), '38', lambda c: None, None)
        self.assertEqual(self.calls, [])

    def test_the_farthest_block_is_moved_first(self):
        """Putting a near block down first would leave the arm reaching over it for the far one."""
        cubes = [dict(map_cm=[0.0, 8.0], yaw_deg=0), dict(map_cm=[0.0, 18.0], yaw_deg=0)]
        moved = S.scramble(cubes, lambda c: dict(plan='x'), None, random.Random(1))
        self.assertEqual([m['frm'] for m in moved], [[0.0, 18.0], [0.0, 8.0]])


if __name__ == '__main__':
    unittest.main()
