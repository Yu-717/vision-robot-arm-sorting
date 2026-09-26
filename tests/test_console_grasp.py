import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'tasks'))
import console_grasp as G  # noqa: E402


class QualityTest(unittest.TestCase):
    def board(self, **kw):
        return dict(dict(label='blue', bbox=[200, 150, 80, 80], method='top-face', map_cm=[-6.0, 13.0]), **kw)

    def test_good_view_needs_no_second_look(self):
        self.assertEqual(G.quality(self.board(), dict(label='blue', map_cm=[-6.1, 12.8])), [])

    def test_partial_or_disagreeing_views_ask_for_a_second_look(self):
        # 2026-09-26: blue at the image border read (-3.78, 13.41) on the board; the phone had (-6.11, 12.84)
        self.assertTrue(G.quality(self.board(bbox=[5, 150, 60, 80]), None))
        self.assertTrue(G.quality(self.board(map_cm=[-3.78, 13.41]), dict(label='blue', map_cm=[-6.11, 12.84])))
        self.assertEqual(G.quality(self.board(method='edges'), None), [])          # sticker-up block, whole view
        self.assertTrue(G.quality(None, dict(label='blue', map_cm=[-6.1, 12.8])))


class WakeTest(unittest.TestCase):
    """The J6 readback, not the console's acceptance, tells that the servo woke up."""

    class Arm:
        """A servo that ignores commands until `wakes_after` jogs have been sent: the grip command does
        nothing and the readback stays where it was, which is what "not_closed" reports."""

        def __init__(self, wakes_after, refuse=(), asleep=True):
            self.j6, self.jogs, self.rearms, self.grips = 29, 0, 0, 0
            self.wakes_after, self.refuse, self.asleep = wakes_after, refuse, asleep
            self.held = False

        def jog(self, joint, delta):
            self.jogs += 1
            if self.jogs >= self.wakes_after:
                self.asleep = False
                self.j6 += delta + 2
            if self.jogs in self.refuse:
                raise G.ConsoleError('servo 6: expected 34, read 29')

        def rearm(self):
            self.rearms += 1

        def grip(self, action):
            self.grips += 1
            if not self.asleep:
                self.held, self.j6 = True, 120

        def status(self):
            grip = ({'result': 'holding', 'readback': 120} if self.held else
                    {'result': 'not_closed', 'readback': self.j6})
            return {'pose': [90, 90, 5, 5, 92, self.j6], 'grip': grip}

    def setUp(self):
        self.saved = G.status, G.log, G.time.sleep
        G.log = lambda *a, **k: None
        G.time.sleep = lambda s: None

    def tearDown(self):
        G.status, G.log, G.time.sleep = self.saved

    def test_an_awake_servo_is_gripped_without_waking_it(self):
        # the wake cost 7.3 s of every 35.5 s block (2026-09-27); an awake servo must not pay it
        arm = self.Arm(wakes_after=1, asleep=False)
        G.status = arm.status
        self.assertEqual(G.wake_and_grip(arm)['result'], 'holding')
        self.assertEqual((arm.jogs, arm.grips), (0, 1))

    def test_an_ignored_grip_is_followed_by_waking_and_gripping_again(self):
        # 2026-09-25 19:30: two +2 steps were accepted while J6 still read 29; the grip was then ignored
        arm = self.Arm(wakes_after=3, refuse=(2,))
        G.status = arm.status
        self.assertEqual(G.wake_and_grip(arm)['result'], 'holding')
        self.assertEqual((arm.jogs, arm.rearms, arm.grips), (3, 1, 2))

    def test_a_servo_that_never_wakes_stops_after_its_tries(self):
        arm = self.Arm(wakes_after=99)
        G.status = arm.status
        self.assertEqual(G.wake_and_grip(arm, tries=2)['result'], 'not_closed')
        self.assertEqual(arm.grips, 2)

    def test_a_servo_that_never_wakes_is_reported(self):
        arm = self.Arm(wakes_after=99)
        G.status = arm.status
        self.assertFalse(G.wake_gripper(arm, steps=4))
        self.assertEqual(arm.jogs, 4)


class PlatformTest(unittest.TestCase):
    def setUp(self):
        self.saved = G.DX, G.DY, G.TABLE_Z

    def tearDown(self):
        G.DX, G.DY, G.TABLE_Z = self.saved

    def test_near_strip_block_plans_clear_of_the_platform(self):
        # the green block of 2026-09-25 19:15 at map (-6.3, 8.1): the 10.5 cm cylinder blocked it
        G.DX, G.DY, G.TABLE_Z = -0.008, 0.0755, -0.0064
        a = G.plan_approach(*G.map_to_base(-0.063, 0.081), 0)
        self.assertIsNotNone(a)
        self.assertTrue(G.clear_of_platform(a['hover']))

    def test_folding_into_the_platform_is_refused(self):
        with self.assertRaises(G.ConsoleError):
            G.check_path([90, 105, 5, 5], [90, 102, -5, -1])     # the folded contact pose
        G.check_path([90, 103, -5, -1], [90, 105, 5, 5])         # leaving it is allowed


class PathTest(unittest.TestCase):
    def test_observe_to_hover_is_allowed(self):
        G.check_path([90, 104, 5, 5], [74, 56, 53, 11])

    def test_rising_from_a_low_grip_pose_is_allowed(self):
        G.check_path([107, 60, 26, 30], [108, 59, 49, 12])   # grip pose (tip ~0) back up to its hover

    def test_rising_from_a_sagged_pose_is_allowed(self):
        G.check_path([88, 65, -3, 60], [88, 66, 20, 38])   # after a servo power cycle the tip rested at -2.1 cm

    def test_path_through_the_table_is_refused(self):
        with self.assertRaises(G.ConsoleError):
            G.check_path([90, 104, 5, 5], [90, 10, 94, 8])   # ends at tip z -3.8 cm, below the table


if __name__ == '__main__':
    unittest.main()


class SortDecisionTest(unittest.TestCase):
    def setUp(self):
        import sort_blocks
        self.S = sort_blocks

    def at(self, marker, dx=0.0, dy=0.0):
        x, y = G.marker_centre(marker)
        return {'map_cm': [x * 100 + dx, y * 100 + dy], 'source': 'phone'}

    def test_decisions(self):
        bins = {c: self.S.SORTING['category_bins'][k] for c, k in self.S.SORTING['colour_category'].items()}
        found = {'green': self.at(bins['green'], 0.5), 'blue': self.at(bins['yellow']), 'red': self.at('42'),
                 'yellow': {'map_cm': [-13.0, 21.5]}}                                    # ~29.6 cm from the base
        self.assertEqual(self.S.decide('green', found)[0], 'skip')          # already on its bin
        self.assertEqual(self.S.decide('blue', found)[0], 'move')           # sitting on yellow's bin
        self.assertEqual(self.S.decide('red', found)[0], 'move')
        self.assertIn('move it closer', self.S.decide('yellow', found)[1])  # ~27 cm from the base
        found['green'] = self.at(bins['red'])
        self.assertIn('occupied', self.S.decide('red', found)[1])
        self.assertEqual(self.S.decide('purple', found)[1], 'not seen')


class RollTest(unittest.TestCase):
    def test_fingers_stay_across_the_pitch_plane(self):
        # 180 deg equivalents only: a 90 deg alternative opened the fingers along the pitch plane and missed
        self.assertEqual(G.grip_roll(90, 0), G.ROLL_ZERO)
        self.assertEqual(G.grip_roll(71, -10.9), 100)                  # verified by eye 2026-09-26
        self.assertEqual(G.grip_roll(88, -16.6), G.ROLL_ZERO)          # needs -14.6: outside the trusted window
        for j1 in range(40, 141, 7):
            for yaw in (-44, -20, 0, 20, 44):
                self.assertTrue(G.J5_RANGE[0] <= G.grip_roll(j1, yaw) <= G.J5_RANGE[1])


class CategoryTest(unittest.TestCase):
    def test_model_evidence_and_colour(self):
        import sort_blocks as S
        self.assertIn('official model', S.category_of('red', {'label': 'Syringe', 'category': 'hazardous', 'conf': 0.8})[1])
        # the model has no 'other' class: a yellow block called Fish_bone stays 'other' and is logged as a disagreement
        cat, src = S.category_of('yellow', {'label': 'Fish_bone', 'category': 'kitchen', 'conf': 0.54})
        self.assertEqual((cat, src.startswith('colour; official model disagreed')), ('other', True))
        self.assertIn('fallback', S.category_of('blue', None)[1])
        self.assertIn('fallback', S.category_of('green', {'label': 'Fish_bone', 'category': 'kitchen', 'conf': 0.3})[1])
        self.assertEqual(S.bin_for('red')[0], '43')


class ResolveTest(unittest.TestCase):
    def test_evidence(self):
        import sort_blocks as S
        fish = {'label': 'Fish_bone', 'category': 'kitchen', 'conf': 0.8}
        self.assertEqual(S.resolve(None, fish)[0], 'kitchen')                        # sticker-up block: model decides
        self.assertEqual(S.resolve('yellow', dict(fish, conf=0.54))[0], 'other')     # model has no 'other'
        self.assertEqual(S.resolve('blue', None)[0], 'recyclable')
        self.assertIsNone(S.resolve(None, None)[0])
        self.assertIsNone(S.resolve(None, dict(fish, conf=0.3))[0])
