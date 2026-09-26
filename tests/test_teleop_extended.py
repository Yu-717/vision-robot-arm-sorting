import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/teleop'))
from core import Controller,SimArm,safe_arm
from motion_policy import home_plan,test_config

class MotionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.c=Controller(self.temp.name)

    def test_profile_is_bounded_and_does_not_mutate_default(self):
        cfg=test_config(self.c.base_cfg)
        self.assertEqual(cfg['soft_limits_deg'][3][0],2)
        self.assertEqual(self.c.base_cfg['soft_limits_deg'][3][0],5)
        self.assertEqual(cfg['course_limits_deg'],self.c.base_cfg['course_limits_deg'])
        self.assertEqual(cfg['soft_limits_deg'][5],self.c.base_cfg['soft_limits_deg'][5])

    def test_profile_requires_enable_and_expires_disarmed(self):
        with self.assertRaises(ValueError):self.c.set_profile('a',True,True)
        self.c.enable('a');self.c.set_profile('a',True,True)
        self.assertEqual(self.c.profile,'boundary')
        with self.assertRaises(ValueError):self.c.jog('a',1,2,self.c.revision)
        self.c.profile_until=time.monotonic()-1
        self.assertEqual(self.c.status()['profile'],'normal')
        self.assertFalse(self.c.status()['armed'])
        self.assertEqual(self.c.cfg['keep_out']['table_margin_m'],.025)

    def test_home_from_out_of_course_start_only_steps_back(self):
        # a start past the course limit may be brought home as long as every segment moves the
        # out-of-range joints back towards their range (see safe_arm.validate_target)
        p=home_plan([90,141,-4,3,95,133],self.c.cfg)
        previous=p['start']
        for q in p['segments']:
            self.assertGreaterEqual(q[2],previous[2]); self.assertGreaterEqual(q[3],previous[3])
            previous=q
        self.assertEqual(p['target'][:4],[90,105,5,5])

    def test_home_keeps_gripper_and_checks_every_segment(self):
        p=home_plan([92,110,8,8,120,133],self.c.cfg)
        self.assertEqual(p['target'],[90,105,5,5,120,133])
        previous=p['start']
        for q in p['segments']:
            self.assertLessEqual(max(abs(a-b) for a,b in zip(previous,q)),2)
            self.assertGreaterEqual(safe_arm.check_keep_out(previous,q,self.c.cfg),0)
            previous=q

    def test_home_stop_cancels_all_remaining_segments(self):
        self.c.arm.pose=[92,110,8,8,120,133];self.c.refresh();self.c.enable('a')
        p=self.c.plan_home();written=[]
        def write(q,duration):
            written.append(q);self.c.arm.pose=q;self.c.stop()
        self.c.arm.write6=write
        with patch('core.time.sleep'):
            with self.assertRaises(ValueError):self.c.home('a',p['id'],p['revision'])
        self.assertEqual(len(written),1)
        self.assertFalse(self.c.status()['armed'])

    def test_home_stale_preview_is_rejected(self):
        self.c.enable('a');p=self.c.plan_home();p['created_at']-=31
        with self.assertRaises(ValueError):self.c.home('a',p['id'],p['revision'])


class PhoneLocalizationTest(unittest.TestCase):
    def test_synthetic_metric_map_and_colour_patch(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('vision runtime required')
        if not hasattr(cv2,'aruco'):self.skipTest('aruco required')
        from localization import locate_phone
        from map_registration import TableMap
        table=TableMap();img=np.full((680,850,3),255,np.uint8)
        def pixel(p):return (round((p[0]+.16)*2500),round((.23-p[1])*2500))
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for mid,pts in table.layout.items():
            x,y=pixel(pts[0]);x2,y2=pixel(pts[2]);w=x2-x
            marker=cv2.aruco.generateImageMarker(dictionary,mid,w)
            marker=cv2.resize(marker,(x2-x,y2-y),interpolation=cv2.INTER_NEAREST)
            img[y:y2,x:x2]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
        centre=(.03,.105)
        a=pixel((centre[0]-.012,centre[1]+.012));b=pixel((centre[0]+.012,centre[1]-.012))
        cv2.rectangle(img,a,b,(0,210,255),-1)
        ok,jpeg=cv2.imencode('.jpg',img)
        result=locate_phone(jpeg.tobytes())
        target=next(o for o in result['objects'] if o['label']=='yellow')
        self.assertLess(math.dist(target['map_cm'],[3,10.5]),.2)
        self.assertFalse(target['height_corrected'])
        self.assertGreaterEqual(len(result['markers']),15)

if __name__=='__main__':unittest.main()
