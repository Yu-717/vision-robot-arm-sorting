import json
import math
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/teleop'))
from core import Controller, SimArm, safe_arm
from calibration import fit, project
from camera import Camera
from server import App, handler, ThreadingHTTPServer


class RecordingArm(SimArm):
    def __init__(self):
        super().__init__()
        self.commands = []
        self.fail = False

    def write6(self, pose, duration_ms):
        self.commands.append(list(pose))
        if not self.fail:
            super().write6(pose, duration_ms)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.arm = RecordingArm()
        self.c = Controller(self.tmp.name, self.arm)

    def move(self, joint=1, delta=1, revision=None):
        with patch('core.time.sleep'):
            return self.c.jog('a',joint,delta,self.c.revision if revision is None else revision)

    def test_locked_start_and_owner(self):
        with self.assertRaises(ValueError):
            self.move()
        self.c.enable('a')
        with self.assertRaises(ValueError):
            self.c.enable('b')
        self.assertEqual(self.arm.commands,[])

    def test_small_step_and_no_duplicate_revision(self):
        self.c.enable('a')
        self.move()
        self.assertEqual(self.arm.commands,[[91,105,5,5,95,30]])
        with self.assertRaises(ValueError):
            self.move(revision=0)
        self.assertEqual(len(self.arm.commands),1)

    def test_limits_keepout_and_bad_arguments(self):
        self.c.enable('a')
        with self.assertRaises(ValueError):
            self.move(delta=100)
        with self.assertRaises(safe_arm.MotionError):
            self.move(joint=3,delta=-1)
        self.assertEqual(self.arm.commands,[])
        self.arm.pose=[90,85,5,5,95,30]
        self.c.refresh(); self.c.enable('a')
        with self.assertRaises(safe_arm.MotionError):
            self.move()
        self.assertEqual(self.arm.commands,[])

    def test_read_failure_never_writes(self):
        self.c.enable('a')
        self.arm.pose[2]=None
        with self.assertRaises(ValueError):
            self.move()
        self.assertEqual(self.arm.commands,[])
        self.assertFalse(self.c.status()['armed'])

    def test_deviation_holds_at_readback_without_retreat(self):
        self.c.cfg['motion']['readback_tolerance_deg']=.5
        self.arm.fail=True
        self.c.enable('a')
        with self.assertRaises(ValueError):
            self.move(delta=2)
        # the step that met resistance, then one command to hold exactly where the arm is (not the start)
        self.assertEqual(self.arm.commands,[[92,105,5,5,95,30],[90,105,5,5,95,30]])
        self.assertEqual(self.arm.commands[1],self.arm.read())
        self.assertFalse(self.c.status()['armed'])

    def test_stop_and_lease_expiry(self):
        self.c.enable('a'); self.c.stop()
        with self.assertRaises(ValueError):
            self.move()
        self.c.enable('a'); self.c.deadline=time.monotonic()-1
        self.c.heartbeat('a')
        with self.assertRaises(ValueError):
            self.move()
        self.assertEqual(self.arm.commands,[])

    def test_old_command_cannot_resume_after_reenable(self):
        self.c.enable('a')
        revision=self.c.revision
        self.c.stop();self.c.enable('a')
        with self.assertRaises(ValueError):self.move(revision=revision)
        self.assertEqual(self.arm.commands,[])

    def test_stop_during_read_prevents_dispatch(self):
        self.c.enable('a')
        original=self.arm.read
        def read_and_stop():
            self.c.stop()
            return original()
        self.arm.read=read_and_stop
        with self.assertRaises(ValueError):
            self.move()
        self.assertEqual(self.arm.commands,[])

    def test_concurrent_commands_not_queued(self):
        entered, release = threading.Event(), threading.Event()
        original=self.arm.read
        def slow_read():
            entered.set();release.wait(2);return original()
        self.c.enable('a'); self.arm.read=slow_read
        errors=[]
        def work():
            try:self.move()
            except Exception as e:errors.append(e)
        t=threading.Thread(target=work);t.start();entered.wait(1)
        try:
            with self.assertRaises(ValueError):self.move()
        finally:
            release.set();t.join(3)
        self.assertFalse(errors)
        self.assertEqual(len(self.arm.commands),1)

    def test_hardware_requires_live_capture_and_onsite(self):
        cam=Camera();self.c.hardware=True;self.c.camera=cam
        with self.assertRaises(ValueError):self.c.enable('a',True)
        cam.publish(b'\xff\xd8\xff\xd9',time.time(),'receive')
        with self.assertRaises(ValueError):self.c.enable('a',True)
        cam.publish(b'\xff\xd8\xff\xd9',time.time(),'capture')
        with self.assertRaises(ValueError):self.c.enable('a',False)
        self.c.enable('a',True)
        cam.stamp-=10
        with self.assertRaises(ValueError):self.move()
        self.assertEqual(self.arm.commands,[])


class CalibrationTest(unittest.TestCase):
    def test_independent_point_and_distance(self):
        h=fit([[0,0,0,0],[640,0,32,0],[640,480,32,24],[0,480,0,24]])['h']
        self.assertAlmostEqual(math.dist(project(h,[100,100]),project(h,[160,180])),5)
        self.assertAlmostEqual(project(h,[320,240])[0],16)

    def test_degenerate_and_nonfinite_rejected(self):
        for points in ([],[[i,i,i,i] for i in range(4)],[[0,0,float('nan'),0]]*4):
            with self.assertRaises(ValueError):fit(points)

    def test_perspective_mapping(self):
        expected=[.8,.1,.02,-.2,.7,.05,.2,-.1]
        pixels=[[30,40],[600,40],[590,450],[20,440],[320,240]]
        actual=fit([p+project(expected,p) for p in pixels])['h']
        self.assertLess(math.dist(project(actual,[200,300]),project(expected,[200,300])),1e-7)


class HTTPTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.camera=Camera(replay=ROOT/'data/captures/2026-09-24/map_v2_placed.jpg')
        self.control=Controller(self.tmp.name,camera=self.camera)
        self.app=App(self.control,self.camera)
        self.http=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app))
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.http.server_close);self.addCleanup(self.http.shutdown)
        self.url=f'http://127.0.0.1:{self.http.server_port}'

    def post(self,path,body,token=True,origin=None):
        headers={'Content-Type':'application/json'}
        if token:headers['X-Console-Token']=self.app.token
        if origin:headers['Origin']=origin
        req=urllib.request.Request(self.url+path,json.dumps({'owner':'test',**body}).encode(),headers)
        with urllib.request.urlopen(req) as r:return json.load(r)

    def test_cross_site_and_missing_token_rejected(self):
        for kwargs in ({'token':False},{'origin':'http://evil.example'}):
            with self.assertRaises(urllib.error.HTTPError) as e:self.post('/api/enable',{},**kwargs)
            self.assertEqual(e.exception.code,409)
        self.assertFalse(self.control.status()['armed'])

    def test_freeze_calibration_measure_sample_export(self):
        frozen=self.post('/api/freeze',{})['frozen']
        points=[[0,0,0,0],[640,0,32,0],[640,480,32,24],[0,480,0,24]]
        self.post('/api/calibrate',{'frame_id':frozen['id'],'points':points,'plane':'test'})
        result=self.post('/api/measure',{'frame_id':frozen['id'],'pixels':[[100,100],[160,180]]})
        self.assertAlmostEqual(result['distance_cm'],5)
        result=self.post('/api/sample',{'label':'offline test'})
        self.assertEqual(result['evidence'],'simulation')
        self.assertTrue((Path(self.tmp.name)/result['image']).is_file())
        with urllib.request.urlopen(self.url+'/api/export') as r:export=json.load(r)
        self.assertEqual(len(export['measurements']),1)
        self.assertEqual(len(export['samples']),1)
        self.post('/api/freeze',{})
        with self.assertRaises(urllib.error.HTTPError):
            self.post('/api/measure',{'frame_id':frozen['id'],'pixels':[[10,10]]})

    def test_simulation_does_not_import_hardware_library(self):
        self.assertNotIn('Arm_Lib',sys.modules)
        self.assertEqual(self.app.status()['mode'],'simulation')


class BrowserGeometryTest(unittest.TestCase):
    def test_frontend_pose_and_camera_match_python_kinematics(self):
        import shutil
        import subprocess
        if not shutil.which('node'):
            self.skipTest('Node is needed for the independent frontend geometry check')
        hand=json.loads((ROOT/'config/calibration/hand_eye.json').read_text())['cam2gripper']
        poses=[[90,105,5,5,95,30],[45,130,20,30,160,90],[90,90,90,90,95,30],[130,100,10,15,120,80]]
        script="import {geometry} from './src/teleop/web/scene.js'; const input="+json.dumps([poses,hand])+"; console.log(JSON.stringify(input[0].map(q=>geometry(q,input[1]))));"
        out=json.loads(subprocess.check_output(['node','--input-type=module','-e',script],cwd=ROOT,text=True))
        for q,result in zip(poses,out):
            expected=safe_arm.workspace.kin.forward(q)['tip']
            self.assertLess(math.dist(result['tip'],expected),.0001)
            frame=safe_arm.workspace.kin.camera_mount_pose(q)  # the camera rides on the wrist, not with J5
            cam=[[sum(frame[i][k]*hand[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
            self.assertLess(math.dist(result['cam'],[row[3] for row in cam[:3]]),.0001)
            for j in range(3):
                self.assertLess(math.dist(result['camAxes'][j],[cam[i][j] for i in range(3)]),1e-10)

if __name__ == '__main__':unittest.main()
