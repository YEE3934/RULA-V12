"""V12 no-side-mixing, no frontal replacements, no baseline score mutation."""
import math
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from angle_lab_core import candidates_from_landmarks,choose_candidate,refine_neck_cross_view_conflicts
from rula_cross_view_bridge import examine_frames,attach_research_score


def landmark_fixture():
    # Camera-independent landmark definition: left 11,13,15 must remain the left.
    points={7:(.38,.12),8:(.62,.12),11:(.40,.32),12:(.60,.32),
            13:(.34,.49),14:(.67,.53),15:(.47,.59),16:(.72,.65),
            23:(.41,.73),24:(.60,.73)}
    return [SimpleNamespace(x=points.get(i,(.5,.5))[0],y=points.get(i,(.5,.5))[1],visibility=.95)
            for i in range(33)]

class DummyPose:
    def __init__(self):self.calls=0
    def process(self,img):
        self.calls+=1
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmark_fixture()))

class V12Integration(unittest.TestCase):
    def test_front_visibility_high_still_never_eligible(self):
        lm=landmark_fixture()
        coords={i:{'x':p.x,'y':p.y,'visibility':p.visibility} for i,p in enumerate(lm)}
        c=candidates_from_landmarks(coords,600,800,'Front','左手側')
        self.assertFalse(c[0]['eligible']);self.assertFalse(c[1]['eligible'])
        self.assertEqual(c[0]['body_side'],'左手側')
        self.assertTrue(np.isfinite(c[0]['angle_deg']))

    def test_no_opposite_side_mix(self):
        lm=landmark_fixture();coords={i:{'x':p.x,'y':p.y,'visibility':p.visibility} for i,p in enumerate(lm)}
        left=candidates_from_landmarks(coords,600,800,'Left','左手側')[0]['angle_deg']
        right=candidates_from_landmarks(coords,600,800,'Left','右手側')[0]['angle_deg']
        self.assertNotEqual(left,right)
        self.assertEqual(left,candidates_from_landmarks(coords,600,800,'Right','左手側')[0]['angle_deg'])

    def test_end_to_end_frame_diagnostics_camera_one_per_frame(self):
        frames={v:np.zeros((800,600,3),np.uint8) for v in ('Front','Left','Right')}
        pose=DummyPose()
        review=examine_frames(frames,pose,18,5.2)
        self.assertEqual(pose.calls,3)
        self.assertEqual(len(review['candidate_rows']),12)
        self.assertEqual(len(review['summary_rows']),4)
        for s in ('左手側','右手側'):
            for p in ('前臂','頸部'):
                self.assertNotEqual(review['choices'][(s,p)]['selected_camera'],'Front')

    def test_score_kept_invariant_when_same_camera(self):
        data={'左手上臂分數':2,'左手前臂分數':1,'左手手腕分數':1,'左手手腕扭轉分數':1,
              '頸部分數':2,'軀幹分數':2,'腿部分數':1,'肌肉使用加分':0,'負重施力加分':0,
              '左手 Grand Score':4,'左手 AL':'AL2','左手前臂角度':80.,'頸部角度':15.}
        before=data.copy()
        review={'choices':{('左手側','前臂'):{'baseline_angle_deg':81.,'selected_angle_deg':81.,'baseline_camera':'Left','selected_camera':'Left','decision':'維持原側向視角','decision_note':'ok'},
                           ('左手側','頸部'):{'baseline_angle_deg':10.,'selected_angle_deg':10.,'baseline_camera':'Left','selected_camera':'Left','decision':'維持原側向視角','decision_note':'ok'}},
                'neck_opposite_gap':np.nan}
        def must_not_call(*a,**kw):raise AssertionError('baseline never rescores')
        attach_research_score(data,'左手側',review,must_not_call,must_not_call)
        self.assertEqual(data['左手 Grand Score'],before['左手 Grand Score'])
        self.assertEqual(data['V12_研究試算GrandScore'],4)
        self.assertEqual(data['V12_研究與原總分差'],0)

    def test_other_camera_scenario_separate_no_original_mutation(self):
        data={'左手上臂分數':2,'左手前臂分數':2,'左手手腕分數':1,'左手手腕扭轉分數':1,
              '頸部分數':2,'軀幹分數':2,'腿部分數':1,'肌肉使用加分':0,'負重施力加分':0,
              '左手 Grand Score':4,'左手 AL':'AL2','左手前臂角度':40.,'頸部角度':15.,
              '左手橫越中線補正':'否','左手前臂向外側工作補正':'否'}
        review={'choices':{('左手側','前臂'):{'baseline_angle_deg':40.,'selected_angle_deg':81.,'baseline_camera':'Left','selected_camera':'Right','decision':'研究候選：另一側鏡頭（待人工驗證）','decision_note':'candidate'},
                           ('左手側','頸部'):{'baseline_angle_deg':15.,'selected_angle_deg':15.,'baseline_camera':'Left','selected_camera':'Left','decision':'維持原側向視角','decision_note':'ok'}},
                'neck_opposite_gap':22.0}
        called=[]
        def lower(angle,is_across_midline,is_out_to_side):
            self.assertEqual(angle,81.);self.assertFalse(is_across_midline);self.assertFalse(is_out_to_side)
            return 1
        def score(*args):
            called.append(args)
            return (1,2,2,3,3,'AL2')
        attach_research_score(data,'左手側',review,lower,score)
        self.assertEqual(data['左手 Grand Score'],4)
        self.assertEqual(data['V12_研究試算GrandScore'],3)
        self.assertEqual(data['V12_研究與原總分差'],-1)
        self.assertTrue(data['V12_頸部跨側分歧警示'])
        self.assertEqual(len(called),1)

    def test_empty_side_frames_graceful(self):
        review=examine_frames({'Front':None,'Left':None,'Right':None},DummyPose(),9,1.2)
        self.assertEqual(len(review['summary_rows']),4)
        self.assertEqual(review['choices'][('左手側','前臂')]['selected_camera'],'無')

if __name__=='__main__':unittest.main()
