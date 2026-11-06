import io
import zipfile
import math
from v13_oop_review import assess_one,evaluate_case,merge_decisions,audit_flags,append_audit_zip


def cand(camera,side='左手側',part='前臂',angle=70,vis=.9,width=.20,segment=.5,eligible=True):
    return {'camera':camera,'body_side':side,'part':part,'angle_deg':angle,
        'min_visibility':vis,'shoulder_width_torso_ratio':width,'shoulder_line_visibility':.9,
        'forearm_upper_ratio':1.0,'projected_length_ratio':segment,
        'view_applicable':eligible,'eligible':eligible,'cross_view_conflict':False}


def case(*cands):
    return {'candidate_rows':list(cands),'choices':{(c['body_side'],c['part']):{
        'selected_camera':c['camera'],'selected_angle_deg':c['angle_deg']}
        for c in cands if c['camera']=='Left'}}


def test_side_is_preserved_even_when_candidate_in_opposite_camera():
    a=cand('Left',angle=50); b=cand('Right',angle=90)
    got=assess_one('左手側','前臂',case(a,b)['choices'],[a,b],18,6.0)
    assert got['評估側']=='左手側'
    assert got['原始鏡頭']=='Left'
    assert got['替代側面鏡頭']=='Right'
    assert got['替代側面候選角度']==90
    assert '同一人體側別' in got['可能原因']


def test_front_only_never_selected_as_sagittal_substitute():
    a=cand('Left',angle=50); f=cand('Front',angle=110)
    got=assess_one('左手側','前臂',{},[a,f],18,6.0)
    assert got['可用替代候選(待人工確認)']==False
    assert got['正面候選角度(僅診斷)']==110


def test_no_warning_does_not_certify_correct_angle():
    a=cand('Left',angle=60);b=cand('Right',angle=62)
    got=assess_one('左手側','前臂',{},[a,b],1,2)
    assert got['品質狀態'].startswith('未觸發疑慮')
    assert '非2D量角準確性證明' in got['品質狀態']


def test_shoulder_rotation_screen_flags():
    a=cand('Left',width=.80);b=cand('Right',width=.20)
    got=assess_one('左手側','前臂',{},[a,b],18,2)
    assert '肩投影偏寬' in got['可能原因']


def test_segment_projection_warns_not_3d_truth():
    a=cand('Left',segment=.1);b=cand('Right',segment=.6)
    got=assess_one('左手側','前臂',{},[a,b],2,2)
    assert '投影長度差異顯著' in got['可能原因']
    assert '疑似' in got['品質狀態']


def test_neck_cross_camera_conflict():
    a=cand('Left',part='頸部',angle=12);b=cand('Right',part='頸部',angle=45)
    got=assess_one('左手側','頸部',{},[a,b],3,5)
    assert '頸角差≥15' in got['可能原因']


def test_missing_side_does_not_pretend_quality():
    a=cand('Front',angle=15)
    got=assess_one('左手側','前臂',{},[a],1,0)
    assert '原鏡頭缺少可計算角度' in got['可能原因']


def test_review_keeps_rula_when_only_angle_is_invalid():
    a=cand('Left');b=cand('Right')
    rows=evaluate_case(case(a,b),1,1,('左手側',))
    saved={(1,'左手側','前臂'):{'人工處理':'無可靠角度；仍可判RULA分級','人工確認備註':'觀察其他鏡頭'}}
    df=audit_flags(merge_decisions(rows,saved))
    fore=df[df['部位']=='前臂'].iloc[0]
    assert not fore['人工角度參考可用']
    assert fore['RULA分級仍可評估']


def test_review_reports_unassessable_rula():
    rows=evaluate_case({'candidate_rows':[],'choices':{}},1,1,('左手側',))
    saved={(1,'左手側','前臂'):{'人工處理':'無可靠角度；RULA分級也無法確認'}}
    df=audit_flags(merge_decisions(rows,saved))
    assert not df.iloc[0]['RULA分級仍可評估']


def test_zip_keeps_existing_and_adds_review():
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as z:z.writestr('原始分數.csv','test,123')
    rows=evaluate_case({'candidate_rows':[],'choices':{}},18,6,('左手側',))
    saved={(18,'左手側','前臂'):{'人工處理':'無可靠角度；仍可判RULA分級','人工確認備註':'雙側都遮擋'}}
    updated=append_audit_zip(blob.getvalue(),rows,saved)
    with zipfile.ZipFile(io.BytesIO(updated)) as z:
        assert z.read('原始分數.csv')==b'test,123'
        assert 'Diagnostics/V13_人工局部覆核決策.csv' in z.namelist()
        contents=z.read('Diagnostics/V13_人工局部覆核決策.csv').decode('utf-8-sig')
        assert '雙側都遮擋' in contents
        assert '無可靠角度；仍可判RULA分級' in contents
