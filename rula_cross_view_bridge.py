"""V12 cross-view *research diagnostics* for the original RULA 222.py.

All exported angles are 2D projections, not calibrated joint angles. The
original per-side RULA Grand Score is never overwritten by this module.
"""
from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from angle_lab_core import (
    SIDES, CAMERAS, PARTS, candidates_from_landmarks,
    refine_relative_projection, refine_neck_cross_view_conflicts,
    choose_candidate,
)

CROSS_VIEW_GUIDE = """RULA V12 跨視角實驗研究說明（不是量測準確率證明）
1. 人體評估側（左手側／右手側）與攝影機方向（Left／Right／Front）完全不同；切換攝影機時不換人體側別。
2. 前臂屈曲：固定肩11/12、肘13/14、腕15/16，顯示 abs(180 - 肘內角)。人工若量的是肘內角，先換成 180-內角才比較。
3. 正面 Front 前臂及頸部前屈角僅供診斷，不會自動替代側面角度；可見度或品質分數不是實際精度。
4. 另一側攝影機即使通過初步品質檢查，仍屬研究候選，可能受遮擋與離平面影響。
5. 跨視角研究試算分數只在研究欄位顯示，不覆寫原系統 Grand Score、AL、複核結果或工作條件補正。
6. 頸部左右相差大時只提出警示；不同身體側參考點／視角的 2D 耳肩髖投影不可直接平均。
7. 符合初篩不等於準確；拒絕樣本不是零誤差。完整驗證需人工真值及未參與調參的獨立樣本。
8. 各攝影機需預先確認同步，攝影機影像不能鏡像；鏡像資料需另行驗證左右關鍵點映射。
9. 如需真實 3D 關節角，應使用經校準多視角三角定位或獨立量測設備；本研究版未進行 3D 重建。
"""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _number(value: Any) -> float:
    return float(value) if _finite(value) else float("nan")


def _landmarks_to_dict(landmarks) -> dict[int, dict[str, float]]:
    return {i:{'x':float(p.x),'y':float(p.y),'visibility':float(getattr(p,'visibility',0.0))}
            for i,p in enumerate(landmarks)}


def examine_frames(frames: dict[str, Any], pose_model, sample_id: int,
                   time_sec: float, resize_max: int = 1000) -> dict:
    """Independent 3-camera landmarks, both anatomical sides, once per frame.

    When media/pose fails, return a diagnostic record instead of aborting RULA.
    Does not calculate clinical 3D or official RULA scores.
    """
    import cv2
    candidates=[]
    errors=[]
    for camera in CAMERAS:
        frame=frames.get(camera)
        if frame is None or getattr(frame,'size',0)==0:
            continue
        try:
            h,w=frame.shape[:2]
            if h<=0 or w<=0:
                raise ValueError('空影像尺寸')
            scale=min(1.0,float(resize_max)/max(h,w))
            if scale < 1.0:
                frame=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
            hh,ww=frame.shape[:2]
            result=pose_model.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            landmarks=getattr(getattr(result,'pose_landmarks',None),'landmark',None)
            if landmarks is None:
                errors.append(f'{camera}:未偵測到人體')
                continue
            points=_landmarks_to_dict(landmarks)
            for side in SIDES:
                candidates.extend(candidates_from_landmarks(points,ww,hh,camera,side))
        except (ValueError,TypeError,AttributeError,RuntimeError,cv2.error) as exc:
            errors.append(f'{camera}:{type(exc).__name__}: {exc}')
    candidates=refine_relative_projection(candidates)
    candidates=refine_neck_cross_view_conflicts(candidates,threshold_deg=15.0)
    choices={}
    for side in SIDES:
        for part in PARTS:
            choices[(side,part)]=choose_candidate(candidates,side,part)

    # Opposite anatomical sides may have different 2D reference lines. Only flag;
    # do not mix anatomical-side projections into an averaged neck 'truth'.
    left_base=next((c for c in candidates if c['camera']=='Left' and c['body_side']=='左手側' and c['part']=='頸部'),None)
    right_base=next((c for c in candidates if c['camera']=='Right' and c['body_side']=='右手側' and c['part']=='頸部'),None)
    if left_base and right_base and _finite(left_base.get('angle_deg')) and _finite(right_base.get('angle_deg')):
        opposite_gap=round(abs(left_base['angle_deg']-right_base['angle_deg']),2)
    else:
        opposite_gap=float('nan')

    rows=[]
    for side in SIDES:
        for part in PARTS:
            item=choices[(side,part)]
            rows.append({
                '樣本編號':int(sample_id),'時間(秒)':round(float(time_sec),3),
                '評估側':side,'部位':part,
                '原攝影機':item['baseline_camera'],
                '原攝影機研究角度':item['baseline_angle_deg'],
                '候選攝影機':item['selected_camera'],
                '候選角度':item['selected_angle_deg'],
                '候選品質':item['selected_quality'],
                '選擇結果':item['decision'],
                '選擇說明':item['decision_note'],
                '人體左右側頸代理角分歧(度)':opposite_gap if part=='頸部' else float('nan'),
                '跨側頸角重大分歧警示':bool(part=='頸部' and _finite(opposite_gap) and opposite_gap>=15),
                '模型診斷':'; '.join(errors),
            })
    candidate_rows=[{'樣本編號':int(sample_id),'時間(秒)':round(float(time_sec),3),**c}
                    for c in candidates]
    return {'choices':choices,'summary_rows':rows,'candidate_rows':candidate_rows,
            'neck_opposite_gap':opposite_gap,'errors':errors}


def _has_modifier(data: dict, key: str) -> bool:
    return str(data.get(key,'')).startswith('是')


def attach_research_score(data: dict, side: str, review: dict | None,
                          get_lower_arm_score: Callable, compute_rula_full: Callable) -> dict:
    """Attach diagnostic and an independently recomputed *scenario* RULA score.

    The input dictionary's original Grand Score is NEVER changed. Recomputing is
    gated: a new score is emitted only for a candidate chosen by the V12 logic.
    """
    if not isinstance(data,dict) or not review:
        return data
    prefix='右手' if side=='右手側' else '左手'
    fore=review['choices'][(side,'前臂')]
    neck=review['choices'][(side,'頸部')]
    reason=fore['decision']
    data.update({
        'V12_前臂原側面角度_研究重算':_number(fore['baseline_angle_deg']),
        'V12_前臂原系統正式角度':_number(data.get(prefix+'前臂角度')),
        'V12_前臂候選角度':_number(fore['selected_angle_deg']),
        'V12_前臂候選攝影機':fore['selected_camera'],
        'V12_前臂選擇結果':reason,
        'V12_前臂選擇說明':fore['decision_note'],
        'V12_頸部原系統正式角度':_number(data.get('頸部角度')),
        'V12_頸部原側面角度_研究重算':_number(neck['baseline_angle_deg']),
        'V12_頸部候選角度':_number(neck['selected_angle_deg']),
        'V12_頸部候選攝影機':neck['selected_camera'],
        'V12_頸部選擇結果':neck['decision'],
        'V12_頸部選擇說明':neck['decision_note'],
        'V12_人體左右側頸代理角差':_number(review.get('neck_opposite_gap')),
        'V12_頸部跨側分歧警示':bool(_finite(review.get('neck_opposite_gap')) and review['neck_opposite_gap']>=15),
        'V12_正式分數未變動':True,
        'V12_研究試算GrandScore':float('nan'),
        'V12_研究試算AL':'不適用',
        'V12_研究試算前臂分數':float('nan'),
        'V12_研究試算來源':'無可用候選角度',
    })
    if not _finite(fore.get('selected_angle_deg')):
        return data
    # Keep the official baseline exactly unchanged when the selected camera
    # remains the original. Re-running Pose on the same image can yield small
    # inference fluctuations; those are not cross-view improvements.
    if fore['selected_camera']==fore['baseline_camera']:
        data['V12_研究試算前臂分數']=int(data[prefix+'前臂分數'])
        data['V12_研究試算GrandScore']=int(data[prefix+' Grand Score'])
        data['V12_研究試算AL']=str(data[prefix+' AL'])
        data['V12_研究試算來源']='沿用原攝影機；試算分數與正式分數一致'
        data['V12_研究與原總分差']=0
        return data
    try:
        new_forearm_score=int(get_lower_arm_score(
            float(fore['selected_angle_deg']),
            is_across_midline=_has_modifier(data,prefix+'橫越中線補正'),
            is_out_to_side=_has_modifier(data,prefix+'前臂向外側工作補正'),
        ))
        rula=compute_rula_full(
            int(data[prefix+'上臂分數']),new_forearm_score,
            int(data[prefix+'手腕分數']),int(data[prefix+'手腕扭轉分數']),
            int(data['頸部分數']),int(data['軀幹分數']),int(data['腿部分數']),
            int(data['肌肉使用加分']),int(data['負重施力加分']))
        data['V12_研究試算前臂分數']=new_forearm_score
        data['V12_研究試算GrandScore']=int(rula[4])
        data['V12_研究試算AL']=str(rula[5])
        data['V12_研究試算來源']=('沿用原攝影機（研究品質檢查）' if fore['selected_camera']==fore['baseline_camera']
                               else '另一側鏡頭同一人體側別（未驗證候選，不是正式分數）')
        data['V12_研究與原總分差']=int(rula[4])-int(data[prefix+' Grand Score'])
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as exc:
        data['V12_研究試算來源']='研究試算失敗：'+type(exc).__name__
    return data


def research_score_row(data: dict, side: str, sample_id: int, time_sec: float) -> dict:
    """Compact audit row exported independently of the existing user report filter."""
    prefix='右手' if side=='右手側' else '左手'
    fields=(
        'V12_前臂原系統正式角度','V12_前臂原側面角度_研究重算',
        'V12_前臂候選角度','V12_前臂候選攝影機','V12_前臂選擇結果','V12_前臂選擇說明',
        'V12_頸部原系統正式角度','V12_頸部原側面角度_研究重算',
        'V12_頸部候選角度','V12_頸部候選攝影機','V12_頸部選擇結果','V12_頸部選擇說明',
        'V12_人體左右側頸代理角差','V12_頸部跨側分歧警示',
        'V12_研究試算前臂分數','V12_研究試算GrandScore',
        'V12_研究試算AL','V12_研究試算來源','V12_研究與原總分差',
        'V12_正式分數未變動',
    )
    return {'樣本編號':int(sample_id),'時間(秒)':round(float(time_sec),3),'評估側':side,
            '正式GrandScore':data.get(prefix+' Grand Score'),
            '正式AL':data.get(prefix+' AL'),
            **{field:data.get(field) for field in fields}}
