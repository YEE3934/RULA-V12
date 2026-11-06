"""V13: exploratory out-of-plane / projection-risk screening and human review.

IMPORTANT: warning == suspected unreliable 2D projection, NOT proven 3D motion.
Nothing in this module alters official 222.py RULA scores or generates 3D angles.
"""
from __future__ import annotations
import io
import math
import zipfile
from typing import Any
import pandas as pd

REVIEW_OPTIONS = (
    '未審核',
    '人工確認：原角度可用',
    '人工確認：替代側向角度可用',
    '無可靠角度；仍可判RULA分級',
    '無可靠角度；RULA分級也無法確認',
    '非離平面；其他品質原因',
)
GUIDE = '''V13 離平面／2D投影品質研究功能（非真實3D離平面定量器）
- 「疑似離平面」是肩寬投影、肢段投影縮短、不同鏡頭角度分歧、關節可見度等間接線索，不是旋轉角度真值。
- 前臂的 Left/Right 候選仍必須抓同一人體側別的肩肘腕；Front 不直接取代前臂屈曲角。
- 頸部 Front 不直接取代前屈角。左右側不同參考點的角度不能平均。
- 低品質影格不自動剔除，也不覆寫 RULA Grand Score。人工覆核決定角度參考是否有效與 RULA 分級是否有依據。
- 『無可靠角度；仍可判RULA分級』：角度主要 MAE 不使用該部位該影格，RULA 分級可在有證據時保留。
- 『RULA分級也無法確認』：要另報評估失敗／可用率，不能作為已驗證正確的分數。
- 角度缺失不等於零誤差；所有原始影格與標記保留以供稽核。實驗仍需人工標註警示的真陽性、假陽性與漏判。
- 現有分數／角度正式計算均不因V13新功能變更。
'''

def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

def _choice(choices: dict, side: str, part: str):
    return choices.get((side,part),{}) if isinstance(choices,dict) else {}

def assess_one(side: str, part: str, choices: dict, candidates: list[dict], sample_id: int, sec: float) -> dict:
    """Conservative projection risk flags independent of human ground-truth angles.

    Positive flags are not a validated out-of-plane classification.
    Also record no-signal cases as unverified, never certified in-plane.
    """
    choice=_choice(choices,side,part)
    original = 'Left' if side=='左手側' else 'Right'
    views={c.get('camera'):c for c in candidates if c.get('body_side')==side and c.get('part')==part}
    base=views.get(original,{})
    alternate='Right' if original=='Left' else 'Left'
    alt=views.get(alternate,{})
    front=views.get('Front',{})
    reasons=[]
    evidence=[]
    r=base.get('shoulder_width_torso_ratio')
    if finite(r):
        evidence.append(f'原側面肩寬/軀幹投影比={float(r):.3f}')
        if float(r)>.55 and finite(base.get('shoulder_line_visibility')) and float(base['shoulder_line_visibility'])>=.35:
            reasons.append('原側面雙肩投影偏寬：疑似軀幹偏離正側面')
    else:
        evidence.append('原側面肩線資料不足')
    if not bool(base.get('view_applicable',False)):
        reasons.append('原側面不通過側向適用性初篩')
    if finite(base.get('min_visibility')) and float(base['min_visibility'])<.55:
        reasons.append('指定部位關鍵點可見度偏低／可能遮擋')
    if part=='前臂':
        fore=base.get('forearm_upper_ratio')
        if finite(fore) and (float(fore)<.40 or float(fore)>2.70):
            reasons.append('前臂/上臂投影肢段比不合理')
        # Compare only same anatomical arm across side cameras: each angular disagreement is warning, not proof.
        if finite(base.get('angle_deg')) and finite(alt.get('angle_deg')):
            diff=abs(float(base['angle_deg'])-float(alt['angle_deg']))
            evidence.append(f'同人體側別之側向鏡頭前臂角差={diff:.1f}°')
            if diff>=20:
                reasons.append('同一人體側別不同側向鏡頭前臂角差≥20°（需檢查投影/遮擋/同步）')
        # Relative projected segment shortening is a warning (only if both candidate landmarks reliable).
        a,b=base.get('projected_length_ratio'),alt.get('projected_length_ratio')
        if finite(a) and finite(b) and float(a)>0 and float(b)>0:
            ratio=min(float(a),float(b))/max(float(a),float(b))
            evidence.append(f'雙側面肘腕投影相對短長比={ratio:.3f}')
            if ratio<.60:
                reasons.append('側向鏡頭肘腕投影長度差異顯著（可能離平面或遮擋）')
    elif part=='頸部':
        if base.get('cross_view_conflict'):
            reasons.append('同側頸部跨鏡頭代理角度存在顯著衝突')
        if finite(base.get('angle_deg')) and finite(alt.get('angle_deg')):
            diff=abs(float(base['angle_deg'])-float(alt['angle_deg']))
            evidence.append(f'同人體側別跨鏡頭頸角差={diff:.1f}°')
            if diff>=15:
                reasons.append('同人體側別跨鏡頭頸角差≥15°')
    if not base or not finite(base.get('angle_deg')):
        reasons.append('原鏡頭缺少可計算角度')
    # Review requirement is downstream manual; V12 fallback can't certify quality.
    if reasons:
        status='疑似離平面／投影或關鍵點品質異常（非確診）'
        require='建議局部人工覆核'
    else:
        status='未觸發疑慮（非2D量角準確性證明）'
        require='一般抽查'
    replacement=(choice.get('selected_camera') if choice.get('selected_camera') not in ('',original,None) and
                 finite(choice.get('selected_angle_deg')) else '')
    return {
        '樣本編號':int(sample_id),'時間(秒)':round(float(sec),3),'評估側':side,'部位':part,
        '原始鏡頭':original,'原側面研究候選角度':base.get('angle_deg'),
        '替代側面鏡頭':alternate,'替代側面候選角度':alt.get('angle_deg'),
        '正面候選角度(僅診斷)':front.get('angle_deg'),
        'V12_研究建議鏡頭':choice.get('selected_camera') or '',
        'V12_研究建議角度':choice.get('selected_angle_deg'),
        '品質狀態':status,'品質證據':'；'.join(evidence),
        '可能原因':'；'.join(dict.fromkeys(reasons)) if reasons else '無明顯警示，仍需抽查',
        '建議處理':require,
        '可用替代候選(待人工確認)':bool(replacement),
        '人工處理':REVIEW_OPTIONS[0],
        '人工採用角度(度)':float('nan'),
        '人工確認備註':'',
    }

def evaluate_case(case: dict, sample_id: int, sec: float, sides: tuple[str,...] = ('左手側','右手側')) -> list[dict]:
    return [assess_one(s,p,case.get('choices',{}),case.get('candidate_rows',[]),sample_id,sec)
            for s in sides for p in ('前臂','頸部')]

def _key(row) -> tuple:
    return (int(row['樣本編號']),str(row['評估側']),str(row['部位']))

def merge_decisions(rows:list[dict], saved:dict) -> pd.DataFrame:
    df=pd.DataFrame(rows)
    if df.empty:return df
    for idx, rec in df.iterrows():
        change=saved.get(_key(rec),{})
        for name in ('人工處理','人工採用角度(度)','人工確認備註'):
            if name in change:
                df.at[idx,name]=change[name]
    return df

def audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:return df
    x=df.copy()
    status=x['人工處理'].fillna('未審核')
    x['人工已覆核']=status.ne('未審核')
    x['人工角度參考可用']=status.isin(['人工確認：原角度可用','人工確認：替代側向角度可用'])
    x['RULA分級仍可評估']=~status.eq('無可靠角度；RULA分級也無法確認')
    x['角度MAE納入建議']='需再核對人工與系統角度定義及獨立真值；不能由此欄自動當成已驗證'
    x['角度不可靠比例統計分類']=status.map({
        '未審核':'尚未審核',
        '人工確認：原角度可用':'角度可用',
        '人工確認：替代側向角度可用':'角度可用（人工確認候選）',
        '無可靠角度；仍可判RULA分級':'角度不可用/RULA級距可能可用',
        '無可靠角度；RULA分級也無法確認':'角度及RULA級距均不可用',
        '非離平面；其他品質原因':'非離平面品質異常'}).fillna('待確認')
    return x

def append_audit_zip(blob:bytes, rows:list[dict], saved:dict) -> bytes:
    """Augment final ZIP with V13 audit; no mutation of any existing score files."""
    df=audit_flags(merge_decisions(rows,saved))
    sink=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob),'r') as src,zipfile.ZipFile(sink,'w',compression=zipfile.ZIP_DEFLATED) as dest:
        for name in src.namelist():
            dest.writestr(src.getinfo(name),src.read(name))
        dest.writestr('Diagnostics/V13_疑似離平面與量角品質篩檢.csv',df.to_csv(index=False).encode('utf-8-sig'))
        reviews=df[df['人工已覆核']] if not df.empty else df
        dest.writestr('Diagnostics/V13_人工局部覆核決策.csv',reviews.to_csv(index=False).encode('utf-8-sig'))
        dest.writestr('Diagnostics/V13_使用限制與統計說明.txt',GUIDE.encode('utf-8'))
    return sink.getvalue()
