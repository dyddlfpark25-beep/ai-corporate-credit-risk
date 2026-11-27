import os
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# ==========================================================
# 0. 페이지 설정 & 폰트 + IBCS 미니멀 스타일
# ==========================================================
st.set_page_config(
    page_title="Credit Risk Report | AI 기업여신 심사",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)

if os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
    fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams['font.family'] = fm.FontProperties(
        fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    ).get_name()
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# IBCS 핵심 3색 팔레트
C_CHARCOAL = "#333333"
C_DARK     = "#1A1A1A"
C_GRAY     = "#737373"
C_LIGHT    = "#E5E5E5"
C_GREEN    = "#6CA62B"
C_RED      = "#D93829"

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #222222;
    font-variant-numeric: tabular-nums;
}

/* 상단 Streamlit Cloud 툴바와 겹치지 않도록 여백 확보 */
.block-container {
    padding-top: 3.8rem !important;
    padding-bottom: 4.5rem !important;
    max-width: 1420px;
}

/* 사이드바: 얇은 선으로만 구분 */
section[data-testid="stSidebar"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E5E5;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2.2rem !important;
}
.sb-title { font-size: 0.96rem; font-weight: 800; color: #1A1A1A; letter-spacing: -0.01em; margin-bottom: 2px; }
.sb-sub   { font-size: 0.74rem; color: #737373; margin-bottom: 14px; }
.sb-sec   {
    font-size: 0.69rem; font-weight: 700; color: #737373;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-top: 4px; margin-bottom: 6px;
}

/* IBCS 헤더 영역 */
.ibcs-meta-top { font-size: 0.84rem; font-weight: 700; color: #1A1A1A; line-height: 1.45; }
.ibcs-meta-sub { font-size: 0.78rem; color: #737373; line-height: 1.45; }
.ibcs-headline {
    font-size: 1.42rem;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: -0.025em;
    line-height: 1.35;
}
.ibcs-headline b { font-weight: 700; }
.ibcs-subline {
    font-size: 0.82rem;
    color: #737373;
    margin-top: 6px;
}

/* 테두리 없는 KPI 스트립 */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    padding: 16px 0 12px 0;
}
.kpi-cell { padding-right: 8px; }
.kpi-num {
    font-size: 2.05rem;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: -0.03em;
    line-height: 1.1;
    white-space: nowrap;
}
.kpi-delta-red   { font-size: 0.80rem; font-weight: 700; color: #D93829; margin-left: 5px; }
.kpi-delta-green { font-size: 0.80rem; font-weight: 700; color: #6CA62B; margin-left: 5px; }
.kpi-delta-gray  { font-size: 0.80rem; font-weight: 600; color: #737373; margin-left: 5px; }
.kpi-lbl { font-size: 0.78rem; color: #525252; font-weight: 600; margin-top: 6px; }
.kpi-sub { font-size: 0.73rem; color: #8C8C8C; margin-top: 2px; white-space: nowrap; }

/* 얇은 섹션 구분선 & 타이틀 */
.ibcs-rule { border: 0; border-top: 1px solid #E5E5E5; margin: 14px 0 18px 0; }
.ibcs-sec-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #222222;
    margin-bottom: 2px;
}
.ibcs-sec-sub {
    font-size: 0.75rem;
    color: #737373;
    margin-bottom: 10px;
}

/* IBCS 인라인 편차 막대 표 */
.ibcs-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.80rem;
    margin-top: 4px;
}
.ibcs-table th {
    text-align: left;
    font-size: 0.72rem;
    font-weight: 700;
    color: #737373;
    padding: 6px 6px 8px 6px;
    border-bottom: 1.5px solid #333333;
    white-space: nowrap;
}
.ibcs-table td {
    padding: 7px 6px;
    border-bottom: 1px solid #EFEFEF;
    vertical-align: middle;
    color: #222222;
}
.ibcs-table tr:hover td { background-color: #FAFAFA; }
.td-lbl   { font-weight: 600; white-space: nowrap; }
.td-num   { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; color: #333333; }
.td-muted { text-align: right; color: #737373; white-space: nowrap; }

.bar-wrap { display: flex; align-items: center; gap: 6px; }
.bar-track { width: 76px; height: 11px; background: #F3F3F3; position: relative; flex-shrink: 0; }
.bar-fill  { height: 11px; background: #333333; }

.var-cell { display: flex; align-items: center; justify-content: center; }
.var-half-l { width: 50px; display: flex; justify-content: flex-end; align-items: center; padding-right: 1px; }
.var-axis   { width: 1.5px; height: 18px; background: #333333; flex-shrink: 0; }
.var-half-r { width: 50px; display: flex; justify-content: flex-start; align-items: center; padding-left: 1px; }
.var-red    { height: 10px; background: #D93829; }
.var-green  { height: 10px; background: #6CA62B; }

.dot-red   { display:inline-block; width:7px; height:7px; border-radius:50%; background:#D93829; margin-right:5px; }
.dot-amber { display:inline-block; width:7px; height:7px; border-radius:50%; background:#D97706; margin-right:5px; }
.dot-green { display:inline-block; width:7px; height:7px; border-radius:50%; background:#6CA62B; margin-right:5px; }

/* 심사의견 & 체크리스트 (종이 리포트 스타일) */
.memo-clean {
    border-left: 2.5px solid #333333;
    padding: 4px 0 4px 16px;
    font-size: 0.87rem;
    line-height: 1.72;
    color: #262626;
}
.memo-clean p { margin: 0 0 10px 0; }
.memo-clean p:last-child { margin-bottom: 0; }

.doc-row {
    padding: 8px 0;
    border-bottom: 1px solid #EFEFEF;
    font-size: 0.83rem;
    line-height: 1.58;
    color: #333333;
}
.doc-row:last-child { border-bottom: none; }
.doc-cat {
    font-size: 0.72rem;
    font-weight: 700;
    color: #D93829;
    letter-spacing: 0.02em;
    margin-bottom: 2px;
}
.doc-cat-ok { color: #6CA62B; }

/* 미니멀 차콜 다운로드 버튼 */
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1.5px solid #333333 !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
.stDownloadButton > button:hover {
    background-color: #333333 !important;
    color: #FFFFFF !important;
}

.footnote {
    font-size: 0.73rem;
    color: #8C8C8C;
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #EFEFEF;
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. 자산 로드
# ==========================================================
@st.cache_resource
def load_assets():
    data = joblib.load('preprocessed_data.pkl')
    model_bundle = joblib.load('best_calibrated_model.pkl')

    cal_model = model_bundle['champion_calibrated']
    cc = cal_model.calibrated_classifiers_[0]
    base_lgbm = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
    if hasattr(base_lgbm, 'estimator'):
        base_lgbm = base_lgbm.estimator

    feature_cols = list(data['X_train'].columns)

    ref_raw, ref_y = None, None
    if os.path.exists('data.csv'):
        try:
            tmp = pd.read_csv('data.csv')
            tmp.columns = tmp.columns.str.strip()
            if 'Bankrupt?' in tmp.columns and all(c in tmp.columns for c in feature_cols):
                ref_raw = tmp[feature_cols].copy()
                ref_y = tmp['Bankrupt?'].copy()
        except Exception:
            ref_raw = None

    if ref_raw is None:
        ref_raw = data['X_test_raw'][feature_cols].copy()
        ref_y = data['y_test'].copy()

    normal_med   = ref_raw[ref_y == 0].median()
    bankrupt_med = ref_raw[ref_y == 1].median()

    return data, model_bundle, cal_model, base_lgbm, feature_cols, normal_med, bankrupt_med

data, model_bundle, cal_model, base_lgbm, feature_cols, normal_med, bankrupt_med = load_assets()

X_test_proc  = data['X_test'][feature_cols]
X_test_raw   = data['X_test_raw'][feature_cols]
y_test       = data['y_test']
cont_cols    = [c for c in data['cont_cols'] if c in feature_cols]
lower_bounds = data['lower_bounds']
upper_bounds = data['upper_bounds']
scaler       = data['scaler']

all_probs = cal_model.predict_proba(X_test_proc)[:, 1]
idx_safe = int(np.argmin(all_probs))
idx_mid  = int(np.argmin(np.abs(all_probs - 0.048)))
idx_risk = int(np.argmax(all_probs))

# 한 줄에 깔끔하게 떨어지는 지표명 구성
VERIFIED_METRICS = [
    ('Persistent EPS in the Last Four Seasons', '지속 EPS (4분기)',   '수익성',       'higher_better'),
    ('Operating Profit Rate',                   '영업이익률',         '수익성',       'higher_better'),
    ('Quick Ratio',                             '당좌비율',          '유동성',       'higher_better'),
    ('Current Liability to Assets',             '유동부채/총자산',     '유동성',       'higher_worse'),
    ('Borrowing dependency',                    '차입금 의존도',       '부채·레버리지', 'higher_worse'),
    ('Debt ratio %',                            '부채비율',          '부채·레버리지', 'higher_worse'),
    ('Equity to Liability',                     '자기자본/부채',       '부채·레버리지', 'higher_better'),
    ('Cash flow rate',                          '영업현금흐름 비율',    '현금흐름',     'higher_better'),
]
ACTIVE_METRICS = [m for m in VERIFIED_METRICS if m[0] in X_test_raw.columns]

def map_feature_category(col_name):
    c = col_name.lower()
    if any(k in c for k in ['roa', 'eps', 'profit', 'income', 'margin', 'tax', 'earning']):
        return '수익성'
    elif any(k in c for k in ['debt', 'borrowing', 'liability to equity', 'equity to liability', 'net worth', 'interest']):
        return '부채·레버리지'
    elif any(k in c for k in ['quick', 'current', 'working capital', 'liquidity']):
        return '단기 유동성'
    elif any(k in c for k in ['cash', 'turnover', 'inventory', 'receivable', 'asset']):
        return '현금흐름·활동성'
    else:
        return '기타 재무구조'

FTP_BASE   = 0.0350
OPEX_RATE  = 0.0080
CAP_RATIO  = 0.1050
TARGET_ROE = 0.1000

def evaluate_credit_and_pricing(pd_val, ead_krw, term_yrs, lgd_val, rate_adj_bp=0.0):
    if pd_val < 0.01:
        grade, decision, tone, rw = 'AAA~A (우량)', '승인 · 우대 금리 적용', 'green', 0.50
    elif pd_val < 0.03:
        grade, decision, tone, rw = 'BBB (양호)', '정상 승인', 'green', 0.75
    elif pd_val < 0.07:
        grade, decision, tone, rw = 'BB (주의)', '조건부 승인 · 담보 보강 및 한도 축소', 'amber', 1.00
    elif pd_val < 0.15:
        grade, decision, tone, rw = 'B (위험)', '정밀 심사 요망 · 고금리 적용', 'red', 1.50
    else:
        grade, decision, tone, rw = 'CCC~D (고위험)', '여신 거절 (Reject)', 'red', 1.50

    term_prem    = (term_yrs - 1) * 0.0015
    el_rate      = pd_val * lgd_val
    cap_chg      = rw * CAP_RATIO * TARGET_ROE
    fair_rate    = FTP_BASE + term_prem + OPEX_RATE + el_rate + cap_chg
    applied_rate = max(0.0, fair_rate + (rate_adj_bp / 10000.0))

    cum_pd     = 1.0 - ((1.0 - pd_val) ** term_yrs)
    el_amt     = ead_krw * el_rate
    rwa_amt    = ead_krw * rw
    req_cap    = rwa_amt * CAP_RATIO
    net_spread = applied_rate - (FTP_BASE + term_prem + OPEX_RATE + el_rate)
    net_profit = ead_krw * net_spread
    raroc      = (net_profit / req_cap) if req_cap > 0 else 0.0

    return {
        'pd': pd_val, 'cum_pd': cum_pd, 'grade': grade, 'decision': decision,
        'tone': tone, 'rw': rw, 'term_prem': term_prem,
        'el_rate': el_rate, 'cap_chg': cap_chg, 'fair_rate': fair_rate,
        'applied_rate': applied_rate, 'el_amt': el_amt, 'rwa_amt': rwa_amt,
        'req_cap': req_cap, 'net_profit': net_profit, 'raroc': raroc
    }

def preprocess_row(raw_df, macro_shock=0.0, adj_profit=0.0, adj_debt=0.0):
    proc = raw_df.copy()
    clipped = proc[cont_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1).clip(lower=0)
    proc[cont_cols] = np.log1p(clipped)
    proc[cont_cols] = scaler.transform(proc[cont_cols])

    profit_cols = [c for c in proc.columns if any(k in c for k in ['ROA', 'Net Income', 'Profit', 'Cash flow', 'EPS'])]
    debt_cols   = [c for c in proc.columns if any(k in c for k in ['Debt', 'Liability', 'Borrowing'])]

    proc[profit_cols] = proc[profit_cols] - macro_shock + adj_profit
    proc[debt_cols]   = proc[debt_cols] + macro_shock + adj_debt
    return proc

# ==========================================================
# 2. 사이드바 (모든 필터·시뮬레이션 입력 통합)
# ==========================================================
st.sidebar.markdown('<div class="sb-title">FILTERS & CONTROLS</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sb-sub">기업여신 심사 조건 및 스트레스 시뮬레이션</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-sec">01. 심사 대상 기업</div>', unsafe_allow_html=True)
sample_choice = st.sidebar.radio(
    "심사 대상 선택",
    [
        f"우량 기업 샘플 (#{idx_safe})",
        f"주의 기업 샘플 (#{idx_mid})",
        f"고위험 기업 샘플 (#{idx_risk})",
        "직접 기업 번호 입력 (0~1363)"
    ],
    label_visibility="collapsed"
)
if "우량" in sample_choice:     selected_idx = idx_safe
elif "주의" in sample_choice:   selected_idx = idx_mid
elif "고위험" in sample_choice: selected_idx = idx_risk
else:
    selected_idx = int(st.sidebar.number_input("Test 기업 번호", 0, len(X_test_raw) - 1, 0))

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sb-sec">02. 여신 신청 조건</div>', unsafe_allow_html=True)
firm_name = st.sidebar.text_input("기업명", value=f"심사대상 기업 #{selected_idx}")
ead_krw   = st.sidebar.number_input("대출 신청액 EAD (억 원)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
term_yrs  = st.sidebar.slider("대출 만기 (년)", 1, 5, 3)

collateral_map = {
    "무담보 · 신용대출 (LGD 45%)": 0.45,
    "부분 담보 · 보증서 (LGD 35%)": 0.35,
    "우량 부동산 담보 (LGD 25%)":   0.25
}
collat_choice = st.sidebar.selectbox("담보 조건 (LGD)", list(collateral_map.keys()))
lgd_val = collateral_map[collat_choice]

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sb-sec">03. 거시 스트레스 & 금리 협상</div>', unsafe_allow_html=True)
macro_opt = st.sidebar.selectbox(
    "거시경제 시나리오",
    ["평상시 (Baseline)", "경기 둔화 (Mild: ±0.5σ)", "금융 위기 (Severe: ±1.0σ)"]
)
macro_shock = 0.5 if "경기 둔화" in macro_opt else (1.0 if "금융 위기" in macro_opt else 0.0)
adj_profit  = st.sidebar.slider("수익성 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
adj_debt    = st.sidebar.slider("부채·차입 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
rate_adj_bp = st.sidebar.slider("영업 협상 금리 가감 (bp)", -150, 150, 0, 10)

firm_raw = X_test_raw.iloc[[selected_idx]].copy()
actual_label = int(y_test.iloc[selected_idx])

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sb-sec">04. 재무비율 직접 조정 (What-If)</div>', unsafe_allow_html=True)
for cat_title in ['수익성', '유동성', '부채·레버리지', '현금흐름']:
    cat_items = [m for m in ACTIVE_METRICS if m[2] == cat_title]
    if not cat_items:
        continue
    with st.sidebar.expander(f"{cat_title} 지표", expanded=False):
        for col, label, _, _ in cat_items:
            orig_val = float(firm_raw[col].values[0])
            step_val = 0.01 if orig_val <= 1.0 else round(orig_val * 0.05, 4)
            new_val = st.number_input(
                label=label, min_value=0.0, value=round(orig_val, 4),
                step=max(step_val, 0.001), format="%.4f", key=f"inp_{selected_idx}_{col}"
            )
            firm_raw[col] = new_val

# ==========================================================
# 3. 모델 예측, 시나리오 민감도, Exact TreeSHAP 산출
# ==========================================================
proc_base = preprocess_row(firm_raw, 0.0, 0.0, 0.0)
pd_base   = float(cal_model.predict_proba(proc_base)[:, 1][0])
res_base  = evaluate_credit_and_pricing(pd_base, ead_krw, term_yrs, lgd_val, 0.0)

proc_curr = preprocess_row(firm_raw, macro_shock, adj_profit, adj_debt)
pd_curr   = float(cal_model.predict_proba(proc_curr)[:, 1][0])
res_curr  = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, lgd_val, rate_adj_bp)

scen_data = []
for s_lbl, s_sh in [("평상시\n(0σ)", 0.0), ("경기둔화\n(±0.5σ)", 0.5), ("금융위기\n(±1.0σ)", 1.0)]:
    p_s = float(cal_model.predict_proba(preprocess_row(firm_raw, s_sh, adj_profit, adj_debt))[:, 1][0])
    r_s = evaluate_credit_and_pricing(p_s, ead_krw, term_yrs, lgd_val, 0.0)
    scen_data.append((s_lbl, p_s * 100, r_s['fair_rate'] * 100, r_s['el_amt'], r_s['grade'].split(' ')[0]))

shap_raw = np.asarray(base_lgbm.predict(proc_curr, pred_contrib=True))[0, :-1]
shap_series = pd.Series(shap_raw, index=feature_cols)
cat_shap = {}
for col, s_val in shap_series.items():
    cat = map_feature_category(col)
    cat_shap[cat] = cat_shap.get(cat, 0.0) + float(s_val)
cat_shap_df = pd.DataFrame(
    [{'재무 영역': k, 'SHAP': v} for k, v in cat_shap.items()]
).sort_values('SHAP', ascending=True)

# ==========================================================
# 4. ROW 1: IBCS 헤더 (좌: 메타 + 핵심 결론 + KPI / 우: 시나리오 차트)
# ==========================================================
today_str = datetime.date.today().strftime("%Y-%m-%d")
top_left, top_right = st.columns([2.2, 1.0], gap="large")

with top_left:
    h_meta, h_msg = st.columns([0.95, 2.25], gap="medium")
    with h_meta:
        st.markdown(
            f'<div class="ibcs-meta-top">Credit Risk in KRW</div>'
            f'<div class="ibcs-meta-sub">{firm_name}<br>심사일 · {today_str}<br>{macro_opt.split("(")[0].strip()}</div>',
            unsafe_allow_html=True
        )
    with h_msg:
        tone_color = C_GREEN if res_curr['tone'] == 'green' else (C_RED if res_curr['tone'] == 'red' else '#D97706')
        st.markdown(
            f'<div class="ibcs-headline">'
            f'<span style="color:{tone_color}; font-weight:700;">{res_curr["grade"]}</span> — '
            f'<b>{res_curr["decision"]}</b></div>'
            f'<div class="ibcs-subline">'
            f'신청 여신 {ead_krw:,.1f}억 원 · 만기 {term_yrs}년 · 담보 LGD {lgd_val*100:.0f}% · '
            f'위험가중치(RW) {res_curr["rw"]*100:.0f}%</div>',
            unsafe_allow_html=True
        )

    delta_pd   = (res_curr['pd'] - res_base['pd']) * 100
    delta_rate = (res_curr['applied_rate'] - res_base['fair_rate']) * 100
    delta_el   = res_curr['el_amt'] - res_base['el_amt']
    raroc_pct  = res_curr['raroc'] * 100

    d_pd_html  = f'<span class="kpi-delta-red">{delta_pd:+.2f}%p</span>' if delta_pd > 0.005 else (f'<span class="kpi-delta-green">{delta_pd:+.2f}%p</span>' if delta_pd < -0.005 else '<span class="kpi-delta-gray">0.00%p</span>')
    d_rt_html  = f'<span class="kpi-delta-red">{delta_rate:+.2f}%p</span>' if delta_rate > 0.005 else (f'<span class="kpi-delta-green">{delta_rate:+.2f}%p</span>' if delta_rate < -0.005 else '<span class="kpi-delta-gray">기준</span>')
    d_el_html  = f'<span class="kpi-delta-red">{delta_el:+.2f}억</span>' if delta_el > 0.005 else (f'<span class="kpi-delta-green">{delta_el:+.2f}억</span>' if delta_el < -0.005 else '')
    d_rar_html = '<span class="kpi-delta-green">충족</span>' if raroc_pct >= 9.95 else '<span class="kpi-delta-red">미달</span>'

    st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-cell">
    <div class="kpi-num">{res_curr['pd']*100:.2f}%{d_pd_html}</div>
    <div class="kpi-lbl">연간 보정 부도확률 (PD)</div>
    <div class="kpi-sub">{term_yrs}년 누적 PD {res_curr['cum_pd']*100:.2f}%</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-num">{res_curr['applied_rate']*100:.2f}%{d_rt_html}</div>
    <div class="kpi-lbl">적용 대출금리</div>
    <div class="kpi-sub">권고 적정금리 {res_curr['fair_rate']*100:.2f}%</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-num">{res_curr['el_amt']:.2f}억{d_el_html}</div>
    <div class="kpi-lbl">연간 예상손실 (EL)</div>
    <div class="kpi-sub">EL률 {res_curr['el_rate']*100:.2f}% · RWA {res_curr['rwa_amt']:.1f}억</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-num">{raroc_pct:.1f}%{d_rar_html}</div>
    <div class="kpi-lbl">예상 세전 RAROC</div>
    <div class="kpi-sub">필요자본 {res_curr['req_cap']:.2f}억 · 목표 10.0%</div>
  </div>
</div>
""", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="ibcs-sec-title">거시 시나리오별 적정금리 (vs 평상시)</div>', unsafe_allow_html=True)
    fig_sc, ax_sc = plt.subplots(figsize=(4.4, 2.35), facecolor='#FFFFFF')
    ax_sc.set_facecolor('#FFFFFF')

    x_pos = np.arange(len(scen_data))
    base_rt_0 = scen_data[0][2]
    labels_sc = [f"{s[0]}\nPD {s[1]:.1f}%" for s in scen_data]
    rates_sc  = [s[2] for s in scen_data]

    for i, r_val in enumerate(rates_sc):
        diff = r_val - base_rt_0
        if i == 0 or abs(diff) < 1e-4:
            ax_sc.bar(i, r_val, width=0.52, color=C_CHARCOAL)
            ax_sc.text(i, r_val * 0.5, f"{r_val:.2f}%", ha='center', va='center', color='#FFFFFF', fontsize=8, fontweight='bold')
        elif diff > 0:
            ax_sc.bar(i, base_rt_0, width=0.52, color=C_CHARCOAL)
            ax_sc.bar(i, diff, bottom=base_rt_0, width=0.52, color=C_RED)
            ax_sc.text(i, base_rt_0 * 0.5, f"{r_val:.2f}%", ha='center', va='center', color='#FFFFFF', fontsize=8, fontweight='bold')
            ax_sc.text(i, r_val + max(rates_sc)*0.03, f"+{diff:.2f}%p", ha='center', va='bottom', color=C_RED, fontsize=8, fontweight='bold')
        else:
            ax_sc.bar(i, r_val, width=0.52, color=C_CHARCOAL)
            ax_sc.bar(i, abs(diff), bottom=r_val, width=0.52, color=C_GREEN, alpha=0.85)
            ax_sc.text(i, r_val * 0.5, f"{r_val:.2f}%", ha='center', va='center', color='#FFFFFF', fontsize=8, fontweight='bold')
            ax_sc.text(i, base_rt_0 + max(rates_sc)*0.03, f"{diff:.2f}%p", ha='center', va='bottom', color=C_GREEN, fontsize=8, fontweight='bold')

    ax_sc.set_xticks(x_pos)
    ax_sc.set_xticklabels(labels_sc, fontsize=7.8, color=C_CHARCOAL)
    ax_sc.set_ylim(0, max(rates_sc) * 1.24 + 0.5)
    ax_sc.set_yticks([])
    for sp in ['top', 'right', 'left']:
        ax_sc.spines[sp].set_visible(False)
    ax_sc.spines['bottom'].set_color(C_CHARCOAL)
    ax_sc.spines['bottom'].set_linewidth(1.0)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_sc)
    plt.close(fig_sc)

st.markdown('<hr class="ibcs-rule">', unsafe_allow_html=True)

# ==========================================================
# 5. ROW 2: 메인 분석 그리드 (좌: 워터폴 & SHAP / 우: IBCS 편차 표 & 민감도)
# ==========================================================
col_l, col_r = st.columns([1.0, 1.28], gap="large")

with col_l:
    st.markdown('<div class="ibcs-sec-title">대출금리 산출 구조</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibcs-sec-sub">조달원가에서 예상손실(EL)·자본비용(UL)이 누적되는 금리 분해</div>', unsafe_allow_html=True)

    wf_items = [
        ('FTP\n조달', FTP_BASE * 100, 'base'),
        ('만기\n프리미엄', res_curr['term_prem'] * 100, 'step'),
        ('업무\n원가', OPEX_RATE * 100, 'step'),
        ('예상손실\nPD×LGD', res_curr['el_rate'] * 100, 'risk'),
        ('자본비용\nUL', res_curr['cap_chg'] * 100, 'step'),
    ]
    if rate_adj_bp != 0:
        wf_items.append(('협상\n가감', rate_adj_bp / 100.0, 'adj'))
        wf_items.append(('적용\n금리', res_curr['applied_rate'] * 100, 'total'))
    else:
        wf_items.append(('권고\n적정금리', res_curr['fair_rate'] * 100, 'total'))

    fig_wf, ax_wf = plt.subplots(figsize=(5.6, 2.85), facecolor='#FFFFFF')
    ax_wf.set_facecolor('#FFFFFF')

    cum = 0.0
    max_h = max(res_curr['fair_rate'], res_curr['applied_rate']) * 100
    x_wf = np.arange(len(wf_items))

    for i, (lbl, val, kind) in enumerate(wf_items):
        if kind == 'base':
            ax_wf.bar(i, val, width=0.54, color='#9CA3AF')
            ax_wf.text(i, val + max_h*0.03, f"{val:.2f}%", ha='center', va='bottom', fontsize=7.8, color=C_CHARCOAL, fontweight='bold')
            prev_top = val
            cum = val
        elif kind == 'total':
            ax_wf.bar(i, val, width=0.58, color=C_CHARCOAL)
            ax_wf.text(i, val + max_h*0.03, f"{val:.2f}%", ha='center', va='bottom', fontsize=8.4, color=C_DARK, fontweight='bold')
            ax_wf.plot([i - 0.73, i - 0.29], [prev_top, prev_top], color='#B0B0B0', linewidth=0.8, linestyle=':')
        else:
            bottom = cum if val >= 0 else cum + val
            color = C_RED if (kind == 'risk' or (kind == 'adj' and val < 0)) else (C_GREEN if (kind == 'adj' and val > 0) else '#525252')
            ax_wf.bar(i, abs(val), bottom=bottom, width=0.54, color=color)
            ax_wf.plot([i - 0.73, i - 0.27], [cum, cum], color='#B0B0B0', linewidth=0.8, linestyle=':')
            top_y = max(cum, cum + val)
            sign_s = f"{val:+.2f}" if kind == 'adj' else f"+{val:.2f}"
            ax_wf.text(i, top_y + max_h*0.03, sign_s, ha='center', va='bottom', fontsize=7.6, color=color, fontweight='bold')
            cum = cum + val
            prev_top = cum

    ax_wf.set_xticks(x_wf)
    ax_wf.set_xticklabels([x[0] for x in wf_items], fontsize=7.5, color=C_CHARCOAL)
    ax_wf.set_ylim(0, max_h * 1.24 + 0.4)
    ax_wf.set_yticks([])
    for sp in ['top', 'right', 'left']:
        ax_wf.spines[sp].set_visible(False)
    ax_wf.spines['bottom'].set_color(C_CHARCOAL)
    ax_wf.spines['bottom'].set_linewidth(1.0)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_wf)
    plt.close(fig_wf)

    st.markdown('<hr class="ibcs-rule" style="margin:12px 0 14px 0;">', unsafe_allow_html=True)

    st.markdown('<div class="ibcs-sec-title">재무영역별 위험 기여도 (TreeSHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibcs-sec-sub">좌측 그린(-): 부도위험 방어 · 우측 레드(+): 부도위험 상승 (Log-Odds)</div>', unsafe_allow_html=True)

    fig_sh, ax_sh = plt.subplots(figsize=(5.6, 2.35), facecolor='#FFFFFF')
    ax_sh.set_facecolor('#FFFFFF')
    s_vals = cat_shap_df['SHAP'].values
    s_lbls = cat_shap_df['재무 영역'].values
    s_cols = [C_RED if v > 0 else C_GREEN for v in s_vals]

    ax_sh.barh(s_lbls, s_vals, color=s_cols, height=0.52)
    ax_sh.axvline(0, color=C_CHARCOAL, linewidth=1.1)

    max_abs = max(np.max(np.abs(s_vals)), 0.25)
    ax_sh.set_xlim(-max_abs * 1.38, max_abs * 1.38)
    for y_i, v in enumerate(s_vals):
        offset = max_abs * 0.05
        if v >= 0:
            ax_sh.text(v + offset, y_i, f"+{v:.2f}", va='center', ha='left', fontsize=8.0, color=C_RED, fontweight='bold')
        else:
            ax_sh.text(v - offset, y_i, f"{v:.2f}", va='center', ha='right', fontsize=8.0, color=C_GREEN, fontweight='bold')

    ax_sh.set_xticks([])
    ax_sh.tick_params(axis='y', labelsize=8.4, colors=C_CHARCOAL, length=0)
    for sp in ['top', 'right', 'bottom', 'left']:
        ax_sh.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.35)
    st.pyplot(fig_sh)
    plt.close(fig_sh)

with col_r:
    st.markdown('<div class="ibcs-sec-title">핵심 재무지표 벤치마크 (대상 기업 vs 정상·파산 중앙값)</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibcs-sec-sub">AC: 대상 기업 수치 · Δ vs 정상 중앙값: 우측 그린(양호) / 좌측 레드(열위)</div>', unsafe_allow_html=True)

    risk_flags = []
    rows_html = []
    for col, label, cat_name, direction in ACTIVE_METRICS:
        val   = float(firm_raw[col].values[0])
        n_med = float(normal_med[col])
        b_med = float(bankrupt_med[col])

        if direction == 'higher_better':
            is_bad = (val < (n_med + b_med) / 2.0)
            status = "위험" if val <= b_med else ("주의" if is_bad else "양호")
            fav_raw = val - n_med
        else:
            is_bad = (val > (n_med + b_med) / 2.0)
            status = "위험" if val >= b_med else ("주의" if is_bad else "양호")
            fav_raw = n_med - val

        if status in ["위험", "주의"]:
            risk_flags.append((cat_name, label, status, val, n_med, b_med, direction))

        ref_max = max(val, n_med, b_med, 1e-5)
        ac_pct = min(max((val / ref_max) * 100.0, 4.0), 100.0)

        span = max(abs(n_med - b_med) * 1.5, abs(n_med) * 0.25, 1e-4)
        norm_fav = max(min(fav_raw / span, 1.0), -1.0)
        bar_px = int(abs(norm_fav) * 46)
        bar_px = max(bar_px, 2) if abs(fav_raw) > 1e-6 else 0

        if norm_fav >= 0:
            left_bar, right_bar = "", f'<div class="var-green" style="width:{bar_px}px;"></div>'
            diff_color = C_GREEN
        else:
            left_bar, right_bar = f'<div class="var-red" style="width:{bar_px}px;"></div>', ""
            diff_color = C_RED

        diff_val = val - n_med
        dot_cls = "dot-red" if status == "위험" else ("dot-amber" if status == "주의" else "dot-green")

        rows_html.append(
            f'<tr>'
            f'<td class="td-lbl">{label}</td>'
            f'<td><div class="bar-wrap"><div class="bar-track"><div class="bar-fill" style="width:{ac_pct:.0f}%;"></div></div><span class="td-num">{val:.4f}</span></div></td>'
            f'<td><div class="var-cell"><div class="var-half-l">{left_bar}</div><div class="var-axis"></div><div class="var-half-r">{right_bar}</div></div></td>'
            f'<td class="td-num" style="color:{diff_color}; font-weight:700;">{diff_val:+.4f}</td>'
            f'<td class="td-muted">{n_med:.4f} / {b_med:.4f}</td>'
            f'<td style="white-space:nowrap; font-weight:600;"><span class="{dot_cls}"></span>{status}</td>'
            f'</tr>'
        )

    table_html = (
        '<table class="ibcs-table">'
        '<thead><tr>'
        '<th>재무지표</th>'
        '<th>AC (대상 기업)</th>'
        '<th style="text-align:center;">Δ 정상 대비 방향</th>'
        '<th style="text-align:right;">편차</th>'
        '<th style="text-align:right;">정상 / 파산 중앙값</th>'
        '<th>진단</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<hr class="ibcs-rule" style="margin:16px 0 14px 0;">', unsafe_allow_html=True)

    st.markdown('<div class="ibcs-sec-title">담보 조건별 금리·예상손실 민감도 (Collateral Sensitivity)</div>', unsafe_allow_html=True)
    base_unsec = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, 0.45)
    collat_html = []
    for c_name, c_lgd in [("무담보 · 신용대출 (LGD 45%)", 0.45), ("부분 담보 · 보증서 (LGD 35%)", 0.35), ("우량 부동산 담보 (LGD 25%)", 0.25)]:
        rc = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, c_lgd, 0.0)
        saved_el = base_unsec['el_amt'] - rc['el_amt']
        saved_rt = (rc['fair_rate'] - base_unsec['fair_rate']) * 100
        is_sel = "background-color:#F5F5F5; font-weight:700;" if abs(c_lgd - lgd_val) < 1e-4 else ""
        sav_txt = f'<span style="color:{C_GREEN}; font-weight:700;">-{saved_el:.2f}억 ({saved_rt:.2f}%p)</span>' if saved_el > 0.001 else '<span style="color:#8C8C8C;">기준</span>'
        collat_html.append(
            f'<tr style="{is_sel}">'
            f'<td>{c_name}</td>'
            f'<td class="td-num">{rc["fair_rate"]*100:.2f}%</td>'
            f'<td class="td-num">{rc["el_amt"]:.2f}억 원</td>'
            f'<td class="td-num">{sav_txt}</td>'
            f'</tr>'
        )

    st.markdown(
        '<table class="ibcs-table">'
        '<thead><tr><th>담보 조건</th><th style="text-align:right;">권고 적정금리</th>'
        '<th style="text-align:right;">연간 예상손실(EL)</th><th style="text-align:right;">무담보 대비 절감</th></tr></thead>'
        f'<tbody>{"".join(collat_html)}</tbody></table>',
        unsafe_allow_html=True
    )

st.markdown('<hr class="ibcs-rule">', unsafe_allow_html=True)

# ==========================================================
# 6. ROW 3: 근거 기반 자동 심사 의견 & 추가 징구 서류 + 다운로드
# ==========================================================
pos_shap = cat_shap_df[cat_shap_df['SHAP'] > 0.05].sort_values('SHAP', ascending=False)
top_risk_cats = list(pos_shap['재무 영역'].head(2))
flagged_cats  = sorted(list(set([f[0] for f in risk_flags])))
severe_items  = [f[1] for f in risk_flags if f[2] == "위험"]
caution_items = [f[1] for f in risk_flags if f[2] == "주의"]

memo_paras = [
    f"<b>[종합 의견]</b> {firm_name}의 연간 보정 PD는 <b>{res_curr['pd']*100:.2f}%</b>({term_yrs}년 누적 {res_curr['cum_pd']*100:.2f}%)로 "
    f"<b>{res_curr['grade']}</b> 등급 및 <b>'{res_curr['decision']}'</b> 대상입니다. 신청 금액 {ead_krw:,.1f}억 원(LGD {lgd_val*100:.0f}%) 기준 "
    f"연간 예상손실(EL)은 <b>{res_curr['el_amt']:.2f}억 원</b>, 목표 RAROC(10%) 달성 적정금리는 <b>연 {res_curr['fair_rate']*100:.2f}%</b>입니다."
]
if severe_items or caution_items:
    d_txt = ""
    if severe_items:
        d_txt += f"파산 기업 중앙값 수준에 도달한 핵심 취약 지표는 <b>{', '.join(severe_items)}</b>이며, "
    if caution_items:
        d_txt += f"정상 기업 대비 열위 구간인 주의 지표는 <b>{', '.join(caution_items)}</b>입니다. "
    if top_risk_cats:
        d_txt += f"TreeSHAP 기준 부도확률 상승을 주도한 영역은 <b>{', '.join(top_risk_cats)}</b>으로 분석됩니다."
    memo_paras.append(f"<b>[정량 위험 진단]</b> {d_txt}")
else:
    memo_paras.append("<b>[정량 위험 진단]</b> 검증된 핵심 재무지표 전반이 정상 기업 중앙값 대비 양호한 구간에 위치하여 단기 재무 부실 징후가 제한적입니다.")

if rate_adj_bp < 0 and res_curr['raroc'] < TARGET_ROE:
    memo_paras.append(
        f"<b>[수익성 경고]</b> 금리 할인({rate_adj_bp:+d}bp) 반영 시 예상 RAROC가 <b>{res_curr['raroc']*100:.1f}%</b>로 목표치(10.0%)를 하회합니다. "
        "금리 조정, 담보 보강(LGD 하향) 또는 부수거래 연계가 필요합니다."
    )

DOC_MAP = {
    '부채·레버리지': (
        "금융거래확인서, 전 금융권 차입금 만기상환 스케줄표, 우발채무 명세서",
        "단기 차입금 차환(Roll-over) 리스크 및 이자보상배율 적정성 점검"
    ),
    '유동성': (
        "향후 6개월 자금수지계획표, 매출채권 연령분석표(Aging), 매입채무 결제조건 내역",
        "단기 유동부채 만기 도래 시 현금화 가능 자산의 실질 회수 가능성 확인"
    ),
    '수익성': (
        "최근 분기 가결산 재무제표, 주요 거래처별 매출처원장, 수주잔고 현황표",
        "영업이익률 하락이 일시적 원가 상승인지 구조적 경쟁력 약화인지 구분"
    ),
    '현금흐름': (
        "최근 6개월 주거래 통장 입출금 내역, 영업현금흐름(OCF) 대비 순이익 괴리 소명서",
        "흑자 도산 가능성 점검(재고자산 급증·매출채권 장기 미회수 여부)"
    )
}

bot_l, bot_r = st.columns([1.15, 1.0], gap="large")

with bot_l:
    st.markdown('<div class="ibcs-sec-title">근거 기반 자동 심사 의견 (AI Underwriting Rationale)</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibcs-sec-sub">입력된 재무수치·SHAP·여신 가격결정 결과에 100% 근거하여 자동 생성</div>', unsafe_allow_html=True)
    p_html = "".join([f"<p>{m}</p>" for m in memo_paras])
    st.markdown(f'<div class="memo-clean">{p_html}</div>', unsafe_allow_html=True)

with bot_r:
    st.markdown('<div class="ibcs-sec-title">추가 제출자료 및 심사역 검토 체크리스트</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibcs-sec-sub">위험 신호가 감지된 재무 영역과 1:1 매핑된 필수 징구 서류</div>', unsafe_allow_html=True)

    req_docs_text = []
    if flagged_cats:
        doc_blocks = []
        for fc in flagged_cats:
            docs, chk = DOC_MAP[fc]
            doc_blocks.append(
                f'<div class="doc-row"><div class="doc-cat">{fc} 위험 대응</div>'
                f'<div><b>징구 서류:</b> {docs}</div>'
                f'<div><b>검토 사항:</b> {chk}</div></div>'
            )
            req_docs_text.append(f"[{fc}] 서류: {docs} / 검토: {chk}")
        st.markdown("".join(doc_blocks), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="doc-row"><div class="doc-cat doc-cat-ok">표준 심사 절차</div>'
            '<div><b>기본 서류:</b> 최근 3개년 감사보고서(표준재무제표증명), 국세·지방세 완납증명서</div>'
            '<div><b>검토 사항:</b> 비재무 리스크(대표자 평판, 업황 변동성, 담보 감정가 적정성) 최종 확인</div></div>',
            unsafe_allow_html=True
        )
        req_docs_text.append("[표준] 최근 3개년 재무제표, 납세증명서 및 비재무 리스크 확인")

    plain_memo = "\n".join([
        "==================================================",
        f"  AI 기업여신 심사 검토 의견서 ({today_str})",
        "==================================================",
        f"1. 기업명: {firm_name}",
        f"2. 신청 조건: 대출금액 {ead_krw:,.1f}억 원 / 만기 {term_yrs}년 / LGD {lgd_val*100:.0f}% (RW {res_curr['rw']*100:.0f}%)",
        f"3. 시나리오: {macro_opt} (수익성 {adj_profit:+.1f}σ, 부채 {adj_debt:+.1f}σ, 금리조정 {rate_adj_bp:+d}bp)",
        "--------------------------------------------------",
        f"- 연간 보정 PD: {res_curr['pd']*100:.2f}% ({term_yrs}년 누적 PD: {res_curr['cum_pd']*100:.2f}%)",
        f"- 신용등급 / 의견: {res_curr['grade']} / {res_curr['decision']}",
        f"- 권고 적정금리: {res_curr['fair_rate']*100:.2f}% (적용금리: {res_curr['applied_rate']*100:.2f}%)",
        f"- 연간 EL: {res_curr['el_amt']:.2f}억 원 / 필요 자기자본: {res_curr['req_cap']:.2f}억 원 / RAROC: {res_curr['raroc']*100:.1f}%",
        "--------------------------------------------------",
        "[추가 징구 서류 및 검토 사항]",
        *req_docs_text,
        "=================================================="
    ])

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="심사 검토 의견서 다운로드 (.txt)",
        data=plain_memo.encode('utf-8-sig'),
        file_name=f"여신심사의견서_{firm_name.replace(' ', '_')}_{today_str}.txt",
        mime="text/plain",
        use_container_width=True
    )

# 최하단 절제된 검증 각주
actual_str = f'<span style="color:{C_RED}; font-weight:700;">실제 파산 (1)</span>' if actual_label == 1 else f'<span style="color:{C_GREEN}; font-weight:700;">실제 정상 (0)</span>'
st.markdown(
    f'<div class="footnote">'
    f'<span>검증 참고 (과거 실제 결과): {actual_str}</span>'
    f'<span>Model: LightGBM + {model_bundle["best_cal_name"]} Calibration · 15-Fold Leak-Free CV ROC-AUC 0.9362 / PR-AUC 0.4536</span>'
    f'</div>',
    unsafe_allow_html=True
)
