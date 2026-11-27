import os
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# ==========================================================
# 0. 페이지 설정 & 폰트(Windows/Linux 공용) + 금융 SaaS 스타일링
# ==========================================================
st.set_page_config(
    page_title="Credit Risk Cockpit | AI 기업여신 심사",
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

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif !important;
        font-variant-numeric: tabular-nums;
    }

    /* 상단 기본 여백 정리 */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1440px;
    }

    /* 사이드바 경계선 및 정돈 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem !important;
    }
    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }
    .brand-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 3px;
        background: #1E3A8A;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 14px;
    }
    .section-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* 메인 리스크 패스포트 헤더 카드 */
    .passport-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .firm-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 2px 0 4px 0;
    }
    .firm-meta {
        font-size: 0.86rem;
        color: #475569;
    }
    .firm-meta b { color: #0F172A; }

    /* 등급 배지 */
    .badge-safe   { background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; padding:5px 12px; border-radius:999px; font-weight:700; font-size:0.82rem; display:inline-block; }
    .badge-warn   { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; padding:5px 12px; border-radius:999px; font-weight:700; font-size:0.82rem; display:inline-block; }
    .badge-danger { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; padding:5px 12px; border-radius:999px; font-weight:700; font-size:0.82rem; display:inline-block; }
    .decision-text {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 7px;
    }

    /* 커스텀 KPI 카드 (어색한 화살표 제거) */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 17px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.48rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 500;
    }

    /* 탭 컨테이너 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        padding: 0 16px;
        font-size: 0.86rem;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* 콘텐츠 화이트 패널 */
    .content-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .panel-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .panel-caption {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 12px;
    }

    /* 자동 심사 의견서 박스 */
    .memo-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1E3A8A;
        padding: 18px 20px;
        border-radius: 10px;
        color: #1E293B;
        line-height: 1.72;
        font-size: 0.93rem;
        margin-bottom: 16px;
    }
    .doc-item {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 13px 16px;
        margin-bottom: 10px;
    }
    .doc-tag {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        color: #1E3A8A;
        background: #EFF6FF;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .doc-line {
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.55;
        margin: 2px 0;
    }

    /* 우측 다크 PROJECTIONS 카드 */
    .projection-box {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        border-radius: 14px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 6px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.14);
    }
    .proj-label {
        font-size: 0.70rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .proj-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.22);
    }
    .proj-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.83rem;
        color: #CBD5E1;
        margin-bottom: 7px;
    }
    .proj-row b { color: #FFFFFF; font-weight: 700; }
    .proj-footer {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(148, 163, 184, 0.22);
        font-size: 0.76rem;
        color: #94A3B8;
        line-height: 1.55;
    }
    .pill-ok   { color: #34D399; font-weight: 700; }
    .pill-warn { color: #FBBF24; font-weight: 700; }
    .pill-bad  { color: #F87171; font-weight: 700; }

    /* 다운로드 버튼 (다크 CTA 스타일) */
    .stDownloadButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }

    /* 우측 Bordered 컨테이너 배경 순백화 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 14px !important;
        border-color: #E2E8F0 !important;
        padding: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. 자산 로드 (모델, 전처리기, 정상/파산 중앙값 벤치마크)
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

VERIFIED_METRICS = [
    ('Persistent EPS in the Last Four Seasons', '최근 4분기 지속 EPS',     '수익성',       'higher_better'),
    ('Operating Profit Rate',                   '영업이익률',              '수익성',       'higher_better'),
    ('Quick Ratio',                             '당좌비율 (단기유동성)',     '유동성',       'higher_better'),
    ('Current Liability to Assets',             '총자산 대비 유동부채',     '유동성',       'higher_worse'),
    ('Borrowing dependency',                    '차입금 의존도',           '부채·레버리지', 'higher_worse'),
    ('Debt ratio %',                            '부채비율',               '부채·레버리지', 'higher_worse'),
    ('Equity to Liability',                     '자기자본 대비 부채 여력',   '부채·레버리지', 'higher_better'),
    ('Cash flow rate',                          '영업현금흐름 비율',        '현금흐름',     'higher_better'),
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
        grade, decision, badge_cls, rw = 'AAA~A (우량)', '승인 · 우대 금리 적용', 'badge-safe', 0.50
    elif pd_val < 0.03:
        grade, decision, badge_cls, rw = 'BBB (양호)', '정상 승인', 'badge-safe', 0.75
    elif pd_val < 0.07:
        grade, decision, badge_cls, rw = 'BB (주의)', '조건부 승인 · 담보 보강 / 한도 축소', 'badge-warn', 1.00
    elif pd_val < 0.15:
        grade, decision, badge_cls, rw = 'B (위험)', '정밀 심사 요망 · 고금리 적용', 'badge-warn', 1.50
    else:
        grade, decision, badge_cls, rw = 'CCC~D (고위험)', '여신 거절 (Reject)', 'badge-danger', 1.50

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
        'badge_cls': badge_cls, 'rw': rw, 'term_prem': term_prem,
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

def render_kpi_card(label, value, sub, val_color="#0F172A"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{val_color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.tick_params(colors='#334155', labelsize=9)
    ax.xaxis.label.set_color('#475569')
    ax.xaxis.label.set_fontsize(9)
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='#94A3B8')
    ax.set_axisbelow(True)

# ==========================================================
# 2. 좌측 사이드바
# ==========================================================
st.sidebar.markdown("""
<div class="sidebar-brand"><span class="brand-dot"></span>Risk Cockpit</div>
<div class="sidebar-sub">AI 기업여신 심사 · 바젤 III 가격결정</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-eyebrow">01. 심사 대상 선택</div>', unsafe_allow_html=True)
sample_choice = st.sidebar.radio(
    "심사 대상 기업 선택",
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
st.sidebar.markdown('<div class="section-eyebrow">02. 기업 및 여신 조건</div>', unsafe_allow_html=True)
firm_name = st.sidebar.text_input("기업명 (보고서 표기용)", value=f"심사대상 기업 #{selected_idx}")
ead_krw   = st.sidebar.number_input("대출 신청액 EAD (억 원)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
term_yrs  = st.sidebar.slider("대출 기간 (만기, 년)", min_value=1, max_value=5, value=3)

collateral_map = {
    "무담보 · 신용대출 (LGD 45%)": 0.45,
    "부분 담보 · 보증서 (LGD 35%)": 0.35,
    "우량 부동산 담보 (LGD 25%)":   0.25
}
collat_choice = st.sidebar.selectbox("담보 조건 (LGD 적용)", list(collateral_map.keys()))
lgd_val = collateral_map[collat_choice]

firm_raw = X_test_raw.iloc[[selected_idx]].copy()
actual_label = int(y_test.iloc[selected_idx])

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-eyebrow">03. 핵심 재무비율 조정 (What-If)</div>', unsafe_allow_html=True)
st.sidebar.caption("방향성이 검증된 핵심 지표만 반영됩니다.")

for cat_title in ['수익성', '유동성', '부채·레버리지', '현금흐름']:
    cat_items = [m for m in ACTIVE_METRICS if m[2] == cat_title]
    if not cat_items:
        continue
    with st.sidebar.expander(f"{cat_title} 지표", expanded=False):
        for col, label, _, _ in cat_items:
            orig_val = float(firm_raw[col].values[0])
            step_val = 0.01 if orig_val <= 1.0 else round(orig_val * 0.05, 4)
            new_val = st.number_input(
                label=label,
                min_value=0.0,
                value=round(orig_val, 4),
                step=max(step_val, 0.001),
                format="%.4f",
                key=f"inp_{selected_idx}_{col}"
            )
            firm_raw[col] = new_val

# ==========================================================
# 3. 메인 2단 레이아웃: 중앙(Risk Passport) + 우측(Stress Tester)
# ==========================================================
col_center, col_right = st.columns([2.35, 1.0], gap="large")

# --- 우측 컨트롤 패널 (border=True 카드로 명확히 분리) ---
with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">STRESS TESTER</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">거시 충격 · 금리 시뮬레이터</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-caption">시나리오 조정 시 좌측 리포트가 즉시 재산출됩니다.</div>', unsafe_allow_html=True)

        macro_opt = st.selectbox(
            "거시경제 시나리오",
            ["평상시 (Baseline)", "경기 둔화 (Mild Shock: ±0.5σ)", "금융 위기 (Severe Shock: ±1.0σ)"]
        )
        macro_shock = 0.5 if "경기 둔화" in macro_opt else (1.0 if "금융 위기" in macro_opt else 0.0)

        adj_profit  = st.slider("수익성 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
        adj_debt    = st.slider("부채·차입 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
        rate_adj_bp = st.slider(
            "영업 협상 금리 가감 (bp)", -150, 150, 0, 10,
            help="-50bp = 권고 적정금리 대비 0.50%p 우대 할인"
        )

        # Baseline vs Stressed 계산
        proc_base = preprocess_row(firm_raw, 0.0, 0.0, 0.0)
        pd_base   = float(cal_model.predict_proba(proc_base)[:, 1][0])
        res_base  = evaluate_credit_and_pricing(pd_base, ead_krw, term_yrs, lgd_val, 0.0)

        proc_curr = preprocess_row(firm_raw, macro_shock, adj_profit, adj_debt)
        pd_curr   = float(cal_model.predict_proba(proc_curr)[:, 1][0])
        res_curr  = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, lgd_val, rate_adj_bp)

        # Exact TreeSHAP 계산
        shap_raw = np.asarray(base_lgbm.predict(proc_curr, pred_contrib=True))[0, :-1]
        shap_series = pd.Series(shap_raw, index=feature_cols)

        cat_shap = {}
        for col, s_val in shap_series.items():
            cat = map_feature_category(col)
            cat_shap[cat] = cat_shap.get(cat, 0.0) + float(s_val)
        cat_shap_df = pd.DataFrame(
            [{'재무 영역': k, 'SHAP 기여도': v} for k, v in cat_shap.items()]
        ).sort_values('SHAP 기여도', ascending=True)

        delta_pd   = (res_curr['pd'] - res_base['pd']) * 100
        delta_rate = (res_curr['applied_rate'] - res_base['fair_rate']) * 100
        delta_el   = res_curr['el_amt'] - res_base['el_amt']
        raroc_pct  = res_curr['raroc'] * 100

        raroc_html = (
            '<span class="pill-ok">Hurdle 충족</span>'
            if raroc_pct >= 9.95 else
            '<span class="pill-warn">Hurdle 미달</span>'
        )
        actual_html = (
            '<span class="pill-bad">실제 파산 (1)</span>'
            if actual_label == 1 else
            '<span class="pill-ok">실제 정상 (0)</span>'
        )

        st.markdown(f"""
        <div class="projection-box">
            <div class="proj-label">LIVE PROJECTIONS · STRESS APPLIED</div>
            <div class="proj-val">{res_curr['grade'].split(' ')[0]} &nbsp;|&nbsp; PD {res_curr['pd']*100:.2f}%</div>
            <div class="proj-row"><span>평상시 대비 PD 변화</span><b>{delta_pd:+.2f}%p</b></div>
            <div class="proj-row"><span>적용 대출금리</span><b>{res_curr['applied_rate']*100:.2f}% ({delta_rate:+.2f}%p)</b></div>
            <div class="proj-row"><span>연간 예상손실 (EL)</span><b>{res_curr['el_amt']:.2f}억 원 ({delta_el:+.2f}억)</b></div>
            <div class="proj-row"><span>예상 RAROC (목표 10%)</span><b>{raroc_pct:.1f}% · {raroc_html}</b></div>
            <div class="proj-footer">
                검증 참고(과거 실제): {actual_html}<br>
                LightGBM + {model_bundle['best_cal_name']} · Leak-Free CV AUC 0.9362
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# 4. 중앙 Risk Passport
# ==========================================================
with col_center:
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    st.markdown(f"""
    <div class="passport-card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:16px;">
            <div>
                <div class="section-eyebrow" style="margin-bottom:3px;">CREDIT RISK PASSPORT · {today_str}</div>
                <div class="firm-title">{firm_name}</div>
                <div class="firm-meta">
                    신청 여신 <b>{ead_krw:,.1f}억 원</b> &nbsp;·&nbsp; 만기 <b>{term_yrs}년</b> &nbsp;·&nbsp;
                    담보 LGD <b>{lgd_val*100:.0f}%</b> (RW {res_curr['rw']*100:.0f}%)
                </div>
            </div>
            <div style="text-align:right; flex-shrink:0;">
                <span class="{res_curr['badge_cls']}">{res_curr['grade']}</span>
                <div class="decision-text">{res_curr['decision']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 커스텀 4-KPI 카드 (화살표 제거 + 깔끔한 박스)
    pd_color = "#B91C1C" if res_curr['pd'] >= 0.15 else ("#B45309" if res_curr['pd'] >= 0.03 else "#0F172A")
    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown(render_kpi_card("연간 보정 PD", f"{res_curr['pd']*100:.2f}%", f"{term_yrs}년 누적 PD {res_curr['cum_pd']*100:.2f}%", pd_color), unsafe_allow_html=True)
    k2.markdown(render_kpi_card("권고 적정금리", f"{res_curr['fair_rate']*100:.2f}%", f"협상 적용금리 {res_curr['applied_rate']*100:.2f}%"), unsafe_allow_html=True)
    k3.markdown(render_kpi_card("연간 예상손실 (EL)", f"{res_curr['el_amt']:.2f}억 원", f"예상손실률 {res_curr['el_rate']*100:.2f}%"), unsafe_allow_html=True)
    k4.markdown(render_kpi_card("필요 자기자본 (K)", f"{res_curr['req_cap']:.2f}억 원", f"위험가중자산(RWA) {res_curr['rwa_amt']:.1f}억"), unsafe_allow_html=True)

    # 3개 분석 탭 (이모지 제거, 필 스타일)
    tab1, tab2, tab3 = st.tabs([
        "01. 위험요인(SHAP) & 재무비교",
        "02. 바젤 III 금리구조 & 민감도",
        "03. 자동 심사의견서 & 징구서류"
    ])

    # 위험 플래그 미리 산출 (Tab 1 & Tab 3 공용)
    comp_rows, risk_flags = [], []
    for col, label, cat_name, direction in ACTIVE_METRICS:
        val = float(firm_raw[col].values[0])
        n_med = float(normal_med[col])
        b_med = float(bankrupt_med[col])

        if direction == 'higher_better':
            is_bad = (val < (n_med + b_med) / 2.0)
            status = "위험" if val <= b_med else ("주의" if is_bad else "양호")
        else:
            is_bad = (val > (n_med + b_med) / 2.0)
            status = "위험" if val >= b_med else ("주의" if is_bad else "양호")

        if status in ["위험", "주의"]:
            risk_flags.append((cat_name, label, status, val, n_med, b_med, direction))

        comp_rows.append({
            '재무지표': label,
            '대상 기업': round(val, 4),
            '정상 중앙값': round(n_med, 4),
            '파산 중앙값': round(b_med, 4),
            '진단': status
        })
    comp_df = pd.DataFrame(comp_rows)

    def style_status(val):
        if val == "위험":
            return "background-color: #FEF2F2; color: #B91C1C; font-weight: 700;"
        elif val == "주의":
            return "background-color: #FFFBEB; color: #B45309; font-weight: 700;"
        return "background-color: #ECFDF5; color: #047857; font-weight: 700;"

    # ------------------------------------------------------
    # [TAB 1] SHAP & 정상/파산 중앙값 비교
    # ------------------------------------------------------
    with tab1:
        with st.container(border=True):
            t1_left, t1_right = st.columns([1.0, 1.18], gap="medium")

            with t1_left:
                st.markdown('<div class="panel-title">재무영역별 위험 기여도 (TreeSHAP)</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-caption">우측(+): 부도위험 상승 요인 · 좌측(-): 부도위험 방어 요인</div>', unsafe_allow_html=True)
                fig_s, ax_s = plt.subplots(figsize=(5.2, 3.6), facecolor='#FFFFFF')
                ax_s.set_facecolor('#FFFFFF')
                s_colors = ['#DC2626' if v > 0 else '#1E3A8A' for v in cat_shap_df['SHAP 기여도']]
                ax_s.barh(cat_shap_df['재무 영역'], cat_shap_df['SHAP 기여도'], color=s_colors, height=0.52)
                ax_s.axvline(0, color='#334155', linewidth=1.0)
                ax_s.set_xlabel('SHAP Log-Odds 기여도 합계')
                style_axes(ax_s)
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)

            with t1_right:
                st.markdown('<div class="panel-title">정상 vs 파산 기업 중앙값 비교</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-caption">극단치 왜곡 차단을 위해 전체 표본 중앙값(Median) 기준 비교</div>', unsafe_allow_html=True)
                styled_comp = (
                    comp_df.style
                    .map(style_status, subset=['진단'])
                    .format({'대상 기업': '{:.4f}', '정상 중앙값': '{:.4f}', '파산 중앙값': '{:.4f}'})
                )
                st.dataframe(styled_comp, use_container_width=True, hide_index=True, height=322)

    # ------------------------------------------------------
    # [TAB 2] 바젤 III 금리구조 & 민감도
    # ------------------------------------------------------
    with tab2:
        with st.container(border=True):
            p_left, p_right = st.columns([1.05, 1.05], gap="medium")

            with p_left:
                st.markdown('<div class="panel-title">바젤 III 권고 적정금리 구성</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-caption">FTP + 업무원가 + 예상손실률(EL) + 규제자본비용(UL)</div>', unsafe_allow_html=True)
                fig_p, ax_p = plt.subplots(figsize=(5.2, 3.5), facecolor='#FFFFFF')
                ax_p.set_facecolor('#FFFFFF')
                p_labels = [
                    '자금조달원가 (FTP)',
                    f'만기 프리미엄 ({term_yrs}년)',
                    '업무원가 (OpEx)',
                    '예상손실률 (PD×LGD)',
                    f'자본비용 (RW {res_curr["rw"]*100:.0f}%)'
                ]
                p_vals = [
                    FTP_BASE * 100,
                    res_curr['term_prem'] * 100,
                    OPEX_RATE * 100,
                    res_curr['el_rate'] * 100,
                    res_curr['cap_chg'] * 100
                ]
                p_colors = ['#64748B', '#94A3B8', '#0284C7', '#DC2626', '#1E3A8A']
                bars = ax_p.barh(p_labels, p_vals, color=p_colors, height=0.52)
                ax_p.set_xlabel('금리 비중 (%)')
                ax_p.set_xlim(0, max(p_vals) * 1.38 + 0.5)
                for bar, v in zip(bars, p_vals):
                    ax_p.text(v + 0.08, bar.get_y() + bar.get_height()/2, f'{v:.2f}%',
                              va='center', fontsize=9, fontweight='bold', color='#0F172A')
                style_axes(ax_p)
                plt.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

            with p_right:
                st.markdown('<div class="panel-title">담보 조건별 금리·손실 민감도</div>', unsafe_allow_html=True)
                base_unsec_el = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, 0.45)['el_amt']
                collat_rows = []
                for c_name, c_lgd in [("무담보 (45%)", 0.45), ("부분담보 (35%)", 0.35), ("우량담보 (25%)", 0.25)]:
                    r_c = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, c_lgd, 0.0)
                    collat_rows.append({
                        '담보 조건': c_name,
                        '적정금리': f"{r_c['fair_rate']*100:.2f}%",
                        '연간 EL': f"{r_c['el_amt']:.2f}억",
                        '절감 EL': f"{(base_unsec_el - r_c['el_amt']):+.2f}억"
                    })
                st.dataframe(pd.DataFrame(collat_rows), use_container_width=True, hide_index=True)

                st.markdown('<div class="panel-title" style="margin-top:12px;">거시경제 충격별 민감도</div>', unsafe_allow_html=True)
                macro_rows = []
                for m_name, m_sh in [("평상시 (0σ)", 0.0), ("경기 둔화 (±0.5σ)", 0.5), ("금융 위기 (±1.0σ)", 1.0)]:
                    p_m = float(cal_model.predict_proba(preprocess_row(firm_raw, m_sh, adj_profit, adj_debt))[:, 1][0])
                    r_m = evaluate_credit_and_pricing(p_m, ead_krw, term_yrs, lgd_val, 0.0)
                    macro_rows.append({
                        '시나리오': m_name,
                        '보정 PD': f"{p_m*100:.2f}%",
                        '등급': r_m['grade'].split(' ')[0],
                        '적정금리': f"{r_m['fair_rate']*100:.2f}%",
                        '연간 EL': f"{r_m['el_amt']:.2f}억"
                    })
                st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # [TAB 3] Rule-Based 자동 심사 의견서 & 징구 서류
    # ------------------------------------------------------
    with tab3:
        with st.container(border=True):
            pos_shap = cat_shap_df[cat_shap_df['SHAP 기여도'] > 0.05].sort_values('SHAP 기여도', ascending=False)
            top_risk_cats = list(pos_shap['재무 영역'].head(2))

            flagged_cats  = sorted(list(set([f[0] for f in risk_flags])))
            severe_items  = [f[1] for f in risk_flags if f[2] == "위험"]
            caution_items = [f[1] for f in risk_flags if f[2] == "주의"]

            memo_lines = []
            memo_lines.append(
                f"<b>[종합 심사 의견]</b> {firm_name}의 연간 보정 부도확률(PD)은 <b>{res_curr['pd']*100:.2f}%</b>"
                f"({term_yrs}년 누적 {res_curr['cum_pd']*100:.2f}%)로 산출되어 신용등급 <b>{res_curr['grade']}</b> 및 "
                f"<b>'{res_curr['decision']}'</b> 대상에 해당합니다. 신청 여신 {ead_krw:,.1f}억 원(LGD {lgd_val*100:.0f}%) 기준 "
                f"연간 예상손실(EL)은 <b>{res_curr['el_amt']:.2f}억 원</b>, 바젤 III 자본비용을 반영한 목표 RAROC(10%) 달성 적정금리는 "
                f"<b>연 {res_curr['fair_rate']*100:.2f}%</b>입니다."
            )

            if severe_items or caution_items:
                detail_txt = ""
                if severe_items:
                    detail_txt += f"파산 기업 중앙값 수준에 도달한 핵심 취약 지표는 <b>{', '.join(severe_items)}</b>이며, "
                if caution_items:
                    detail_txt += f"정상 기업 대비 열위 구간인 주의 지표는 <b>{', '.join(caution_items)}</b>으로 확인되었습니다. "
                if top_risk_cats:
                    detail_txt += f"모델(TreeSHAP) 기준 부도확률 상승을 주도한 재무 영역은 <b>{', '.join(top_risk_cats)}</b>입니다."
                memo_lines.append(f"<b>[정량 위험 진단]</b> {detail_txt}")
            else:
                memo_lines.append(
                    "<b>[정량 위험 진단]</b> 검증된 핵심 재무지표 전반이 정상 기업 중앙값 대비 양호한 구간에 위치하고 있어 "
                    "단기적 재무 부실 징후는 제한적입니다."
                )

            if rate_adj_bp < 0 and res_curr['raroc'] < TARGET_ROE:
                memo_lines.append(
                    f"<b>[수익성 경고]</b> 협상 금리 조정({rate_adj_bp:+d}bp) 반영 시 예상 RAROC가 <b>{res_curr['raroc']*100:.1f}%</b>로 "
                    "은행 목표 기준(10.0%)을 하회합니다. 금리 인상, 담보 보강(LGD 하향), 또는 부수거래(외환·수신) 연계가 필요합니다."
                )

            st.markdown('<div class="section-eyebrow">AI UNDERWRITING RATIONALE</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">근거 기반 자동 심사 의견</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='memo-box'>{'<br><br>'.join(memo_lines)}</div>", unsafe_allow_html=True)

            DOC_MAP = {
                '부채·레버리지': (
                    "금융거래확인서, 전 금융권 차입금 만기상환 스케줄표, 우발채무(지급보증) 명세서",
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

            st.markdown('<div class="panel-title" style="margin-bottom:8px;">추가 제출자료 및 심사 담당자 검토 체크리스트</div>', unsafe_allow_html=True)
            req_docs_text = []
            if flagged_cats:
                for fc in flagged_cats:
                    docs, check_pt = DOC_MAP[fc]
                    st.markdown(f"""
                    <div class="doc-item">
                        <span class="doc-tag">{fc} 위험 대응</span>
                        <div class="doc-line"><b>추가 징구 서류:</b> {docs}</div>
                        <div class="doc-line"><b>심사역 검토 사항:</b> {check_pt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    req_docs_text.append(f"[{fc}] 서류: {docs} / 검토: {check_pt}")
            else:
                st.markdown("""
                <div class="doc-item">
                    <span class="doc-tag">표준 심사 절차</span>
                    <div class="doc-line"><b>기본 징구 서류:</b> 최근 3개년 감사보고서(또는 표준재무제표증명), 국세·지방세 완납증명서</div>
                    <div class="doc-line"><b>심사역 검토 사항:</b> 비재무 리스크(대표자 평판, 업황 변동성, 담보 감정가 적정성) 최종 확인</div>
                </div>
                """, unsafe_allow_html=True)
                req_docs_text.append("[표준] 최근 3개년 재무제표, 납세증명서 및 비재무 리스크 확인")

            plain_memo = "\n".join([
                "==================================================",
                f"  AI 기업여신 심사 검토 의견서 ({today_str})",
                "==================================================",
                f"1. 기업명: {firm_name}",
                f"2. 신청 조건: 대출금액 {ead_krw:,.1f}억 원 / 만기 {term_yrs}년 / LGD {lgd_val*100:.0f}% (RW {res_curr['rw']*100:.0f}%)",
                f"3. 시나리오: {macro_opt} (수익성 {adj_profit:+.1f}σ, 부채 {adj_debt:+.1f}σ, 금리조정 {rate_adj_bp:+d}bp)",
                "--------------------------------------------------",
                "[신용분석 및 바젤 III 가격결정 결과]",
                f"- 연간 보정 PD: {res_curr['pd']*100:.2f}% ({term_yrs}년 누적 PD: {res_curr['cum_pd']*100:.2f}%)",
                f"- 산출 신용등급: {res_curr['grade']} -> {res_curr['decision']}",
                f"- 권고 적정금리: {res_curr['fair_rate']*100:.2f}% (적용금리: {res_curr['applied_rate']*100:.2f}%)",
                f"- 연간 예상손실(EL): {res_curr['el_amt']:.2f}억 원 / 필요 자기자본(K): {res_curr['req_cap']:.2f}억 원",
                f"- 예상 세전 RAROC: {res_curr['raroc']*100:.1f}% (목표 Hurdle Rate: 10.0%)",
                "--------------------------------------------------",
                "[추가 징구 서류 및 검토 사항]",
                *req_docs_text,
                "=================================================="
            ])

            st.download_button(
                label="심사 검토 의견서 다운로드 (.txt)",
                data=plain_memo.encode('utf-8-sig'),
                file_name=f"여신심사의견서_{firm_name.replace(' ', '_')}_{today_str}.txt",
                mime="text/plain",
                use_container_width=True
            )
