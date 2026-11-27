import os
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================================
# 0. 페이지 기본 설정 & 깔끔한 금융 대시보드 스타일링
# ==========================================================
st.set_page_config(
    page_title="AI Corporate Credit Risk Cockpit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

import matplotlib.font_manager as fm
if os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
    fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams['font.family'] = fm.FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf').get_name()
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .passport-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .projection-box {
        background-color: #0F172A;
        color: #F8FAFC;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .proj-label { font-size: 0.78rem; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 2px; }
    .proj-val   { font-size: 1.55rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px; }
    .proj-sub   { font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px; }
    .badge-safe   { background:#DCFCE7; color:#166534; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.85rem; }
    .badge-warn   { background:#FEF3C7; color:#92400E; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.85rem; }
    .badge-danger { background:#FEE2E2; color:#991B1B; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.85rem; }
    .memo-box {
        background-color: #F8FAFC;
        border-left: 4px solid #1E3A5F;
        padding: 16px 18px;
        border-radius: 6px;
        color: #1F2937;
        line-height: 1.65;
        font-size: 0.96rem;
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

    # 정상 vs 파산 중앙값 벤치마크 계산 (data.csv 우선, 없으면 X_test_raw 활용)
    ref_raw = None
    ref_y = None
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

# 대표 샘플 기업 추출 (우량 / 주의 / 고위험)
all_probs = cal_model.predict_proba(X_test_proc)[:, 1]
idx_safe = int(np.argmin(all_probs))
idx_mid  = int(np.argmin(np.abs(all_probs - 0.048)))
idx_risk = int(np.argmax(all_probs))

# 방향성이 검증된 핵심 재무지표 정의 (1번 감사에서 오염/역전 변수 제외)
# direction: 'higher_better' (높을수록 안전), 'higher_worse' (높을수록 위험)
VERIFIED_METRICS = [
    ('Persistent EPS in the Last Four Seasons', '최근 4분기 지속 EPS',     '수익성',   'higher_better'),
    ('Operating Profit Rate',                   '영업이익률',              '수익성',   'higher_better'),
    ('Quick Ratio',                             '당좌비율 (단기유동성)',     '유동성',   'higher_better'),
    ('Current Liability to Assets',             '총자산 대비 유동부채',     '유동성',   'higher_worse'),
    ('Borrowing dependency',                    '차입금 의존도',           '부채·레버리지', 'higher_worse'),
    ('Debt ratio %',                            '부채비율',               '부채·레버리지', 'higher_worse'),
    ('Equity to Liability',                     '자기자본 대비 부채 여력',   '부채·레버리지', 'higher_better'),
    ('Cash flow rate',                          '영업현금흐름 비율',        '현금흐름', 'higher_better'),
]
ACTIVE_METRICS = [m for m in VERIFIED_METRICS if m[0] in X_test_raw.columns]

# 4대 재무 카테고리 매핑 (TreeSHAP 합산용)
def map_feature_category(col_name):
    c = col_name.lower()
    if any(k in c for k in ['roa', 'eps', 'profit', 'income', 'margin', 'tax', 'earning']):
        return '수익성 (Profitability)'
    elif any(k in c for k in ['debt', 'borrowing', 'liability to equity', 'equity to liability', 'net worth', 'interest']):
        return '부채·레버리지 (Leverage)'
    elif any(k in c for k in ['quick', 'current', 'working capital', 'liquidity']):
        return '단기 유동성 (Liquidity)'
    elif any(k in c for k in ['cash', 'turnover', 'inventory', 'receivable', 'asset']):
        return '현금흐름·활동성 (Cash & Activity)'
    else:
        return '기타 재무구조 (Others)'

# 바젤 III 가격결정 & 등급 판정 함수
FTP_BASE   = 0.0350   # 조달원가 3.50%
OPEX_RATE  = 0.0080   # 업무원가 0.80%
CAP_RATIO  = 0.1050   # 규제자본비율 10.5%
TARGET_ROE = 0.1000   # 목표 ROE 10.0%

def evaluate_credit_and_pricing(pd_val, ead_krw, term_yrs, lgd_val, rate_adj_bp=0.0):
    if pd_val < 0.01:
        grade, decision, badge_cls, rw = 'AAA~A (우량)', '승인 (우대 금리 적용)', 'badge-safe', 0.50
    elif pd_val < 0.03:
        grade, decision, badge_cls, rw = 'BBB (양호)', '정상 승인', 'badge-safe', 0.75
    elif pd_val < 0.07:
        grade, decision, badge_cls, rw = 'BB (주의)', '조건부 승인 (담보 보강 / 한도 축소)', 'badge-warn', 1.00
    elif pd_val < 0.15:
        grade, decision, badge_cls, rw = 'B (위험)', '정밀 심사 요망 / 고금리 적용', 'badge-warn', 1.50
    else:
        grade, decision, badge_cls, rw = 'CCC~D (고위험)', '여신 거절 (Reject)', 'badge-danger', 1.50

    term_prem = (term_yrs - 1) * 0.0015         # 만기 유동성 프리미엄 (연 +15bp)
    el_rate   = pd_val * lgd_val                # 연간 예상손실률
    cap_chg   = rw * CAP_RATIO * TARGET_ROE     # 바젤 자본비용률
    fair_rate = FTP_BASE + term_prem + OPEX_RATE + el_rate + cap_chg
    applied_rate = max(0.0, fair_rate + (rate_adj_bp / 10000.0))

    cum_pd    = 1.0 - ((1.0 - pd_val) ** term_yrs)
    el_amt    = ead_krw * el_rate               # 연간 예상손실액 (억원)
    rwa_amt   = ead_krw * rw                    # 위험가중자산 (억원)
    req_cap   = rwa_amt * CAP_RATIO             # 필요 자기자본 (억원)

    net_spread = applied_rate - (FTP_BASE + term_prem + OPEX_RATE + el_rate)
    net_profit = ead_krw * net_spread           # 연간 세전 예상순이익 (억원)
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

# ==========================================================
# 2. 좌측 사이드바: 1) 기업·여신 조건 + 2) 재무비율 직접 입력
# ==========================================================
st.sidebar.markdown("## 🏦 Risk Cockpit")
st.sidebar.caption("AI 기업여신 심사 · 바젤 III 가격결정 시스템")

sample_choice = st.sidebar.radio(
    "심사 대상 기업 선택",
    [
        f"우량 기업 샘플 (#{idx_safe})",
        f"주의 기업 샘플 (#{idx_mid})",
        f"고위험 기업 샘플 (#{idx_risk})",
        "직접 기업 번호 입력 (0~1363)"
    ]
)

if "우량" in sample_choice:     selected_idx = idx_safe
elif "주의" in sample_choice:   selected_idx = idx_mid
elif "고위험" in sample_choice: selected_idx = idx_risk
else:
    selected_idx = int(st.sidebar.number_input("Test 기업 번호", 0, len(X_test_raw) - 1, 0))

st.sidebar.markdown("---")
st.sidebar.subheader("1. 기업 및 대출 신청 조건")
firm_name = st.sidebar.text_input("기업명 (보고서 표기용)", value=f"심사대상 기업 #{selected_idx}")
ead_krw   = st.sidebar.number_input("대출 신청액 EAD (억 원)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
term_yrs  = st.sidebar.slider("대출 기간 (만기, 년)", min_value=1, max_value=5, value=3)

collateral_map = {
    "무담보 / 신용대출 (LGD 45%, 바젤 표준)": 0.45,
    "부분 담보 / 보증서 (LGD 35%)":        0.35,
    "우량 부동산 담보 (LGD 25%)":          0.25
}
collat_choice = st.sidebar.selectbox("담보 조건 (LGD 적용)", list(collateral_map.keys()))
lgd_val = collateral_map[collat_choice]

# 원본 데이터 복사 및 What-If 직접 입력
firm_raw = X_test_raw.iloc[[selected_idx]].copy()
actual_label = int(y_test.iloc[selected_idx])

st.sidebar.markdown("---")
st.sidebar.subheader("2. 핵심 재무비율 입력 (What-If)")
st.sidebar.caption("검증된 핵심 지표만 노출(단위 오염 변수 제외). 수치 변경 시 즉시 재심사됩니다.")

for cat_title in ['수익성', '유동성', '부채·레버리지', '현금흐름']:
    cat_items = [m for m in ACTIVE_METRICS if m[2] == cat_title]
    if not cat_items:
        continue
    with st.sidebar.expander(f"📂 {cat_title} 지표 조정", expanded=(cat_title in ['수익성', '부채·레버리지'])):
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
# 3. 메인 2단 분할: 중앙(Risk Passport) + 우측(Stress Tester)
# ==========================================================
col_center, col_right = st.columns([2.25, 1.0], gap="large")

# --- 우측 컨트롤 먼저 읽어 실시간 계산에 반영 ---
with col_right:
    st.markdown("### ⚡ STRESS TESTER")
    st.caption("거시경제 충격 및 금리 협상 시뮬레이터")

    macro_opt = st.selectbox(
        "거시경제 시나리오",
        ["평상시 (Baseline)", "경기 둔화 (Mild Shock: ±0.5σ)", "금융 위기 (Severe Shock: ±1.0σ)"]
    )
    macro_shock = 0.5 if "경기 둔화" in macro_opt else (1.0 if "금융 위기" in macro_opt else 0.0)

    adj_profit = st.slider("수익성 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
    adj_debt   = st.slider("부채·차입 추가 충격 (σ)", -2.0, 2.0, 0.0, 0.1)
    rate_adj_bp = st.slider("영업 협상 금리 가감 (bp)", -150, 150, 0, 10,
                            help="-50bp = 적정금리 대비 0.50%p 우대 할인")

# --- Baseline vs Stressed 계산 실행 ---
proc_base   = preprocess_row(firm_raw, macro_shock=0.0, adj_profit=0.0, adj_debt=0.0)
pd_base     = float(cal_model.predict_proba(proc_base)[:, 1][0])
res_base    = evaluate_credit_and_pricing(pd_base, ead_krw, term_yrs, lgd_val, rate_adj_bp=0.0)

proc_curr   = preprocess_row(firm_raw, macro_shock=macro_shock, adj_profit=adj_profit, adj_debt=adj_debt)
pd_curr     = float(cal_model.predict_proba(proc_curr)[:, 1][0])
res_curr    = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, lgd_val, rate_adj_bp=rate_adj_bp)

# Exact TreeSHAP 계산 (외부 shap 패키지 없이 LightGBM 내장 기능 활용)
shap_raw = np.asarray(base_lgbm.predict(proc_curr, pred_contrib=True))[0, :-1]
shap_series = pd.Series(shap_raw, index=feature_cols)

cat_shap = {}
for col, s_val in shap_series.items():
    cat = map_feature_category(col)
    cat_shap[cat] = cat_shap.get(cat, 0.0) + float(s_val)
cat_shap_df = pd.DataFrame(
    [{'재무 영역': k, 'SHAP 기여도': v} for k, v in cat_shap.items()]
).sort_values('SHAP 기여도', ascending=True)

# 우측 PROJECTIONS 다크 카드 렌더링
with col_right:
    delta_pd   = (res_curr['pd'] - res_base['pd']) * 100
    delta_rate = (res_curr['applied_rate'] - res_base['fair_rate']) * 100
    delta_el   = res_curr['el_amt'] - res_base['el_amt']
    raroc_pct  = res_curr['raroc'] * 100
    raroc_flag = "✅ Hurdle(10%) 충족" if raroc_pct >= 9.95 else "⚠️ Hurdle(10%) 미달"

    st.markdown(f"""
    <div class="projection-box">
        <div class="proj-label">LIVE PROJECTIONS (STRESS APPLIED)</div>
        <div class="proj-val">{res_curr['grade'].split(' ')[0]} &nbsp;|&nbsp; PD {res_curr['pd']*100:.2f}%</div>
        <div class="proj-sub">• 평상시 대비 PD 변화: <b>{delta_pd:+.2f}%p</b></div>
        <div class="proj-sub">• 적용 대출금리: <b>{res_curr['applied_rate']*100:.2f}%</b> ({delta_rate:+.2f}%p)</div>
        <div class="proj-sub">• 연간 예상손실(EL): <b>{res_curr['el_amt']:.2f}억 원</b> ({delta_el:+.2f}억)</div>
        <div class="proj-sub">• 예상 RAROC: <b>{raroc_pct:.1f}%</b> ({raroc_flag})</div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        f"**과거 실제 결과 (검증 참고):** {'🔴 실제 파산 (1)' if actual_label == 1 else '🟢 실제 정상 (0)'}\n\n"
        f"**모델 기준:** LightGBM + {model_bundle['best_cal_name']} Calibration\n"
        f"(15회 Leak-Free CV ROC-AUC 0.9362)"
    )

# ==========================================================
# 4. 중앙 Risk Passport (헤더 + 3개 분석 탭)
# ==========================================================
with col_center:
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    st.markdown(f"""
    <div class="passport-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.8rem; color:#64748B; font-weight:600;">📋 CREDIT RISK PASSPORT · {today_str}</div>
                <div style="font-size:1.55rem; font-weight:800; color:#0F172A; margin-top:2px;">{firm_name}</div>
                <div style="font-size:0.88rem; color:#475569; margin-top:2px;">
                    신청 여신: <b>{ead_krw:,.1f}억 원</b> &nbsp;|&nbsp; 만기: <b>{term_yrs}년</b> &nbsp;|&nbsp; 담보 LGD: <b>{lgd_val*100:.0f}%</b> (RW {res_curr['rw']*100:.0f}%)
                </div>
            </div>
            <div style="text-align:right;">
                <span class="{res_curr['badge_cls']}">{res_curr['grade']}</span>
                <div style="font-size:0.9rem; font-weight:700; color:#1E293B; margin-top:6px;">{res_curr['decision']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4개 핵심 KPI 카드
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("연간 보정 PD", f"{res_curr['pd']*100:.2f}%", f"{term_yrs}년 누적 {res_curr['cum_pd']*100:.2f}%", delta_color="off")
    k2.metric("권고 적정금리", f"{res_curr['fair_rate']*100:.2f}%", f"적용금리 {res_curr['applied_rate']*100:.2f}%", delta_color="off")
    k3.metric("연간 예상손실 (EL)", f"{res_curr['el_amt']:.2f}억 원", f"EL률 {res_curr['el_rate']*100:.2f}%", delta_color="inverse")
    k4.metric("필요 자기자본 (K)", f"{res_curr['req_cap']:.2f}억 원", f"RWA {res_curr['rwa_amt']:.1f}억", delta_color="off")

    # 중앙 세부 분석 탭 3개
    tab1, tab2, tab3 = st.tabs([
        "🔍 1. 위험요인(SHAP) & 중앙값 비교",
        "💰 2. 바젤 III 금리구조 & 민감도",
        "📝 3. 자동 심사의견서 & 징구서류"
    ])

    # ------------------------------------------------------
    # [TAB 1] 왜 위험한가 (SHAP) + 정상/파산 중앙값 비교
    # ------------------------------------------------------
    with tab1:
        t1_left, t1_right = st.columns([1.05, 1.15])

        with t1_left:
            st.markdown("#### 4대 재무영역별 위험 기여도 (TreeSHAP)")
            st.caption("빨간색(+): 부도위험 상승 요인 / 파란색(-): 부도위험 방어 요인")
            fig_s, ax_s = plt.subplots(figsize=(5.5, 3.6))
            s_colors = ['#C44E52' if v > 0 else '#1E3A5F' for v in cat_shap_df['SHAP 기여도']]
            ax_s.barh(cat_shap_df['재무 영역'], cat_shap_df['SHAP 기여도'], color=s_colors, height=0.55)
            ax_s.axvline(0, color='#334155', linewidth=0.9)
            ax_s.set_xlabel('SHAP Log-Odds 기여도 합계')
            ax_s.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

        with t1_right:
            st.markdown("#### 정상 vs 파산 기업 중앙값 비교")
            st.caption("극단치 왜곡을 막기 위해 평균 대신 전체 기업 중앙값(Median)과 비교")
            comp_rows = []
            risk_flags = []
            for col, label, cat_name, direction in ACTIVE_METRICS:
                val = float(firm_raw[col].values[0])
                n_med = float(normal_med[col])
                b_med = float(bankrupt_med[col])

                # 위험 신호 판정 (정상 중앙값 대비 불리 + 파산 중앙값에 근접/악화)
                if direction == 'higher_better':
                    is_bad = (val < (n_med + b_med) / 2.0)
                    status = "🔴 위험" if val <= b_med else ("🟡 주의" if is_bad else "🟢 양호")
                else:
                    is_bad = (val > (n_med + b_med) / 2.0)
                    status = "🔴 위험" if val >= b_med else ("🟡 주의" if is_bad else "🟢 양호")

                if status in ["🔴 위험", "🟡 주의"]:
                    risk_flags.append((cat_name, label, status, val, n_med, b_med, direction))

                comp_rows.append({
                    '재무지표': label,
                    '대상 기업': round(val, 4),
                    '정상 중앙값': round(n_med, 4),
                    '파산 중앙값': round(b_med, 4),
                    '진단': status
                })

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True, height=318)

    # ------------------------------------------------------
    # [TAB 2] 바젤 III 금리 구성 + 담보/거시 민감도 분석
    # ------------------------------------------------------
    with tab2:
        p_left, p_right = st.columns([1.1, 1.0])

        with p_left:
            st.markdown("#### 바젤 III 권고 적정금리 구성 내역")
            fig_p, ax_p = plt.subplots(figsize=(5.5, 3.4))
            p_labels = [
                '자금조달원가 (FTP)',
                f'만기 프리미엄 ({term_yrs}년)',
                '업무원가 (OpEx)',
                f'예상손실률 (PD×LGD)',
                f'자본비용 (RW {res_curr["rw"]*100:.0f}%)'
            ]
            p_vals = [
                FTP_BASE * 100,
                res_curr['term_prem'] * 100,
                OPEX_RATE * 100,
                res_curr['el_rate'] * 100,
                res_curr['cap_chg'] * 100
            ]
            p_colors = ['#475569', '#64748B', '#0284C7', '#DC2626', '#1E3A5F']
            bars = ax_p.barh(p_labels, p_vals, color=p_colors, height=0.55)
            ax_p.set_xlabel('금리 비중 (%)')
            ax_p.set_xlim(0, max(p_vals) * 1.38 + 0.5)
            for bar, v in zip(bars, p_vals):
                ax_p.text(v + 0.08, bar.get_y() + bar.get_height()/2, f'{v:.2f}%', va='center', fontsize=9, fontweight='bold')
            ax_p.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        with p_right:
            st.markdown("#### 담보 조건별 금리·손실 민감도")
            collat_rows = []
            for c_name, c_lgd in [("무담보 (45%)", 0.45), ("부분담보 (35%)", 0.35), ("우량담보 (25%)", 0.25)]:
                r_c = evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, c_lgd, rate_adj_bp=0.0)
                collat_rows.append({
                    '담보 조건': c_name,
                    '적정금리(%)': f"{r_c['fair_rate']*100:.2f}%",
                    '연간 EL(억원)': f"{r_c['el_amt']:.2f}",
                    '절감 EL(억원)': f"{(evaluate_credit_and_pricing(pd_curr, ead_krw, term_yrs, 0.45)['el_amt'] - r_c['el_amt']):+.2f}"
                })
            st.dataframe(pd.DataFrame(collat_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 거시경제 충격별 민감도")
            macro_rows = []
            for m_name, m_sh in [("평상시 (0σ)", 0.0), ("경기 둔화 (±0.5σ)", 0.5), ("금융 위기 (±1.0σ)", 1.0)]:
                p_m = float(cal_model.predict_proba(preprocess_row(firm_raw, m_sh, adj_profit, adj_debt))[:, 1][0])
                r_m = evaluate_credit_and_pricing(p_m, ead_krw, term_yrs, lgd_val, rate_adj_bp=0.0)
                macro_rows.append({
                    '시나리오': m_name,
                    '보정 PD(%)': f"{p_m*100:.2f}%",
                    '신용등급': r_m['grade'].split(' ')[0],
                    '적정금리(%)': f"{r_m['fair_rate']*100:.2f}%",
                    '연간 EL(억)': f"{r_m['el_amt']:.2f}"
                })
            st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # [TAB 3] 100% Rule-Based 자동 심사 의견서 & 징구 서류
    # ------------------------------------------------------
    with tab3:
        # 1) SHAP 상승 기여 상위 영역 추출
        pos_shap = cat_shap_df[cat_shap_df['SHAP 기여도'] > 0.05].sort_values('SHAP 기여도', ascending=False)
        top_risk_cats = list(pos_shap['재무 영역'].head(2))

        # 2) 검증된 지표 중 위험/주의 신호 요약
        flagged_cats = sorted(list(set([f[0] for f in risk_flags])))
        severe_items = [f[1] for f in risk_flags if f[2] == "🔴 위험"]
        caution_items = [f[1] for f in risk_flags if f[2] == "🟡 주의"]

        # 3) Rule-Based 심사 의견 문장 조립 (입력값·모델 결과 외 사실 생성 금지)
        memo_lines = []
        memo_lines.append(
            f"<b>[종합 심사 의견]</b> {firm_name}의 연간 보정 부도확률(PD)은 <b>{res_curr['pd']*100:.2f}%</b>"
            f"({term_yrs}년 누적 {res_curr['cum_pd']*100:.2f}%)로 산출되어 신용등급 <b>{res_curr['grade']}</b> 및 "
            f"<b>'{res_curr['decision']}'</b> 대상에 해당합니다. 신청 금액 {ead_krw:,.1f}억 원(LGD {lgd_val*100:.0f}%) 기준 "
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

        st.markdown("#### AI Underwriting Rationale (근거 기반 자동 심사 의견)")
        st.markdown(f"<div class='memo-box'>{'<br><br>'.join(memo_lines)}</div>", unsafe_allow_html=True)

        # 4) 위험 카테고리별 1:1 매핑 추가 제출자료 및 심사역 검토 포인트
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

        st.markdown("#### 📋 추가 제출자료 및 심사 담당자 검토 체크리스트")
        req_docs_text = []
        if flagged_cats:
            for fc in flagged_cats:
                docs, check_pt = DOC_MAP[fc]
                st.markdown(f"- **[{fc} 위험 대응]**\n  - **추가 징구 서류:** {docs}\n  - **심사역 검토 사항:** {check_pt}")
                req_docs_text.append(f"[{fc}] 서류: {docs} / 검토: {check_pt}")
        else:
            st.markdown("- **표준 심사 서류:** 최근 3개년 감사보고서(또는 표준재무제표증명), 국세·지방세 완납증명서")
            st.markdown("- **심사역 검토 사항:** 비재무 리스크(대표자 평판, 업황 변동성, 담보 감정가 적정성) 최종 확인")
            req_docs_text.append("[표준] 최근 3개년 재무제표, 납세증명서 및 비재무 리스크 확인")

        # 5) 실제 동작하는 심사 의견서 다운로드 버튼
        plain_memo = "\n".join([
            f"==================================================",
            f"  AI 기업여신 심사 검토 의견서 ({today_str})",
            f"==================================================",
            f"1. 기업명: {firm_name}",
            f"2. 신청 조건: 대출금액 {ead_krw:,.1f}억 원 / 만기 {term_yrs}년 / LGD {lgd_val*100:.0f}% (RW {res_curr['rw']*100:.0f}%)",
            f"3. 시나리오: {macro_opt} (수익성 {adj_profit:+.1f}σ, 부채 {adj_debt:+.1f}σ, 금리조정 {rate_adj_bp:+d}bp)",
            f"--------------------------------------------------",
            f"[신용분석 및 바젤 III 가격결정 결과]",
            f"- 연간 보정 PD: {res_curr['pd']*100:.2f}% ({term_yrs}년 누적 PD: {res_curr['cum_pd']*100:.2f}%)",
            f"- 산출 신용등급: {res_curr['grade']} -> {res_curr['decision']}",
            f"- 권고 적정금리: {res_curr['fair_rate']*100:.2f}% (적용금리: {res_curr['applied_rate']*100:.2f}%)",
            f"- 연간 예상손실(EL): {res_curr['el_amt']:.2f}억 원 / 필요 자기자본(K): {res_curr['req_cap']:.2f}억 원",
            f"- 예상 세전 RAROC: {res_curr['raroc']*100:.1f}% (목표 Hurdle Rate: 10.0%)",
            f"--------------------------------------------------",
            f"[추가 징구 서류 및 검토 사항]",
            *req_docs_text,
            f"=================================================="
        ])

        st.download_button(
            label="📥 심사 검토 의견서 다운로드 (.txt)",
            data=plain_memo.encode('utf-8-sig'),
            file_name=f"여신심사의견서_{firm_name.replace(' ', '_')}_{today_str}.txt",
            mime="text/plain",
            use_container_width=True
        )
