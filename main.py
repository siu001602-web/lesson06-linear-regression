python

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

CUTOFF_YEAR = 2025
MIN_OBSERVATIONS = 300

st.title("🌡️ 서울 기온 예측기")
st.caption(
    "서울의 연평균기온 데이터를 이용한 선형회귀 예측 모델 "
    "| 학습 데이터 기준: 2025년까지"
)


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"필수 열이 없습니다: {', '.join(sorted(missing))}"
        )

    # 날짜와 기온을 올바른 자료형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준 연도까지만 사용
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 유지
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("인터넷 연결과 데이터 파일의 열 이름을 확인해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 하려면 유효한 연도가 최소 2개 필요합니다.")
    st.stop()


# --------------------------------------------------
# 선형회귀 및 상관계수
# --------------------------------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

# y = slope * x + intercept
slope, intercept = np.polyfit(x, y, 1)
correlation = float(np.corrcoef(x, y)[0, 1])

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

# 선택 가능한 모든 연도에 대한 예측값
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

predicted_temp = slope * selected_year + intercept


# --------------------------------------------------
# 핵심 지표
# --------------------------------------------------
st.subheader(f"{selected_year}년 예상 연평균기온")

st.metric(
    label="선형회귀 예상 기온",
    value=f"{predicted_temp:.2f} °C",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("상관계수 (r)", f"{correlation:.4f}")

with col2:
    st.metric("회귀 분석 연도 수", f"{n_years:,}개")

with col3:
    st.metric("시작 연도", f"{start_year}년")

with col4:
    st.metric("끝 연도", f"{end_year}년")

st.caption(
    f"회귀식: 연평균기온 = {slope:.5f} × 연도 "
    f"{intercept:+.2f}"
)

st.info(
    f"분석 기준: 2025년까지의 자료 중 연간 관측일이 "
    f"{MIN_OBSERVATIONS}일 이상인 {n_years}개 연도 "
    f"({start_year}~{end_year}년)."
)


# --------------------------------------------------
# Plotly 산점도 + 회귀 직선
# --------------------------------------------------
fig = go.Figure()

# 관측된 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="관측 연평균기온",
        marker=dict(
            color="#2F80ED",
            size=7,
            opacity=0.8,
        ),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 학습 데이터 범위에 맞춘 회귀 직선
line_x = np.linspace(start_year, end_year, 200)
line_y = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형회귀 직선",
        line=dict(color="#EB5757", width=3),
        hovertemplate=(
            "연도: %{x:.1f}년<br>"
            "회귀값: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 슬라이더로 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상 기온",
        marker=dict(
            color="#F2C94C",
            size=14,
            symbol="diamond",
            line=dict(color="#333333", width=1),
        ),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 선형회귀",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    hovermode="closest",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=90, b=20),
)

fig.update_xaxes(
    range=[min(start_year, selected_year) - 2,
           max(end_year, selected_year) + 2]
)

st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# 연도별 집계 결과 확인
# --------------------------------------------------
with st.expander("연도별 분석 데이터 확인"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    st.dataframe(display_df, width="stretch", hide_index=True)

    csv = yearly.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "연도별 집계 데이터 CSV 다운로드",
        data=csv,
        file_name="seoul_yearly_temperature.csv",
        mime="text/csv",
    )

st.caption(
    "주의: 예상 기온은 과거 연도와 연평균기온의 선형 관계를 "
    "외삽한 값입니다. 실제 미래 기온이나 기후변화를 "
    "정밀하게 예측하는 결과는 아닙니다."
)
