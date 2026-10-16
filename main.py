import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 연도별 평균기온 분석 및 예측기")
st.markdown(
    "서울의 기온 데이터를 바탕으로 연평균 기온 추세를 분석하고, 전체 기간과 최근 20년의 기온 상승 속도를 비교합니다."
)


# 데이터 로드 함수
@st.cache_data
def load_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")
  return df


# 데이터 불러오기
try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 컬럼명 공백 제거
df.columns = df.columns.str.strip()

# 날짜 데이터 전처리
df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 1. 2025년까지의 데이터만 필터링
df_filtered = df[df["연도"] <= 2025].copy()

# 연도별 관측일수 및 평균기온 계산
yearly_stats = (
    df_filtered.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean"),
    )
    .reset_index()
)

# 2. 관측일이 300일이 안 되는 해는 제외
yearly_data = yearly_stats[yearly_stats["관측일수"] >= 300].dropna(
    subset=["연평균기온"]
)

if yearly_data.empty:
  st.warning("조건을 만족하는 데이터가 없습니다.")
  st.stop()

# 메타데이터 추출 (전체 기간)
start_year = int(yearly_data["연도"].min())
end_year = int(yearly_data["연도"].max())
num_years = len(yearly_data)

# 선형 회귀 분석 (전체 기간)
X_all = yearly_data["연도"].values
y_all = yearly_data["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
correlation_all = np.corrcoef(X_all, y_all)[0, 1]

# 최근 20년 데이터 필터링 (예: 2006년 ~ 2025년)
recent_start_year = max(end_year - 19, start_year)
recent_data = yearly_data[yearly_data["연도"] >= recent_start_year]
X_recent = recent_data["연도"].values
y_recent = recent_data["연평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)

# 100년당 기온 상승 폭으로 변환 (기울기 * 100)
per_century_all = slope_all * 100
per_century_recent = slope_recent * 100

st.markdown("---")
st.subheader("📊 기온 상승 속도 비교 (100년당 상승 폭)")

# 화면 상단 비교 메트릭 표시
col1, col2 = st.columns(2)

with col1:
  st.markdown(
      f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <h4 style="margin: 0; color: #31333F;">전체 기간 ({start_year}~{end_year}년, {num_years}개년)</h4>
            <h2 style="margin: 10px 0 0 0; color: #1f77b4;">100년당 +{per_century_all:.2f} °C</h2>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">상관계수(r): {correlation_all:.4f}</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col2:
  st.markdown(
      f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <h4 style="margin: 0; color: #31333F;">최근 20년 ({recent_start_year}~{end_year}년)</h4>
            <h2 style="margin: 10px 0 0 0; color: #ff4b4b;">100년당 +{per_century_recent:.2f} °C</h2>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">최근 속도가 전체 평균보다 약 {(per_century_recent / per_century_all):.1f}배 빠름</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.markdown("---")

# Plotly 그래프 생성
yearly_data["회귀예측값"] = slope_all * yearly_data["연도"] + intercept_all

fig = px.scatter(
    yearly_data,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 및 전체 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
)

# 전체 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["회귀예측값"],
        mode="lines",
        name=f"전체 회귀선 (100년당 +{per_century_all:.2f}°C)",
        line=dict(color="blue", width=2.5),
    )
)

# 최근 20년 회귀 직선 추가
recent_data_line_y = slope_recent * recent_data["연도"] + intercept_recent
fig.add_trace(
    go.Scatter(
        x=recent_data["연도"],
        y=recent_data_line_y,
        mode="lines",
        name=f"최근 20년 회귀선 (100년당 +{per_century_recent:.2f}°C)",
        line=dict(color="red", width=2.5, dash="dash"),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔮 연도별 예상 기온 조회 (전체 회귀 기준)")

# 슬라이더 (1900 ~ 2100년)
selected_year = st.slider(
    "조회할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택한 연도의 예상 기온 계산
predicted_temp = slope_all * selected_year + intercept_all

# 결과 강조 표시
st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h3 style="margin: 0; color: #31333F;">선택 연도: {selected_year}년</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b;">예상 연평균기온: {predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
