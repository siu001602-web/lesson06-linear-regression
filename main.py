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
    "서울의 기온 데이터를 바탕으로 연평균 기온 추세를 분석하고, 회귀 분석을 통해 미래 연도의 기온을 예측합니다."
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

# 메타데이터 추출
start_year = int(yearly_data["연도"].min())
end_year = int(yearly_data["연도"].max())
num_years = len(yearly_data)

# 선형 회귀 분석 (최소제곱법)
X = yearly_data["연도"].values
y = yearly_data["연평균기온"].values

slope, intercept = np.polyfit(X, y, 1)
correlation = np.corrcoef(X, y)[0, 1]

# 화면 상단 메타데이터 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("시작 연도", f"{start_year}년")
col2.metric("끝 연도", f"{end_year}년")
col3.metric("사용한 연도 개수", f"{num_years}개")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

st.markdown("---")

# Plotly 그래프 생성
yearly_data["회귀예측값"] = slope * yearly_data["연도"] + intercept

fig = px.scatter(
    yearly_data,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["회귀예측값"],
        mode="lines",
        name=f"회귀선 (y = {slope:.4f}x + {intercept:.2f})",
        line=dict(color="red", width=2.5),
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
st.subheader("🔮 연도별 예상 기온 조회")

# 슬라이더 (1900 ~ 2100년)
selected_year = st.slider(
    "조회할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택한 연도의 예상 기온 계산
predicted_temp = slope * selected_year + intercept

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
