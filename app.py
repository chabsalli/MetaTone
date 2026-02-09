# app.py
import os
import json
import uuid
import sqlite3
import math
import re
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px

# ============================
# 1. 앱 설정 및 상수
# ============================
DB_PATH = "metatone_pro.db"

SOFT_SKILLS = ["문제해결", "의사소통", "협업", "리더십", "자기관리/회복탄력성", "학습역량"]
CATEGORIES = ["학습", "프로젝트", "리더십·동아리", "대외활동", "관계·협업", "생활·루틴"]

# 주간 단위(사용자 확정)
# - goal_week: "YYYY-Www" 형태로 저장
def iso_week_key(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"

# 템플릿(사용자 확정 5-1)
TEMPLATES = {
    "자유 기록": {
        "behavior_label": "1. 행동 (Behavior)",
        "behavior_ph": "예: 팀 프로젝트 회의에서 갈등을 중재하고 일정을 다시 짰습니다.",
        "emotion_label": "2. 감정 (Emotion)",
        "emotion_ph": "예: 처음엔 당황스러웠지만 점차 책임감을 느꼈습니다.",
        "result_label": "3. 결과 (Result)",
        "result_ph": "예: 지연되었던 일정을 3일 단축했고 팀 분위기가 좋아졌습니다.",
    },
    "갈등 중재": {
        "behavior_label": "1. 행동 (갈등 상황에서 내가 한 말/행동)",
        "behavior_ph": "예: A와 B의 의견 차이를 정리해 쟁점을 2개로 나누고, 합의 가능한 기준부터 제안했습니다.",
        "emotion_label": "2. 감정 (내 감정 + 상대 반응)",
        "emotion_ph": "예: 답답했지만 침착하려고 했고, 상대는 방어적으로 반응했습니다.",
        "result_label": "3. 결과 (관계/결정/성과 측면)",
        "result_ph": "예: 결론은 늦어졌지만 기준이 생겨 이후 충돌이 줄었습니다.",
    },
    "마감/압박": {
        "behavior_label": "1. 행동 (압박 속에서 취한 전략)",
        "behavior_ph": "예: 일정이 밀려 우선순위를 재정의하고, 핵심 산출물부터 완성했습니다.",
        "emotion_label": "2. 감정 (스트레스/집중/회복)",
        "emotion_ph": "예: 불안했지만 체크리스트가 생기니 안정됐습니다.",
        "result_label": "3. 결과 (품질/속도/학습)",
        "result_ph": "예: 품질을 유지하면서 마감에 맞췄고, 다음엔 사전 리스크 체크가 필요함을 느꼈습니다.",
    },
    "피드백 주고받기": {
        "behavior_label": "1. 행동 (피드백을 어떻게 전달/수용했나)",
        "behavior_ph": "예: 사실-영향-요청 구조로 피드백을 주고, 반박 대신 질문으로 확인했습니다.",
        "emotion_label": "2. 감정 (불편함/수용/방어)",
        "emotion_ph": "예: 서운했지만 성장 기회로 해석하려고 했습니다.",
        "result_label": "3. 결과 (관계/성과/다음 액션)",
        "result_ph": "예: 관계는 유지됐고, 다음 회의부터 합의된 기준으로 진행했습니다.",
    },
    "리더 역할": {
        "behavior_label": "1. 행동 (리더로서 의사결정/조율/지원)",
        "behavior_ph": "예: 역할 분담을 재정의하고, 병목을 맡아 해결했습니다.",
        "emotion_label": "2. 감정 (책임/부담/동기)",
        "emotion_ph": "예: 부담이 컸지만 팀이 안정되는 느낌이 있었습니다.",
        "result_label": "3. 결과 (팀/성과/학습)",
        "result_ph": "예: 일정이 안정됐고, 다음엔 초기에 기준/리스크를 더 명확히 해야 함을 배웠습니다.",
    },
}

# LLM 모델(분석/정체성/한줄피드백)
MODEL_ANALYZE = "gpt-4o-mini"
MODEL_ONE_LINER = "gpt-4o-mini"
MODEL_IDENTITY = "gpt-4o-mini"

# ============================
# 2. 데이터베이스 레이어
# ============================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _col_exists(conn, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # 메인 기록 테이블 (기존 스키마 + template_name, mood)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            entry_date TEXT,
            category TEXT,
            template_name TEXT,
            mood INTEGER,
            behavior TEXT,
            emotion TEXT,
            result TEXT,
            analysis_json TEXT,
            top_skill TEXT
        )""")

        # 성장 루틴(2+2 메모) 테이블 + 이번 주 목표
        cur.execute("""
        CREATE TABLE IF NOT EXISTS growth_notes (
            id TEXT PRIMARY KEY,
            entry_id TEXT,
            note_type TEXT, -- 'practice', 'question'
            content TEXT,
            user_memo TEXT,
            is_completed INTEGER DEFAULT 0,
            is_weekly_goal INTEGER DEFAULT 0,
            goal_week TEXT,
            FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
        )""")

        # 정체성 문장(6-1)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS identity_statement (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT,
            updated_at TEXT,
            is_pinned INTEGER DEFAULT 0
        )""")
        # 기본 row 확보
        cur.execute("INSERT OR IGNORE INTO identity_statement (id, content, updated_at, is_pinned) VALUES (1, '', '', 0)")

        conn.commit()

def bump_db_version():
    st.session_state["db_ver"] = st.session_state.get("db_ver", 0) + 1

@st.cache_data
def fetch_entries(db_ver: int):
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM entries ORDER BY entry_date DESC", conn)

@st.cache_data
def fetch_growth_notes(db_ver: int):
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM growth_notes", conn)

def get_identity():
    with get_conn() as conn:
        row = conn.execute("SELECT content, updated_at, is_pinned FROM identity_statement WHERE id=1").fetchone()
    return {"content": row[0] or "", "updated_at": row[1] or "", "is_pinned": int(row[2] or 0)}

def set_identity(content: str, is_pinned: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE identity_statement SET content=?, updated_at=?, is_pinned=? WHERE id=1",
            (content, datetime.now().isoformat(timespec="seconds"), int(is_pinned)),
        )
    bump_db_version()

# ============================
# 3. 분석 로직 (LLM)
# ============================
def _openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def analyze_experience(behavior, emotion, result, api_key, mood: int, template_name: str, category: str):
    """
    메타인지 강화 프롬프트 포함:
    - counterfactuals 2개(사용자 확정)
    - blind_spot, next_signal
    JSON only
    """
    client = _openai_client(api_key)

    system = (
        "당신은 메타인지 기반 커리어 코치이자 분석가다. "
        "사용자의 경험을 구조화해 소프트스킬을 추정하되, 추측은 '가정'임을 명확히 한다. "
        "현실적/실행 가능한 대안만 제시한다(마법처럼 상대 마음을 바꾼다 금지). "
        "반드시 JSON만 반환한다."
    )

    prompt = f"""
아래 사용자의 기록을 분석해 주세요.

[카테고리]: {category}
[템플릿]: {template_name}
[기분(1~10)]: {mood}

[행동]: {behavior}
[감정(서술)]: {emotion}
[결과]: {result}

반드시 아래 스키마 그대로 JSON으로만 응답하세요(키 이름 고정).

{{
  "soft_skills": [
    {{
      "name": "({SOFT_SKILLS} 중 1개)",
      "reason": "왜 그렇게 판단했는지 1~2문장",
      "confidence": 0.0
    }}
  ],
  "growth_plan": {{
    "top_skill": "({SOFT_SKILLS} 중 1개)",
    "practices": ["실천 제안 1", "실천 제안 2"],
    "questions": ["성찰 질문 1", "성찰 질문 2"]
  }},
  "metacognition": {{
    "counterfactuals": [
      {{
        "alt_action": "당시 갈등/의사결정 상황에서 내가 선택할 수 있었던 현실적인 대안 행동 1(구체적)",
        "expected_outcome_change": "결과가 어떻게 달라질 가능성이 있는지 2~3문장(불확실성/가정 표시)",
        "reflection_question": "사용자가 답할 수 있는 맞춤형 성찰 질문 1문장"
      }},
      {{
        "alt_action": "대안 행동 2(다른 접근)",
        "expected_outcome_change": "2~3문장(불확실성/가정 표시)",
        "reflection_question": "맞춤형 성찰 질문 1문장"
      }}
    ],
    "blind_spot": "사용자가 놓쳤을 가능성이 있는 관점 1문장",
    "next_signal": "다음에 비슷한 상황이 오면 스스로 감지할 신호(몸/생각/대화 패턴) 1문장"
  }}
}}

규칙:
- confidence는 0~1 사이 숫자.
- soft_skills는 1~3개.
- counterfactuals는 정확히 2개.
- 문장은 간결하지만 구체적으로.
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_ANALYZE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def generate_one_liner(api_key: str, recent_entries: list[dict]):
    """
    1-1: 오늘의 한 줄 피드백(초저비용)
    - 최근 3개 요약을 바탕으로 1~2문장
    """
    if not api_key or not recent_entries:
        return ""

    client = _openai_client(api_key)
    system = (
        "당신은 짧고 날카로운 메타인지 코치다. "
        "사용자가 오늘 바로 실행할 수 있는 1~2문장 피드백만 준다. "
        "과장 금지. 판단 근거는 암시만 하고 길게 설명하지 않는다."
    )
    payload = [
        {
            "date": e.get("entry_date"),
            "top_skill": e.get("top_skill"),
            "mood": e.get("mood"),
            "behavior": (e.get("behavior") or "")[:240],
            "result": (e.get("result") or "")[:240],
        }
        for e in recent_entries[:3]
    ]
    user = (
        "최근 기록 3개를 보고, 오늘의 한 줄 피드백(1~2문장)을 만들어줘.\n"
        "- 첫 문장: 패턴 인식(좋은 점 or 위험 신호)\n"
        "- 둘째 문장: 오늘 실천 1개(아주 구체적으로)\n"
        f"입력:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_ONE_LINER,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

def generate_identity_statement(api_key: str, entries: pd.DataFrame):
    """
    6-1: 정체성 문장 자동 생성(20개 이상)
    - 1~2문장, 사용자가 수정/고정 가능
    """
    if not api_key:
        return None, "OpenAI API Key가 필요합니다."
    if entries is None or len(entries) < 20:
        return None, "정체성 문장은 기록이 20개 이상일 때 생성할 수 있어요."

    client = _openai_client(api_key)
    system = (
        "당신은 사용자의 반복 패턴을 바탕으로 정체성 문장을 만드는 코치다. "
        "자기찬양/허세 없이, 관찰 기반으로 1~2문장만. 한국어."
    )

    # 비용 절감: 최근 30개만
    df = entries.sort_values("entry_date", ascending=False).head(30)
    summary = []
    for _, r in df.iterrows():
        summary.append({
            "date": r.get("entry_date"),
            "category": r.get("category"),
            "top_skill": r.get("top_skill"),
            "mood": r.get("mood"),
            "behavior": (r.get("behavior") or "")[:200],
            "result": (r.get("result") or "")[:200],
        })

    user = (
        "아래 기록 요약을 바탕으로 사용자의 '정체성 문장'을 1~2문장으로 작성해줘.\n"
        "- 반드시 행동/선택의 경향을 담아줘.\n"
        "- 과장 금지.\n"
        f"입력:\n{json.dumps(summary, ensure_ascii=False)}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_IDENTITY,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip(), None
    except Exception as e:
        return None, f"정체성 문장 생성 실패: {e}"

# ============================
# 4. 분석/시각화 유틸
# ============================
def shannon_entropy(proportions):
    # proportions: list of floats summing to 1
    ent = 0.0
    for p in proportions:
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent

def balance_score_from_counts(counts: pd.Series) -> float:
    # 0~100 스케일: 0(한 스킬 올인) ~ 100(고르게)
    total = counts.sum()
    if total <= 0:
        return 0.0
    props = (counts / total).tolist()
    ent = shannon_entropy(props)
    max_ent = math.log(len(SOFT_SKILLS), 2)
    if max_ent == 0:
        return 0.0
    return float((ent / max_ent) * 100.0)

def mood_bucket(m: int) -> str:
    # 1~10 → 3구간
    if m <= 3:
        return "낮음(1~3)"
    if m <= 7:
        return "중간(4~7)"
    return "높음(8~10)"

def safe_json_load(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

# ============================
# 5. UI 및 메인 로직
# ============================
def main():
    st.set_page_config(page_title="MetaTone Pro", layout="wide")
    init_db()

    if "db_ver" not in st.session_state:
        st.session_state["db_ver"] = 0
    if "one_liner_cache" not in st.session_state:
        st.session_state["one_liner_cache"] = {"date": "", "text": ""}

    st.sidebar.title("💎 MetaTone Pro")
    menu = st.sidebar.radio("메뉴", ["성장 대시보드", "새 기록 작성", "회고 보관소"])
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    # 공통 데이터
    df_entries = fetch_entries(st.session_state["db_ver"])
    df_notes = fetch_growth_notes(st.session_state["db_ver"])

    # ================
    # 페이지 1: 새 기록 작성
    # ================
    if menu == "새 기록 작성":
        st.header("✍️ 구조화된 오늘의 경험 기록")
        st.caption("템플릿을 고르면 질문이 달라져요. (MetaTone: 메타인지 향상에 최적화)")

        with st.form("entry_form"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                entry_date = st.date_input("날짜", date.today())
            with col2:
                category = st.selectbox("카테고리", CATEGORIES)
            with col3:
                template_name = st.selectbox("템플릿", list(TEMPLATES.keys()), index=0)

            # mood 숫자(시각화/패턴 분석용)
            mood = st.slider("기분(1~10)", 1, 10, 6)

            t = TEMPLATES[template_name]
            st.subheader(t["behavior_label"])
            behavior = st.text_area(" ", key="behavior", placeholder=t["behavior_ph"], height=120)

            st.subheader(t["emotion_label"])
            emotion = st.text_area("  ", key="emotion", placeholder=t["emotion_ph"], height=100)

            st.subheader(t["result_label"])
            result = st.text_area("   ", key="result", placeholder=t["result_ph"], height=110)

            submit = st.form_submit_button("역량 분석 및 저장", type="primary")

        if submit:
            if not api_key:
                st.error("OpenAI API Key를 입력해주세요.")
                return
            if not behavior.strip() or not result.strip():
                st.error("행동과 결과는 최소한 작성해 주세요. (감정은 짧아도 OK)")
                return

            with st.spinner("전문 코치가 분석 중입니다..."):
                try:
                    analysis = analyze_experience(
                        behavior=behavior,
                        emotion=emotion,
                        result=result,
                        api_key=api_key,
                        mood=int(mood),
                        template_name=template_name,
                        category=category,
                    )
                except Exception as e:
                    st.error(f"분석 실패: {e}")
                    return

            entry_id = str(uuid.uuid4())
            top_skill = (analysis.get("growth_plan", {}) or {}).get("top_skill", "")

            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO entries (id, entry_date, category, template_name, mood, behavior, emotion, result, analysis_json, top_skill) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        entry_id,
                        entry_date.isoformat(),
                        category,
                        template_name,
                        int(mood),
                        behavior,
                        emotion,
                        result,
                        json.dumps(analysis, ensure_ascii=False),
                        top_skill,
                    ),
                )

                # 2+2 성장 메모 저장
                gp = analysis.get("growth_plan", {}) or {}
                for p in (gp.get("practices") or [])[:2]:
                    cur.execute(
                        "INSERT INTO growth_notes (id, entry_id, note_type, content) VALUES (?,?,?,?)",
                        (str(uuid.uuid4()), entry_id, "practice", p),
                    )
                for q in (gp.get("questions") or [])[:2]:
                    cur.execute(
                        "INSERT INTO growth_notes (id, entry_id, note_type, content) VALUES (?,?,?,?)",
                        (str(uuid.uuid4()), entry_id, "question", q),
                    )

                conn.commit()

            bump_db_version()
            st.success(f"분석 완료! 오늘의 핵심 역량은 **[{top_skill}]** 입니다.")
            st.rerun()

    # ================
    # 페이지 2: 성장 대시보드
    # ================
    elif menu == "성장 대시보드":
        st.header("📊 역량 성장 리포트 (주간 단위)")
        if df_entries.empty:
            st.info("아직 기록이 없어요. '새 기록 작성'에서 첫 기록을 남겨보세요!")
            return

        # --- 1-1 오늘의 한 줄 피드백 (대시보드 상단 고정) ---
        st.subheader("🧭 오늘의 한 줄 피드백")
        recent3 = df_entries.head(3).to_dict(orient="records")
        today_key = date.today().isoformat()
        cached = st.session_state["one_liner_cache"]

        colA, colB = st.columns([4, 1])
        with colB:
            regen = st.button("재생성", use_container_width=True)
        if regen or cached["date"] != today_key or not cached["text"]:
            if api_key:
                text = generate_one_liner(api_key, recent3)
                st.session_state["one_liner_cache"] = {"date": today_key, "text": text}
            else:
                st.session_state["one_liner_cache"] = {"date": today_key, "text": ""}

        one_liner = st.session_state["one_liner_cache"]["text"]
        if one_liner:
            st.success(one_liner)
        else:
            st.info("OpenAI API Key를 입력하면 한 줄 피드백이 생성돼요.")

        st.divider()

        # --- 2-2 스킬 편향 경고 (최근 10개 기준) ---
        st.subheader("⚖️ 역량 편향 & 균형도")
        last_n = 10
        df_last = df_entries.head(last_n)
        counts_last = df_last["top_skill"].value_counts()
        if not counts_last.empty:
            top = counts_last.index[0]
            share = counts_last.iloc[0] / max(1, counts_last.sum())
            balance = balance_score_from_counts(counts_last.reindex(SOFT_SKILLS).fillna(0))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("최근 10개 최빈 역량", top)
            with m2:
                st.metric("편향 비중", f"{share*100:.0f}%")
            with m3:
                st.metric("균형도(0~100)", f"{balance:.0f}")

            if share >= 0.60:
                st.warning(
                    f"최근 {last_n}개 기록의 **{share*100:.0f}%**가 **{top}**에 몰려 있어요.\n\n"
                    "MetaTone 관점: 편향이 나쁘진 않지만, **다른 프레임도 함께 강화**하면 성장 폭이 커져요."
                )

        # --- 전체 분포 (현재 상태) ---
        dist_counts = df_entries["top_skill"].value_counts().reindex(SOFT_SKILLS).fillna(0).astype(int).reset_index()
        dist_counts.columns = ["top_skill", "count"]
        fig_dist = px.bar(dist_counts, x="top_skill", y="count", title="전체 기록 기준 핵심 역량 분포")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()

        # --- 시간에 따른 성장(주간) ---
        st.subheader("📆 주간 변화(성장 추세)")

        df_t = df_entries.copy()
        df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
        df_t["week"] = df_t["entry_date"].dt.date.apply(iso_week_key)

        # 주차별 스킬 비중(100% stacked area)
        pivot = (
            df_t.pivot_table(index="week", columns="top_skill", values="id", aggfunc="count", fill_value=0)
            .reindex(columns=SOFT_SKILLS, fill_value=0)
            .sort_index()
        )
        pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
        pivot_pct = pivot_pct.reset_index().melt(id_vars="week", var_name="top_skill", value_name="pct")

        fig_area = px.area(
            pivot_pct,
            x="week",
            y="pct",
            color="top_skill",
            title="주차별 핵심 역량 비중(%)",
        )
        fig_area.update_layout(yaxis_title="비중(%)", xaxis_title="주(ISO Week)")
        st.plotly_chart(fig_area, use_container_width=True)

        # 주차별 균형도 추세
        bal_rows = []
        for wk, g in df_t.groupby("week"):
            c = g["top_skill"].value_counts().reindex(SOFT_SKILLS).fillna(0)
            bal_rows.append({"week": wk, "balance": balance_score_from_counts(c)})
        df_bal = pd.DataFrame(bal_rows).sort_values("week")
        fig_bal = px.line(df_bal, x="week", y="balance", markers=True, title="주차별 균형도(0~100) 추세")
        fig_bal.update_layout(yaxis_title="균형도", xaxis_title="주(ISO Week)")
        st.plotly_chart(fig_bal, use_container_width=True)

        # (보너스) Mood x Skill 히트맵: 메타인지 트리거
        st.subheader("🧠 기분-역량 패턴(메타인지 트리거)")
        df_m = df_entries.copy()
        df_m["mood"] = pd.to_numeric(df_m["mood"], errors="coerce").fillna(0).astype(int)
        df_m["mood_bucket"] = df_m["mood"].apply(lambda x: mood_bucket(x) if x > 0 else "미입력")
        heat = (
            df_m.pivot_table(index="top_skill", columns="mood_bucket", values="id", aggfunc="count", fill_value=0)
            .reindex(index=SOFT_SKILLS, fill_value=0)
            .reset_index()
            .melt(id_vars="top_skill", var_name="mood_bucket", value_name="count")
        )
        fig_heat = px.density_heatmap(
            heat,
            x="mood_bucket",
            y="top_skill",
            z="count",
            histfunc="sum",
            title="기분 구간별 기록된 핵심 역량 빈도",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # --- 3-1 이번 주 목표(최대 3개) ---
        st.subheader("🎯 이번 주 집중 루틴 (최대 3개)")
        current_week = iso_week_key(date.today())

        # 이번 주 목표 목록
        weekly = df_notes[
            (df_notes["note_type"] == "practice")
            & (df_notes["is_weekly_goal"] == 1)
            & (df_notes["goal_week"] == current_week)
        ].copy()

        # 후보: 최근 practice 중에서 아직 weekly_goal 아닌 것 (최근 30개)
        recent_entry_ids = set(df_entries.head(30)["id"].tolist())
        candidates = df_notes[
            (df_notes["note_type"] == "practice")
            & (df_notes["entry_id"].isin(recent_entry_ids))
        ].copy()
        candidates["label"] = candidates["content"].fillna("")
        # 중복 제거(같은 문장 많을 수 있음)
        candidates = candidates.drop_duplicates(subset=["label"]).head(30)

        left, right = st.columns([2, 1])
        with left:
            if weekly.empty:
                st.info("이번 주 목표가 아직 없어요. 아래에서 최대 3개를 골라 고정해보세요.")
            else:
                st.write(f"**이번 주({current_week}) 목표**")
                for _, r in weekly.iterrows():
                    done = bool(int(r.get("is_completed") or 0))
                    new_done = st.checkbox(r["content"], value=done, key=f"wk_done_{r['id']}")
                    if new_done != done:
                        with get_conn() as conn:
                            conn.execute("UPDATE growth_notes SET is_completed=? WHERE id=?", (1 if new_done else 0, r["id"]))
                        bump_db_version()
                        st.rerun()

            # 선택 UI
            selected = st.multiselect(
                "이번 주 목표로 승격할 실천 제안 선택(최대 3개)",
                options=candidates["label"].tolist(),
                default=[],
            )

            if len(selected) > 3:
                st.error("최대 3개까지만 선택할 수 있어요.")
            else:
                if st.button("이번 주 목표로 저장", type="primary"):
                    # 현재 주 목표 3개 제한 강제
                    with get_conn() as conn:
                        existing = conn.execute(
                            "SELECT COUNT(*) FROM growth_notes WHERE is_weekly_goal=1 AND goal_week=?",
                            (current_week,),
                        ).fetchone()[0]
                        if existing + len(selected) > 3:
                            st.error("이미 저장된 목표가 있어요. 합쳐서 최대 3개까지만 가능합니다.")
                        else:
                            # candidates에서 content로 id 찾아 업데이트
                            for content in selected:
                                # content 매칭되는 행(아무거나 1개)
                                row = conn.execute(
                                    "SELECT id FROM growth_notes WHERE note_type='practice' AND content=? LIMIT 1",
                                    (content,),
                                ).fetchone()
                                if row:
                                    conn.execute(
                                        "UPDATE growth_notes SET is_weekly_goal=1, goal_week=? WHERE id=?",
                                        (current_week, row[0]),
                                    )
                            conn.commit()
                    bump_db_version()
                    st.success("이번 주 목표로 저장했어요.")
                    st.rerun()

        with right:
            # 간단한 진행률
            if not weekly.empty:
                total = len(weekly)
                done = int((weekly["is_completed"].fillna(0).astype(int) == 1).sum())
                pct = int(round(done / max(1, total) * 100))
                st.metric("이번 주 진행률", f"{pct}%", f"{done}/{total} 완료")
            else:
                st.metric("이번 주 진행률", "0%", "0/0")

        st.divider()

        # --- 6-1 정체성 문장 ---
        st.subheader("🪞 나의 정체성 문장 (MetaTone)")
        ident = get_identity()
        can_generate = len(df_entries) >= 20

        col1, col2 = st.columns([3, 1])
        with col2:
            gen_btn = st.button("정체성 문장 생성/갱신", disabled=(not can_generate or not api_key), use_container_width=True)

        if gen_btn:
            with st.spinner("패턴을 요약해 정체성 문장을 생성 중..."):
                text, err = generate_identity_statement(api_key, df_entries)
            if err:
                st.error(err)
            else:
                # 핀 유지
                set_identity(text, ident["is_pinned"])
                st.success("정체성 문장을 업데이트했어요.")
                st.rerun()

        pinned = st.checkbox("고정(핀)", value=bool(ident["is_pinned"]))
        content = st.text_area(
            "정체성 문장(직접 수정 가능)",
            value=ident["content"] or ("기록이 20개 이상이면 생성할 수 있어요." if not can_generate else ""),
            height=80,
        )
        save = st.button("저장")
        if save:
            set_identity(content, int(pinned))
            st.success("저장했어요.")
            st.rerun()

        if ident["updated_at"]:
            st.caption(f"최근 업데이트: {ident['updated_at']}")

    # ================
    # 페이지 3: 회고 보관소 (2+2 메모 관리 + 메타인지 질문 표시)
    # ================
    elif menu == "회고 보관소":
        st.header("📚 누적 기록 및 성장 관리")
        if df_entries.empty:
            st.info("아직 기록이 없어요. '새 기록 작성'에서 기록을 남겨보세요!")
            return

        # notes를 entry_id로 빠르게 접근하기 위해 dict 구성
        notes_by_entry = {}
        if not df_notes.empty:
            for _, n in df_notes.iterrows():
                notes_by_entry.setdefault(n["entry_id"], []).append(n)

        current_week = iso_week_key(date.today())

        for _, row in df_entries.iterrows():
            title = f"📅 {row['entry_date']} | {row['category']} | 템플릿: {row.get('template_name','')} | 핵심: {row['top_skill']}"
            with st.expander(title):
                st.write(f"**기분(1~10)**: {row.get('mood', '')}")
                st.write(f"**[행동]** {row['behavior']}")
                st.write(f"**[감정]** {row['emotion']}")
                st.write(f"**[결과]** {row['result']}")

                analysis = safe_json_load(row.get("analysis_json"))
                meta = (analysis.get("metacognition") or {}) if isinstance(analysis, dict) else {}

                # 메타인지: 맞춤형 대안행동/성찰 질문
                if meta:
                    st.divider()
                    st.subheader("🧠 메타인지 질문(대안 행동 시뮬레이션)")
                    cfs = meta.get("counterfactuals") or []
                    for i, cf in enumerate(cfs[:2], start=1):
                        st.markdown(f"**대안 행동 {i}**: {cf.get('alt_action','')}")
                        st.write(f"가능한 변화(가정): {cf.get('expected_outcome_change','')}")
                        st.info(cf.get("reflection_question", ""))

                    if meta.get("blind_spot"):
                        st.write(f"**놓친 관점(Blind spot)**: {meta.get('blind_spot')}")
                    if meta.get("next_signal"):
                        st.write(f"**다음에 감지할 신호(Next signal)**: {meta.get('next_signal')}")

                st.divider()
                st.subheader("🌱 성장 루틴 (Top 스킬 2+2)")

                entry_notes = notes_by_entry.get(row["id"], [])
                if not entry_notes:
                    st.info("저장된 성장 메모가 없습니다.")
                    continue

                df_en = pd.DataFrame(entry_notes)

                col_a, col_b = st.columns(2)

                # Practices: 완료/이번주목표 승격
                with col_a:
                    st.write("**실천 제안 (Practices)**")
                    practices = df_en[df_en["note_type"] == "practice"]
                    if practices.empty:
                        st.caption("실천 제안이 없어요.")
                    else:
                        for _, n in practices.iterrows():
                            done = bool(int(n.get("is_completed") or 0))
                            new_done = st.checkbox(n["content"], key=f"p_{n['id']}", value=done)
                            if new_done != done:
                                with get_conn() as conn:
                                    conn.execute(
                                        "UPDATE growth_notes SET is_completed=? WHERE id=?",
                                        (1 if new_done else 0, n["id"]),
                                    )
                                bump_db_version()
                                st.rerun()

                            # 3-1: 이번 주 목표로 승격(최대 3개 제한)
                            is_goal = bool(int(n.get("is_weekly_goal") or 0)) and (n.get("goal_week") == current_week)
                            goal_toggle = st.checkbox(
                                "이번 주 목표로 고정",
                                key=f"goal_{n['id']}",
                                value=is_goal,
                            )
                            if goal_toggle != is_goal:
                                with get_conn() as conn:
                                    if goal_toggle:
                                        cnt = conn.execute(
                                            "SELECT COUNT(*) FROM growth_notes WHERE is_weekly_goal=1 AND goal_week=?",
                                            (current_week,),
                                        ).fetchone()[0]
                                        if cnt >= 3:
                                            st.error("이번 주 목표는 최대 3개까지만 가능해요.")
                                        else:
                                            conn.execute(
                                                "UPDATE growth_notes SET is_weekly_goal=1, goal_week=? WHERE id=?",
                                                (current_week, n["id"]),
                                            )
                                    else:
                                        conn.execute(
                                            "UPDATE growth_notes SET is_weekly_goal=0, goal_week=NULL WHERE id=?",
                                            (n["id"],),
                                        )
                                    conn.commit()
                                bump_db_version()
                                st.rerun()

                # Questions: 답변 메모 + 저장
                with col_b:
                    st.write("**성찰 질문 (Questions)**")
                    questions = df_en[df_en["note_type"] == "question"]
                    if questions.empty:
                        st.caption("성찰 질문이 없어요.")
                    else:
                        for _, n in questions.iterrows():
                            st.info(n["content"])
                            memo = st.text_area(
                                "답변 메모",
                                key=f"memo_{n['id']}",
                                value=n.get("user_memo") or "",
                                height=80,
                            )
                            if st.button("메모 저장", key=f"btn_{n['id']}"):
                                with get_conn() as conn:
                                    conn.execute(
                                        "UPDATE growth_notes SET user_memo=? WHERE id=?",
                                        (memo, n["id"]),
                                    )
                                bump_db_version()
                                st.success("메모가 저장되었습니다.")
                                st.rerun()

if __name__ == "__main__":
    main()
