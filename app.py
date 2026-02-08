import os
import re
import json
import uuid
import time
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# OpenAI client (requires openai>=1.0.0)
# ----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================
# App Config
# ============================
APP_TITLE = "MetaTone — 성장 기록 기반 소프트스킬 트래커 (MVP)"
DB_PATH = "metatone.db"

DEFAULT_MODEL = "gpt-4o-mini"
MODEL_OPTIONS = [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o"]

SOFT_SKILLS = ["문제해결", "의사소통", "협업", "리더십", "자기관리/회복탄력성", "학습역량"]

CATEGORIES = ["학습(수업/자격증/독서)", "프로젝트", "리더십·동아리", "대외활동", "관계·협업", "생활·루틴"]

ANALYSIS_ENGINES = ["무료(로컬) — 규칙/TF-IDF", "LLM(OpenAI)"]
DEFAULT_ENGINE = ANALYSIS_ENGINES[0]


# ============================
# DB Utilities
# (기존 스키마 호환: tags/title 컬럼은 남겨두되 MetaTone에서는 사용하지 않음)
# ============================
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)


def init_db() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            category TEXT,
            tags TEXT,
            title TEXT,
            raw_text TEXT NOT NULL,
            artifacts TEXT,
            analysis_json TEXT
        )
        """)
        conn.commit()


def insert_entry(entry: Dict[str, Any]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entries (id, created_at, entry_date, category, tags, title, raw_text, artifacts, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["id"],
            entry["created_at"],
            entry["entry_date"],
            entry.get("category"),
            json.dumps(entry.get("tags", []), ensure_ascii=False),  # MetaTone 미사용(빈 리스트)
            entry.get("title"),  # MetaTone 미사용(None)
            entry["raw_text"],
            json.dumps(entry.get("artifacts", []), ensure_ascii=False),
            json.dumps(entry.get("analysis", {}), ensure_ascii=False)
        ))
        conn.commit()


def update_entry_analysis(entry_id: str, analysis: Dict[str, Any]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE entries SET analysis_json = ? WHERE id = ?
        """, (json.dumps(analysis, ensure_ascii=False), entry_id))
        conn.commit()


def fetch_entries(limit: int = 500) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM entries ORDER BY entry_date DESC, created_at DESC LIMIT ?
        """, conn, params=(limit,))

    def safe_json(x, default):
        if not x:
            return default
        try:
            return json.loads(x)
        except Exception:
            return default

    df["tags_parsed"] = df["tags"].apply(lambda x: safe_json(x, default=[]))
    df["artifacts_parsed"] = df["artifacts"].apply(lambda x: safe_json(x, default=[]))
    df["analysis_parsed"] = df["analysis_json"].apply(lambda x: safe_json(x, default={}))
    return df


def fetch_entry_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
        row = cur.fetchone()

    if not row:
        return None

    cols = ["id", "created_at", "entry_date", "category", "tags", "title", "raw_text", "artifacts", "analysis_json"]
    d = dict(zip(cols, row))

    for k, default in [("tags", []), ("artifacts", []), ("analysis_json", {})]:
        try:
            d[k] = json.loads(d[k]) if d[k] else default
        except Exception:
            d[k] = default
    return d


def delete_entry(entry_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        conn.commit()


# ============================
# Text Similarity (local) + caching
# ============================
@st.cache_resource(show_spinner=False)
def build_similarity_index_cached(corpus: Tuple[str, ...]) -> Tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
    X = vectorizer.fit_transform(list(corpus))
    return vectorizer, X


def get_similar_entries(df: pd.DataFrame, target_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    if top_k <= 0 or df.empty:
        return []
    corpus_list = df["raw_text"].fillna("").tolist()
    if len(corpus_list) < 2:
        return []

    corpus = tuple(corpus_list)
    vectorizer, X = build_similarity_index_cached(corpus)
    try:
        x_target = vectorizer.transform([target_text])
        sims = cosine_similarity(x_target, X).flatten()
    except Exception:
        return []

    pairs = list(zip(df["id"].tolist(), sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


# ============================
# OpenAI helpers
# ============================
def get_openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않거나 버전이 너무 낮습니다. `pip install -U openai` 해주세요.")
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있습니다.")
    return OpenAI(api_key=api_key)


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> str:
    s = strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s


def _json_repair_minimal(s: str) -> str:
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def robust_json_loads(s: str) -> Dict[str, Any]:
    raw = _extract_first_json_object(s)
    try:
        out = json.loads(raw)
    except Exception:
        out = json.loads(_json_repair_minimal(raw))
    if not isinstance(out, dict):
        raise ValueError("JSON 최상위가 객체(dict)가 아닙니다.")
    return out


# ============================
# MetaTone: 분석 스키마(패턴 요약 없음)
# ============================
SKILL_CONCEPTS = {
    "문제해결": "문제를 정의하고 원인을 파악해 실행 가능한 대안을 만들고 검증하는 역량",
    "의사소통": "상대의 이해를 기준으로 정보를 구조화·전달하고 합의를 이끌어내는 역량",
    "협업": "역할·의존성을 맞추고 상호 신뢰를 바탕으로 성과를 함께 만드는 역량",
    "리더십": "방향을 제시하고 의사결정을 돕고 구성원이 움직이게 만드는 영향력",
    "자기관리/회복탄력성": "에너지·감정·시간을 관리하며 압박 속에서도 회복하고 지속하는 역량",
    "학습역량": "학습 목표를 세우고 피드백을 통해 지식을 내 것으로 만드는 역량",
}


def analyze_entry_with_openai(
    api_key: str,
    model: str,
    entry: Dict[str, Any],
    related_summaries: List[Dict[str, Any]],
    output_mode: str = "portfolio"
) -> Dict[str, Any]:
    """
    output_mode:
      - "analysis_only": 상황분석 + 스킬 + 성장계획 + 개념설명
      - "portfolio": 위 + STAR/면접스크립트(선택 유지)
    """
    client = get_openai_client(api_key)

    persona = (
        "당신은 'MetaTone'의 코치입니다. "
        "사용자의 기록을 기반으로 상황을 요약·분석하고, 그 상황에서 쌓인 소프트스킬을 증거(원문 발췌)로 연결합니다. "
        "과장·미사여구 금지, 단정 금지, 근거 중심."
    )

    # related summaries: keep short
    related_block = []
    for rs in (related_summaries or [])[:5]:
        related_block.append({
            "id": rs.get("id"),
            "date": rs.get("entry_date"),
            "one_liner": rs.get("one_liner", ""),
            "skills": rs.get("skills", []),
        })

    want_portfolio = (output_mode == "portfolio")

    output_contract: Dict[str, Any] = {
        "meta": {
            "entry_id": entry["id"],
            "entry_date": entry["entry_date"],
            "category": entry.get("category") or ""
        },
        "situation_analysis": {
            "summary": "2~3문장 상황 요약",
            "challenge": "핵심 난점/제약 1~2개",
            "your_actions": "본인이 실제로 한 행동(구체) 2~4개",
            "outcome": "결과/변화(가능하면 관찰 가능한 표현)",
            "learning": "배운 점 1~2문장"
        },
        "soft_skills": [
            {
                "name": "협업",
                "confidence": 0.0,
                "evidence_quotes": ["원문 그대로 짧게 1~2개(각 80자 이내)"],
                "why_it_counts": "왜 이 역량인지 한 문장",
                "concept": "이 역량의 개념 설명(1문장)"
            }
        ],
        "growth_plan": {
            "what_to_develop_next": ["다음에 발전시키면 좋은 역량 1~2개(소프트스킬 이름)"],
            "how_to_practice": ["내일/다음주에 할 수 있는 연습/루틴 2~4개(행동형)"],
            "reflection_questions": ["다음 기록에 포함하면 좋은 질문 2~3개"]
        }
    }

    if want_portfolio:
        output_contract["portfolio"] = {
            "star_paragraph": "4~6문장",
            "interview_script_1min": "",
            "keywords": []
        }

    user_payload = {
        "entry": {
            "entry_date": entry["entry_date"],
            "category": entry.get("category"),
            "raw_text": entry["raw_text"],
            "artifacts": entry.get("artifacts") or []
        },
        "related_entries_hint": related_block,
        "soft_skill_candidates": SOFT_SKILLS,
        "skill_concepts": SKILL_CONCEPTS,
        "output_contract_example": output_contract
    }

    instructions = (
        "규칙:\n"
        "1) 반드시 JSON만 출력 (마크다운/코드펜스/설명문 금지)\n"
        "2) soft_skills는 1~3개만 선택, confidence는 0~1 숫자\n"
        "3) evidence_quotes는 원문 그대로, 최대 2개, 각 80자 이내\n"
        "4) concept는 제공된 skill_concepts를 참고하되, 문장 1개로 간단히\n"
        "5) 성장계획은 '구체적 행동' 위주로\n"
        "6) 과장/미사여구/단정 금지\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            {"role": "user", "content": instructions},
        ]
    )
    return robust_json_loads(resp.choices[0].message.content or "")


# ============================
# 무료(로컬) 분석: MetaTone 포맷
# ============================
def analyze_entry_local(
    entry: Dict[str, Any],
    related_summaries: List[Dict[str, Any]],
    output_mode: str = "portfolio"
) -> Dict[str, Any]:
    text = (entry.get("raw_text") or "").strip()

    lines = [l.strip() for l in re.split(r"[\n\r]+", text) if l.strip()]
    first = lines[0] if lines else ""
    second = lines[1] if len(lines) >= 2 else ""
    last = lines[-1] if lines else ""

    # 상황 요약(보수적으로)
    summary = first[:160] if first else "기록을 더 구체적으로 쓰면 상황 요약이 선명해집니다."
    challenge = ""
    for kw in ["어려", "문제", "갈등", "압박", "실수", "리스크", "막혔", "힘들"]:
        if kw in text:
            challenge = "기록에서 난점/제약(문제·압박·갈등 등)이 드러납니다."
            break
    if not challenge:
        challenge = "난점/제약을 한 문장으로 덧붙이면 분석 정확도가 올라갑니다."

    your_actions = []
    for kw in ["했다", "진행", "정리", "공유", "설명", "조율", "확인", "개선", "시도", "결정", "분석"]:
        for l in lines:
            if kw in l and l not in your_actions:
                your_actions.append(l[:120])
            if len(your_actions) >= 4:
                break
        if len(your_actions) >= 4:
            break
    if not your_actions:
        your_actions = ["내가 실제로 한 행동(예: 조율/정리/분석/공유)을 2~3개 문장으로 적어보세요."]

    outcome = ""
    for kw in ["결과", "완료", "성공", "개선", "변화", "달성", "줄었", "늘었", "좋아졌"]:
        if kw in text:
            outcome = "기록에서 결과/변화가 언급됩니다. (가능하면 수치·관찰로 보강 추천)"
            break
    if not outcome:
        outcome = second[:160] if second else "결과(무엇이 달라졌는지)를 한 문장으로 추가해보세요."

    learning = ""
    for kw in ["배웠", "깨달", "다음", "개선", "반성", "느꼈", "알게", "교훈", "성찰"]:
        if kw in text:
            learning = last[:160]
            break
    if not learning:
        learning = "배운 점(다음에 적용할 기준/원칙)을 1문장으로 남기면 누적 트래킹이 쉬워져요."

    # 스킬 룰
    skill_rules = {
        "문제해결": ["문제", "원인", "해결", "분석", "디버깅", "구조", "대안", "개선"],
        "의사소통": ["설명", "공유", "발표", "설득", "정리", "문서", "피드백", "합의"],
        "협업": ["팀", "협업", "조율", "역할", "회의", "갈등", "동료", "함께"],
        "리더십": ["주도", "리드", "결정", "방향", "가이드", "코칭", "책임", "기획"],
        "자기관리/회복탄력성": ["시간", "루틴", "회복", "스트레스", "압박", "우선순위", "지속", "컨디션"],
        "학습역량": ["공부", "학습", "정리", "복습", "실험", "개념", "강의", "독서", "연습"],
    }
    scores = {k: 0 for k in SOFT_SKILLS}
    for sk, kws in skill_rules.items():
        for kw in kws:
            if kw in text:
                scores[sk] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = [(k, v) for k, v in ranked if v > 0][:3]
    if not picked:
        picked = [("학습역량", 1)]

    # 근거 문장(보수적으로)
    sentences = re.split(r"[.!?\n]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    evidence = {k: [] for k in SOFT_SKILLS}
    for sk, _v in picked:
        kws = skill_rules.get(sk, [])
        for s in sentences:
            if any(kw in s for kw in kws):
                evidence[sk].append(s[:80])
                if len(evidence[sk]) >= 2:
                    break

    max_score = max(v for _, v in picked) if picked else 1
    soft_skills = []
    for sk, v in picked:
        conf = 0.4 + 0.6 * (v / max_score) if max_score > 0 else 0.5
        soft_skills.append({
            "name": sk,
            "confidence": round(min(max(conf, 0.0), 1.0), 2),
            "evidence_quotes": evidence[sk][:2] if evidence[sk] else (sentences[:1] if sentences else []),
            "why_it_counts": "원문에서 해당 행동 단서(키워드/행동 표현)가 보여서 이 역량을 쌓은 것으로 추정했습니다. (무료 로컬 분석)",
            "concept": SKILL_CONCEPTS.get(sk, "")
        })

    # 성장 플랜(다음 역량): 현재 선택된 것 중 confidence 낮은 것 보완 + 인접 스킬 추천(단순)
    next_candidates = [s["name"] for s in soft_skills[1:3]] or [soft_skills[0]["name"]]
    next_candidates = list(dict.fromkeys(next_candidates))[:2]

    how_to_practice = [
        "기록에 '내가 선택한 기준(우선순위/근거/합의 방식)'을 1문장으로 남기기",
        "결과를 관찰 가능한 표현(전/후 변화, 시간/횟수/품질)로 적기",
        "상대와 상호작용이 있었다면 '내가 한 말/요청/정리'를 한 문장으로 남기기",
    ]
    reflection_questions = [
        "내가 한 선택의 기준은 무엇이었나?",
        "다음에 같은 상황이면 무엇을 유지/변경할까?",
        "결과를 수치나 관찰로 표현한다면 무엇이 될까?",
    ]

    out: Dict[str, Any] = {
        "meta": {
            "entry_id": entry["id"],
            "entry_date": entry["entry_date"],
            "category": entry.get("category") or ""
        },
        "situation_analysis": {
            "summary": summary,
            "challenge": challenge,
            "your_actions": your_actions,
            "outcome": outcome,
            "learning": learning
        },
        "soft_skills": soft_skills,
        "growth_plan": {
            "what_to_develop_next": next_candidates,
            "how_to_practice": how_to_practice,
            "reflection_questions": reflection_questions
        }
    }

    if output_mode == "portfolio":
        # 옵션 유지(원하면 UI에서 숨겨도 됨). 패턴요약은 없음.
        star_parts = [
            f"상황: {summary}",
            f"난점: {challenge}",
            f"행동: {', '.join(your_actions[:3])}" if your_actions else "행동: (기록 보강 필요)",
            f"결과: {outcome}",
            f"배움: {learning}",
        ]
        out["portfolio"] = {
            "star_paragraph": " ".join([p for p in star_parts if p]),
            "interview_script_1min": " ".join([star_parts[0], star_parts[2], star_parts[3], star_parts[4]]),
            "keywords": [s["name"] for s in soft_skills]
        }

    return out


# ============================
# 누적 스킬 계산/표현
# ============================
def compute_skill_totals(df: pd.DataFrame) -> Dict[str, int]:
    totals = {s: 0 for s in SOFT_SKILLS}
    if df.empty:
        return totals
    for an in df["analysis_parsed"].tolist():
        if not isinstance(an, dict):
            continue
        skills = an.get("soft_skills") or []
        if not isinstance(skills, list):
            continue
        for sk in skills:
            if isinstance(sk, dict):
                name = sk.get("name")
                if name in totals:
                    totals[name] += 1
    return totals


def render_skill_totals(totals: Dict[str, int]) -> None:
    st.subheader("📈 소프트스킬 누적(범주별)")
    cols = st.columns(3)
    items = list(totals.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.metric(label=k, value=v)

    # 표로도 제공
    df_tot = pd.DataFrame([{"soft_skill": k, "count": v} for k, v in totals.items()]).sort_values("count", ascending=False)
    st.dataframe(df_tot, use_container_width=True, hide_index=True)


# ============================
# UI Helpers (MetaTone 전용 출력)
# ============================
def format_analysis_block(analysis: Dict[str, Any]) -> None:
    if not analysis or not isinstance(analysis, dict):
        st.info("아직 분석 결과가 없습니다.")
        return

    st.subheader("🧠 상황 분석")
    s = analysis.get("situation_analysis", {}) or {}
    st.markdown("**요약**")
    st.write(s.get("summary", ""))
    st.markdown("**난점/제약**")
    st.write(s.get("challenge", ""))
    st.markdown("**내 행동**")
    actions = s.get("your_actions") or []
    if isinstance(actions, list):
        for a in actions:
            st.write(f"- {a}")
    else:
        st.write(actions)
    st.markdown("**결과/변화**")
    st.write(s.get("outcome", ""))
    st.markdown("**배움**")
    st.write(s.get("learning", ""))

    st.subheader("🎯 오늘 쌓은 소프트 스킬")
    skills = analysis.get("soft_skills", []) or []
    if not skills:
        st.write("선정된 역량이 없습니다.")
    else:
        for sk in skills:
            if not isinstance(sk, dict):
                continue
            name = sk.get("name", "")
            conf = sk.get("confidence", 0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            st.markdown(f"- **{name}** (confidence: {conf:.2f})")

            ev = sk.get("evidence_quotes", []) or []
            if isinstance(ev, list) and ev:
                for q in ev[:2]:
                    st.caption(f"근거: “{q}”")

            st.write(sk.get("why_it_counts", ""))

            concept = sk.get("concept") or SKILL_CONCEPTS.get(name, "")
            if concept:
                st.info(f"개념: {concept}")

    st.subheader("🚀 앞으로 발전시키면 좋은 역량")
    gp = analysis.get("growth_plan", {}) or {}
    nxt = gp.get("what_to_develop_next") or []
    if isinstance(nxt, list) and nxt:
        st.write("다음 역량(추천): " + ", ".join(nxt))
    practice = gp.get("how_to_practice") or []
    if practice:
        st.markdown("**연습/루틴 제안**")
        for p in practice[:6]:
            st.write(f"- {p}")
    qs = gp.get("reflection_questions") or []
    if qs:
        st.markdown("**다음 기록에 도움이 되는 질문**")
        for q in qs[:6]:
            st.write(f"- {q}")

    port = analysis.get("portfolio")
    if isinstance(port, dict) and port.get("star_paragraph"):
        st.subheader("📝 (옵션) STAR/면접 스크립트")
        st.markdown("**STAR 문단**")
        st.write(port.get("star_paragraph", ""))
        st.markdown("**면접 1분 스크립트**")
        st.write(port.get("interview_script_1min", ""))
        st.markdown("**키워드**")
        st.write(", ".join(port.get("keywords", []) or []))


def summarize_for_related(df: pd.DataFrame) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        an = r.get("analysis_parsed") or {}
        one_liner = ""
        skills: List[str] = []
        try:
            s = (an.get("situation_analysis", {}) or {}) if isinstance(an, dict) else {}
            one_liner = (s.get("learning") or s.get("outcome") or s.get("summary") or "")[:80]
            soft = (an.get("soft_skills") or []) if isinstance(an, dict) else []
            skills = [x.get("name") for x in soft if isinstance(x, dict) and x.get("name")]
        except Exception:
            pass
        summaries.append({
            "id": r["id"],
            "entry_date": r["entry_date"],
            "one_liner": one_liner,
            "skills": skills
        })
    return summaries


# ============================
# Engine switch wrapper (LLM 실패 시 무료 fallback)
# ============================
def run_analysis_engine(
    engine: str,
    entry: Dict[str, Any],
    related: List[Dict[str, Any]],
    output_mode: str
) -> Dict[str, Any]:
    if engine.startswith("무료"):
        return analyze_entry_local(entry=entry, related_summaries=related, output_mode=output_mode)

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("LLM 분석을 선택했지만 API Key가 없어 무료(로컬) 분석으로 대체합니다.")
        return analyze_entry_local(entry=entry, related_summaries=related, output_mode=output_mode)

    try:
        return analyze_entry_with_openai(
            api_key=api_key,
            model=st.session_state.get("model", DEFAULT_MODEL),
            entry=entry,
            related_summaries=related,
            output_mode=output_mode
        )
    except Exception as e:
        st.warning(f"LLM 분석 실패 → 무료(로컬) 분석으로 대체합니다.\n\n사유: {e}")
        return analyze_entry_local(entry=entry, related_summaries=related, output_mode=output_mode)


# ============================
# Streamlit App
# ============================
def main():
    st.set_page_config(page_title="MetaTone", layout="wide")
    init_db()

    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (선택)",
        value=st.session_state.get("api_key", api_key_env),
        type="password"
    )
    st.session_state["api_key"] = (api_key_input or "").strip()

    current_model = st.session_state.get("model", DEFAULT_MODEL)
    if current_model not in MODEL_OPTIONS:
        current_model = DEFAULT_MODEL
    model_index = MODEL_OPTIONS.index(current_model)
    st.sidebar.selectbox("Model (LLM 모드에서만 사용)", options=MODEL_OPTIONS, index=model_index, key="model")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 분석 엔진")
    st.sidebar.selectbox("분석 방식", options=ANALYSIS_ENGINES, index=0, key="engine")
    st.sidebar.caption("무료(로컬)은 OpenAI 없이도 동작합니다. (쿼터/결제 이슈 없음)")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("페이지", ["✍️ 오늘의 기록 추가", "📚 기록 목록", "🧪 디버그/로그"])

    df = fetch_entries()
    totals = compute_skill_totals(df)

    st.title(APP_TITLE)
    st.caption("기록 본문에서 상황을 분석하고, 쌓인 소프트스킬과 다음 성장 방향을 정리합니다. (누적 트래킹 포함)")

    # 누적은 모든 페이지 상단에 노출(원하면 특정 페이지만 노출로 바꿀 수 있어요)
    render_skill_totals(totals)
    st.markdown("---")

    if page == "✍️ 오늘의 기록 추가":
        render_new_entry(df)
    elif page == "📚 기록 목록":
        render_history(df)
    else:
        render_debug(df)


# ============================
# Page: New Entry (요구사항 반영)
# 구성: 날짜, 증거/자료 링크(선택), 카테고리(선택/직접입력), 기록본문(필수)
# 분석 옵션은 그대로 유지
# ============================
def render_new_entry(df: pd.DataFrame):
    st.subheader("✍️ 오늘의 기록 추가")

    col1, col2 = st.columns([1, 1])
    with col1:
        entry_date = st.date_input("날짜", value=date.today())

        cat_choice = st.selectbox(
            "카테고리(선택)",
            options=["(선택 안 함)"] + CATEGORIES + ["(직접 입력)"],
            index=0
        )
        cat_custom = ""
        if cat_choice == "(직접 입력)":
            cat_custom = st.text_input("카테고리 직접 입력", placeholder="예: 인턴/현장실습, 개인 프로젝트, 취미 활동 등")
        category = None
        if cat_choice == "(선택 안 함)":
            category = None
        elif cat_choice == "(직접 입력)":
            category = cat_custom.strip() if cat_custom.strip() else None
        else:
            category = cat_choice

    with col2:
        artifacts = st.text_area(
            "증거/자료 링크(선택) — 줄바꿈으로 여러 개",
            placeholder="예: Notion 링크, Google Doc, GitHub, 발표자료 URL 등"
        )
        artifacts_list = [x.strip() for x in (artifacts or "").splitlines() if x.strip()]

    raw_text = st.text_area(
        "기록 본문(필수)",
        height=240,
        placeholder="오늘의 상황/내 역할/내가 한 행동/결과/배움 중심으로 적어주세요."
    )

    st.markdown("### 🔎 분석 옵션")
    do_analysis = st.checkbox("저장 후 분석 실행하기", value=True)
    output_mode_label = st.selectbox("산출물 범위", options=["포트폴리오까지(추천)", "분석만"], index=0)
    top_k = st.slider("유사 기록 추천(top-k)", min_value=0, max_value=10, value=5)

    if st.button("✅ 저장", type="primary"):
        if not (raw_text or "").strip():
            st.error("기록 본문은 필수입니다.")
            return

        entry_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat(timespec="seconds")
        entry = {
            "id": entry_id,
            "created_at": created_at,
            "entry_date": entry_date.isoformat(),
            "category": category,
            "tags": [],            # MetaTone 미사용
            "title": None,         # MetaTone 미사용
            "raw_text": raw_text.strip(),
            "artifacts": artifacts_list,
            "analysis": {}
        }
        insert_entry(entry)
        st.success("기록을 저장했습니다.")

        if not do_analysis:
            return

        # 유사 기록 힌트(선택)
        related: List[Dict[str, Any]] = []
        if top_k > 0 and not df.empty:
            sims = get_similar_entries(df, entry["raw_text"], top_k=top_k)
            hint_rows: List[Dict[str, Any]] = []
            for rid, _score in sims:
                r = fetch_entry_by_id(rid)
                if r:
                    hint_rows.append(r)
            hint_df = pd.DataFrame(hint_rows) if hint_rows else pd.DataFrame()
            if not hint_df.empty:
                hint_df["analysis_parsed"] = hint_df["analysis_json"].apply(
                    lambda x: x if isinstance(x, dict) else (x or {})
                )
                related = summarize_for_related(hint_df)

        engine = st.session_state.get("engine", DEFAULT_ENGINE)
        output_mode = "portfolio" if output_mode_label.startswith("포트폴리오") else "analysis_only"

        with st.spinner("분석 중..."):
            analysis = run_analysis_engine(engine=engine, entry=entry, related=related, output_mode=output_mode)
            update_entry_analysis(entry_id, analysis)
            st.success("분석 완료! 아래 결과를 확인하세요.")
            format_analysis_block(analysis)


# ============================
# Page: History
# ============================
def render_history(df: pd.DataFrame):
    st.subheader("📚 기록 목록")
    if df.empty:
        st.info("아직 저장된 기록이 없습니다. '오늘의 기록 추가'에서 작성해보세요.")
        return

    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        cat = st.selectbox("카테고리 필터", options=["(전체)"] + CATEGORIES)
    with colf2:
        skill_filter = st.selectbox("소프트스킬 필터", options=["(전체)"] + SOFT_SKILLS)
    with colf3:
        q = st.text_input("검색(본문)", placeholder="예: 발표, 조율, 회복, 기준, 피드백...")

    filtered = df.copy()

    if cat != "(전체)":
        filtered = filtered[filtered["category"] == cat]

    if (q or "").strip():
        qq = q.strip().lower()
        filtered = filtered[filtered["raw_text"].str.lower().str.contains(qq, na=False)]

    if skill_filter != "(전체)":
        def has_skill(an):
            if not isinstance(an, dict):
                return False
            skills = an.get("soft_skills", []) or []
            return any((s.get("name") == skill_filter) for s in skills if isinstance(s, dict))
        filtered = filtered[filtered["analysis_parsed"].apply(has_skill)]

    st.caption(f"총 {len(filtered)}개")

    engine = st.session_state.get("engine", DEFAULT_ENGINE)

    for _, r in filtered.iterrows():
        entry_date = r.get("entry_date", "")
        category = r.get("category") or "—"
        an = r.get("analysis_parsed") or {}
        if not isinstance(an, dict):
            an = {}
        skills = [s.get("name") for s in (an.get("soft_skills") or []) if isinstance(s, dict) and s.get("name")]
        skill_text = ", ".join(skills) if skills else "—"

        with st.expander(f"{entry_date} · 카테고리: {category}  |  스킬: {skill_text}"):
            st.write(r["raw_text"])

            artifacts = r.get("artifacts_parsed") or []
            if artifacts:
                st.markdown("**증거/링크**")
                for a in artifacts:
                    st.write(f"- {a}")

            st.markdown("---")
            if an:
                format_analysis_block(an)
            else:
                st.info("아직 분석 결과가 없습니다. 아래 버튼으로 분석을 실행할 수 있어요.")

            colb1, colb2, colb3 = st.columns([1, 1, 1])
            with colb1:
                if st.button("🤖 이 기록 분석하기", key=f"an_{r['id']}"):
                    entry = fetch_entry_by_id(r["id"])
                    if not entry:
                        st.error("기록을 불러오지 못했습니다.")
                    else:
                        other = df[df["id"] != r["id"]].head(80)
                        related = summarize_for_related(other) if not other.empty else []
                        payload = {
                            "id": entry["id"],
                            "entry_date": entry["entry_date"],
                            "category": entry.get("category"),
                            "raw_text": entry["raw_text"],
                            "artifacts": entry.get("artifacts") or []
                        }
                        with st.spinner("분석 중..."):
                            analysis = run_analysis_engine(engine=engine, entry=payload, related=related, output_mode="analysis_only")
                            update_entry_analysis(r["id"], analysis)
                            st.success("분석 완료! 화면을 갱신합니다.")
                            st.rerun()
            with colb2:
                if st.button("🗑️ 삭제", key=f"del_{r['id']}"):
                    delete_entry(r["id"])
                    st.success("삭제했습니다. 화면을 갱신합니다.")
                    st.rerun()
            with colb3:
                port = (an.get("portfolio") or {}) if isinstance(an, dict) else {}
                if isinstance(port, dict) and port.get("star_paragraph"):
                    st.download_button(
                        "⬇️ STAR 문단 다운로드(txt)",
                        data=port["star_paragraph"],
                        file_name=f"STAR_{entry_date}_{r['id'][:6]}.txt"
                    )


# ============================
# Page: Debug
# ============================
def render_debug(df: pd.DataFrame):
    st.subheader("🧪 디버그/로그")
    st.write("현재 DB에 저장된 기록 개수:", len(df))

    st.markdown("### 최근 10개 기록 미리보기")
    if not df.empty:
        st.dataframe(df[["entry_date", "category"]].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("기록이 없습니다.")

    st.markdown("### 환경/설정")
    st.write({
        "engine": st.session_state.get("engine", DEFAULT_ENGINE),
        "has_api_key": bool(st.session_state.get("api_key")),
        "model": st.session_state.get("model", DEFAULT_MODEL),
        "db_path": DB_PATH,
        "journal_mode": "WAL (init_db에서 설정)",
    })

    st.info(
        "MetaTone에서는 패턴 요약을 제거하고, "
        "상황 분석 → 스킬 도출 → 성장 플랜 → 개념 설명 → 누적 트래킹 중심으로 구성했습니다."
    )


if __name__ == "__main__":
    main()
