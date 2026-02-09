# app.py
import os
import re
import json
import uuid
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Charts
import plotly.graph_objects as go

# Local similarity
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

SKILL_CONCEPTS = {
    "문제해결": "문제를 정의하고 원인을 파악해 실행 가능한 대안을 만들고 검증하는 역량",
    "의사소통": "상대의 이해를 기준으로 정보를 구조화·전달하고 합의를 이끌어내는 역량",
    "협업": "역할·의존성을 맞추고 상호 신뢰를 바탕으로 성과를 함께 만드는 역량",
    "리더십": "방향을 제시하고 의사결정을 돕고 구성원이 움직이게 만드는 영향력",
    "자기관리/회복탄력성": "에너지·감정·시간을 관리하며 압박 속에서도 회복하고 지속하는 역량",
    "학습역량": "학습 목표를 세우고 피드백을 통해 지식을 내 것으로 만드는 역량",
}

# 2+2 기본
PRACTICE_N = 2
QUESTION_N = 2
ALT_ACTION_N = 2  # "대안행동 2개"를 성장플랜에 포함(LLM/로컬 둘 다)


# ============================
# DB Utilities
# ============================
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)


def init_db() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # entries: 기존과 최대한 호환 (title/tags는 미사용이어도 유지)
        cur.execute(
            """
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
            """
        )

        # structured inputs: 행동/감정/결과 분리 저장
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entry_structured (
                entry_id TEXT PRIMARY KEY,
                actions_text TEXT,
                emotions_text TEXT,
                results_text TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        # notes: entry_id + skill_name 단위로 practice/question 각각 0..1 저장
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_notes (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                note_type TEXT NOT NULL, -- 'practice' | 'question' | 'alt_action'
                item_index INTEGER NOT NULL, -- 0..N
                item_text TEXT NOT NULL,
                memo_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(entry_id, skill_name, note_type, item_index)
            )
            """
        )

        # 체크박스(실행 여부) 저장
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checklist (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                item_type TEXT NOT NULL, -- 'practice' | 'alt_action'
                item_index INTEGER NOT NULL,
                item_text TEXT NOT NULL,
                is_done INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(entry_id, skill_name, item_type, item_index)
            )
            """
        )

        conn.commit()


def safe_json_loads(x: Any, default: Any) -> Any:
    if not x:
        return default
    try:
        return json.loads(x)
    except Exception:
        return default


def insert_entry(entry: Dict[str, Any]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO entries
            (id, created_at, entry_date, category, tags, title, raw_text, artifacts, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"],
                entry["created_at"],
                entry["entry_date"],
                entry.get("category"),
                json.dumps(entry.get("tags", []), ensure_ascii=False),
                entry.get("title"),
                entry["raw_text"],
                json.dumps(entry.get("artifacts", []), ensure_ascii=False),
                json.dumps(entry.get("analysis", {}), ensure_ascii=False),
            ),
        )
        conn.commit()


def upsert_structured(entry_id: str, actions_text: str, emotions_text: str, results_text: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO entry_structured (entry_id, actions_text, emotions_text, results_text, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                actions_text = excluded.actions_text,
                emotions_text = excluded.emotions_text,
                results_text = excluded.results_text,
                updated_at = excluded.updated_at
            """,
            (entry_id, actions_text or "", emotions_text or "", results_text or "", now),
        )
        conn.commit()


def fetch_structured(entry_id: str) -> Dict[str, str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT actions_text, emotions_text, results_text FROM entry_structured WHERE entry_id = ?",
            (entry_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"actions_text": "", "emotions_text": "", "results_text": ""}
    return {"actions_text": row[0] or "", "emotions_text": row[1] or "", "results_text": row[2] or ""}


def update_entry_analysis(entry_id: str, analysis: Dict[str, Any]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE entries SET analysis_json = ? WHERE id = ?",
            (json.dumps(analysis, ensure_ascii=False), entry_id),
        )
        conn.commit()


def delete_entry(entry_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        cur.execute("DELETE FROM skill_notes WHERE entry_id = ?", (entry_id,))
        cur.execute("DELETE FROM entry_structured WHERE entry_id = ?", (entry_id,))
        cur.execute("DELETE FROM checklist WHERE entry_id = ?", (entry_id,))
        conn.commit()


def fetch_entries(limit: int = 1000) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM entries ORDER BY entry_date DESC, created_at DESC LIMIT ?",
            conn,
            params=(limit,),
        )

    if df.empty:
        df["tags_parsed"] = []
        df["artifacts_parsed"] = []
        df["analysis_parsed"] = []
        return df

    df["tags_parsed"] = df["tags"].apply(lambda x: safe_json_loads(x, default=[]))
    df["artifacts_parsed"] = df["artifacts"].apply(lambda x: safe_json_loads(x, default=[]))
    df["analysis_parsed"] = df["analysis_json"].apply(lambda x: safe_json_loads(x, default={}))
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

    d["tags"] = safe_json_loads(d.get("tags"), default=[])
    d["artifacts"] = safe_json_loads(d.get("artifacts"), default=[])
    d["analysis_json"] = safe_json_loads(d.get("analysis_json"), default={})
    return d


# ============================
# Notes / Checklist
# ============================
def upsert_skill_note(
    entry_id: str,
    entry_date: str,
    skill_name: str,
    note_type: str,  # practice|question|alt_action
    item_index: int,
    item_text: str,
    memo_text: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO skill_notes
            (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id, skill_name, note_type, item_index) DO UPDATE SET
                item_text = excluded.item_text,
                memo_text = excluded.memo_text,
                updated_at = excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                entry_id,
                entry_date,
                skill_name,
                note_type,
                int(item_index),
                item_text or "",
                memo_text or "",
                now,
                now,
            ),
        )
        conn.commit()


def fetch_skill_notes_for_entry(entry_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE entry_id = ?
            ORDER BY skill_name, note_type, item_index
            """,
            (entry_id,),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "entry_id": r[0],
                "entry_date": r[1],
                "skill_name": r[2],
                "note_type": r[3],
                "item_index": int(r[4]),
                "item_text": r[5] or "",
                "memo_text": r[6] or "",
                "updated_at": r[7] or "",
            }
        )
    return out


def fetch_skill_notes_by_skill(skill_name: str, limit: int = 500) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE skill_name = ?
            ORDER BY entry_date DESC, updated_at DESC
            LIMIT ?
            """,
            (skill_name, limit),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "entry_id": r[0],
                "entry_date": r[1],
                "skill_name": r[2],
                "note_type": r[3],
                "item_index": int(r[4]),
                "item_text": r[5] or "",
                "memo_text": r[6] or "",
                "updated_at": r[7] or "",
            }
        )
    return out


def fetch_skill_notes_by_date(entry_date: str, limit: int = 500) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE entry_date = ?
            ORDER BY skill_name, note_type, item_index
            LIMIT ?
            """,
            (entry_date, limit),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "entry_id": r[0],
                "entry_date": r[1],
                "skill_name": r[2],
                "note_type": r[3],
                "item_index": int(r[4]),
                "item_text": r[5] or "",
                "memo_text": r[6] or "",
                "updated_at": r[7] or "",
            }
        )
    return out


def upsert_checklist(
    entry_id: str,
    entry_date: str,
    skill_name: str,
    item_type: str,  # practice|alt_action
    item_index: int,
    item_text: str,
    is_done: bool,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO checklist
            (id, entry_id, entry_date, skill_name, item_type, item_index, item_text, is_done, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id, skill_name, item_type, item_index) DO UPDATE SET
                item_text = excluded.item_text,
                is_done = excluded.is_done,
                updated_at = excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                entry_id,
                entry_date,
                skill_name,
                item_type,
                int(item_index),
                item_text or "",
                1 if is_done else 0,
                now,
            ),
        )
        conn.commit()


def fetch_checklist_for_entry(entry_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT entry_id, entry_date, skill_name, item_type, item_index, item_text, is_done, updated_at
            FROM checklist
            WHERE entry_id = ?
            ORDER BY skill_name, item_type, item_index
            """,
            (entry_id,),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "entry_id": r[0],
                "entry_date": r[1],
                "skill_name": r[2],
                "item_type": r[3],
                "item_index": int(r[4]),
                "item_text": r[5] or "",
                "is_done": bool(r[6]),
                "updated_at": r[7] or "",
            }
        )
    return out


def group_notes(notes: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[int, Dict[str, str]]]:
    out: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for n in notes:
        k = (n["skill_name"], n["note_type"])
        out.setdefault(k, {})
        out[k][int(n["item_index"])] = {"item_text": n.get("item_text", ""), "memo_text": n.get("memo_text", "")}
    return out


def group_checklist(items: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]:
    out: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for it in items:
        k = (it["skill_name"], it["item_type"])
        out.setdefault(k, {})
        out[k][int(it["item_index"])] = it
    return out


# ============================
# Similarity (local) + caching
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
# Robust JSON parsing (LLM)
# ============================
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
            return s[start : i + 1]
    return s


def _json_repair_minimal(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing comma
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
# Analysis engines
# - 요약/STAR 없음
# - 행동/배움
# - 스킬 1~3개
# - 성장플랜: top 스킬 기준 practices(2) + questions(2) + alt_actions(2)
# ============================
def get_openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않거나 버전이 너무 낮습니다. `pip install -U openai` 해주세요.")
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있습니다.")
    return OpenAI(api_key=api_key)


def analyze_entry_with_openai(
    api_key: str,
    model: str,
    entry: Dict[str, Any],
    related_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    client = get_openai_client(api_key)

    persona = (
        "당신은 MetaTone의 코치입니다. "
        "사용자의 기록에서 '행동'과 '배움'을 뽑고, 그 기록에서 드러난 소프트스킬(1~3개)을 근거 인용과 함께 제시합니다. "
        "과장/미사여구/단정 금지. 근거 중심."
    )

    related_block: List[Dict[str, Any]] = []
    for rs in (related_summaries or [])[:5]:
        related_block.append(
            {
                "id": rs.get("id"),
                "date": rs.get("entry_date"),
                "one_liner": rs.get("one_liner", ""),
                "skills": rs.get("skills", []),
            }
        )

    output_contract: Dict[str, Any] = {
        "meta": {"entry_id": entry["id"], "entry_date": entry["entry_date"], "category": entry.get("category") or ""},
        "situation_analysis": {
            "actions": ["내가 실제로 한 행동 2~4개(짧은 문장)"],
            "learnings": ["배움 1~2개(짧은 문장)"],
        },
        "soft_skills": [
            {
                "name": "협업",
                "confidence": 0.0,
                "evidence_quotes": ["원문 그대로 1~2개(각 80자 이내)"],
                "why_it_counts": "왜 이 역량인지 1문장",
                "concept": "개념 1문장",
            }
        ],
        "growth_plan": {
            "top_skill": "협업",
            "practices": ["연습/루틴 1", "연습/루틴 2"],
            "questions": ["다음 기록 질문 1", "다음 기록 질문 2"],
            "alt_actions": ["대안행동 1", "대안행동 2"],
        },
    }

    user_payload = {
        "entry": {
            "entry_date": entry["entry_date"],
            "category": entry.get("category"),
            "raw_text": entry["raw_text"],
            "artifacts": entry.get("artifacts") or [],
            "structured": entry.get("structured") or {},
        },
        "related_entries_hint": related_block,
        "soft_skill_candidates": SOFT_SKILLS,
        "skill_concepts": SKILL_CONCEPTS,
        "output_contract_example": output_contract,
        "constraints": {"practice_n": PRACTICE_N, "question_n": QUESTION_N, "alt_action_n": ALT_ACTION_N},
    }

    instructions = (
        "규칙:\n"
        "1) 반드시 JSON만 출력(마크다운/코드펜스/설명문 금지)\n"
        "2) soft_skills는 1~3개, confidence는 0~1 숫자\n"
        "3) evidence_quotes는 원문 그대로 최대 2개, 각 80자 이내\n"
        "4) situation_analysis는 actions/learnings만 (요약 금지)\n"
        f"5) growth_plan.practices는 정확히 {PRACTICE_N}개, questions는 정확히 {QUESTION_N}개, alt_actions는 정확히 {ALT_ACTION_N}개\n"
        "6) growth_plan.top_skill은 soft_skills 중 confidence가 가장 높은 스킬명\n"
        "7) concept는 skill_concepts를 참고해 1문장으로 간단히\n"
        "8) 과장/미사여구/단정 금지\n"
        "9) alt_actions는 '당시 갈등/어려움 상황에서 다른 선택을 했다면?' 관점으로, 구체 행동 2개를 제시\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "user", "content": instructions},
            ],
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenAI 호출 실패: {e}\n\n"
            f"점검:\n- API Key 유효 여부\n- 모델({model}) 접근 권한/이름\n- 사용량/쿼터/결제 상태"
        )

    out = robust_json_loads(resp.choices[0].message.content or "")
    return out


def analyze_entry_local(entry: Dict[str, Any], related_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = (entry.get("raw_text") or "").strip()

    # 행동/배움: 보수적 추출
    lines = [l.strip() for l in re.split(r"[\n\r]+", text) if l.strip()]
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]

    action_markers = ["했다", "함", "진행", "정리", "공유", "설명", "조율", "확인", "개선", "시도", "결정", "분석", "제안", "요청"]
    actions: List[str] = []
    for l in lines:
        if any(m in l for m in action_markers):
            actions.append(l[:140])
        if len(actions) >= 4:
            break
    if not actions:
        actions = [sentences[0][:140]] if sentences else ["(행동) 내가 실제로 한 일을 2~3문장으로 적어보세요."]

    learning_markers = ["배웠", "깨달", "다음", "개선", "반성", "느꼈", "알게", "교훈", "성찰"]
    learnings: List[str] = []
    for l in reversed(lines):
        if any(m in l for m in learning_markers):
            learnings.append(l[:140])
        if len(learnings) >= 2:
            break
    learnings = list(reversed(learnings))
    if not learnings:
        learnings = ["(배움) 오늘 얻은 교훈/다음 기준을 1문장으로 남겨보세요."]

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

    evidence: Dict[str, List[str]] = {k: [] for k in SOFT_SKILLS}
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
        soft_skills.append(
            {
                "name": sk,
                "confidence": round(min(max(conf, 0.0), 1.0), 2),
                "evidence_quotes": evidence[sk][:2] if evidence[sk] else (sentences[:1] if sentences else []),
                "why_it_counts": "원문에서 해당 행동 단서(키워드/표현)가 보여 이 역량이 드러난 것으로 추정했습니다. (무료 로컬 분석)",
                "concept": SKILL_CONCEPTS.get(sk, ""),
            }
        )

    top_skill = soft_skills[0]["name"]

    practices = [
        "다음 기록에서 '내가 선택한 기준(우선순위/근거)'을 1문장으로 남기기",
        "결과를 관찰 가능한 표현(전/후 변화, 시간/횟수/품질)로 적기",
    ][:PRACTICE_N]

    questions = [
        "내가 한 선택의 기준은 무엇이었나?",
        "다음에 같은 상황이면 무엇을 유지/변경할까?",
    ][:QUESTION_N]

    alt_actions = [
        "갈등/어려움 상황에서 먼저 상대의 요구·우려를 2문장으로 요약해 확인하기",
        "결정 전에 '대안 2개 + 각각의 리스크 1개'를 적고 팀과 5분만 공유하기",
    ][:ALT_ACTION_N]

    return {
        "meta": {"entry_id": entry["id"], "entry_date": entry["entry_date"], "category": entry.get("category") or ""},
        "situation_analysis": {"actions": actions[:4], "learnings": learnings[:2]},
        "soft_skills": soft_skills,
        "growth_plan": {"top_skill": top_skill, "practices": practices, "questions": questions, "alt_actions": alt_actions},
    }


def run_analysis_engine(engine: str, entry: Dict[str, Any], related: List[Dict[str, Any]]) -> Dict[str, Any]:
    if engine.startswith("무료"):
        return analyze_entry_local(entry=entry, related_summaries=related)

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("LLM 분석을 선택했지만 API Key가 없어 무료(로컬) 분석으로 대체합니다.")
        return analyze_entry_local(entry=entry, related_summaries=related)

    try:
        return analyze_entry_with_openai(
            api_key=api_key,
            model=st.session_state.get("model", DEFAULT_MODEL),
            entry=entry,
            related_summaries=related,
        )
    except Exception as e:
        st.warning(f"LLM 분석 실패 → 무료(로컬) 분석으로 대체합니다.\n\n사유: {e}")
        return analyze_entry_local(entry=entry, related_summaries=related)


# ============================
# Aggregations / Related summaries
# ============================
def summarize_for_related(df: pd.DataFrame) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        an = r.get("analysis_parsed") or {}
        one_liner = ""
        skills: List[str] = []
        try:
            sa = (an.get("situation_analysis", {}) or {}) if isinstance(an, dict) else {}
            learnings = sa.get("learnings") or []
            if isinstance(learnings, list) and learnings:
                one_liner = (learnings[0] or "")[:80]
            soft = (an.get("soft_skills") or []) if isinstance(an, dict) else []
            skills = [x.get("name") for x in soft if isinstance(x, dict) and x.get("name")]
        except Exception:
            pass
        summaries.append({"id": r["id"], "entry_date": r["entry_date"], "one_liner": one_liner, "skills": skills})
    return summaries


def compute_skill_vector(df: pd.DataFrame) -> Dict[str, float]:
    """
    방사형 차트용 점수(0~1):
    기간 내 스킬 confidence 평균(없으면 0)
    """
    sums = {s: 0.0 for s in SOFT_SKILLS}
    cnts = {s: 0 for s in SOFT_SKILLS}
    if df.empty:
        return {s: 0.0 for s in SOFT_SKILLS}

    for an in df["analysis_parsed"].tolist():
        if not isinstance(an, dict):
            continue
        skills = an.get("soft_skills") or []
        if not isinstance(skills, list):
            continue
        for sk in skills:
            if not isinstance(sk, dict):
                continue
            name = sk.get("name")
            if name not in sums:
                continue
            try:
                conf = float(sk.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            sums[name] += max(0.0, min(1.0, conf))
            cnts[name] += 1

    vec: Dict[str, float] = {}
    for s in SOFT_SKILLS:
        vec[s] = (sums[s] / cnts[s]) if cnts[s] > 0 else 0.0
    return vec


def render_radar(vec: Dict[str, float], title: str) -> None:
    labels = SOFT_SKILLS
    values = [float(vec.get(s, 0.0)) for s in labels]
    # close the loop
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
            )
        ]
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================
# UI helpers
# ============================
def get_top_skill_from_analysis(analysis: Dict[str, Any]) -> Optional[str]:
    if not isinstance(analysis, dict):
        return None
    gp = analysis.get("growth_plan", {}) or {}
    top = gp.get("top_skill")
    if top in SOFT_SKILLS:
        return top
    skills = analysis.get("soft_skills") or []
    if isinstance(skills, list) and skills and isinstance(skills[0], dict):
        name = skills[0].get("name")
        if name in SOFT_SKILLS:
            return name
    return None


def get_growth_items(analysis: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    gp = (analysis.get("growth_plan", {}) or {}) if isinstance(analysis, dict) else {}
    practices = gp.get("practices") or []
    questions = gp.get("questions") or []
    alt_actions = gp.get("alt_actions") or []
    if not isinstance(practices, list):
        practices = []
    if not isinstance(questions, list):
        questions = []
    if not isinstance(alt_actions, list):
        alt_actions = []
    practices = (practices + [""] * PRACTICE_N)[:PRACTICE_N]
    questions = (questions + [""] * QUESTION_N)[:QUESTION_N]
    alt_actions = (alt_actions + [""] * ALT_ACTION_N)[:ALT_ACTION_N]
    return practices, questions, alt_actions


def ensure_notes_initialized(entry_id: str, entry_date: str, skill_name: str, practices: List[str], questions: List[str], alt_actions: List[str]) -> None:
    existing = fetch_skill_notes_for_entry(entry_id)
    exists_keys = {(n["skill_name"], n["note_type"], int(n["item_index"])) for n in existing}

    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()

        for i in range(PRACTICE_N):
            key = (skill_name, "practice", i)
            if key not in exists_keys:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO skill_notes
                    (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), entry_id, entry_date, skill_name, "practice", i, practices[i] if i < len(practices) else "", "", now, now),
                )

        for i in range(QUESTION_N):
            key = (skill_name, "question", i)
            if key not in exists_keys:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO skill_notes
                    (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), entry_id, entry_date, skill_name, "question", i, questions[i] if i < len(questions) else "", "", now, now),
                )

        for i in range(ALT_ACTION_N):
            key = (skill_name, "alt_action", i)
            if key not in exists_keys:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO skill_notes
                    (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), entry_id, entry_date, skill_name, "alt_action", i, alt_actions[i] if i < len(alt_actions) else "", "", now, now),
                )

        conn.commit()


def ensure_checklist_initialized(entry_id: str, entry_date: str, skill_name: str, practices: List[str], alt_actions: List[str]) -> None:
    existing = fetch_checklist_for_entry(entry_id)
    exists_keys = {(x["skill_name"], x["item_type"], int(x["item_index"])) for x in existing}

    # practices
    for i in range(PRACTICE_N):
        key = (skill_name, "practice", i)
        if key not in exists_keys:
            upsert_checklist(entry_id, entry_date, skill_name, "practice", i, practices[i] if i < len(practices) else "", False)

    # alt actions
    for i in range(ALT_ACTION_N):
        key = (skill_name, "alt_action", i)
        if key not in exists_keys:
            upsert_checklist(entry_id, entry_date, skill_name, "alt_action", i, alt_actions[i] if i < len(alt_actions) else "", False)


def render_analysis_block(analysis: Dict[str, Any]) -> None:
    if not analysis or not isinstance(analysis, dict):
        st.info("아직 분석 결과가 없습니다.")
        return

    st.subheader("🧠 상황분석")
    sa = analysis.get("situation_analysis", {}) or {}
    actions = sa.get("actions") or []
    learnings = sa.get("learnings") or []

    st.markdown("**행동**")
    if isinstance(actions, list) and actions:
        for a in actions:
            st.write(f"- {a}")
    else:
        st.write("—")

    st.markdown("**배움**")
    if isinstance(learnings, list) and learnings:
        for l in learnings:
            st.write(f"- {l}")
    else:
        st.write("—")

    st.subheader("🎯 오늘 드러난 소프트 스킬")
    skills = analysis.get("soft_skills", []) or []
    if not isinstance(skills, list) or not skills:
        st.write("선정된 역량이 없습니다.")
        return

    for sk in skills:
        if not isinstance(sk, dict):
            continue
        name = sk.get("name", "")
        try:
            conf = float(sk.get("confidence", 0))
        except Exception:
            conf = 0.0

        with st.expander(f"{name} (confidence: {conf:.2f})", expanded=False):
            ev = sk.get("evidence_quotes", []) or []
            if isinstance(ev, list) and ev:
                st.markdown("**근거(원문 인용)**")
                for q in ev[:2]:
                    st.caption(f"“{q}”")

            st.markdown("**왜 이 스킬인가**")
            st.write(sk.get("why_it_counts", ""))

            st.markdown("**개념 설명**")
            st.info(sk.get("concept") or SKILL_CONCEPTS.get(name, ""))


def render_growth_and_memo(entry_id: str, entry_date: str, analysis: Dict[str, Any]) -> None:
    top_skill = get_top_skill_from_analysis(analysis)
    if not top_skill:
        st.info("top 스킬을 결정할 수 없어 메모/체크리스트를 표시하지 않았습니다.")
        return

    practices, questions, alt_actions = get_growth_items(analysis)
    ensure_notes_initialized(entry_id, entry_date, top_skill, practices, questions, alt_actions)
    ensure_checklist_initialized(entry_id, entry_date, top_skill, practices, alt_actions)

    notes = fetch_skill_notes_for_entry(entry_id)
    grouped = group_notes(notes)

    checklist_items = fetch_checklist_for_entry(entry_id)
    cgroup = group_checklist(checklist_items)

    st.subheader(f"✅ 실행 체크 + ✍️ 메모 (Top 스킬: {top_skill})")
    st.caption("체크박스는 '실제로 행동으로 옮겼는지'를 기록합니다. 질문에는 답변(메모)을 남겨 DB에 영구 저장됩니다.")

    # --- Practices: checkbox + editable text + memo
    st.markdown("### 1) 연습/루틴 (2) — 체크박스")
    for i in range(PRACTICE_N):
        cur = grouped.get((top_skill, "practice"), {}).get(i, {"item_text": "", "memo_text": ""})
        ck = cgroup.get((top_skill, "practice"), {}).get(i, {"is_done": False, "item_text": cur["item_text"]})

        item_key = f"item_{entry_id}_{top_skill}_practice_{i}"
        memo_key = f"memo_{entry_id}_{top_skill}_practice_{i}"
        done_key = f"done_{entry_id}_{top_skill}_practice_{i}"

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.checkbox("실행함", value=bool(ck.get("is_done", False)), key=done_key)
        with col_b:
            st.text_input(f"연습 {i+1}", value=cur["item_text"], key=item_key)
            st.text_area("메모", value=cur["memo_text"], key=memo_key, height=80)

    # --- Questions: user answers stored in memo_text
    st.markdown("### 2) 성찰 질문 (2) — 답변 저장")
    for i in range(QUESTION_N):
        cur = grouped.get((top_skill, "question"), {}).get(i, {"item_text": "", "memo_text": ""})
        item_key = f"item_{entry_id}_{top_skill}_question_{i}"
        memo_key = f"memo_{entry_id}_{top_skill}_question_{i}"

        st.text_input(f"질문 {i+1}", value=cur["item_text"], key=item_key)
        st.text_area("내 답변", value=cur["memo_text"], key=memo_key, height=110)

    # --- Alternative actions: checkbox + memo (메타인지 강화)
    st.markdown("### 3) 대안행동 (2) — ‘그때 이렇게 했다면?’")
    for i in range(ALT_ACTION_N):
        cur = grouped.get((top_skill, "alt_action"), {}).get(i, {"item_text": "", "memo_text": ""})
        ck = cgroup.get((top_skill, "alt_action"), {}).get(i, {"is_done": False, "item_text": cur["item_text"]})

        item_key = f"item_{entry_id}_{top_skill}_alt_{i}"
        memo_key = f"memo_{entry_id}_{top_skill}_alt_{i}"
        done_key = f"done_{entry_id}_{top_skill}_alt_{i}"

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.checkbox("실행/적용해봄", value=bool(ck.get("is_done", False)), key=done_key)
        with col_b:
            st.text_input(f"대안행동 {i+1}", value=cur["item_text"], key=item_key)
            st.text_area("메모(적용/상상 결과)", value=cur["memo_text"], key=memo_key, height=90)

    if st.button("💾 체크/메모 저장", key=f"save_all_{entry_id}_{top_skill}"):
        # Save practices + checklist
        for i in range(PRACTICE_N):
            item_key = f"item_{entry_id}_{top_skill}_practice_{i}"
            memo_key = f"memo_{entry_id}_{top_skill}_practice_{i}"
            done_key = f"done_{entry_id}_{top_skill}_practice_{i}"

            item_text = st.session_state.get(item_key, "")
            memo_text = st.session_state.get(memo_key, "")
            is_done = bool(st.session_state.get(done_key, False))

            upsert_skill_note(entry_id, entry_date, top_skill, "practice", i, item_text, memo_text)
            upsert_checklist(entry_id, entry_date, top_skill, "practice", i, item_text, is_done)

        # Save questions (answers into memo_text)
        for i in range(QUESTION_N):
            item_key = f"item_{entry_id}_{top_skill}_question_{i}"
            memo_key = f"memo_{entry_id}_{top_skill}_question_{i}"

            item_text = st.session_state.get(item_key, "")
            memo_text = st.session_state.get(memo_key, "")
            upsert_skill_note(entry_id, entry_date, top_skill, "question", i, item_text, memo_text)

        # Save alt actions + checklist
        for i in range(ALT_ACTION_N):
            item_key = f"item_{entry_id}_{top_skill}_alt_{i}"
            memo_key = f"memo_{entry_id}_{top_skill}_alt_{i}"
            done_key = f"done_{entry_id}_{top_skill}_alt_{i}"

            item_text = st.session_state.get(item_key, "")
            memo_text = st.session_state.get(memo_key, "")
            is_done = bool(st.session_state.get(done_key, False))

            upsert_skill_note(entry_id, entry_date, top_skill, "alt_action", i, item_text, memo_text)
            upsert_checklist(entry_id, entry_date, top_skill, "alt_action", i, item_text, is_done)

        st.success("저장했습니다.")


# ============================
# Pages
# ============================
def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("📊 대시보드")
    st.caption("방사형 차트로 ‘강점(점수가 높은 스킬)’과 ‘기록이 부족한 영역(점수가 낮은 스킬)’을 한눈에 봅니다.")

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("기간", ["최근 7일", "최근 30일", "전체"], index=0)
        if mode == "최근 7일":
            start = (date.today() - timedelta(days=6)).isoformat()
            dfw = df[df["entry_date"] >= start] if not df.empty else df
        elif mode == "최근 30일":
            start = (date.today() - timedelta(days=29)).isoformat()
            dfw = df[df["entry_date"] >= start] if not df.empty else df
        else:
            dfw = df

        st.metric("기록 수", int(len(dfw)))

        # 간단한 부족 영역 Top-2
        vec = compute_skill_vector(dfw)
        low2 = sorted(vec.items(), key=lambda x: x[1])[:2]
        st.write("**지금 부족해 보이는 영역(Top-2)**")
        for k, v in low2:
            st.write(f"- {k}: {v:.2f}")

    with col2:
        vec = compute_skill_vector(dfw)
        render_radar(vec, title=f"스킬 프로필(0~1) — {mode}")

    st.markdown("---")
    st.subheader("🗓️ 주간 흐름(최근 4주)")
    if df.empty:
        st.info("아직 기록이 없습니다.")
        return

    # 주 단위(월~일)로 스킬 언급 횟수
    tmp = df.copy()
    tmp["entry_date_dt"] = pd.to_datetime(tmp["entry_date"], errors="coerce")
    tmp = tmp.dropna(subset=["entry_date_dt"])
    tmp["week_start"] = (tmp["entry_date_dt"] - pd.to_timedelta(tmp["entry_date_dt"].dt.weekday, unit="D")).dt.date

    rows: List[Dict[str, Any]] = []
    for _, r in tmp.iterrows():
        an = r.get("analysis_parsed") or {}
        if not isinstance(an, dict):
            continue
        for sk in (an.get("soft_skills") or []):
            if isinstance(sk, dict) and sk.get("name") in SOFT_SKILLS:
                rows.append({"week_start": r["week_start"], "skill": sk["name"]})

    if not rows:
        st.info("아직 분석 결과가 부족해 주간 흐름을 표시할 데이터가 없습니다.")
        return

    df_rows = pd.DataFrame(rows)
    pivot = (
        df_rows.groupby(["week_start", "skill"])
        .size()
        .reset_index(name="count")
        .pivot(index="week_start", columns="skill", values="count")
        .fillna(0)
        .sort_index(ascending=False)
        .head(4)
        .sort_index()
    )
    st.dataframe(pivot, use_container_width=True)


def render_new_entry(df: pd.DataFrame) -> None:
    st.subheader("✍️ 오늘의 기록 추가")
    st.info("일기처럼 나열 대신, **행동 / 감정 / 결과** 3칸에 나누어 적어보세요.")

    col1, col2 = st.columns([1, 1])
    with col1:
        entry_date = st.date_input("날짜", value=date.today())

        cat_choice = st.selectbox(
            "카테고리(선택)",
            options=["(선택 안 함)"] + CATEGORIES + ["(직접 입력)"],
            index=0,
        )
        cat_custom = ""
        if cat_choice == "(직접 입력)":
            cat_custom = st.text_input("카테고리 직접 입력", placeholder="예: 인턴/현장실습, 개인 프로젝트, 취미 활동 등")

        if cat_choice == "(선택 안 함)":
            category = None
        elif cat_choice == "(직접 입력)":
            category = cat_custom.strip() if cat_custom.strip() else None
        else:
            category = cat_choice

    with col2:
        artifacts = st.text_area(
            "증거/자료 링크(선택) — 줄바꿈으로 여러 개",
            placeholder="예: Notion 링크, Google Doc, GitHub, 발표자료 URL 등",
        )
        artifacts_list = [x.strip() for x in (artifacts or "").splitlines() if x.strip()]

    st.markdown("### 🧩 기록 입력(구조화)")
    a_col, e_col, r_col = st.columns(3)
    with a_col:
        actions_text = st.text_area("행동", height=160, placeholder="내가 실제로 한 행동을 구체적으로")
    with e_col:
        emotions_text = st.text_area("감정", height=160, placeholder="그때 느낀 감정/몸 상태/압박감 등")
    with r_col:
        results_text = st.text_area("결과", height=160, placeholder="결과(변화/피드백/수치/관찰) + 배움")

    # raw_text는 호환을 위해 3칸을 합쳐 저장
    raw_text = "\n".join(
        [
            f"[행동]\n{actions_text.strip()}",
            f"[감정]\n{emotions_text.strip()}",
            f"[결과]\n{results_text.strip()}",
        ]
    ).strip()

    st.markdown("### 🔎 분석 옵션")
    do_analysis = st.checkbox("저장 후 분석 실행하기", value=True)
    top_k = st.slider("유사 기록 힌트(top-k)", min_value=0, max_value=10, value=5)

    if st.button("✅ 저장", type="primary"):
        if not (actions_text.strip() or emotions_text.strip() or results_text.strip()):
            st.error("행동/감정/결과 중 최소 하나는 입력해주세요.")
            return

        entry_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat(timespec="seconds")

        entry = {
            "id": entry_id,
            "created_at": created_at,
            "entry_date": entry_date.isoformat(),
            "category": category,
            "tags": [],
            "title": None,
            "raw_text": raw_text,
            "artifacts": artifacts_list,
            "analysis": {},
        }
        insert_entry(entry)
        upsert_structured(entry_id, actions_text, emotions_text, results_text)

        st.success("기록을 저장했습니다.")

        if not do_analysis:
            return

        # related hints
        related: List[Dict[str, Any]] = []
        if top_k > 0 and not df.empty:
            sims = get_similar_entries(df, entry["raw_text"], top_k=top_k)
            hint_rows: List[Dict[str, Any]] = []
            for rid, _score in sims:
                r = fetch_entry_by_id(rid)
                if r:
                    hint_rows.append(
                        {
                            "id": r["id"],
                            "entry_date": r["entry_date"],
                            "category": r.get("category"),
                            "raw_text": r["raw_text"],
                            "artifacts": r.get("artifacts") or [],
                            "analysis_json": json.dumps(r.get("analysis_json") or {}, ensure_ascii=False),
                        }
                    )
            hint_df = pd.DataFrame(hint_rows) if hint_rows else pd.DataFrame()
            if not hint_df.empty:
                hint_df["analysis_parsed"] = hint_df["analysis_json"].apply(lambda x: safe_json_loads(x, default={}))
                related = summarize_for_related(hint_df)

        engine = st.session_state.get("engine", DEFAULT_ENGINE)

        payload = {
            "id": entry_id,
            "entry_date": entry["entry_date"],
            "category": category,
            "raw_text": raw_text,
            "artifacts": artifacts_list,
            "structured": {"actions": actions_text, "emotions": emotions_text, "results": results_text},
        }

        with st.spinner("분석 중..."):
            analysis = run_analysis_engine(engine=engine, entry=payload, related=related)
            update_entry_analysis(entry_id, analysis)

        st.success("분석 완료!")
        render_analysis_block(analysis)
        st.markdown("---")
        render_growth_and_memo(entry_id, entry["entry_date"], analysis)


def render_history(df: pd.DataFrame) -> None:
    st.subheader("📚 기록 목록")
    if df.empty:
        st.info("아직 저장된 기록이 없습니다. '오늘의 기록 추가'에서 작성해보세요.")
        return

    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        cat = st.selectbox("카테고리 필터", options=["(전체)"] + CATEGORIES, index=0)
    with colf2:
        skill_filter = st.selectbox("소프트스킬 필터", options=["(전체)"] + SOFT_SKILLS, index=0)
    with colf3:
        q = st.text_input("검색(본문)", placeholder="예: 발표, 조율, 회복, 기준, 피드백...")

    filtered = df.copy()

    if cat != "(전체)":
        filtered = filtered[filtered["category"] == cat]

    if (q or "").strip():
        qq = q.strip().lower()
        filtered = filtered[filtered["raw_text"].str.lower().str.contains(qq, na=False)]

    if skill_filter != "(전체)":
        def has_skill(an: Any) -> bool:
            if not isinstance(an, dict):
                return False
            skills = an.get("soft_skills", []) or []
            return any((s.get("name") == skill_filter) for s in skills if isinstance(s, dict))
        filtered = filtered[filtered["analysis_parsed"].apply(has_skill)]

    st.caption(f"총 {len(filtered)}개")
    engine = st.session_state.get("engine", DEFAULT_ENGINE)

    for _, r in filtered.iterrows():
        entry_id = r["id"]
        entry_date = r.get("entry_date", "")
        category = r.get("category") or "—"
        an = r.get("analysis_parsed") or {}
        if not isinstance(an, dict):
            an = {}

        skills = [s.get("name") for s in (an.get("soft_skills") or []) if isinstance(s, dict) and s.get("name")]
        skill_text = ", ".join(skills) if skills else "—"

        with st.expander(f"{entry_date} · 카테고리: {category} | 스킬: {skill_text}"):
            st.write(r["raw_text"])

            artifacts = r.get("artifacts_parsed") or []
            if artifacts:
                st.markdown("**증거/링크**")
                for a in artifacts:
                    st.write(f"- {a}")

            # 구조화 입력도 보여주기
            st.markdown("---")
            s = fetch_structured(entry_id)
            if any(v.strip() for v in s.values()):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**행동**")
                    st.write(s["actions_text"] or "—")
                with c2:
                    st.markdown("**감정**")
                    st.write(s["emotions_text"] or "—")
                with c3:
                    st.markdown("**결과**")
                    st.write(s["results_text"] or "—")

            st.markdown("---")
            if an:
                render_analysis_block(an)
                st.markdown("---")
                render_growth_and_memo(entry_id, entry_date, an)
            else:
                st.info("아직 분석 결과가 없습니다. 아래 버튼으로 분석을 실행할 수 있어요.")

            colb1, colb2 = st.columns([1, 1])
            with colb1:
                if st.button("🤖 이 기록 분석하기", key=f"an_{entry_id}"):
                    entry = fetch_entry_by_id(entry_id)
                    if not entry:
                        st.error("기록을 불러오지 못했습니다.")
                    else:
                        other = df[df["id"] != entry_id].head(80)
                        related = summarize_for_related(other) if not other.empty else []
                        structured = fetch_structured(entry_id)

                        payload = {
                            "id": entry["id"],
                            "entry_date": entry["entry_date"],
                            "category": entry.get("category"),
                            "raw_text": entry["raw_text"],
                            "artifacts": entry.get("artifacts") or [],
                            "structured": {"actions": structured["actions_text"], "emotions": structured["emotions_text"], "results": structured["results_text"]},
                        }
                        with st.spinner("분석 중..."):
                            analysis = run_analysis_engine(engine=engine, entry=payload, related=related)
                            update_entry_analysis(entry_id, analysis)
                        st.success("분석 완료! 화면을 갱신합니다.")
                        st.rerun()

            with colb2:
                if st.button("🗑️ 삭제", key=f"del_{entry_id}"):
                    delete_entry(entry_id)
                    st.success("삭제했습니다. 화면을 갱신합니다.")
                    st.rerun()


def render_memos(df: pd.DataFrame) -> None:
    st.subheader("📒 메모/답변/체크리스트")
    st.caption("Top 스킬의 연습/대안행동 체크, 성찰 질문 답변(메모)을 날짜별/스킬별로 확인합니다.")

    tab1, tab2 = st.tabs(["날짜별", "스킬별"])

    with tab1:
        dates = sorted(df["entry_date"].dropna().unique().tolist(), reverse=True) if not df.empty else []
        if not dates:
            st.info("아직 기록/메모가 없습니다.")
        else:
            d = st.selectbox("날짜 선택", options=dates, index=0)
            notes = fetch_skill_notes_by_date(d)
            if not notes:
                st.info("이 날짜에 저장된 메모가 없습니다.")
            else:
                by_entry: Dict[str, List[Dict[str, Any]]] = {}
                for n in notes:
                    by_entry.setdefault(n["entry_id"], []).append(n)

                for entry_id, ns in by_entry.items():
                    with st.expander(f"{d} · entry_id: {entry_id[:8]} · 항목 {len(ns)}개"):
                        by_skill: Dict[str, List[Dict[str, Any]]] = {}
                        for n in ns:
                            by_skill.setdefault(n["skill_name"], []).append(n)

                        # 체크리스트도 로드
                        citems = fetch_checklist_for_entry(entry_id)
                        cgroup = group_checklist(citems)

                        for sk, sk_notes in by_skill.items():
                            st.markdown(f"**{sk}**")

                            # practice / alt_action 체크 표시
                            for nt in ["practice", "alt_action"]:
                                items = [x for x in sk_notes if x["note_type"] == nt]
                                if not items:
                                    continue
                                st.caption("연습/루틴" if nt == "practice" else "대안행동")
                                for it in sorted(items, key=lambda x: int(x["item_index"])):
                                    ck = cgroup.get((sk, nt), {}).get(int(it["item_index"]), {"is_done": False})
                                    badge = "✅" if ck.get("is_done") else "⬜"
                                    st.write(f"- {badge} {it['item_text']}")
                                    if (it.get("memo_text") or "").strip():
                                        st.write(f"  ↳ 메모: {it['memo_text']}")

                            # question 답변
                            qitems = [x for x in sk_notes if x["note_type"] == "question"]
                            if qitems:
                                st.caption("성찰 질문/답변")
                                for it in sorted(qitems, key=lambda x: int(x["item_index"])):
                                    st.write(f"- Q: {it['item_text']}")
                                    st.write(f"  A: {(it.get('memo_text') or '').strip() or '—'}")

    with tab2:
        skill = st.selectbox("스킬 선택", options=SOFT_SKILLS, index=0)
        notes = fetch_skill_notes_by_skill(skill, limit=500)
        if not notes:
            st.info("이 스킬에 저장된 메모가 없습니다.")
            return

        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for n in notes:
            by_date.setdefault(n["entry_date"], []).append(n)

        for d in sorted(by_date.keys(), reverse=True):
            with st.expander(f"{d} · 항목 {len(by_date[d])}개"):
                items = by_date[d]
                for nt in ["practice", "alt_action", "question"]:
                    sub = [x for x in items if x["note_type"] == nt]
                    if not sub:
                        continue
                    st.caption("연습/루틴" if nt == "practice" else ("대안행동" if nt == "alt_action" else "성찰 질문/답변"))
                    for it in sorted(sub, key=lambda x: int(x["item_index"])):
                        if nt == "question":
                            st.write(f"- Q: {it['item_text']}")
                            st.write(f"  A: {(it.get('memo_text') or '').strip() or '—'}")
                        else:
                            st.write(f"- {it['item_text']}")
                            if (it.get("memo_text") or "").strip():
                                st.write(f"  ↳ 메모: {it['memo_text']}")
                        st.caption(f"entry_id: {it['entry_id'][:8]} · updated: {it['updated_at']}")


def render_debug(df: pd.DataFrame) -> None:
    st.subheader("🧪 디버그/로그")
    st.write("현재 DB에 저장된 기록 개수:", len(df))

    st.markdown("### 최근 10개 기록 미리보기")
    if not df.empty:
        st.dataframe(df[["entry_date", "category"]].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("기록이 없습니다.")

    st.markdown("### 환경/설정")
    st.write(
        {
            "engine": st.session_state.get("engine", DEFAULT_ENGINE),
            "has_api_key": bool(st.session_state.get("api_key")),
            "model": st.session_state.get("model", DEFAULT_MODEL),
            "db_path": DB_PATH,
            "policy": "입력(행동/감정/결과) + 분석(행동/배움/스킬) + 성장(연습2/질문2/대안행동2) + 체크/메모 저장",
        }
    )

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM skill_notes")
        note_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM checklist")
        ck_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM entry_structured")
        st_count = cur.fetchone()[0]

    st.write("저장된 메모 개수(skill_notes):", note_count)
    st.write("저장된 체크리스트 항목(checklist):", ck_count)
    st.write("구조화 입력 저장(entry_structured):", st_count)


# ============================
# Main
# ============================
def main() -> None:
    st.set_page_config(page_title="MetaTone", layout="wide")
    init_db()

    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (선택)",
        value=st.session_state.get("api_key", api_key_env),
        type="password",
    )
    st.session_state["api_key"] = (api_key_input or "").strip()

    current_model = st.session_state.get("model", DEFAULT_MODEL)
    if current_model not in MODEL_OPTIONS:
        current_model = DEFAULT_MODEL
    st.sidebar.selectbox(
        "Model (LLM 모드에서만 사용)",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(current_model),
        key="model",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 분석 엔진")
    st.sidebar.selectbox("분석 방식", options=ANALYSIS_ENGINES, index=0, key="engine")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("페이지", ["📊 대시보드", "✍️ 오늘의 기록 추가", "📚 기록 목록", "📒 메모", "🧪 디버그/로그"])

    df = fetch_entries()

    st.title(APP_TITLE)
    st.caption("기록(행동/감정/결과) → 분석(행동/배움/스킬) → 성장(연습2/질문2/대안행동2) → 체크박스/답변/메모를 DB에 저장")

    st.markdown("---")

    if page == "📊 대시보드":
        render_dashboard(df)
    elif page == "✍️ 오늘의 기록 추가":
        render_new_entry(df)
    elif page == "📚 기록 목록":
        render_history(df)
    elif page == "📒 메모":
        render_memos(df)
    else:
        render_debug(df)


if __name__ == "__main__":
    main()
