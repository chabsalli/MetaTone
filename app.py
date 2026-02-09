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

SKILL_CONCEPTS = {
    "문제해결": "문제를 정의하고 원인을 파악해 실행 가능한 대안을 만들고 검증하는 역량",
    "의사소통": "상대의 이해를 기준으로 정보를 구조화·전달하고 합의를 이끌어내는 역량",
    "협업": "역할·의존성을 맞추고 상호 신뢰를 바탕으로 성과를 함께 만드는 역량",
    "리더십": "방향을 제시하고 의사결정을 돕고 구성원이 움직이게 만드는 영향력",
    "자기관리/회복탄력성": "에너지·감정·시간을 관리하며 압박 속에서도 회복하고 지속하는 역량",
    "학습역량": "학습 목표를 세우고 피드백을 통해 지식을 내 것으로 만드는 역량",
}

# 2+2 메모 기본 개수
PRACTICE_N = 2
QUESTION_N = 2


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

        # entries (기존 스키마 유지: tags/title은 MetaTone에서 미사용)
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

        # notes: entry_id + skill_name 단위로, practice/question 각각 0..1 저장
        cur.execute("""
        CREATE TABLE IF NOT EXISTS skill_notes (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            note_type TEXT NOT NULL,          -- 'practice' | 'question'
            item_index INTEGER NOT NULL,      -- 0..1
            item_text TEXT NOT NULL,
            memo_text TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(entry_id, skill_name, note_type, item_index)
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
            json.dumps(entry.get("tags", []), ensure_ascii=False),
            entry.get("title"),
            entry["raw_text"],
            json.dumps(entry.get("artifacts", []), ensure_ascii=False),
            json.dumps(entry.get("analysis", {}), ensure_ascii=False)
        ))
        conn.commit()


def update_entry_analysis(entry_id: str, analysis: Dict[str, Any]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE entries SET analysis_json = ? WHERE id = ?",
                    (json.dumps(analysis, ensure_ascii=False), entry_id))
        conn.commit()


def delete_entry(entry_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        # notes도 같이 삭제
        cur.execute("DELETE FROM skill_notes WHERE entry_id = ?", (entry_id,))
        conn.commit()


def fetch_entries(limit: int = 500) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM entries ORDER BY entry_date DESC, created_at DESC LIMIT ?",
            conn,
            params=(limit,),
        )

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


def upsert_skill_note(
    entry_id: str,
    entry_date: str,
    skill_name: str,
    note_type: str,
    item_index: int,
    item_text: str,
    memo_text: str
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO skill_notes (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id, skill_name, note_type, item_index)
            DO UPDATE SET
                item_text = excluded.item_text,
                memo_text = excluded.memo_text,
                updated_at = excluded.updated_at
        """, (
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
        ))
        conn.commit()


def fetch_skill_notes_for_entry(entry_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE entry_id = ?
            ORDER BY skill_name, note_type, item_index
        """, (entry_id,))
        rows = cur.fetchall()

    out = []
    for r in rows:
        out.append({
            "entry_id": r[0],
            "entry_date": r[1],
            "skill_name": r[2],
            "note_type": r[3],
            "item_index": r[4],
            "item_text": r[5],
            "memo_text": r[6],
            "updated_at": r[7],
        })
    return out


def fetch_skill_notes_by_skill(skill_name: str, limit: int = 300) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE skill_name = ?
            ORDER BY entry_date DESC, updated_at DESC
            LIMIT ?
        """, (skill_name, limit))
        rows = cur.fetchall()

    out = []
    for r in rows:
        out.append({
            "entry_id": r[0],
            "entry_date": r[1],
            "skill_name": r[2],
            "note_type": r[3],
            "item_index": r[4],
            "item_text": r[5],
            "memo_text": r[6],
            "updated_at": r[7],
        })
    return out


def fetch_skill_notes_by_date(entry_date: str, limit: int = 300) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, updated_at
            FROM skill_notes
            WHERE entry_date = ?
            ORDER BY skill_name, note_type, item_index
            LIMIT ?
        """, (entry_date, limit))
        rows = cur.fetchall()

    out = []
    for r in rows:
        out.append({
            "entry_id": r[0],
            "entry_date": r[1],
            "skill_name": r[2],
            "note_type": r[3],
            "item_index": r[4],
            "item_text": r[5],
            "memo_text": r[6],
            "updated_at": r[7],
        })
    return out


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
# Robust JSON parsing
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
# Analysis engines
# (요약 없음 / 패턴요약 없음 / STAR 없음)
# 상황분석: 행동, 배움
# 성장플랜: top 스킬 기준 2 practice + 2 question
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
    output_mode: str = "analysis_only",
) -> Dict[str, Any]:
    client = get_openai_client(api_key)

    persona = (
        "당신은 MetaTone의 코치입니다. "
        "사용자의 기록에서 '행동'과 '배움'을 뽑고, 그 기록에서 드러난 소프트스킬(1~3개)을 근거 인용과 함께 제시합니다. "
        "과장/미사여구/단정 금지. 근거 중심."
    )

    related_block = []
    for rs in (related_summaries or [])[:5]:
        related_block.append({
            "id": rs.get("id"),
            "date": rs.get("entry_date"),
            "one_liner": rs.get("one_liner", ""),
            "skills": rs.get("skills", []),
        })

    # JSON 계약(요약 없음, 행동/배움만)
    output_contract: Dict[str, Any] = {
        "meta": {
            "entry_id": entry["id"],
            "entry_date": entry["entry_date"],
            "category": entry.get("category") or ""
        },
        "situation_analysis": {
            "actions": ["내가 실제로 한 행동 2~4개(짧은 문장)"],
            "learnings": ["배움 1~2개(짧은 문장)"]
        },
        "soft_skills": [
            {
                "name": "협업",
                "confidence": 0.0,
                "evidence_quotes": ["원문 그대로 1~2개(각 80자 이내)"],
                "why_it_counts": "왜 이 역량인지 1문장",
                "concept": "개념 1문장"
            }
        ],
        "growth_plan": {
            "top_skill": "협업",
            "practices": ["연습/루틴 1", "연습/루틴 2"],
            "questions": ["다음 기록 질문 1", "다음 기록 질문 2"]
        }
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
        "output_contract_example": output_contract,
        "constraints": {
            "practice_n": PRACTICE_N,
            "question_n": QUESTION_N
        }
    }

    instructions = (
        "규칙:\n"
        "1) 반드시 JSON만 출력(마크다운/코드펜스/설명문 금지)\n"
        "2) soft_skills는 1~3개, confidence는 0~1 숫자\n"
        "3) evidence_quotes는 원문 그대로 최대 2개, 각 80자 이내\n"
        "4) situation_analysis는 actions/learnings만 (요약 금지)\n"
        f"5) growth_plan의 practices는 정확히 {PRACTICE_N}개, questions는 정확히 {QUESTION_N}개\n"
        "6) growth_plan.top_skill은 soft_skills 중 confidence가 가장 높은 스킬명\n"
        "7) concept는 skill_concepts를 참고해 1문장으로 간단히\n"
        "8) 과장/미사여구/단정 금지\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "user", "content": instructions},
            ]
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenAI 호출 실패: {e}\n\n"
            f"점검:\n- API Key 유효 여부\n- 모델({model}) 접근 권한/이름\n- 사용량/쿼터/결제 상태"
        )

    out = robust_json_loads(resp.choices[0].message.content or "")
    return out


def analyze_entry_local(
    entry: Dict[str, Any],
    related_summaries: List[Dict[str, Any]],
    output_mode: str = "analysis_only",
) -> Dict[str, Any]:
    text = (entry.get("raw_text") or "").strip()

    # 행동/배움: 문장/줄에서 간단 추출(보수적)
    lines = [l.strip() for l in re.split(r"[\n\r]+", text) if l.strip()]
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]

    # 행동 후보: 동사/표현 기반
    action_markers = ["했다", "함", "진행", "정리", "공유", "설명", "조율", "확인", "개선", "시도", "결정", "분석", "제안", "요청"]
    actions: List[str] = []
    for l in lines:
        if any(m in l for m in action_markers):
            actions.append(l[:140])
        if len(actions) >= 4:
            break
    if not actions:
        # fallback: 앞 문장 일부
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
            "why_it_counts": "원문에서 해당 행동 단서(키워드/표현)가 보여 이 역량이 드러난 것으로 추정했습니다. (무료 로컬 분석)",
            "concept": SKILL_CONCEPTS.get(sk, "")
        })

    # top_skill = confidence max
    top_skill = soft_skills[0]["name"]
    # growth_plan: top skill 기준 2+2
    practices = [
        "다음 기록에서 '내가 선택한 기준(우선순위/근거)'을 1문장으로 남기기",
        "결과를 관찰 가능한 표현(전/후 변화, 시간/횟수/품질)로 적기",
    ][:PRACTICE_N]
    questions = [
        "내가 한 선택의 기준은 무엇이었나?",
        "다음에 같은 상황이면 무엇을 유지/변경할까?",
    ][:QUESTION_N]

    out: Dict[str, Any] = {
        "meta": {
            "entry_id": entry["id"],
            "entry_date": entry["entry_date"],
            "category": entry.get("category") or ""
        },
        "situation_analysis": {
            "actions": actions[:4],
            "learnings": learnings[:2],
        },
        "soft_skills": soft_skills,
        "growth_plan": {
            "top_skill": top_skill,
            "practices": practices,
            "questions": questions
        }
    }
    return out


def run_analysis_engine(
    engine: str,
    entry: Dict[str, Any],
    related: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
            related_summaries=related
        )
    except Exception as e:
        st.warning(f"LLM 분석 실패 → 무료(로컬) 분석으로 대체합니다.\n\n사유: {e}")
        return analyze_entry_local(entry=entry, related_summaries=related)


# ============================
# Aggregations / Related summaries
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
        summaries.append({
            "id": r["id"],
            "entry_date": r["entry_date"],
            "one_liner": one_liner,
            "skills": skills
        })
    return summaries


# ============================
# Notes initialization + UI helpers
# ============================
def ensure_notes_initialized(
    entry_id: str,
    entry_date: str,
    skill_name: str,
    practices: List[str],
    questions: List[str],
) -> None:
    """
    notes 테이블에 기본 row가 없으면 만들어둔다.
    (이미 있으면 upsert로 덮어쓰지 않음: 사용자가 수정한 텍스트/메모를 보호)
    """
    existing = fetch_skill_notes_for_entry(entry_id)
    exists_keys = set()
    for n in existing:
        exists_keys.add((n["skill_name"], n["note_type"], int(n["item_index"])))

    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        # practices
        for i in range(PRACTICE_N):
            key = (skill_name, "practice", i)
            if key in exists_keys:
                continue
            item_text = practices[i] if i < len(practices) else ""
            cur.execute("""
                INSERT OR IGNORE INTO skill_notes
                (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), entry_id, entry_date, skill_name, "practice", i,
                item_text, "", now, now
            ))

        # questions
        for i in range(QUESTION_N):
            key = (skill_name, "question", i)
            if key in exists_keys:
                continue
            item_text = questions[i] if i < len(questions) else ""
            cur.execute("""
                INSERT OR IGNORE INTO skill_notes
                (id, entry_id, entry_date, skill_name, note_type, item_index, item_text, memo_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), entry_id, entry_date, skill_name, "question", i,
                item_text, "", now, now
            ))
        conn.commit()


def group_notes(notes: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[int, Dict[str, str]]]:
    """
    return: {(skill_name, note_type): {idx: {"item_text":..., "memo_text":...}}}
    """
    out: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for n in notes:
        k = (n["skill_name"], n["note_type"])
        out.setdefault(k, {})
        out[k][int(n["item_index"])] = {
            "item_text": n.get("item_text") or "",
            "memo_text": n.get("memo_text") or ""
        }
    return out


def render_skill_totals(totals: Dict[str, int]) -> None:
    st.subheader("📈 소프트스킬 누적(범주별)")
    cols = st.columns(3)
    items = list(totals.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.metric(label=k, value=v)

    df_tot = (
        pd.DataFrame([{"soft_skill": k, "count": v} for k, v in totals.items()])
        .sort_values("count", ascending=False)
    )
    st.dataframe(df_tot, use_container_width=True, hide_index=True)


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

    st.subheader("🎯 오늘 쌓은 소프트 스킬")
    skills = analysis.get("soft_skills", []) or []
    if not isinstance(skills, list) or not skills:
        st.write("선정된 역량이 없습니다.")
        return

    for sk in skills:
        if not isinstance(sk, dict):
            continue
        name = sk.get("name", "")
        conf = sk.get("confidence", 0)
        try:
            conf = float(conf)
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


def get_top_skill_from_analysis(analysis: Dict[str, Any]) -> Optional[str]:
    if not isinstance(analysis, dict):
        return None
    gp = analysis.get("growth_plan", {}) or {}
    top = gp.get("top_skill")
    if top in SOFT_SKILLS:
        return top

    # fallback: soft_skills[0]
    skills = analysis.get("soft_skills") or []
    if isinstance(skills, list) and skills and isinstance(skills[0], dict):
        name = skills[0].get("name")
        if name in SOFT_SKILLS:
            return name
    return None


def get_growth_items_for_top_skill(analysis: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    gp = (analysis.get("growth_plan", {}) or {}) if isinstance(analysis, dict) else {}
    practices = gp.get("practices") or []
    questions = gp.get("questions") or []
    if not isinstance(practices, list):
        practices = []
    if not isinstance(questions, list):
        questions = []
    # 정확히 2개로 맞추기(부족하면 빈 값)
    practices = (practices + [""] * PRACTICE_N)[:PRACTICE_N]
    questions = (questions + [""] * QUESTION_N)[:QUESTION_N]
    return practices, questions


def render_memo_editor_for_skill(
    entry_id: str,
    entry_date: str,
    skill_name: str,
    default_practices: List[str],
    default_questions: List[str],
) -> None:
    """
    - item_text 수정 가능
    - memo_text 입력 가능
    - 저장 버튼으로 upsert
    """
    ensure_notes_initialized(entry_id, entry_date, skill_name, default_practices, default_questions)
    notes = fetch_skill_notes_for_entry(entry_id)
    grouped = group_notes(notes)

    st.markdown(f"### ✍️ 메모 — {skill_name}")
    st.caption("연습/질문 문구도 수정할 수 있어요. 저장을 눌러 반영하세요.")

    # practices
    st.markdown("**연습/루틴 (2)**")
    for i in range(PRACTICE_N):
        cur = grouped.get((skill_name, "practice"), {}).get(i, {"item_text": "", "memo_text": ""})
        item_key = f"item_{entry_id}_{skill_name}_practice_{i}"
        memo_key = f"memo_{entry_id}_{skill_name}_practice_{i}"

        st.text_input(f"연습 {i+1}", value=cur["item_text"], key=item_key)
        st.text_area("메모", value=cur["memo_text"], key=memo_key, height=80)

    # questions
    st.markdown("**다음 기록 질문 (2)**")
    for i in range(QUESTION_N):
        cur = grouped.get((skill_name, "question"), {}).get(i, {"item_text": "", "memo_text": ""})
        item_key = f"item_{entry_id}_{skill_name}_question_{i}"
        memo_key = f"memo_{entry_id}_{skill_name}_question_{i}"

        st.text_input(f"질문 {i+1}", value=cur["item_text"], key=item_key)
        st.text_area("메모", value=cur["memo_text"], key=memo_key, height=80)

    if st.button("💾 메모 저장", key=f"save_{entry_id}_{skill_name}"):
        # practices save
        for i in range(PRACTICE_N):
            item_key = f"item_{entry_id}_{skill_name}_practice_{i}"
            memo_key = f"memo_{entry_id}_{skill_name}_practice_{i}"
            upsert_skill_note(
                entry_id=entry_id,
                entry_date=entry_date,
                skill_name=skill_name,
                note_type="practice",
                item_index=i,
                item_text=st.session_state.get(item_key, ""),
                memo_text=st.session_state.get(memo_key, ""),
            )
        # questions save
        for i in range(QUESTION_N):
            item_key = f"item_{entry_id}_{skill_name}_question_{i}"
            memo_key = f"memo_{entry_id}_{skill_name}_question_{i}"
            upsert_skill_note(
                entry_id=entry_id,
                entry_date=entry_date,
                skill_name=skill_name,
                note_type="question",
                item_index=i,
                item_text=st.session_state.get(item_key, ""),
                memo_text=st.session_state.get(memo_key, ""),
            )
        st.success("저장했습니다.")


# ============================
# Streamlit Pages
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
    st.sidebar.selectbox(
        "Model (LLM 모드에서만 사용)",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(current_model),
        key="model"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 분석 엔진")
    st.sidebar.selectbox("분석 방식", options=ANALYSIS_ENGINES, index=0, key="engine")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("페이지", ["✍️ 오늘의 기록 추가", "📚 기록 목록", "📒 메모", "🧪 디버그/로그"])

    df = fetch_entries()
    totals = compute_skill_totals(df)

    st.title(APP_TITLE)
    st.caption("기록 본문에서 행동/배움을 뽑고, 오늘 드러난 소프트스킬과 (top 스킬 기준) 2+2 루틴/질문 메모를 누적합니다.")

    render_skill_totals(totals)
    st.markdown("---")

    if page == "✍️ 오늘의 기록 추가":
        render_new_entry(df)
    elif page == "📚 기록 목록":
        render_history(df)
    elif page == "📒 메모":
        render_notes_page(df)
    else:
        render_debug(df)


def render_new_entry(df: pd.DataFrame):
    st.subheader("✍️ 오늘의 기록 추가")

    st.info("한 박스에 자유롭게 적되, 가능하면 **행동 → 경험한 감정 → 결과** 순서로 써보세요.\n"
            "예) 오늘 내가 한 행동 / 그때 느낀 감정 / 결과와 배움")

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
        height=260,
        placeholder="행동 → 감정 → 결과 순서로 적어보세요.\n(예: 내가 한 행동 / 느낀 감정 / 결과 + 배움)"
    )

    st.markdown("### 🔎 분석 옵션")
    do_analysis = st.checkbox("저장 후 분석 실행하기", value=True)
    top_k = st.slider("유사 기록 힌트(top-k)", min_value=0, max_value=10, value=5)

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
            "tags": [],
            "title": None,
            "raw_text": raw_text.strip(),
            "artifacts": artifacts_list,
            "analysis": {}
        }
        insert_entry(entry)
        st.success("기록을 저장했습니다.")

        if not do_analysis:
            return

        # related hints (optional)
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
        with st.spinner("분석 중..."):
            analysis = run_analysis_engine(engine=engine, entry=entry, related=related)
            update_entry_analysis(entry_id, analysis)

        st.success("분석 완료!")
        render_analysis_block(analysis)

        # top skill memo editor (기본: top 1개만)
        top_skill = get_top_skill_from_analysis(analysis)
        if top_skill:
            practices, questions = get_growth_items_for_top_skill(analysis)
            st.markdown("---")
            render_memo_editor_for_skill(
                entry_id=entry_id,
                entry_date=entry["entry_date"],
                skill_name=top_skill,
                default_practices=practices,
                default_questions=questions
            )

            # 옵션: 다른 스킬도 메모하기
            other_skills = []
            skills = analysis.get("soft_skills") or []
            if isinstance(skills, list):
                for sk in skills:
                    if isinstance(sk, dict) and sk.get("name") and sk.get("name") != top_skill:
                        other_skills.append(sk.get("name"))
            if other_skills:
                if st.toggle("다른 스킬도 메모하기", value=False, key=f"toggle_other_{entry_id}"):
                    st.markdown("---")
                    st.subheader("➕ 다른 스킬 메모")
                    st.caption("다른 스킬은 기본 템플릿(2+2)로 시작하며, 문구/메모 모두 수정 가능합니다.")
                    for osk in other_skills:
                        default_pr = [
                            f"{osk}을(를) 강화하기 위해, 다음 기록에 '내 행동'을 더 구체적으로 1문장 추가하기",
                            f"{osk} 관련 결과를 관찰 가능한 표현으로 1문장 추가하기",
                        ][:PRACTICE_N]
                        default_qs = [
                            f"오늘 {osk} 관점에서 내가 한 선택의 기준은 무엇이었나?",
                            f"다음엔 {osk} 관점에서 무엇을 바꾸면 더 좋아질까?",
                        ][:QUESTION_N]
                        render_memo_editor_for_skill(
                            entry_id=entry_id,
                            entry_date=entry["entry_date"],
                            skill_name=osk,
                            default_practices=default_pr,
                            default_questions=default_qs
                        )
        else:
            st.info("top 스킬을 결정할 수 없어 메모 섹션을 표시하지 않았습니다. (분석 결과에 soft_skills가 필요합니다.)")


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
                render_analysis_block(an)
            else:
                st.info("아직 분석 결과가 없습니다. 아래 버튼으로 분석을 실행할 수 있어요.")

            colb1, colb2 = st.columns([1, 1])
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
                            analysis = run_analysis_engine(engine=engine, entry=payload, related=related)
                            update_entry_analysis(r["id"], analysis)
                        st.success("분석 완료! 화면을 갱신합니다.")
                        st.rerun()

            with colb2:
                if st.button("🗑️ 삭제", key=f"del_{r['id']}"):
                    delete_entry(r["id"])
                    st.success("삭제했습니다. 화면을 갱신합니다.")
                    st.rerun()

            # Memo section (분석 결과가 있을 때만)
            an_now = r.get("analysis_parsed") or {}
            if isinstance(an_now, dict) and an_now.get("soft_skills"):
                top_skill = get_top_skill_from_analysis(an_now)
                if top_skill:
                    practices, questions = get_growth_items_for_top_skill(an_now)
                    st.markdown("---")
                    render_memo_editor_for_skill(
                        entry_id=r["id"],
                        entry_date=r["entry_date"],
                        skill_name=top_skill,
                        default_practices=practices,
                        default_questions=questions
                    )

                    other_skills = []
                    skills = an_now.get("soft_skills") or []
                    if isinstance(skills, list):
                        for sk in skills:
                            if isinstance(sk, dict) and sk.get("name") and sk.get("name") != top_skill:
                                other_skills.append(sk.get("name"))

                    if other_skills:
                        if st.toggle("다른 스킬도 메모하기", value=False, key=f"toggle_other_hist_{r['id']}"):
                            st.markdown("---")
                            st.subheader("➕ 다른 스킬 메모")
                            for osk in other_skills:
                                default_pr = [
                                    f"{osk}을(를) 강화하기 위해, 다음 기록에 '내 행동'을 더 구체적으로 1문장 추가하기",
                                    f"{osk} 관련 결과를 관찰 가능한 표현으로 1문장 추가하기",
                                ][:PRACTICE_N]
                                default_qs = [
                                    f"오늘 {osk} 관점에서 내가 한 선택의 기준은 무엇이었나?",
                                    f"다음엔 {osk} 관점에서 무엇을 바꾸면 더 좋아질까?",
                                ][:QUESTION_N]
                                render_memo_editor_for_skill(
                                    entry_id=r["id"],
                                    entry_date=r["entry_date"],
                                    skill_name=osk,
                                    default_practices=default_pr,
                                    default_questions=default_qs
                                )


def render_notes_page(df: pd.DataFrame):
    st.subheader("📒 메모")
    st.caption("메모는 기록(entry) 기준으로도, 소프트스킬 기준으로도 확인할 수 있어요.")

    tab1, tab2 = st.tabs(["날짜별", "스킬별"])

    with tab1:
        # 날짜 선택: entries에서 날짜 목록 생성
        dates = sorted(df["entry_date"].dropna().unique().tolist(), reverse=True) if not df.empty else []
        if not dates:
            st.info("아직 기록/메모가 없습니다.")
        else:
            d = st.selectbox("날짜 선택", options=dates, index=0)
            notes = fetch_skill_notes_by_date(d)
            if not notes:
                st.info("이 날짜에 저장된 메모가 없습니다.")
            else:
                # entry_id별 그룹
                by_entry: Dict[str, List[Dict[str, Any]]] = {}
                for n in notes:
                    by_entry.setdefault(n["entry_id"], []).append(n)

                for entry_id, ns in by_entry.items():
                    with st.expander(f"{d} · entry_id: {entry_id[:8]} · 메모 {len(ns)}개"):
                        # skill별 그룹
                        by_skill: Dict[str, List[Dict[str, Any]]] = {}
                        for n in ns:
                            by_skill.setdefault(n["skill_name"], []).append(n)
                        for sk, sk_notes in by_skill.items():
                            st.markdown(f"**{sk}**")
                            # practice/question 분리 출력
                            for nt in ["practice", "question"]:
                                items = [x for x in sk_notes if x["note_type"] == nt]
                                if not items:
                                    continue
                                st.caption("연습/루틴" if nt == "practice" else "다음 기록 질문")
                                for it in sorted(items, key=lambda x: int(x["item_index"])):
                                    st.write(f"- {it['item_text']}")
                                    if (it.get("memo_text") or "").strip():
                                        st.write(f"  ↳ 메모: {it['memo_text']}")

    with tab2:
        skill = st.selectbox("스킬 선택", options=SOFT_SKILLS, index=0)
        notes = fetch_skill_notes_by_skill(skill, limit=300)
        if not notes:
            st.info("이 스킬에 저장된 메모가 없습니다.")
        else:
            # entry_date별 그룹
            by_date: Dict[str, List[Dict[str, Any]]] = {}
            for n in notes:
                by_date.setdefault(n["entry_date"], []).append(n)

            for d in sorted(by_date.keys(), reverse=True):
                with st.expander(f"{d} · 메모 {len(by_date[d])}개"):
                    items = by_date[d]
                    for nt in ["practice", "question"]:
                        sub = [x for x in items if x["note_type"] == nt]
                        if not sub:
                            continue
                        st.caption("연습/루틴" if nt == "practice" else "다음 기록 질문")
                        for it in sorted(sub, key=lambda x: int(x["item_index"])):
                            st.write(f"- {it['item_text']}")
                            if (it.get("memo_text") or "").strip():
                                st.write(f"  ↳ 메모: {it['memo_text']}")
                            st.caption(f"entry_id: {it['entry_id'][:8]} · updated: {it['updated_at']}")


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
        "note_policy": "기본: top 스킬만 2+2 메모. 토글로 다른 스킬 확장.",
    })

    # notes count
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM skill_notes")
        note_count = cur.fetchone()[0]
    st.write("저장된 메모 개수(skill_notes):", note_count)

    st.info(
        "MetaTone 분석은 요약/STAR/패턴요약 없이, 행동·배움 + 스킬 근거/개념 + (top 스킬 기준) 2+2 메모 루틴에 집중합니다."
    )


if __name__ == "__main__":
    main()
