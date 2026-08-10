import streamlit as st

st.title("P.article Bank")

import os
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st
import trafilatura
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# Database
# ============================================================

DB_PATH = Path(__file__).parent / "articles.db"

THEMES = ["Human Rights & Justice", "Development & Sustainability", "Peace & Conflict"]

# Minimum fit score for an article to count as "in" a theme when filtering/counting.
# Every article still gets a score against all three themes, stored regardless of threshold,
# so the full gradient is always available for display.
THEME_FIT_THRESHOLD = 0.35


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT NOT NULL,
            source TEXT,
            date_published TEXT,
            date_added TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS article_themes (
            article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            theme_id INTEGER NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
            fit_score REAL NOT NULL,
            PRIMARY KEY (article_id, theme_id)
        );

        CREATE TABLE IF NOT EXISTS analyses (
            article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
            claims_evidence TEXT NOT NULL,
            framing_bias TEXT NOT NULL,
            political_theory TEXT NOT NULL,
            actors_stakes TEXT NOT NULL,
            comparisons TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    for name in THEMES:
        conn.execute("INSERT OR IGNORE INTO themes (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_article(title, url, source, date_published, raw_text, embedding: np.ndarray) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO articles (url, title, source, date_published, date_added, raw_text, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, title, source, date_published, _now(), raw_text, embedding.astype(np.float32).tobytes()),
    )
    conn.commit()
    article_id = cur.lastrowid
    conn.close()
    return article_id


def save_theme_scores(article_id: int, scores: dict[str, float]) -> None:
    conn = get_connection()
    for name, score in scores.items():
        row = conn.execute("SELECT id FROM themes WHERE name = ?", (name,)).fetchone()
        if not row:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO article_themes (article_id, theme_id, fit_score) VALUES (?, ?, ?)",
            (article_id, row["id"], score),
        )
    conn.commit()
    conn.close()


def save_analysis(
    article_id, claims_evidence, framing_bias, political_theory, actors_stakes, comparisons
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO analyses
            (article_id, claims_evidence, framing_bias, political_theory, actors_stakes, comparisons, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            json.dumps(claims_evidence),
            json.dumps(framing_bias),
            json.dumps(political_theory),
            json.dumps(actors_stakes),
            json.dumps(comparisons),
            _now(),
        ),
    )
    conn.commit()
    conn.close()


def list_themes(min_fit: float = THEME_FIT_THRESHOLD) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT themes.id, themes.name,
               COUNT(CASE WHEN article_themes.fit_score >= ? THEN 1 END) AS article_count
        FROM themes
        LEFT JOIN article_themes ON themes.id = article_themes.theme_id
        GROUP BY themes.id
        ORDER BY themes.name
        """,
        (min_fit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_articles(theme_id: int | None = None, min_fit: float = THEME_FIT_THRESHOLD) -> list[dict]:
    conn = get_connection()
    if theme_id:
        rows = conn.execute(
            """
            SELECT articles.*, article_themes.fit_score AS theme_fit_score
            FROM articles
            JOIN article_themes ON articles.id = article_themes.article_id
            WHERE article_themes.theme_id = ? AND article_themes.fit_score >= ?
            ORDER BY article_themes.fit_score DESC
            """,
            (theme_id, min_fit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM articles ORDER BY date_added DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_article_theme_scores(article_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT themes.name, article_themes.fit_score
        FROM themes
        JOIN article_themes ON themes.id = article_themes.theme_id
        WHERE article_themes.article_id = ?
        ORDER BY article_themes.fit_score DESC
        """,
        (article_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis(article_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM analyses WHERE article_id = ?", (article_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "claims_evidence": json.loads(row["claims_evidence"]),
        "framing_bias": json.loads(row["framing_bias"]),
        "political_theory": json.loads(row["political_theory"]),
        "actors_stakes": json.loads(row["actors_stakes"]),
        "comparisons": json.loads(row["comparisons"]),
    }


def embedding_of(article_row: dict) -> np.ndarray:
    return np.frombuffer(article_row["embedding"], dtype=np.float32)


# ============================================================
# Embeddings
# ============================================================

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embed_model() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL_NAME)


def embed(text: str) -> np.ndarray:
    return _embed_model().encode(text, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def top_similar(
    query_vec: np.ndarray,
    candidates: list[tuple[dict, np.ndarray]],
    top_n: int = 3,
    min_similarity: float = 0.3,
) -> list[tuple[float, dict]]:
    scored = [(cosine_sim(query_vec, vec), article) for article, vec in candidates]
    scored = [pair for pair in scored if pair[0] >= min_similarity]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_n]


# ============================================================
# Article extraction
# ============================================================


def is_url(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("http://") or stripped.startswith("https://")


def fetch_article(url: str) -> dict:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch {url}")

    result = trafilatura.extract(
        downloaded, include_comments=False, output_format="json", with_metadata=True
    )
    if not result:
        raise ValueError(f"Could not extract article text from {url}")

    data = json.loads(result)
    text = data.get("text") or ""
    if not text.strip():
        raise ValueError(f"No article text found at {url}")

    return {
        "title": data.get("title") or url,
        "source": data.get("sitename") or "",
        "date_published": data.get("date") or "",
        "text": text,
    }


# ============================================================
# Claude analysis
# ============================================================

MODEL = os.environ.get("POLARTICLES_MODEL", "claude-sonnet-5")

POLITICAL_THEORIES = [
    "Classical Realism",
    "Defensive Realism",
    "Offensive Realism",
    "Liberalism",
    "Critical Theory",
    "Constructivism",
    "Liberal Institutionalism",
]

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


ANALYSIS_TOOL = {
    "name": "record_analysis",
    "description": "Record structured analysis of a political article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims_evidence": {
                "type": "array",
                "description": "The main claims made in the article and how well each is backed up.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_cited": {"type": "string"},
                        "evidence_strength": {
                            "type": "string",
                            "enum": ["strong", "moderate", "weak", "absent"],
                        },
                    },
                    "required": ["claim", "evidence_cited", "evidence_strength"],
                },
            },
            "framing_bias": {
                "type": "object",
                "description": "How the article frames the subject and what perspectives it favors or omits.",
                "properties": {
                    "tone": {"type": "string"},
                    "loaded_language": {"type": "array", "items": {"type": "string"}},
                    "perspectives_missing": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["tone", "loaded_language", "perspectives_missing", "notes"],
            },
            "political_theory": {
                "type": "array",
                "description": (
                    "Which political theory lens(es) best explain the dynamics in this article. "
                    "Be selective - only include theories that genuinely apply, not all of them."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "theory": {"type": "string", "enum": POLITICAL_THEORIES},
                        "explanation": {
                            "type": "string",
                            "description": "Why this theory's lens fits what's happening in the article.",
                        },
                    },
                    "required": ["theory", "explanation"],
                },
            },
            "actors_stakes": {
                "type": "array",
                "description": "Who is involved in the article and what they stand to gain or lose.",
                "items": {
                    "type": "object",
                    "properties": {
                        "actor": {"type": "string"},
                        "role": {"type": "string"},
                        "stake": {"type": "string"},
                    },
                    "required": ["actor", "role", "stake"],
                },
            },
        },
        "required": ["claims_evidence", "framing_bias", "political_theory", "actors_stakes"],
    },
}

THEME_FIT_TOOL = {
    "name": "score_theme_fit",
    "description": "Score how strongly an article fits each of a fixed set of themes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string", "enum": THEMES},
                        "score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "0 = no fit at all, 1 = perfect fit.",
                        },
                    },
                    "required": ["theme", "score"],
                },
                "minItems": len(THEMES),
                "maxItems": len(THEMES),
                "description": "One entry per theme, in any order.",
            }
        },
        "required": ["scores"],
    },
}

COMPARISON_TOOL = {
    "name": "record_comparisons",
    "description": "For each candidate article, explain what is different about the new article's situation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "comparisons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {
                            "type": "integer",
                            "description": "The bracketed index of the candidate article being compared to.",
                        },
                        "what_is_different": {"type": "string"},
                    },
                    "required": ["candidate_index", "what_is_different"],
                },
            }
        },
        "required": ["comparisons"],
    },
}


def analyze_article(title: str, text: str) -> dict:
    response = client().messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "record_analysis"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this political article using the record_analysis tool. "
                    "Be specific and concrete - quote or closely paraphrase the article "
                    "rather than generalizing. Keep the framing/bias notes honest about "
                    "slant rather than presenting the article as neutral by default. "
                    "For political theory, only name lenses that genuinely fit this "
                    "article's dynamics, not a checklist of every theory.\n\n"
                    f"Title: {title}\n\nText:\n{text[:12000]}"
                ),
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_analysis":
            return block.input
    raise RuntimeError("Claude did not return a structured analysis")


def score_theme_fit(title: str, text: str) -> dict[str, float]:
    response = client().messages.create(
        model=MODEL,
        max_tokens=256,
        tools=[THEME_FIT_TOOL],
        tool_choice={"type": "tool", "name": "score_theme_fit"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Score how strongly this article fits each of the following fixed themes, "
                    "from 0 (no fit) to 1 (perfect fit). An article can score meaningfully on "
                    "more than one theme if it genuinely overlaps between them - give an honest "
                    "gradient rather than forcing a single choice.\n\n"
                    f"Themes: {', '.join(THEMES)}\n\n"
                    f"Title: {title}\n\nText:\n{text[:4000]}"
                ),
            }
        ],
    )
    scores = {name: 0.0 for name in THEMES}
    for block in response.content:
        if block.type == "tool_use" and block.name == "score_theme_fit":
            for entry in block.input["scores"]:
                if entry["theme"] in scores:
                    scores[entry["theme"]] = float(entry["score"])
    return scores


def compare_to_similar(title: str, text: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    candidate_blocks = [
        f"[{i}] {c['title']}\n{c['raw_text'][:2000]}" for i, c in enumerate(candidates)
    ]
    response = client().messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[COMPARISON_TOOL],
        tool_choice={"type": "tool", "name": "record_comparisons"},
        messages=[
            {
                "role": "user",
                "content": (
                    "The new article below is topically similar to each candidate article. "
                    "For each candidate, explain what is specifically different about the new "
                    "article's situation - new developments, different actors, a different "
                    "outcome, escalation/de-escalation, etc. Don't just restate the similarity.\n\n"
                    f"New article - Title: {title}\n\nText:\n{text[:8000]}\n\n"
                    f"Candidate articles:\n\n" + "\n\n".join(candidate_blocks)
                ),
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_comparisons":
            return block.input["comparisons"]
    return []


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Political Article Analyst", layout="wide")
init_db()

st.title("Political Article Analyst")

with st.sidebar:
    st.header("Add an article")
    add_input = st.text_area("Paste a URL or the article text", height=120)
    if st.button("Add", type="primary", use_container_width=True):
        if not add_input.strip():
            st.warning("Paste a URL or some text first.")
        else:
            with st.spinner("Fetching and analyzing..."):
                try:
                    if is_url(add_input):
                        article = fetch_article(add_input.strip())
                        url = add_input.strip()
                    else:
                        stripped = add_input.strip()
                        article = {
                            "title": stripped.splitlines()[0][:80],
                            "source": "",
                            "date_published": "",
                            "text": stripped,
                        }
                        url = None

                    vec = embed(article["text"])

                    # Find comparable articles before inserting, so the new article
                    # never compares itself against itself.
                    existing_articles = list_articles()
                    candidates = [(a, embedding_of(a)) for a in existing_articles]
                    similar = top_similar(vec, candidates)

                    article_id = insert_article(
                        title=article["title"],
                        url=url,
                        source=article["source"],
                        date_published=article["date_published"],
                        raw_text=article["text"],
                        embedding=vec,
                    )

                    theme_scores = score_theme_fit(article["title"], article["text"])
                    save_theme_scores(article_id, theme_scores)

                    result = analyze_article(article["title"], article["text"])

                    comparison_hits = compare_to_similar(
                        article["title"], article["text"], [c for _, c in similar]
                    )
                    comparisons = []
                    for hit in comparison_hits:
                        idx = hit["candidate_index"]
                        if 0 <= idx < len(similar):
                            score, candidate = similar[idx]
                            comparisons.append(
                                {
                                    "compared_to_id": candidate["id"],
                                    "compared_to_title": candidate["title"],
                                    "similarity": score,
                                    "what_is_different": hit["what_is_different"],
                                }
                            )

                    save_analysis(
                        article_id,
                        result["claims_evidence"],
                        result["framing_bias"],
                        result["political_theory"],
                        result["actors_stakes"],
                        comparisons,
                    )

                    st.success(f"Added: {article['title']}")
                except Exception as e:
                    st.error(f"Failed to add article: {e}")

    st.divider()
    st.header("Themes")
    themes = list_themes()
    theme_options = {"All articles": None}
    for t in themes:
        theme_options[f"{t['name']} ({t['article_count']})"] = t["id"]
    selected_label = st.radio("Filter by theme", list(theme_options.keys()))
    selected_theme_id = theme_options[selected_label]

st.subheader("Search")
query = st.text_input("What are you interested in?", placeholder="e.g. coverage of the debt ceiling fight")

articles = list_articles(theme_id=selected_theme_id)
scores: dict[int, float] = {}

if query.strip():
    q_vec = embed(query.strip())
    scored = [(cosine_sim(q_vec, embedding_of(a)), a) for a in articles]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    articles = [a for _, a in scored]
    scores = {a["id"]: score for score, a in scored}

st.caption(f"{len(articles)} article(s)")

for a in articles:
    theme_scores = get_article_theme_scores(a["id"])
    score_label = f" — relevance {scores[a['id']]:.2f}" if a["id"] in scores else ""
    with st.expander(f"{a['title']}{score_label}"):
        meta_bits = []
        if a.get("source"):
            meta_bits.append(a["source"])
        if a.get("date_published"):
            meta_bits.append(a["date_published"])
        if meta_bits:
            st.caption(" | ".join(meta_bits))
        if a.get("url"):
            st.markdown(f"[Original article]({a['url']})")

        st.markdown("**Theme fit**")
        for t in theme_scores:
            st.progress(t["fit_score"], text=f"{t['name']} — {t['fit_score'] * 100:.0f}%")

        result = get_analysis(a["id"])
        if not result:
            st.info("No analysis stored for this article.")
            continue

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Claims & Evidence", "Framing & Bias", "Political Theory", "Actors & Stakes", "Comparisons"]
        )

        with tab1:
            for item in result["claims_evidence"]:
                st.markdown(f"**Claim:** {item['claim']}")
                st.markdown(f"*Evidence:* {item['evidence_cited']} — **{item['evidence_strength']}**")
                st.markdown("---")

        with tab2:
            fb = result["framing_bias"]
            st.markdown(f"**Tone:** {fb['tone']}")
            if fb["loaded_language"]:
                st.markdown("**Loaded language:** " + ", ".join(fb["loaded_language"]))
            if fb["perspectives_missing"]:
                st.markdown("**Missing perspectives:** " + ", ".join(fb["perspectives_missing"]))
            if fb["notes"]:
                st.markdown(fb["notes"])

        with tab3:
            if not result["political_theory"]:
                st.info("No clear theoretical lens identified.")
            for item in result["political_theory"]:
                st.markdown(f"**{item['theory']}**")
                st.markdown(item["explanation"])
                st.markdown("---")

        with tab4:
            for item in result["actors_stakes"]:
                st.markdown(f"**{item['actor']}** ({item['role']}) — {item['stake']}")

        with tab5:
            if not result["comparisons"]:
                st.info("No comparable articles yet.")
            for item in result["comparisons"]:
                st.markdown(f"**vs. {item['compared_to_title']}** ({item['similarity'] * 100:.0f}% similar)")
                st.markdown(item["what_is_different"])
                st.markdown("---")

    )
