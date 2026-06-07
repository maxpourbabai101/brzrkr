"""PostmortemDB — SQLite-backed knowledge base of quant failure modes.

Each entry is a :class:`Lesson` with a stable `id`, a `category`, a
short `title`, a longer `description`, a `symptom` (how to detect it
in *this* system), and a `mitigation` (what to do about it).

Two sources populate the DB:

1. **Seed knowledge** — published failures from quant history,
   bundled in :mod:`src.learning.seed_knowledge`. Loaded once via
   :meth:`PostmortemDB.bootstrap_if_empty`.

2. **Self-observation** — the :class:`SessionObserver` writes new
   lessons / increments confirmation counts based on what the agent
   actually does. This is the "self-updating" half.

Storage is SQLite (``data/brzrkr.db``). The public API is identical to
the old JSONL-backed version so all callers continue to work unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.db import get_db, init_schema

logger = logging.getLogger(__name__)

# kept for backward-compat; nothing uses it for IO any more
DEFAULT_DB_PATH = Path("data/postmortems.jsonl")


@dataclass
class Lesson:
    id: str
    category: str
    title: str
    description: str
    symptom: str
    mitigation: str
    severity: int = 3
    source: str = "seed"
    confirmed_count: int = 0
    first_seen: str = ""
    last_confirmed: str = ""
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lesson":
        return cls(
            id=d["id"],
            category=d["category"],
            title=d["title"],
            description=d.get("description", ""),
            symptom=d.get("symptom", ""),
            mitigation=d.get("mitigation", ""),
            severity=int(d.get("severity", 3)),
            source=d.get("source", "seed"),
            confirmed_count=int(d.get("confirmed_count", 0)),
            first_seen=d.get("first_seen", ""),
            last_confirmed=d.get("last_confirmed", ""),
            references=list(d.get("references", [])),
            tags=list(d.get("tags", [])),
        )


@dataclass
class PostmortemDB:
    path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)  # ignored; kept for compat

    def __post_init__(self) -> None:
        init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_lesson(row) -> Lesson:
        d = dict(row)
        d["references"] = json.loads(d.pop("references_json") or "[]")
        d["tags"] = json.loads(d.pop("tags_json") or "[]")
        return Lesson.from_dict(d)

    def _upsert(self, lesson: Lesson) -> None:
        conn = get_db()
        conn.execute(
            """INSERT INTO postmortems
                   (id, category, title, description, symptom, mitigation,
                    severity, source, confirmed_count, first_seen, last_confirmed,
                    references_json, tags_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   title           = excluded.title,
                   description     = excluded.description,
                   symptom         = excluded.symptom,
                   mitigation      = excluded.mitigation,
                   severity        = excluded.severity,
                   source          = excluded.source,
                   confirmed_count = excluded.confirmed_count,
                   first_seen      = COALESCE(postmortems.first_seen, excluded.first_seen),
                   last_confirmed  = excluded.last_confirmed,
                   references_json = excluded.references_json,
                   tags_json       = excluded.tags_json""",
            (
                lesson.id,
                lesson.category,
                lesson.title,
                lesson.description,
                lesson.symptom,
                lesson.mitigation,
                lesson.severity,
                lesson.source,
                lesson.confirmed_count,
                lesson.first_seen,
                lesson.last_confirmed,
                json.dumps(lesson.references),
                json.dumps(lesson.tags),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Legacy IO stubs (kept so subclasses / tests that call _load don't break)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        pass  # data lives in SQLite; nothing to load into memory

    def reload(self) -> None:
        pass  # no-op: reads go directly to DB

    # ------------------------------------------------------------------
    # Public API  (identical signatures to the old JSONL version)
    # ------------------------------------------------------------------

    @property
    def all_lessons(self) -> List[Lesson]:
        rows = get_db().execute("SELECT * FROM postmortems").fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def get(self, lesson_id: str) -> Optional[Lesson]:
        row = get_db().execute(
            "SELECT * FROM postmortems WHERE id=?", (lesson_id,)
        ).fetchone()
        return self._row_to_lesson(row) if row else None

    def find(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        min_severity: int = 1,
        source: Optional[str] = None,
    ) -> List[Lesson]:
        rows = get_db().execute("SELECT * FROM postmortems").fetchall()
        out = []
        for r in rows:
            lesson = self._row_to_lesson(r)
            if category and lesson.category != category:
                continue
            if tag and tag not in lesson.tags:
                continue
            if lesson.severity < min_severity:
                continue
            if source and lesson.source != source:
                continue
            out.append(lesson)
        return sorted(out, key=lambda x: (-x.severity, -x.confirmed_count, x.id))

    def add_or_update(self, lesson: Lesson) -> bool:
        """Returns True if a new record was added, False if updated."""
        existing = self.get(lesson.id)
        is_new = existing is None
        now = datetime.now(timezone.utc).isoformat()
        if is_new:
            lesson.first_seen = lesson.first_seen or now
        else:
            lesson.title       = lesson.title       or existing.title
            lesson.description = lesson.description or existing.description
            lesson.symptom     = lesson.symptom     or existing.symptom
            lesson.mitigation  = lesson.mitigation  or existing.mitigation
            lesson.severity    = max(existing.severity, lesson.severity)
            lesson.references  = list(set(existing.references + lesson.references))
            lesson.tags        = list(set(existing.tags + lesson.tags))
            lesson.confirmed_count = existing.confirmed_count
            lesson.first_seen  = existing.first_seen
        self._upsert(lesson)
        return is_new

    def confirm(self, lesson_id: str) -> None:
        conn = get_db()
        conn.execute(
            """UPDATE postmortems
               SET confirmed_count = confirmed_count + 1,
                   last_confirmed  = ?
               WHERE id = ?""",
            (datetime.now(timezone.utc).isoformat(), lesson_id),
        )
        conn.commit()

    def remove(self, lesson_id: str) -> bool:
        conn = get_db()
        cur = conn.execute("DELETE FROM postmortems WHERE id=?", (lesson_id,))
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_if_empty(self) -> int:
        count = get_db().execute("SELECT COUNT(*) FROM postmortems").fetchone()[0]
        if count:
            return 0
        from src.learning.seed_knowledge import SEED_LESSONS
        now = datetime.now(timezone.utc).isoformat()
        for d in SEED_LESSONS:
            lesson = Lesson.from_dict({**d, "source": "seed", "first_seen": now})
            self._upsert(lesson)
        n = get_db().execute("SELECT COUNT(*) FROM postmortems").fetchone()[0]
        logger.info("PostmortemDB seeded with %d lessons", n)
        return n

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown(self, *, top: int = 40) -> str:
        lessons = sorted(
            self.all_lessons,
            key=lambda x: (-x.severity, -x.confirmed_count),
        )[:top]
        total = get_db().execute("SELECT COUNT(*) FROM postmortems").fetchone()[0]
        lines = [f"# Postmortem DB ({total} lessons; showing top {len(lessons)})", ""]
        for l in lessons:
            sev   = "🔴" * l.severity + "⚪" * (5 - l.severity)
            stars = f"  ({l.confirmed_count}× confirmed)" if l.confirmed_count else ""
            lines.append(f"## {l.title}{stars}")
            lines.append(
                f"`{l.id}` · category=`{l.category}` · severity {sev} · source=`{l.source}`"
            )
            lines.append("")
            lines.append(f"**What it is:** {l.description}")
            lines.append("")
            lines.append(f"**How to detect:** {l.symptom}")
            lines.append("")
            lines.append(f"**Mitigation:** {l.mitigation}")
            if l.references:
                lines.append("")
                lines.append("**References:** " + "; ".join(l.references))
            if l.tags:
                lines.append("")
                lines.append("**Tags:** " + ", ".join(f"`{t}`" for t in l.tags))
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        by_cat: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_sev: Dict[int, int] = {}
        confirmed = 0
        for l in self.all_lessons:
            by_cat[l.category]   = by_cat.get(l.category, 0) + 1
            by_source[l.source]  = by_source.get(l.source, 0) + 1
            by_sev[l.severity]   = by_sev.get(l.severity, 0) + 1
            if l.confirmed_count > 0:
                confirmed += 1
        return {
            "total": sum(by_cat.values()),
            "confirmed_by_observation": confirmed,
            "by_category": by_cat,
            "by_source": by_source,
            "by_severity": by_sev,
        }
