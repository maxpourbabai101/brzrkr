"""PostmortemDB — JSONL-backed knowledge base of quant failure modes.

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

File format is JSONL so it's human-readable, greppable, and survives
a corrupted line (the bad line is skipped and the rest still loads).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/postmortems.jsonl")


@dataclass
class Lesson:
    id: str
    category: str
    title: str
    description: str
    symptom: str
    mitigation: str
    severity: int = 3                       # 1 (advisory) .. 5 (account-ending)
    source: str = "seed"                    # "seed" | "observer" | "user"
    confirmed_count: int = 0                # incremented when observer sees the pattern
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
    path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Lesson] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self._index.clear()
        if not self.path.exists():
            self._loaded = True
            return
        for ln in self.path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
                lesson = Lesson.from_dict(d)
                self._index[lesson.id] = lesson
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed postmortem line: %s (%s)",
                                ln[:120], exc)
        self._loaded = True

    def _persist(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            for lesson in self._index.values():
                f.write(json.dumps(lesson.to_dict()) + "\n")
        tmp.replace(self.path)

    def reload(self) -> None:
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def all_lessons(self) -> List[Lesson]:
        if not self._loaded:
            self._load()
        return list(self._index.values())

    def get(self, lesson_id: str) -> Optional[Lesson]:
        if not self._loaded:
            self._load()
        return self._index.get(lesson_id)

    def find(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        min_severity: int = 1,
        source: Optional[str] = None,
    ) -> List[Lesson]:
        if not self._loaded:
            self._load()
        out = []
        for l in self._index.values():
            if category and l.category != category:
                continue
            if tag and tag not in l.tags:
                continue
            if l.severity < min_severity:
                continue
            if source and l.source != source:
                continue
            out.append(l)
        return sorted(out, key=lambda x: (-x.severity, -x.confirmed_count, x.id))

    def add_or_update(self, lesson: Lesson) -> bool:
        """Returns True if a new record was added, False if updated."""
        if not self._loaded:
            self._load()
        existing = self._index.get(lesson.id)
        is_new = existing is None
        now = datetime.now(timezone.utc).isoformat()
        if is_new:
            lesson.first_seen = lesson.first_seen or now
            self._index[lesson.id] = lesson
        else:
            # Merge: preserve confirmed_count, update timestamps.
            existing.title = lesson.title or existing.title
            existing.description = lesson.description or existing.description
            existing.symptom = lesson.symptom or existing.symptom
            existing.mitigation = lesson.mitigation or existing.mitigation
            existing.severity = max(existing.severity, lesson.severity)
            existing.references = list(set(existing.references + lesson.references))
            existing.tags = list(set(existing.tags + lesson.tags))
        self._persist()
        return is_new

    def confirm(self, lesson_id: str) -> None:
        """Increment the confirmation count for a lesson the system just
        saw evidence of. Quietly no-ops if the lesson is unknown."""
        if not self._loaded:
            self._load()
        l = self._index.get(lesson_id)
        if l is None:
            return
        l.confirmed_count += 1
        l.last_confirmed = datetime.now(timezone.utc).isoformat()
        self._persist()

    def remove(self, lesson_id: str) -> bool:
        if not self._loaded:
            self._load()
        existed = lesson_id in self._index
        self._index.pop(lesson_id, None)
        if existed:
            self._persist()
        return existed

    # ------------------------------------------------------------------
    # Bootstrap with seed knowledge on first use
    # ------------------------------------------------------------------
    def bootstrap_if_empty(self) -> int:
        """Seed the DB the first time it's used. Idempotent."""
        if not self._loaded:
            self._load()
        if self._index:
            return 0
        from src.learning.seed_knowledge import SEED_LESSONS
        for d in SEED_LESSONS:
            self._index[d["id"]] = Lesson.from_dict({
                **d,
                "source": "seed",
                "first_seen": datetime.now(timezone.utc).isoformat(),
            })
        self._persist()
        logger.info("PostmortemDB seeded with %d lessons", len(self._index))
        return len(self._index)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_markdown(self, *, top: int = 40) -> str:
        if not self._loaded:
            self._load()
        lessons = sorted(self._index.values(),
                          key=lambda x: (-x.severity, -x.confirmed_count))[:top]
        lines = [f"# Postmortem DB ({len(self._index)} lessons; showing top {len(lessons)})",
                 ""]
        for l in lessons:
            sev = "🔴" * l.severity + "⚪" * (5 - l.severity)
            stars = f"  ({l.confirmed_count}× confirmed)" if l.confirmed_count else ""
            lines.append(f"## {l.title}{stars}")
            lines.append(f"`{l.id}` · category=`{l.category}` · severity {sev} · source=`{l.source}`")
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
        if not self._loaded:
            self._load()
        by_cat: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_sev: Dict[int, int] = {}
        confirmed = 0
        for l in self._index.values():
            by_cat[l.category] = by_cat.get(l.category, 0) + 1
            by_source[l.source] = by_source.get(l.source, 0) + 1
            by_sev[l.severity] = by_sev.get(l.severity, 0) + 1
            if l.confirmed_count > 0:
                confirmed += 1
        return {
            "total": len(self._index),
            "confirmed_by_observation": confirmed,
            "by_category": by_cat,
            "by_source": by_source,
            "by_severity": by_sev,
        }
