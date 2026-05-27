"""Self-learning subsystem.

Persistent knowledge base of (a) documented failure modes from quant
history and (b) the system's own observations of its behavior over
time. Consulted before trades and training runs; appended to by the
observer after every session.
"""

from src.learning.postmortem_db import Lesson, PostmortemDB
from src.learning.observer import SessionObserver
from src.learning.preflight import PreflightReport, run_preflight

__all__ = [
    "Lesson", "PostmortemDB",
    "SessionObserver",
    "PreflightReport", "run_preflight",
]
