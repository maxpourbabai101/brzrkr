"""System page — sci-fi CPU/RAM/disk gauges + process list.

Live performance monitor for the entire trading_enhancer program.
Updates every 2 seconds via a separate Tk after() loop (doesn't
piggyback on broker polling — system metrics change faster than that).
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, FONT_SERIF
from brzrkr_app.widgets import (
    BarMeter, BloodMetric, CodexBox, GothicCard, InkDivider,
    PageTitle, SciFiGauge, SectionHeader, StatusBeacon,
)

ROOT = Path(__file__).resolve().parent.parent.parent


class SystemPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C.NIGHT)
        self._scroll.pack(fill="both", expand=True)
        body = self._scroll
        body.grid_columnconfigure((0, 1, 2, 3), weight=1)

        PageTitle(body, f"{G.GEAR} Anatomical Reading",
                   subtitle="cpu · memory · disk · network · processes"
                   ).grid(row=0, column=0, columnspan=4, sticky="ew",
                          pady=(0, 4))
        InkDivider(body, length=720).grid(row=1, column=0, columnspan=4,
                                            sticky="w", pady=(0, 12))

        if not _HAS_PSUTIL:
            ctk.CTkLabel(
                body,
                text=(f"{G.SKULL}  psutil not installed.\n\n"
                      "Install:  ./venv/bin/python -m pip install psutil\n"
                      "Then restart BRZRKR."),
                text_color=C.WOUND,
                font=ctk.CTkFont(family=FONT_MONO[0], size=12),
                justify="left",
            ).grid(row=2, column=0, columnspan=4, padx=20, pady=20, sticky="w")
            return

        # Top: machine identity
        ident = GothicCard(body)
        ident.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        ident.grid_columnconfigure(0, weight=1)
        SectionHeader(ident, "Vessel", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        self.ident_lbl = ctk.CTkLabel(
            ident, text="—",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            justify="left", anchor="w",
        )
        self.ident_lbl.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        # Gauge row: CPU, Memory, Disk
        self.cpu_gauge = SciFiGauge(body, "CPU", max_value=100, unit="%", size=160)
        self.mem_gauge = SciFiGauge(body, "Memory", max_value=100, unit="%", size=160)
        self.disk_gauge = SciFiGauge(body, "Disk", max_value=100, unit="%", size=160)
        self.proc_gauge = SciFiGauge(body, "Trader RAM", max_value=4096,
                                       unit="MB", size=160,
                                       color=C.SIGIL)
        self.cpu_gauge.grid(row=3, column=0, padx=(0, 6), sticky="nsew")
        self.mem_gauge.grid(row=3, column=1, padx=6, sticky="nsew")
        self.disk_gauge.grid(row=3, column=2, padx=6, sticky="nsew")
        self.proc_gauge.grid(row=3, column=3, padx=(6, 0), sticky="nsew")

        # Per-core CPU bars
        cores_card = GothicCard(body)
        cores_card.grid(row=4, column=0, columnspan=2, sticky="nsew",
                          padx=(0, 6), pady=(12, 0))
        cores_card.grid_columnconfigure(0, weight=1)
        SectionHeader(cores_card, "Cores", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")
        self.cores_frame = ctk.CTkFrame(cores_card, fg_color="transparent")
        self.cores_frame.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")
        self.core_bars: list[BarMeter] = []
        n_cores = psutil.cpu_count(logical=True) or 1
        for i in range(n_cores):
            b = BarMeter(self.cores_frame, label=f"c{i:02d}",
                          width=240, max_value=100, unit="%")
            b.grid(row=i, column=0, pady=2, sticky="ew")
            self.core_bars.append(b)

        # Memory breakdown
        mem_card = GothicCard(body)
        mem_card.grid(row=4, column=2, columnspan=2, sticky="nsew",
                       padx=(6, 0), pady=(12, 0))
        mem_card.grid_columnconfigure(0, weight=1)
        SectionHeader(mem_card, "Memory Detail", glyph=G.RUNE_F).grid(
            row=0, column=0, sticky="ew")
        body_m = ctk.CTkFrame(mem_card, fg_color="transparent")
        body_m.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")
        body_m.grid_columnconfigure(0, weight=1)

        self.mem_total = BarMeter(body_m, label="total", width=320,
                                    max_value=64 * 1024, unit="MB")
        self.mem_used  = BarMeter(body_m, label="used",  width=320,
                                    max_value=64 * 1024, unit="MB")
        self.mem_avail = BarMeter(body_m, label="avail", width=320,
                                    max_value=64 * 1024, unit="MB",
                                    color=C.SAGE)
        self.swap      = BarMeter(body_m, label="swap",  width=320,
                                    max_value=8 * 1024, unit="MB",
                                    color=C.OMEN)
        self.mem_total.grid(row=0, column=0, pady=2, sticky="ew")
        self.mem_used.grid(row=1, column=0, pady=2, sticky="ew")
        self.mem_avail.grid(row=2, column=0, pady=2, sticky="ew")
        self.swap.grid(row=3, column=0, pady=2, sticky="ew")

        # Network rate
        net_card = GothicCard(body)
        net_card.grid(row=5, column=0, columnspan=2, sticky="nsew",
                       padx=(0, 6), pady=(12, 0))
        net_card.grid_columnconfigure(0, weight=1)
        SectionHeader(net_card, "Network", glyph=G.RUNE_R).grid(
            row=0, column=0, sticky="ew")
        n_body = ctk.CTkFrame(net_card, fg_color="transparent")
        n_body.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")
        self.net_in  = BarMeter(n_body, label="in/s",  width=320,
                                  max_value=10 * 1024, unit="KB",
                                  color=C.LIFE)
        self.net_out = BarMeter(n_body, label="out/s", width=320,
                                  max_value=10 * 1024, unit="KB",
                                  color=C.WOUND)
        self.net_in.grid(row=0, column=0, pady=2, sticky="ew")
        self.net_out.grid(row=1, column=0, pady=2, sticky="ew")

        # Process list
        proc_card = GothicCard(body)
        proc_card.grid(row=5, column=2, columnspan=2, sticky="nsew",
                        padx=(6, 0), pady=(12, 0))
        proc_card.grid_columnconfigure(0, weight=1)
        SectionHeader(proc_card, "Trader Processes", glyph=G.GEAR).grid(
            row=0, column=0, sticky="ew")
        self.proc_box = CodexBox(proc_card, height=200)
        self.proc_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        # Internal state for network rate calc
        self._last_net = None
        self._last_net_ts = None

        # Identity once
        self._set_identity()

        # Start the tick loop (every 2s)
        self.after(500, self._tick)

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        # System tab runs its own faster loop; nothing to do here.
        pass

    # ------------------------------------------------------------------
    def _set_identity(self) -> None:
        uname = platform.uname()
        py = platform.python_version()
        proc_cpu = ""
        if platform.system() == "Darwin":
            try:
                proc_cpu = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True).strip()
            except Exception:
                proc_cpu = ""
        text = (
            f"  {G.RIGHT}  {uname.system} {uname.release}  ·  arch: {uname.machine}\n"
            f"  {G.RIGHT}  host: {uname.node}\n"
            f"  {G.RIGHT}  python: {py}\n"
        )
        if proc_cpu:
            text += f"  {G.RIGHT}  cpu: {proc_cpu}\n"
        n_cores = psutil.cpu_count(logical=True)
        n_phys = psutil.cpu_count(logical=False)
        text += f"  {G.RIGHT}  cores: {n_phys} physical, {n_cores} logical\n"
        text += f"  {G.RIGHT}  cwd: {os.getcwd()}"
        self.ident_lbl.configure(text=text)

    def _tick(self) -> None:
        if not _HAS_PSUTIL:
            return
        try:
            self._refresh()
        except Exception:
            pass
        self.after(1000, self._tick)   # 1 Hz; psutil reads are cheap

    def _refresh(self) -> None:
        # CPU
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        overall = sum(per_core) / len(per_core) if per_core else 0
        self.cpu_gauge.set(overall)
        for i, bar in enumerate(self.core_bars):
            if i < len(per_core):
                bar.set(per_core[i])

        # Memory
        m = psutil.virtual_memory()
        self.mem_gauge.set(m.percent)
        total_mb = m.total / (1024 * 1024)
        used_mb = (m.total - m.available) / (1024 * 1024)
        avail_mb = m.available / (1024 * 1024)
        # Resize meters' max if needed
        self.mem_total._max = total_mb
        self.mem_used._max = total_mb
        self.mem_avail._max = total_mb
        self.mem_total.set(total_mb)
        self.mem_used.set(used_mb)
        self.mem_avail.set(avail_mb)

        s = psutil.swap_memory()
        swap_mb = s.used / (1024 * 1024)
        self.swap._max = max(self.swap._max, s.total / (1024 * 1024), 1)
        self.swap.set(swap_mb)

        # Disk
        d = psutil.disk_usage("/")
        self.disk_gauge.set(d.percent)

        # Network rate
        now = datetime.now(timezone.utc)
        net = psutil.net_io_counters()
        if self._last_net is not None and self._last_net_ts is not None:
            dt = (now - self._last_net_ts).total_seconds()
            if dt > 0:
                in_kbs = (net.bytes_recv - self._last_net.bytes_recv) / dt / 1024
                out_kbs = (net.bytes_sent - self._last_net.bytes_sent) / dt / 1024
                self.net_in._max = max(self.net_in._max, in_kbs * 2, 100)
                self.net_out._max = max(self.net_out._max, out_kbs * 2, 100)
                self.net_in.set(max(0, in_kbs))
                self.net_out.set(max(0, out_kbs))
        self._last_net = net
        self._last_net_ts = now

        # Trader-related processes
        my_pid = os.getpid()
        rows = []
        total_rss_mb = 0.0
        for p in psutil.process_iter(attrs=["pid", "name", "cmdline",
                                              "memory_info", "cpu_percent"]):
            try:
                info = p.info
                name = info.get("name") or ""
                cmd = " ".join(info.get("cmdline") or [])
                # Heuristic: anything python in our cwd, or named BRZRKR/agent/etc.
                interesting = (
                    "BRZRKR" in cmd
                    or "agent.py" in cmd
                    or "trade.py" in cmd
                    or "train.py" in cmd
                    or "weekend_practice" in cmd
                    or "learn.py" in cmd
                    or info.get("pid") == my_pid
                )
                if not interesting:
                    continue
                mi = info.get("memory_info")
                rss_mb = (mi.rss / (1024 * 1024)) if mi else 0
                total_rss_mb += rss_mb
                cpu_pct = info.get("cpu_percent") or 0
                short_cmd = (cmd[:46] + "…") if len(cmd) > 47 else cmd
                marker = G.RUNE_T if info["pid"] == my_pid else G.RIGHT
                rows.append(
                    f"  {marker}  pid {info['pid']:>6d}  cpu {cpu_pct:>5.1f}%  "
                    f"rss {rss_mb:>6.1f}MB  {short_cmd}"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not rows:
            rows = [f"  {G.DOT_DIM}  no trader processes detected."]
        self.proc_box.set_text("\n".join(rows))

        # Process gauge: total trader RAM
        self.proc_gauge._max = max(self.proc_gauge._max, total_rss_mb * 2, 1024)
        self.proc_gauge.set(total_rss_mb)
