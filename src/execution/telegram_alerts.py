"""Telegram monitoring and alerts for BRZRKR."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


class TelegramAlerter:
    """Send alerts via Telegram bot."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "8312135341")
        self.enabled = enabled and bool(self.bot_token)
        self._lock = threading.Lock()
        self._last_alert_time: Dict[str, float] = {}  # For rate limiting
        
        if self.enabled:
            logger.info(f"TelegramAlerter initialized for chat {self.chat_id}")
        else:
            logger.warning("TelegramAlerter disabled (no bot token)")
    
    def send(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        deduplicate_window_sec: int = 60,
    ) -> bool:
        """Send an alert via Telegram."""
        if not self.enabled:
            return False
        
        # Deduplication key
        dedup_key = f"{level.value}:{title}"
        now = time.time()
        
        with self._lock:
            last_time = self._last_alert_time.get(dedup_key, 0)
            if now - last_time < deduplicate_window_sec:
                logger.debug(f"Telegram alert deduplicated: {dedup_key}")
                return False
            self._last_alert_time[dedup_key] = now
        
        # Build message with emoji
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(level, "📢")
        
        text = f"{emoji} *{title}*\n{message}"
        
        if metadata:
            text += "\n\n*Details:*"
            for k, v in metadata.items():
                text += f"\n  `{k}`: {v}"
        
        text += f"\n\n_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_"
        
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Telegram alert sent: {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_info(self, title: str, message: str, **kwargs) -> bool:
        return self.send(AlertLevel.INFO, title, message, kwargs.get("metadata"))
    
    def send_warning(self, title: str, message: str, **kwargs) -> bool:
        return self.send(AlertLevel.WARNING, title, message, kwargs.get("metadata"))
    
    def send_error(self, title: str, message: str, **kwargs) -> bool:
        return self.send(AlertLevel.ERROR, title, message, kwargs.get("metadata"))
    
    def send_critical(self, title: str, message: str, **kwargs) -> bool:
        return self.send(AlertLevel.CRITICAL, title, message, kwargs.get("metadata"))


class PositionMonitor:
    """Monitor positions and trigger alerts."""
    
    def __init__(self, executor, alerter: TelegramAlerter):
        self.executor = executor
        self.alerter = alerter
        self._last_positions: Dict[str, Dict] = {}
        self._last_equity: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, interval_sec: int = 60) -> None:
        """Start monitoring in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(interval_sec,), daemon=True)
        self._thread.start()
        logger.info(f"PositionMonitor started (interval={interval_sec}s)")
    
    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PositionMonitor stopped")
    
    def _run(self, interval_sec: int) -> None:
        while self._running:
            try:
                self._check_positions()
                self._check_equity()
            except Exception as e:
                logger.error(f"PositionMonitor check failed: {e}")
            time.sleep(interval_sec)
    
    def _check_positions(self) -> None:
        """Check for new/closed positions and significant changes."""
        try:
            positions = self.executor.get_all_positions()
            current = {f"{p['account']}:{p['symbol']}": p for p in positions}
            
            # New positions
            for key, pos in current.items():
                if key not in self._last_positions:
                    self.alerter.send_info(
                        f"New Position: {pos['symbol']} ({pos['account']})",
                        f"Direction: {pos.get('side', 'long').upper()}\n"
                        f"Qty: {pos.get('qty', 0):.0f}\n"
                        f"Entry: ${pos.get('avg_entry_price', 0):.2f}\n"
                        f"Market Value: ${pos.get('market_value', 0):,.2f}",
                        metadata={"account": pos['account'], "symbol": pos['symbol']},
                    )
            
            # Closed positions
            for key, pos in self._last_positions.items():
                if key not in current:
                    acc, sym = key.split(":", 1)
                    pnl = pos.get('unrealized_pl', 0)
                    pnl_pct = pos.get('unrealized_plpc', 0) * 100
                    self.alerter.send_info(
                        f"Position Closed: {sym} ({acc})",
                        f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                        f"Side: {pos.get('side', 'long')}",
                        metadata={"account": acc, "symbol": sym, "pnl": pnl},
                    )
            
            self._last_positions = current
        except Exception as e:
            logger.error(f"Position check failed: {e}")
    
    def _check_equity(self) -> None:
        """Check for significant equity changes."""
        try:
            for acc in self.executor.accounts:
                if not acc.is_configured or not acc._executor:
                    continue
                equity = acc._executor.get_account_equity()
                last = self._last_equity.get(acc.name)
                
                if last is not None:
                    change_pct = (equity - last) / last
                    if abs(change_pct) > 0.02:  # 2% threshold
                        level = AlertLevel.WARNING if change_pct < 0 else AlertLevel.INFO
                        self.alerter.send(
                            level,
                            f"Equity Change: {acc.name}",
                            f"${last:,.2f} → ${equity:,.2f} ({change_pct:+.2%})",
                            metadata={"account": acc.name, "equity": equity, "change_pct": change_pct},
                        )
                
                self._last_equity[acc.name] = equity
        except Exception as e:
            logger.error(f"Equity check failed: {e}")


# Convenience functions for common alerts
class Alerts:
    """Static methods for common alert scenarios."""
    
    def __init__(self, alerter: TelegramAlerter):
        self.alerter = alerter
    
    def daily_start(self, accounts_summary: Dict) -> None:
        total = accounts_summary.get("total_equity", 0)
        self.alerter.send_info("Trading Session Started", f"Total Portfolio: ${total:,.2f}")
    
    def daily_end(self, accounts_summary: Dict) -> None:
        total = accounts_summary.get("total_equity", 0)
        pnl = accounts_summary.get("daily_pnl", 0)
        pnlpct = accounts_summary.get("daily_pnl_pct", 0)
        self.alerter.send_info("Trading Session Ended", f"Total: ${total:,.2f}\nDaily P&L: ${pnl:+,.2f} ({pnlpct:+.2%})")
    
    def signal_generated(self, symbol: str, direction: str, confidence: float, accounts: List[str]) -> None:
        self.alerter.send_info(
            f"Signal: {symbol} {direction.upper()}",
            f"Confidence: {confidence:.1%}\nAccounts: {', '.join(accounts)}",
        )
    
    def signal_executed(self, symbol: str, accounts_results: Dict) -> None:
        successful = [a for a, r in accounts_results.items() if r.submitted]
        failed = [a for a, r in accounts_results.items() if not r.submitted]
        
        if successful:
            self.alerter.send_info(
                f"Order Executed: {symbol}",
                f"Success: {', '.join(successful)}",
            )
        if failed:
            self.alerter.send_warning(
                f"Order Failed: {symbol}",
                f"Failed: {', '.join(failed)}",
            )
    
    def daily_loss_breach(self, account: str, loss_pct: float, limit: float) -> None:
        self.alerter.send_critical(
            f"Daily Loss Breach: {account}",
            f"Loss: {loss_pct:.2%} (limit: {limit:.2%})\nTrading halted for this account.",
        )
    
    def max_positions_reached(self, account: str, count: int, max_pos: int) -> None:
        self.alerter.send_warning(
            f"Max Positions: {account}",
            f"{count}/{max_pos} positions open. No new entries until slots free.",
        )
    
    def error(self, context: str, error: Exception) -> None:
        self.alerter.send_error(
            f"Error in {context}",
            f"{type(error).__name__}: {str(error)[:500]}",
        )


# Global instance getter (configured once at startup)
_alerter: Optional[TelegramAlerter] = None


def get_alerter(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> TelegramAlerter:
    """Get or create global alerter instance."""
    global _alerter
    if _alerter is None:
        _alerter = TelegramAlerter(bot_token=bot_token, chat_id=chat_id)
    return _alerter