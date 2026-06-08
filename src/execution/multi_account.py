"""Multi-account Alpaca executor — manages 1-5 funded accounts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.execution.broker import AlpacaExecutor, ExecutionResult
from src.execution.brokers import InMemoryBroker
from src.execution.portfolio_allocator import PortfolioRiskAllocator
from src.execution.telegram_alerts import TelegramAlerter, Alerts, get_alerter

logger = logging.getLogger(__name__)


@dataclass
class AccountConfig:
    """Configuration for a single Alpaca account."""
    name: str
    api_key_env: str
    secret_key_env: str
    allocation_pct: float
    max_positions: int
    max_daily_loss_pct: float
    allowed_sessions: List[str]
    trade_futures: bool
    trade_crypto: bool
    paper: bool
    
    _executor: Optional[AlpacaExecutor] = field(default=None, init=False, repr=False)
    _starting_equity: float = field(default=0.0, init=False, repr=False)
    _daily_pl: float = field(default=0.0, init=False, repr=False)
    
    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")
    
    @property
    def secret_key(self) -> str:
        return os.getenv(self.secret_key_env, "")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)
    
    def get_executor(self, live_money: bool = False) -> AlpacaExecutor:
        """Get or create the AlpacaExecutor for this account."""
        if self._executor is None:
            self._executor = AlpacaExecutor(
                live_money=live_money and not self.paper,
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            try:
                self._starting_equity = self._executor.get_account_equity()
            except Exception:
                self._starting_equity = 0.0
        return self._executor
    
    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_pl = 0.0
        if self._executor:
            try:
                self._starting_equity = self._executor.get_account_equity()
            except Exception:
                pass
    
    def update_daily_pl(self) -> float:
        """Update and return current daily P&L."""
        if self._executor and self._starting_equity > 0:
            current = self._executor.get_account_equity()
            self._daily_pl = (current - self._starting_equity) / self._starting_equity
        return self._daily_pl
    
    def is_daily_loss_breached(self) -> bool:
        return abs(self._daily_pl) >= self.max_daily_loss_pct and self._daily_pl < 0


@dataclass
class PortfolioConfig:
    """Global portfolio constraints."""
    max_total_exposure_pct: float = 0.80
    max_sector_concentration_pct: float = 0.30
    max_symbol_concentration_pct: float = 0.15
    rebalance_drift_pct: float = 0.10
    correlation_limit: float = 0.70


@dataclass
class MultiAccountExecutor:
    """
    Manages multiple Alpaca accounts as a single logical portfolio.

    Routes signals to accounts based on allocation_pct, current positions,
    risk limits, and correlation constraints.
    """
    accounts: List[AccountConfig]
    portfolio: PortfolioConfig
    live_money: bool = False
    
    _account_map: Dict[str, AccountConfig] = field(default_factory=dict, init=False)
    _total_starting_equity: float = field(default=0.0, init=False)
    
    # Portfolio risk allocator
    _risk_allocator: Optional[PortfolioRiskAllocator] = field(default=None, init=False, repr=False)
    # Telegram alerts
    _alerter: Optional[TelegramAlerter] = field(default=None, init=False, repr=False)
    _alerts: Optional[Alerts] = field(default=None, init=False, repr=False)
    # Position monitor
    _monitor: Optional[Any] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        self._account_map = {acc.name: acc for acc in self.accounts}
        self._validate_config()
        
        # Initialize alerter
        try:
            self._alerter = get_alerter()
            self._alerts = Alerts(self._alerter)
        except Exception as e:
            logger.warning(f"Could not initialize Telegram alerter: {e}")
    
    def _validate_config(self) -> None:
        total_alloc = sum(a.allocation_pct for a in self.accounts)
        if abs(total_alloc - 1.0) > 0.01:
            raise ValueError(f"Account allocations must sum to 1.0, got {total_alloc}")
        
        configured = [a for a in self.accounts if a.is_configured]
        if not configured:
            logger.warning("No accounts have API keys configured!")
        else:
            logger.info(f"MultiAccountExecutor: {len(configured)}/{len(self.accounts)} accounts configured")
    
    def initialize(self) -> None:
        """Initialize all executors and record starting equity."""
        self._total_starting_equity = 0.0
        for acc in self.accounts:
            if acc.is_configured:
                try:
                    ex = acc.get_executor(live_money=self.live_money)
                    eq = ex.get_account_equity()
                    acc._starting_equity = eq
                    self._total_starting_equity += eq
                    logger.info(f"Account {acc.name}: equity=${eq:,.2f} (target alloc={acc.allocation_pct:.0%})")
                except Exception as e:
                    logger.error(f"Failed to init account {acc.name}: {e}")
        
        # Initialize portfolio risk allocator
        if self._total_starting_equity > 0:
            account_dicts = {}
            for acc in self.accounts:
                if acc.is_configured:
                    account_dicts[acc.name] = {
                        "equity": acc._starting_equity,
                        "allocation_pct": acc.allocation_pct,
                        "max_positions": acc.max_positions,
                        "max_daily_loss_pct": acc.max_daily_loss_pct,
                        "is_configured": True,
                    }
            self._risk_allocator = PortfolioRiskAllocator(
                accounts=account_dicts,
                total_equity=self._total_starting_equity,
                max_total_exposure=self.portfolio.max_total_exposure_pct,
                max_symbol_concentration=self.portfolio.max_symbol_concentration_pct,
                max_sector_concentration=self.portfolio.max_sector_concentration_pct,
                correlation_limit=self.portfolio.correlation_limit,
            )
            logger.info("PortfolioRiskAllocator initialized")
        
        # Send daily start alert
        if self._alerts:
            try:
                summary = self.get_portfolio_summary()
                self._alerts.daily_start(summary)
            except Exception as e:
                logger.warning(f"Daily start alert failed: {e}")
    
    def start_monitoring(self, interval_sec: int = 60) -> None:
        """Start position monitoring background thread."""
        if self._monitor is not None:
            logger.warning("Monitoring already started")
            return
        
        try:
            from src.execution.telegram_alerts import PositionMonitor
            self._monitor = PositionMonitor(self, self._alerter)
            self._monitor.start(interval_sec)
            logger.info(f"Position monitoring started (interval={interval_sec}s)")
        except Exception as e:
            logger.warning(f"Could not start position monitor: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop position monitoring."""
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
            logger.info("Position monitoring stopped")
    
    def get_total_equity(self) -> float:
        """Sum of equity across all accounts."""
        total = 0.0
        for acc in self.accounts:
            if acc.is_configured and acc._executor:
                try:
                    total += acc._executor.get_account_equity()
                except Exception:
                    total += acc._starting_equity
        return total
    
    def get_account_equity(self, account_name: str) -> float:
        acc = self._account_map.get(account_name)
        if acc and acc._executor:
            return acc._executor.get_account_equity()
        return acc._starting_equity if acc else 0.0
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Aggregate positions across all accounts with account tags."""
        all_positions = []
        for acc in self.accounts:
            if acc.is_configured and acc._executor:
                try:
                    positions = acc._executor.get_open_positions()
                    for p in positions:
                        p["account"] = acc.name
                        all_positions.append(p)
                except Exception as e:
                    logger.error(f"Failed to get positions for {acc.name}: {e}")
        return all_positions
    
    def get_all_orders(self, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        """Aggregate orders across all accounts."""
        all_orders = []
        for acc in self.accounts:
            if acc.is_configured and acc._executor:
                try:
                    orders = acc._executor.get_orders(status=status, limit=limit)
                    for o in orders:
                        o["account"] = acc.name
                        all_orders.append(o)
                except Exception as e:
                    logger.error(f"Failed to get orders for {acc.name}: {e}")
        return all_orders
    
    def _calculate_target_notional(self, signal_notional: float, account_name: str) -> float:
        """Calculate target notional for a specific account based on allocation."""
        acc = self._account_map.get(account_name)
        if not acc:
            return 0.0
        # Target is signal notional * account's allocation percentage
        return signal_notional * acc.allocation_pct
    
    def _can_add_position(self, account_name: str, symbol: str, notional: float) -> tuple[bool, str]:
        """Check if account can accept a new position."""
        acc = self._account_map.get(account_name)
        if not acc or not acc.is_configured or not acc._executor:
            return False, "account not configured"
        
        # Check daily loss limit
        if acc.is_daily_loss_breached():
            return False, f"daily loss breached ({acc._daily_pl:.1%})"
        
        # Check position count
        positions = acc._executor.get_open_positions()
        if len(positions) >= acc.max_positions:
            return False, f"max positions reached ({len(positions)}/{acc.max_positions})"
        
        # Check symbol concentration (no hedging same symbol)
        for p in positions:
            if p["symbol"] == symbol:
                return False, f"already have position in {symbol}"
        
        # Check notional vs account equity
        equity = acc._executor.get_account_equity()
        max_notional = equity * 0.15  # max 15% per trade per account
        if notional > max_notional:
            return False, f"notional ${notional:,.0f} exceeds 15% of equity ${equity:,.0f}"
        
        return True, "ok"
    
    def submit_signal(self, signal: Dict[str, Any]) -> Dict[str, ExecutionResult]:
        """
        Route a signal to appropriate accounts based on allocation strategy.
        Returns dict of account_name -> ExecutionResult.
        """
        symbol = signal["asset"]
        total_notional = signal["position_size_usd"]
        results = {}
        
        if total_notional <= 0:
            logger.warning(f"Signal for {symbol} has zero notional")
            return results
        
        # Check with risk allocator first
        sector = signal.get("sector")
        viable_accounts = []
        
        if self._risk_allocator:
            # Use sector mapping from config
            ticker_map = self._load_ticker_sector_map()
            sector = ticker_map.get(symbol, sector)
            
            allowed, reason, viable = self._risk_allocator.can_allocate(
                symbol, total_notional, sector=sector
            )
            if not allowed:
                logger.warning(f"Risk allocator blocked {symbol}: {reason}")
                if self._alerts:
                    self._alerts.alerter.send_warning(
                        f"Risk Block: {symbol}",
                        f"Reason: {reason}\nNotional: ${total_notional:,.0f}",
                    )
                return results
            viable_accounts = viable
        else:
            viable_accounts = [a.name for a in self.accounts if a.is_configured]
        
        # Filter and sort viable accounts
        candidate_accounts = [
            a for a in self.accounts 
            if a.is_configured and a.name in viable_accounts
        ]
        
        # Sort by position count (prefer less loaded)
        candidate_accounts.sort(
            key=lambda a: len(a._executor.get_open_positions()) if a._executor else 0
        )
        
        remaining_notional = total_notional
        allocations = {}
        submitted_count = 0
        
        for acc in candidate_accounts:
            if remaining_notional < 100:
                break
            
            target_notional = self._calculate_target_notional(total_notional, acc.name)
            if target_notional < 100:
                continue
            
            can_add, reason = self._can_add_position(acc.name, symbol, target_notional)
            if not can_add:
                logger.debug(f"Account {acc.name} cannot take {symbol}: {reason}")
                continue
            
            acc_signal = signal.copy()
            acc_signal["position_size_usd"] = target_notional
            acc_signal["account"] = acc.name
            
            executor = acc.get_executor(live_money=self.live_money)
            result = executor.submit_signal(acc_signal)
            results[acc.name] = result
            
            if result.submitted:
                remaining_notional -= target_notional
                allocations[acc.name] = target_notional
                submitted_count += 1
                logger.info(
                    f"Routed {symbol} to {acc.name}: ${target_notional:,.0f} "
                    f"(remaining: ${remaining_notional:,.0f})"
                )
            else:
                logger.warning(f"Account {acc.name} rejected {symbol}: {result.reason}")
        
        # Update risk allocator with actual allocations
        if self._risk_allocator and allocations:
            self._risk_allocator.allocate(symbol, sum(allocations.values()), list(allocations.keys()), sector=sector)
        
        # Send alerts
        if self._alerts:
            if submitted_count > 0:
                self._alerts.signal_executed(symbol, results)
            else:
                self._alerts.alerter.send_warning(
                    f"Signal Not Executed: {symbol}",
                    f"No accounts could accept the position. Remaining: ${remaining_notional:,.0f}",
                )
        
        if remaining_notional > total_notional * 0.1:
            logger.warning(f"Could only route ${total_notional - remaining_notional:,.0f} of ${total_notional:,.0f} for {symbol}")
        
        return results
    
    def _load_ticker_sector_map(self) -> Dict[str, str]:
        """Load ticker to sector mapping from config."""
        # Default mapping - can be extended
        return {
            "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
            "META": "technology", "GOOGL": "technology", "GOOG": "technology",
            "AMD": "technology", "TSLA": "consumer_discretionary",
            "JPM": "financials", "BAC": "financials", "GS": "financials",
            "MS": "financials", "XOM": "energy", "CVX": "energy", "COP": "energy",
            "FCX": "materials", "NEM": "materials",
            "SPY": "broad_market", "QQQ": "technology", "IWM": "small_cap",
            "DIA": "broad_market", "XLK": "technology", "XLF": "financials",
            "XLE": "energy", "XLV": "healthcare", "GLD": "commodities",
            "SLV": "commodities", "USO": "energy", "TLT": "bonds",
            "TQQQ": "technology", "SQQQ": "technology",
            "IBIT": "crypto", "GBTC": "crypto",
        }
    
    def close_position(self, account_name: str, symbol: str) -> bool:
        acc = self._account_map.get(account_name)
        if acc and acc._executor:
            return acc._executor.close_position(symbol)
        return False
    
    def cancel_all_orders(self, account_name: Optional[str] = None) -> int:
        total = 0
        accounts = [self._account_map[account_name]] if account_name else self.accounts
        for acc in accounts:
            if acc.is_configured and acc._executor:
                total += acc._executor.cancel_all_orders()
        return total
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get consolidated portfolio view."""
        total_equity = self.get_total_equity()
        all_positions = self.get_all_positions()
        
        # Calculate exposure by symbol
        symbol_exposure = {}
        for p in all_positions:
            sym = p["symbol"]
            mv = abs(p.get("market_value", 0))
            symbol_exposure[sym] = symbol_exposure.get(sym, 0) + mv
        
        # Calculate exposure by account
        account_exposure = {}
        for acc in self.accounts:
            if acc.is_configured and acc._executor:
                eq = acc._executor.get_account_equity()
                acc_positions = [p for p in all_positions if p["account"] == acc.name]
                exp = sum(abs(p.get("market_value", 0)) for p in acc_positions)
                account_exposure[acc.name] = {
                    "equity": eq,
                    "exposure": exp,
                    "exposure_pct": exp / eq if eq > 0 else 0,
                    "target_alloc": acc.allocation_pct,
                    "actual_alloc": eq / total_equity if total_equity > 0 else 0,
                    "positions": len(acc_positions),
                    "max_positions": acc.max_positions,
                    "daily_pl_pct": acc._daily_pl,
                }
        
        return {
            "total_equity": total_equity,
            "total_positions": len(all_positions),
            "symbol_exposure": symbol_exposure,
            "accounts": account_exposure,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        for acc in self.accounts:
            acc.reset_daily()
        
        # Send daily end alert for previous day
        if self._alerts:
            try:
                summary = self.get_portfolio_summary()
                # Calculate daily P&L
                daily_pl = 0
                for acc in self.accounts:
                    if acc.is_configured:
                        daily_pl += acc._daily_pl * acc._starting_equity
                summary["daily_pnl"] = daily_pl
                summary["daily_pnl_pct"] = daily_pl / self._total_starting_equity if self._total_starting_equity > 0 else 0
                self._alerts.daily_end(summary)
            except Exception as e:
                logger.warning(f"Daily end alert failed: {e}")
    
    def update_daily_pl(self) -> Dict[str, float]:
        """Update and return daily P&L for all accounts."""
        return {acc.name: acc.update_daily_pl() for acc in self.accounts if acc.is_configured}


def load_accounts_config(path: Path) -> tuple[List[AccountConfig], PortfolioConfig]:
    """Load accounts.yaml and return parsed configs."""
    data = yaml.safe_load(path.read_text())
    
    accounts = []
    for name, cfg in data.get("accounts", {}).items():
        accounts.append(AccountConfig(name=name, **cfg))
    
    portfolio = PortfolioConfig(**data.get("portfolio", {}))
    
    return accounts, portfolio