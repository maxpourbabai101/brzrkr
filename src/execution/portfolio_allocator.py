"""Portfolio risk allocator for multi-account trading."""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskBudget:
    """Risk budget allocated to a specific account."""
    account_name: str
    max_notional: float
    remaining_notional: float
    max_positions: int
    current_positions: int
    daily_loss_limit: float
    current_daily_loss: float
    is_active: bool = True
    
    @property
    def utilization_pct(self) -> float:
        if self.max_notional <= 0:
            return 0.0
        return 1.0 - (self.remaining_notional / self.max_notional)


class PortfolioRiskAllocator:
    """
    Allocates risk budget across multiple accounts based on:
    - Target allocation percentages
    - Current utilization
    - Daily loss limits
    - Correlation constraints
    """
    
    def __init__(
        self,
        accounts: Dict[str, Dict],
        total_equity: float,
        max_total_exposure: float = 0.80,
        max_symbol_concentration: float = 0.15,
        max_sector_concentration: float = 0.30,
        correlation_limit: float = 0.70,
    ):
        self.accounts = accounts
        self.total_equity = total_equity
        self.max_total_exposure = max_total_exposure
        self.max_symbol_concentration = max_symbol_concentration
        self.max_sector_concentration = max_sector_concentration
        self.correlation_limit = correlation_limit
        
        # Track allocated risk per symbol and sector
        self.symbol_allocation: Dict[str, float] = {}  # symbol -> notional
        self.sector_allocation: Dict[str, float] = {}  # sector -> notional
        
        self._initialize_budgets()
    
    def _initialize_budgets(self) -> None:
        """Initialize risk budgets for each account."""
        self.budgets = {}
        for name, cfg in self.accounts.items():
            if not cfg.get("is_configured", False):
                continue
            equity = cfg.get("equity", 0)
            allocation_pct = cfg.get("allocation_pct", 0)
            max_notional = equity * allocation_pct
            self.budgets[name] = RiskBudget(
                account_name=name,
                max_notional=max_notional,
                remaining_notional=max_notional,
                max_positions=cfg.get("max_positions", 5),
                current_positions=0,
                daily_loss_limit=cfg.get("max_daily_loss_pct", 0.03),
                current_daily_loss=0.0,
            )
        logger.info(f"Initialized risk budgets for {len(self.budgets)} accounts")
    
    def update_account_state(
        self,
        account_name: str,
        equity: float,
        positions: int,
        daily_loss_pct: float,
        symbol_exposure: Dict[str, float],
    ) -> None:
        """Update allocator with current account state."""
        if account_name not in self.budgets:
            return
        
        budget = self.budgets[account_name]
        budget.remaining_notional = budget.max_notional
        budget.current_positions = positions
        budget.current_daily_loss = daily_loss_pct
        
        # Reduce remaining notional by current exposure
        total_exposure = sum(abs(v) for v in symbol_exposure.values())
        budget.remaining_notional = max(0, budget.max_notional - total_exposure)
        
        # Check daily loss breach
        if daily_loss_pct >= budget.daily_loss_limit:
            budget.is_active = False
            logger.warning(f"Account {account_name} daily loss limit breached: {daily_loss_pct:.2%}")
        else:
            budget.is_active = True
    
    def can_allocate(
        self,
        symbol: str,
        notional: float,
        sector: Optional[str] = None,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> tuple[bool, str, List[str]]:
        """
        Check if a new position can be allocated.
        Returns (allowed, reason, viable_accounts).
        """
        # Check symbol concentration
        current_symbol = self.symbol_allocation.get(symbol, 0)
        if current_symbol + notional > self.total_equity * self.max_symbol_concentration:
            return False, f"symbol concentration limit: {symbol} would be {(current_symbol + notional) / self.total_equity:.1%}", []
        
        # Check sector concentration
        if sector:
            current_sector = self.sector_allocation.get(sector, 0)
            if current_sector + notional > self.total_equity * self.max_sector_concentration:
                return False, f"sector concentration limit: {sector} would be {(current_sector + notional) / self.total_equity:.1%}", []
        
        # Check total portfolio exposure
        total_current_exposure = sum(self.symbol_allocation.values())
        if total_current_exposure + notional > self.total_equity * self.max_total_exposure:
            return False, f"total portfolio exposure limit", []
        
        # Find viable accounts
        viable = []
        for name, budget in self.budgets.items():
            if not budget.is_active:
                continue
            if budget.remaining_notional < notional:
                continue
            if budget.current_positions >= budget.max_positions:
                continue
            if budget.current_daily_loss >= budget.daily_loss_limit:
                continue
            viable.append(name)
        
        if not viable:
            return False, "no accounts have capacity", []
        
        return True, "ok", viable
    
    def allocate(
        self,
        symbol: str,
        notional: float,
        accounts: List[str],
        sector: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Allocate notional across viable accounts.
        Returns dict of account_name -> allocated_notional.
        """
        # Proportional allocation based on remaining budget
        total_budget = sum(self.budgets[a].remaining_notional for a in accounts)
        if total_budget <= 0:
            return {}
        
        allocation = {}
        for acc in accounts:
            budget = self.budgets[acc]
            weight = budget.remaining_notional / total_budget
            alloc = min(notional * weight, budget.remaining_notional)
            allocation[acc] = alloc
            budget.remaining_notional -= alloc
        
        # Update global tracking
        self.symbol_allocation[symbol] = self.symbol_allocation.get(symbol, 0) + notional
        if sector:
            self.sector_allocation[sector] = self.sector_allocation.get(sector, 0) + notional
        
        return allocation
    
    def release(
        self,
        symbol: str,
        notional: float,
        sector: Optional[str] = None,
    ) -> None:
        """Release allocation when position is closed."""
        self.symbol_allocation[symbol] = max(0, self.symbol_allocation.get(symbol, 0) - notional)
        if sector:
            self.sector_allocation[sector] = max(0, self.sector_allocation.get(sector, 0) - notional)
    
    def get_allocation_summary(self) -> Dict:
        """Get current allocation state."""
        return {
            "symbol_exposure": dict(self.symbol_allocation),
            "sector_exposure": dict(self.sector_allocation),
            "total_exposure": sum(self.symbol_allocation.values()),
            "accounts": {
                name: {
                    "max_notional": b.max_notional,
                    "remaining_notional": b.remaining_notional,
                    "utilization": b.utilization_pct,
                    "positions": b.current_positions,
                    "max_positions": b.max_positions,
                    "daily_loss": b.current_daily_loss,
                    "daily_loss_limit": b.daily_loss_limit,
                    "is_active": b.is_active,
                }
                for name, b in self.budgets.items()
            }
        }


# Example usage
if __name__ == "__main__":
    accounts = {
        "primary": {"equity": 100000, "allocation_pct": 0.4, "max_positions": 8, "max_daily_loss_pct": 0.025, "is_configured": True},
        "secondary": {"equity": 75000, "allocation_pct": 0.3, "max_positions": 5, "max_daily_loss_pct": 0.03, "is_configured": True},
        "tertiary": {"equity": 50000, "allocation_pct": 0.3, "max_positions": 4, "max_daily_loss_pct": 0.02, "is_configured": True},
    }
    
    allocator = PortfolioRiskAllocator(
        accounts=accounts,
        total_equity=225000,
    )
    
    # Test allocation
    allowed, reason, viable = allocator.can_allocate("AAPL", 15000, sector="technology")
    print(f"Allowed: {allowed}, Reason: {reason}, Viable: {viable}")
    
    if allowed:
        allocation = allocator.allocate("AAPL", 15000, viable, sector="technology")
        print(f"Allocation: {allocation}")
    
    print(allocator.get_allocation_summary())