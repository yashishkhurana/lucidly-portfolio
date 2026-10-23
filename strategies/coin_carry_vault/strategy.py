"""BTC / ETH carry clusters — LTV ladder, carry guard and the delever decision tree.

Clean-room reimplementation of the cbBTC/USDC and wstETH/USDC carry vaults I ran at Lucidly
(Morpho Blue, Ethereum mainnet). The depositor keeps the asset's price exposure; the vault
borrows USDC against it to a target LTV and parks the USDC in a yield vault (syUSD).

    equity (in asset units) = collateral - debt/px + credit/px
    net carry on borrowed USDC = yield_vault_apy (+ staking yield on the collateral for ETH) - borrow_apr
    LTV = debt_usd / collateral_usd ; rises when the asset falls
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ladder:
    min_ltv: float = 0.60       # re-lever below this (if carry positive)
    target_ltv: float = 0.65
    max_ltv: float = 0.70       # de-lever back to target above this
    emergency_ltv: float = 0.74  # bypass gas budget
    liquidation_ltv: float = 0.86  # Morpho market LLTV
    relever_max_borrow_apr: float = 0.05
    min_tx_usd: float = 50.0


def ltv(debt_usd: float, collateral_usd: float) -> float:
    return debt_usd / collateral_usd if collateral_usd > 0 else 0.0


def price_drop_to_liquidation(current_ltv: float, lltv: float) -> float:
    """Asset drawdown that lifts LTV to LLTV: ltv/(1-d) = lltv -> d = 1 - ltv/lltv."""
    return 1 - current_ltv / lltv


def decide(debt_usd: float, collateral_usd: float, borrow_apr: float, carry_apy: float, pending_withdraw_usd: float,
           L: Ladder) -> tuple[str, float]:
    """Return (action, usdc_amount). Safety preempts economics; withdrawals preempt re-levering."""
    cur = ltv(debt_usd, collateral_usd)
    if cur > L.max_ltv:
        repay = debt_usd - L.target_ltv * collateral_usd
        return ("delever_emergency" if cur > L.emergency_ltv else "delever", repay)
    if pending_withdraw_usd > 0:
        return ("service_withdrawal", pending_withdraw_usd)
    if cur < L.min_ltv and borrow_apr <= L.relever_max_borrow_apr and carry_apy > borrow_apr:
        borrow = L.target_ltv * collateral_usd - debt_usd
        return ("relever", borrow) if borrow >= L.min_tx_usd else ("hold", 0.0)
    return ("hold", 0.0)


def delever_route(needed_usd: float, yield_vault_liquidity_usd: float, idle_usdc: float, shares_value_usd: float) -> str:
    """The production decision tree for freeing USDC to repay:
    Q1 yield-vault liquidity >= needed? no -> flash deleverage (sell collateral inside a flashloan)
    Q2 residual after idle > dust? no -> repay with idle (1 tx)
    Q3 vault holds enough shares? no -> flash deleverage ; yes -> two-stage redeem (request, ~60 s, self-solve close)"""
    if yield_vault_liquidity_usd < needed_usd:
        return "flash_deleverage"
    residual = needed_usd - idle_usdc
    if residual <= 1.0:
        return "repay_with_idle"
    if shares_value_usd < residual:
        return "flash_deleverage"
    return "two_stage_redeem"
