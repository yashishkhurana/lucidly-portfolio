# Strategy portfolio — Yashish Khurana

Strategies I designed and built at Lucidly Labs, 2024–26, rebuilt from scratch so they can be rerun. The
production code is Lucidly-owned and is not included; each strategy here is the control logic as pure
Python, a replay on bundled data, and a static page generated from its `results.json`. Every window is
stated as dates on its page; where an input is modelled rather than read from chain the page says so.

## Where the logic is

| file | what it holds |
|---|---|
| [`strategies/dn_ena/strategy.py`](strategies/dn_ena/strategy.py) | PT-sENA + ENA short: entry sizing, liquidation and margin health, reserve top-up (`topup_needed`), roll rule (`should_rotate`) |
| [`strategies/leverage_loop/strategy.py`](strategies/leverage_loop/strategy.py) | loop scoring, caps, net-APY gate and markdown guard state machines (`next_net_state`, `next_markdown_state`) |
| [`strategies/lending_cluster/strategy.py`](strategies/lending_cluster/strategy.py) | lending allocator (`solve_weights`, `plan`), entry/exit gates, junior-tranche risk rule (`depeg_risk_penalty`, `tranche_eligible`) |
| [`strategies/hedged_btc_mm/strategy.py`](strategies/hedged_btc_mm/strategy.py) | hedge sizing (`optimise_hedge`) and rebalance trigger |
| [`strategies/coin_carry_vault/strategy.py`](strategies/coin_carry_vault/strategy.py) | carry LTV ladder (`decide`) and delever routing |
| [`common/cl_engine.py`](common/cl_engine.py) · [`clmm.py`](common/clmm.py) · [`metrics.py`](common/metrics.py) | shared concentrated-liquidity simulator and measured-fee method, v3 tick math, performance metrics |
| `strategies/<name>/backtest.py` | the replays that drive each strategy over the bundled data |
| [`tests/`](tests) | 26 pytest cases for the math and the state machines |

The HTML pages are generated output (each inlines its results data so it opens offline): `pages/index.html`
for the overview, or any `strategies/<name>/index.html`.

| strategy | venue | data | module |
|---|---|---|---|
| dnENA — PT-sENA fixed yield + token-for-token ENA short | Pendle (Ethereum) + HyperCore | Pendle PT-sENA history per maturity, Hyperliquid ENA hourly funding, ENA 1h spot — pulled by `strategies/dn_ena/fetch_data.py` | `strategies/dn_ena` |
| Leveraged siUSD/USDC loop | Morpho Blue, Ethereum | real: 93-day archive-RPC series, 22 Mar – 22 Jun 2026 | `strategies/leverage_loop` |
| USDC lending cluster | Morpho Blue, Ethereum | real: same 93-day series (supply APY = borrow × utilisation) | `strategies/lending_cluster` |
| Hedged WBTC/WETH concentrated-liquidity MM | Uniswap v3 + Aave v3, Arbitrum | real: 25.7 h, 9,940 Swap events, fees measured | `strategies/hedged_btc_mm` |

Also in the repo, without pages: the junior-tranche risk rule in `strategies/lending_cluster/strategy.py` (depeg leverage
priced before allocation, tested; no Royco tranche series is bundled, so no scenario), `strategies/coin_carry_vault/strategy.py`
(LTV-ladder arithmetic, tested) and `strategies/eurc_usdc_mm` (the fee-measurement reconciliation against my public EURC/USDC
Aerodrome backtest: measured per-swap fees were 23 EURC on the log's own window against about 79 (51.0 EURC + 27.9 USDC) the published log accrued — the
old log credited the position too large a share of each swap. The correction is to the fee line: on the like-for-like window it
moved equity by 0.6 bp. Method in `common/cl_engine.py`).

## Run

```
pip install -r requirements.txt
python build_all.py                               # rebuilds every results.json + page
pytest                                            # 26 cases
python -m strategies.leverage_loop.backtest       # one strategy
python -m strategies.dn_ena.fetch_data --from 2025-06-01     # Pendle PT-sENA history (every maturity), HL ENA funding, Binance ENA 1h; needs network
ARBITRUM_RPC_URL=... python -m strategies.cl_swaps.fetch_data --pool wbtc_weth_arb     # Swap logs -> measured fees
python build_all.py                               # the dnENA and CL pages switch to real / measured automatically
```

Keys stay in environment variables; nothing in the repo reads a hard-coded RPC URL.

## Reporting rules (common/report.py)

- A tile or verdict is `good` only against a stated benchmark, on real data, over a window of at least 30 days.
- Nothing under 30 days is annualised; those pages show the window return only.
- Modelled or synthetic inputs are named in the scenario note and in the index table.
- Sharpe is not shown for windows under 100 samples or for stablecoin yield curves.
- Concentrated-liquidity P&L is reported versus holding the seeded tokens (fees − IL − slippage), so inventory revaluation does not pose as LP return.
- CL fees are *measured* from the pool's Swap events when `data/swaps_<pool>.csv` exists (per swap: amount_in × fee tier × L/(L_pool+L)); otherwise *modelled* and labelled as such on the tile.

## Data

- `data/algo_history.csv` — daily archive-RPC reads of three Morpho Blue markets (collateral exchange rate, borrow APR, supply/borrow), 2026-03-22 → 06-22. The loop page uses the siUSD/USDC rows; the lending page uses all three.
- `data/swaps_wbtc_weth_arb.csv`, `data/swaps_eurc_usdc_base.csv` — per-swap Swap events (amounts, price, pool liquidity, historical fee) pulled by `strategies/cl_swaps/fetch_data.py`.
- `data/wbtc_weth_arbitrum_1min.csv` — 1-minute WBTC/WETH 0.05% pool price, Arbitrum, 2026-03-10/11 (fallback path when no swap log is present).
- `data/pendle_pt_sena.csv`, `data/hl_funding_ENA.csv`, `data/binance_ENAUSDT_1h.csv` — written by `strategies/dn_ena/fetch_data.py` (layout documented in its docstring; the layout is the contract).

## Layout

```
common/      metrics.py · clmm.py · cl_engine.py (shared LP simulator, measured-fee method) · data.py · report.py
strategies/  <name>/strategy.py (pure logic) · backtest.py (replay; writes results.json + index.html)
tests/       pytest cases for the math and the state machines
pages/       index.html · template.html
```
