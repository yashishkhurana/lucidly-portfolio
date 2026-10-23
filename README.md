# Strategy portfolio — Yashish Khurana

This is a record of strategies I designed, ran and post-mortemed with real capital at Lucidly Labs
(2024–2026), rebuilt from scratch so the control logic and backtests can be read and rerun. It is DeFi
yield and structured-product work: the transferable parts are the cost-aware control design
(hysteresis, deadbands, churn accounting), the position and hedge maths, and the habit of writing down
what broke. It is not mid-frequency alpha research; that work is in the `quant-research` repo, in
progress. Where a result rests on synthetic data the page says so, and no result here should be read
as a return claim. The production code is Lucidly-owned and is not included.

## Where the logic is

| file | what it holds |
|---|---|
| [`strategies/dn_cluster/strategy.py`](strategies/dn_cluster/strategy.py) | delta-neutral allocation, leg sizing, close rules (`should_close`) |
| [`strategies/leverage_loop/strategy.py`](strategies/leverage_loop/strategy.py) | loop scoring, caps, net-APY gate and markdown guard state machines (`next_net_state`, `next_markdown_state`) |
| [`strategies/coin_carry_vault/strategy.py`](strategies/coin_carry_vault/strategy.py) | carry LTV ladder (`decide`) and delever routing |
| [`strategies/lending_cluster/strategy.py`](strategies/lending_cluster/strategy.py) | lending allocator: entry/exit gates, capacity, utilisation impact |
| [`strategies/hedged_btc_mm/strategy.py`](strategies/hedged_btc_mm/strategy.py) | hedge sizing (`optimise_hedge`) and rebalance trigger |
| [`common/cl_engine.py`](common/cl_engine.py) · [`clmm.py`](common/clmm.py) · [`metrics.py`](common/metrics.py) | shared concentrated-liquidity simulator, v3 tick math, performance metrics |
| `strategies/<name>/backtest.py` | the replays that drive each strategy over real or synthetic data |
| [`tests/`](tests) | 28 pytest cases for the math and the state machines |

The HTML pages are generated output (each inlines its results data so it opens offline): `pages/index.html`
for the overview, or any `strategies/<name>/index.html`.

| strategy | venue | data | module |
|---|---|---|---|
| Leveraged stablecoin loop cluster | Morpho Blue, Ethereum | real (93-day archive series) | `strategies/leverage_loop` |
| Delta-neutral funding-carry cluster (+ PT variant) | Hyperliquid, HyperEVM | real: 90 d HL funding + spot (candle opens), fetched 2026-10-23 | `strategies/dn_cluster` |
| USDC lending cluster + junior-tranche postmortem | Morpho Blue, Royco, Ethereum | real (same series); postmortem from chain data | `strategies/lending_cluster` |
| Hedged WBTC/WETH concentrated-liquidity MM | Uniswap v3 + Aave v3, Arbitrum | real 25.7 h with measured fees (9,940 Swap events); synthetic sweep | `strategies/hedged_btc_mm` |
| EURC/USDC concentrated-liquidity MM | Aerodrome, Base | real: the public log's 5.2 h window with measured fees (Swap events, 76-pip fee); synthetic sweep | `strategies/eurc_usdc_mm` |
| BTC / ETH carry vaults (ladder arithmetic) | Morpho Blue, Ethereum | real borrow APR; synthetic price paths | `strategies/coin_carry_vault` |
| Daily volatility-cone tool | Deribit, Hyperliquid | one archived run | `strategies/vol_cone_forecast` |

## Run

```
pip install -r requirements.txt
python build_all.py                               # rebuilds every results.json + page
pytest                                            # 28 cases
python -m strategies.leverage_loop.backtest       # one strategy
python -m strategies.dn_cluster.fetch_data        # 90 d HL funding + Binance 1h spot (HYPE from HL candles); needs network
ARBITRUM_RPC_URL=... python -m strategies.cl_swaps.fetch_data --pool wbtc_weth_arb     # Swap logs -> measured fees
BASE_RPC_URL=... python -m strategies.cl_swaps.fetch_data --pool eurc_usdc_base --start 2026-01-04T23:00Z --end 2026-01-05T04:17Z   # the public log's window
python build_all.py                               # the DN and CL pages switch to real / measured automatically
```

Keys stay in environment variables; nothing in the repo reads a hard-coded RPC URL.

## Reporting rules (common/report.py)

- A tile or verdict is `good` only against a stated benchmark, on real data, over a window of at least 30 days.
- Nothing under 30 days is annualised; those pages show the window return only.
- Synthetic inputs are named in the scenario note and in the index table.
- Sharpe is not shown for windows under 100 samples or for stablecoin yield curves.
- Concentrated-liquidity P&L is reported versus holding the seeded tokens (fees − IL − slippage), so inventory revaluation does not pose as LP return.
- CL fees are *measured* from the pool's Swap events when `data/swaps_<pool>.csv` exists (per swap: amount_in × fee tier × L/(L_pool+L)); otherwise *modelled* and labelled as such on the tile.

## Data

- `data/algo_history.csv` — daily archive-RPC reads of three Morpho Blue markets (collateral exchange rate,
  borrow APR, supply/borrow), 2026-03-22 → 06-22.
- `data/swaps_wbtc_weth_arb.csv`, `data/swaps_eurc_usdc_base.csv` — per-swap Swap events (amounts, price, pool liquidity, historical fee) pulled by `strategies/cl_swaps/fetch_data.py`; the CL backtests charge measured fees from these.
- `data/wbtc_weth_arbitrum_1min.csv` — 1-minute WBTC/WETH 0.05% pool price, Arbitrum, 2026-03-10/11 (fallback path when no swap log is present).
- `data/eurc_usdc_aero_backtest.csv` and the `*.png` plots — from my public `lucidlylabs/simulations` repo.
- `data/hl_funding_<COIN>.csv`, `data/binance_<SYM>_1h.csv` — 90 days of Hyperliquid hourly funding and 1h spot (HYPE candles from Hyperliquid), fetched 2026-10-23.
- `data/forecast_sample_2026-10-22.json` — one run of the daily forecaster.
- Synthetic paths (GBM / OU prices, AR(1) funding) are generated with fixed seeds inside each backtest.

## Layout

```
common/      metrics.py · clmm.py · cl_engine.py (shared LP simulator) · data.py · report.py
strategies/  <name>/strategy.py (pure logic) · backtest.py (replay; writes results.json + index.html)
tests/       pytest cases for the math and the state machines
pages/       index.html · template.html
```
