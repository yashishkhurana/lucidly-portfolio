# Changes

## v5 (2026-10-23)
- DN cluster: hourly funding and candle timestamps aligned to the hour (an exact-timestamp join had kept 28 of 2,160 rows); candle OPEN used as the acted-on price (no look-ahead), partial last candle dropped; returns on capital; fifth scenario with no funding gate; part-fill seed band and taker-vs-maker caveat stated; "What the real data says" generated from the scenario numbers.
- dn_cluster/strategy.py: allocation weights floored at zero (needed for the no-gate row; no effect on gated rows); `orphaned_leg`, `funding_window_h`, `basis_haircut`, `part_fill_prob`.
- cl_swaps/fetch_data.py: Swap-event puller over JSON-RPC with historical `fee()` per row; common/cl_engine.py measured-fee mode (amount_in × tier × L/(L_pool+L)); both CL backtests switch to it when `data/swaps_<pool>.csv` exists.
- EURC/USDC: deposit in EURC-equivalent of the log's USDC equity; real scenario on the log's own band and window; P&L vs hold decomposed with a named remainder (`reval`); fee provenance and the extended-window rotation stated.
- Hedged WBTC/WETH: real-day note follows the fee mode; sweep and tiles on the same benchmark.
- coin_carry_vault: seizure capped at collateral, residual bad debt socialised, LTV recorded at day open, drop-to-liquidation sign fixed.
- common/report.py: `good` only against a stated benchmark on real data ≥ 30 d; no annualisation under 30 d; simple annualisation everywhere.
- tests/: 28 cases (CL round-trips, IL closed form, haircut conservation, engine decomposition, Swap decoder + measured fees, net-gate hysteresis, markdown sustain, spill-on-cap, solver bounds, liquidation formulas, DN close rules and allocation, ladder decisions).

## v1–v4
Earlier passes: clean-room rewrite of the seven strategies, reporting rules, tests, reviews 1–4 applied.
