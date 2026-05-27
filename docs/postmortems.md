# Postmortem DB (52 lessons; showing top 52)

## Backtest overfitting from too many strategy variations
`l_001` · category=`overfitting` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Trying N variations of a strategy and picking the best by Sharpe gives a fake Sharpe that scales with sqrt(log N). Most published backtests with Sharpe > 2 do not replicate.

**How to detect:** Best-of-K strategy variations all have Sharpe > 1.5 in-sample; live performance is sub-1.0.

**Mitigation:** Use deflated Sharpe ratio (Bailey & López de Prado 2014); penalize by number of variations tried; combinatorial purged CV.

**References:** Bailey, Borwein, López de Prado, Zhu (2014); Bailey & López de Prado (2014) Deflated Sharpe Ratio

**Tags:** `backtest`, `ml`, `sharpe`

---

## Lookahead bias from same-day data
`l_002` · category=`leakage` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Computing today's signal from today's close and trading at today's close is lookahead. Same applies to fundamentals (earnings released after close used in same-day signal).

**How to detect:** Backtest dramatically outperforms paper trading; signal magnitude shrinks with execution lag added.

**Mitigation:** Lag all features by at least one bar. Use point-in-time fundamentals only.

**References:** López de Prado (2018) ch 4

**Tags:** `leakage`, `backtest`

---

## Random K-fold CV on time series
`l_003` · category=`leakage` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Random shuffling for cross-validation lets the model see the future from the past. Common ML mistake that's catastrophic in finance.

**How to detect:** CV scores ~0.65, walk-forward / live scores ~0.51.

**Mitigation:** Use TimeSeriesSplit or purged combinatorial CV. Never shuffle.

**References:** López de Prado (2018) ch 7

**Tags:** `leakage`, `ml`, `cv`

---

## Quant Quake (Aug 6-10, 2007)
`l_007` · category=`regime_shift` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Several large quant funds with similar factor exposures faced simultaneous redemptions. Forced de-leveraging cascaded; models lost 6-15% in 3 days against their backtests.

**How to detect:** Daily losses 4+ standard deviations from backtest; multiple uncorrelated alphas drawing down together.

**Mitigation:** Stress-test for crowdedness; cap position sizes well below the 'optimal' from backtest.

**References:** Khandani & Lo (2011) What Happened to the Quants in August 2007?

**Tags:** `regime`, `crowding`, `factor`

---

## Full-Kelly is too aggressive in practice
`l_009` · category=`risk_sizing` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Kelly criterion assumes known probabilities and zero estimation error. Using estimated edge with full Kelly typically leads to >50% drawdowns even with positive expected value.

**How to detect:** Equity curve has 40%+ drawdowns despite positive long-term return.

**Mitigation:** Use half-Kelly or quarter-Kelly. Cap any single position at 5% of equity.

**References:** MacLean, Thorp, Ziemba (2010); Already implemented in src/risk/risk_manager.py

**Tags:** `kelly`, `analyzer:confirmed_negative`, `sizing`

---

## Long Term Capital Management (1998)
`l_010` · category=`risk_sizing` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Convergence trades with 25x+ leverage. When Russia defaulted and correlations went to 1, LTCM lost $4.6B in 4 months — required Fed-organized bailout.

**How to detect:** Multiple 'uncorrelated' positions move together in a stress event.

**Mitigation:** Stress-test under correlation = 1 assumption; cap gross leverage at 3-5x for retail; never trade convergence at extreme leverage.

**References:** Lowenstein (2000) When Genius Failed

**Tags:** `leverage`, `crisis`, `correlation`

---

## Position sizing as fixed dollars vs % of equity
`l_011` · category=`risk_sizing` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Doubling down on losers (martingale) or fixed-dollar sizing after losses leads to ruin even with positive edge.

**How to detect:** Position sizes increase after losses.

**Mitigation:** Always size as % of current equity (not starting equity). Never increase size after a loss.

**References:** Thorp (1969)

**Tags:** `sizing`, `behavioral`

---

## Knight Capital August 1, 2012
`l_012` · category=`execution` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Code deployment error left old SMARS routing logic active. In 45 minutes Knight sent millions of erroneous orders, losing $440M. Firm bankrupt within a week.

**How to detect:** Order volume spikes wildly above strategy's normal turnover.

**Mitigation:** Kill-switch on order rate; manual sign-off for deploys; canary deploy with size cap.

**References:** SEC enforcement action 2013

**Tags:** `deployment`, `execution`, `operational`

---

## Tail risk shorts blow up in 1-day events
`l_017` · category=`liquidity` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Selling deep OTM puts/calls for premium has positive expected value most years, then loses 5-10 years of profit in one event (VIX Feb 2018, XIV collapse).

**How to detect:** Strategy makes small steady gains, then a single day takes 50%+.

**Mitigation:** Never sell naked vol without delta-hedge or defined risk. Read Taleb 'Dynamic Hedging'.

**References:** Taleb (1997)

**Tags:** `options`, `tail_risk`, `vol_selling`

---

## Swiss Franc removal of EUR peg (Jan 15, 2015)
`l_027` · category=`macro` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** SNB removed its 1.20 EUR/CHF floor without warning. CHF rallied 20% in minutes. Many FX retail brokers and several hedge funds went bankrupt overnight.

**How to detect:** Carry/peg trades with 'no risk' in central bank guidance.

**Mitigation:** Never assume any central bank commitment is permanent. Cap exposure to pegged regimes.

**References:** Various 2015 broker bankruptcy filings

**Tags:** `fx`, `macro`, `tail_risk`

---

## Crypto exchange insolvencies (Mt. Gox, FTX)
`l_030` · category=`counterparty` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Customer funds commingled, lent out, or stolen. Mt. Gox lost 850k BTC in 2014; FTX lost $8B+ in 2022.

**How to detect:** Exchange offering yields on customer balances; withdrawal delays; founder spending lavishly.

**Mitigation:** Self-custody all crypto. Use exchanges only for active trading.

**References:** FTX bankruptcy filings

**Tags:** `counterparty`, `crypto`

---

## API key with too-broad permissions
`l_032` · category=`deployment` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Trading bot's API key has withdrawal permission. Compromised key drains the account.

**How to detect:** API key has more scopes than the bot needs.

**Mitigation:** Use trading-only keys; never enable withdrawals via API. Rotate keys regularly.

**References:** Multiple exchange post-breach reports

**Tags:** `security`, `api`

---

## Negative oil price (April 20, 2020)
`l_044` · category=`macro` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** WTI front-month futures settled at -$37.63 due to storage constraints + futures roll mechanics. Many retail oil ETF holders wiped out.

**How to detect:** Futures backwardation extreme; storage at capacity; settlement approaching.

**Mitigation:** Don't hold front-month commodity futures over settlement. Avoid leveraged commodity ETFs for >1 day.

**References:** CFTC retrospective

**Tags:** `commodities`, `futures`, `tail_risk`

---

## No paper-trading period before live
`l_046` · category=`deployment` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Strategy goes from backtest to live with no paper-trading validation. First live week reveals 3 bugs that erase 20% of starting capital.

**How to detect:** Strategy goes live without 4+ weeks of paper validation.

**Mitigation:** Mandatory paper period before live (enforced by the promotion gate in this repo).

**References:** Universal practitioner consensus

**Tags:** `deployment`, `process`

---

## Selection bias in 'best strategy' from many backtests
`l_050` · category=`overfitting` · severity 🔴🔴🔴🔴🔴 · source=`seed`

**What it is:** Test 100 strategies, the best by Sharpe will have Sharpe 2.5 even on pure noise. Picking it doesn't mean it's good.

**How to detect:** Final strategy is the best of many tried; CV scores cluster near top of distribution.

**Mitigation:** Pre-register hypotheses. Use deflated Sharpe. Walk-forward fold-by-fold accuracy must be consistent, not just average.

**References:** Harvey, Liu, Zhu (2016); Bailey & López de Prado (2014)

**Tags:** `overfitting`, `selection_bias`, `process`

---

## Survivorship bias in universe selection
`l_004` · category=`leakage` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Backtesting on today's S&P 500 components ignores companies that were dropped (often after failing). Inflates returns by 1-3% annually.

**How to detect:** Long-only backtests on 'large caps' look better than they should.

**Mitigation:** Use point-in-time index membership data (CRSP, Compustat, or open-source equivalents).

**References:** Elton, Gruber, Blake (1996)

**Tags:** `data_quality`, `backtest`

---

## Target encoding leakage
`l_005` · category=`leakage` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Using target statistics (e.g., mean return of category) computed on the full dataset leaks future info into features.

**How to detect:** Categorical features with suspiciously high importance.

**Mitigation:** Compute target encodings only on the training fold; recompute per fold.

**References:** Kaggle: any target leakage post-mortem

**Tags:** `leakage`, `ml`, `features`

---

## Momentum crash of 2009
`l_006` · category=`regime_shift` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** After the March 2009 bottom, the cross-sectional momentum factor reversed violently — losers became winners. AQR, Renaissance, and others took 20-30% drawdowns in months.

**How to detect:** Long-momentum portfolios at multi-year lows during sharp risk-on rebounds.

**Mitigation:** Add volatility scaling to momentum sizing; reduce exposure when realized vol spikes.

**References:** Daniel & Moskowitz (2016) Momentum Crashes

**Tags:** `regime`, `momentum`, `factor`

---

## 2022 rates shock killed long-duration trades
`l_008` · category=`regime_shift` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** After 14 years of falling rates, the Fed hiking cycle inverted every long-duration carry trade simultaneously. Long bond funds lost 20%+; many factor portfolios with implicit duration exposure also bled.

**How to detect:** Strategies with no obvious rate exposure correlate strongly with TLT during a regime change.

**Mitigation:** Decompose alpha into factor exposures; monitor rate sensitivity even for non-rate strategies.

**References:** Various 2022 hedge fund letters

**Tags:** `regime`, `macro`, `duration`

---

## Slippage underestimation in backtest
`l_013` · category=`execution` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Most backtests assume mid-price fills. Real fills cost 2-30 bps even for liquid names; more for illiquid. A strategy with 1 bp/trade alpha is unprofitable after costs.

**How to detect:** Backtest Sharpe drops 30-50% when realistic costs added.

**Mitigation:** Apply 2-5 bps per side for liquid US equities; 10+ for small-caps; vary by spread + size.

**References:** Almgren & Chriss (2000)

**Tags:** `execution`, `backtest`, `costs`

---

## Flash Crash (May 6, 2010)
`l_015` · category=`execution` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** $4.1B sell algo caused liquidity vacuum; major indices dropped 9% in minutes then recovered. Stop orders triggered at absurd prices (e.g., Accenture at $0.01).

**How to detect:** Wild quote anomalies in normal-hours trading.

**Mitigation:** Avoid market orders during volatile periods; use limit orders with reasonable bands.

**References:** SEC/CFTC 2010 Report

**Tags:** `execution`, `crisis`, `liquidity`

---

## Liquidity vanishes precisely when needed
`l_016` · category=`liquidity` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** In crisis, bid/ask widens 10-100x. Your hedges can't be lifted; your underwater positions can't be exited.

**How to detect:** Spreads widen sharply during P&L stress.

**Mitigation:** Trade only names with >$10M daily volume; pre-stage exits before vol spikes; maintain cash reserve.

**References:** O'Hara (2015)

**Tags:** `liquidity`, `crisis`

---

## Gamestop / WSB short squeezes (Jan 2021)
`l_018` · category=`crowding` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Retail coordination via Reddit forced short covers on heavily-shorted small/mid-caps. Melvin Capital lost 50%+; multiple funds blew up.

**How to detect:** Short interest > 50% of float in a heavily-discussed-online name.

**Mitigation:** Avoid concentrated shorts in names with high social-media mention rate.

**References:** Various 2021 hedge fund letters

**Tags:** `crowding`, `shorting`, `retail`

---

## Diversification fails in crises
`l_020` · category=`correlation` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Cross-asset correlations spike to 0.8-1.0 during sell-offs. Your 'diversified' portfolio behaves like a single concentrated long.

**How to detect:** All positions red simultaneously on a single risk-off day.

**Mitigation:** Stress-test under correlation=1 scenario; size for the bad case; hold tail hedges that pay out in stress.

**References:** Longin & Solnik (2001) Extreme Correlation of International Equity Markets

**Tags:** `correlation`, `crisis`, `diversification`

---

## Point-in-time vs as-reported fundamentals
`l_022` · category=`data_quality` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Fundamentals databases that don't track revisions let you backtest as if you knew the final restated numbers in real time. Inflates returns 1-3% annually.

**How to detect:** Value/quality strategies backtest much better than they live-trade.

**Mitigation:** Use point-in-time fundamentals (Compustat, Sharadar, S&P Capital IQ).

**References:** López de Prado (2018) ch 4

**Tags:** `data_quality`, `fundamentals`

---

## Discretionary override of systematic stops
`l_023` · category=`behavioral` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Operator overrides a stop because 'it'll come back'. Sometimes it does; the time it doesn't kills the strategy.

**How to detect:** Manual position closures / extensions in the trade log.

**Mitigation:** Either trade fully systematic or fully discretionary. Don't mix without a written protocol.

**References:** Tharp (2007)

**Tags:** `behavioral`, `stops`

---

## Revenge trading after a loss
`l_024` · category=`behavioral` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** After a loss, operator increases size or breaks rules to 'win it back'. Compounds the original loss.

**How to detect:** Position size rises right after a loss; rule violations cluster post-drawdown.

**Mitigation:** Hard-coded cooldown after losses (already in src/risk/countermeasures.py).

**References:** Lo (2017) Adaptive Markets

**Tags:** `behavioral`, `psychology`

---

## Confirmation bias in backtest interpretation
`l_025` · category=`behavioral` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Operator tweaks strategy until backtest looks good. Each tweak is a free parameter; the model is fit to noise.

**How to detect:** Many small parameter changes between backtest version A and 'final' version.

**Mitigation:** Lock the strategy parameters before seeing any out-of-sample data. Walk-forward CV with frozen hyperparameters.

**References:** López de Prado (2018) ch 11

**Tags:** `behavioral`, `backtest`

---

## Russia default 1998
`l_026` · category=`macro` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Russia defaulted on sovereign debt, triggering flight to quality. EM bonds collapsed; LTCM convergence trades blew up.

**How to detect:** EM credit spreads widening 200+ bps in days.

**Mitigation:** Cap EM exposure; have tail-risk hedges; reduce leverage when global liquidity tightens.

**References:** Lowenstein (2000)

**Tags:** `macro`, `em`, `crisis`

---

## COVID-19 March 2020 vol spike
`l_028` · category=`macro` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** VIX hit 82 (vs ~12 normal). Most vol-targeting strategies cut leverage at exactly the wrong time; many short-vol funds blew up.

**How to detect:** Realized vol moves multiple standard deviations above your training distribution.

**Mitigation:** Vol-target with a floor; don't blindly de-leverage to zero in panics.

**References:** Multiple 2020 fund letters

**Tags:** `vol`, `macro`, `crisis`

---

## MF Global bankruptcy (2011)
`l_029` · category=`counterparty` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** MF Global misused $1.6B of customer segregated funds. Customers got back ~89% after years of legal process.

**How to detect:** Broker with rapidly deteriorating balance sheet or repeated regulatory actions.

**Mitigation:** Don't keep more than 2-3 months of trading capital at any single broker. Use SIPC-insured brokers.

**References:** CFTC Order 2013

**Tags:** `counterparty`, `broker`

---

## No idempotency on order placement
`l_033` · category=`deployment` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Retry after network timeout sends duplicate orders. End up with 2x intended position.

**How to detect:** Filled qty exceeds requested qty.

**Mitigation:** Use client_order_id deduplication on every order (already done in src/execution/broker.py).

**References:** Standard distributed systems pattern

**Tags:** `operational`, `execution`

---

## Hyperparameter tuning on the same data you validate on
`l_036` · category=`overfitting` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Pick best hyperparameters by CV score, then report that CV score. The CV is no longer unbiased — you've used it for selection.

**How to detect:** Reported CV scores feel 'just right' across many trial configs.

**Mitigation:** Hold out a final test set never touched during tuning. Report performance on it once.

**References:** López de Prado (2018) ch 9

**Tags:** `ml`, `cv`

---

## Model has more parameters than samples after labeling
`l_037` · category=`overfitting` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** XGBoost with 1000 trees on 500 samples will memorize noise. Same for deep networks on small labeled finance datasets.

**How to detect:** Train accuracy » CV accuracy.

**Mitigation:** Constrain complexity (depth, n_estimators, regularization); use early stopping; favor simpler models for small data.

**References:** López de Prado (2018) ch 6

**Tags:** `ml`, `overfitting`

---

## Model not retrained as regime changes
`l_038` · category=`regime_shift` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Model trained on 2010-2019 (low-rate, low-vol) performed terribly in 2020-2022 (vol regime + rate hikes).

**How to detect:** Model accuracy degrades over time on rolling out-of-sample.

**Mitigation:** Schedule monthly retraining. Monitor feature distribution drift.

**References:** Standard MLOps practice

**Tags:** `ml`, `regime`, `mlops`

---

## Overnight gap risk on swing trades
`l_041` · category=`execution` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Stops don't trigger on gaps. A 5% stop becomes a 15% loss if the stock gaps down on earnings overnight.

**How to detect:** Realized losses materially larger than stop-loss intent.

**Mitigation:** Reduce or close before earnings/news events. Use options for defined-risk overnight exposure.

**References:** Common retail blowup pattern

**Tags:** `overnight`, `gap`, `earnings`

---

## Risk parity unwinds in 2020 / 2022
`l_043` · category=`crowding` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Risk-parity funds had built up huge bond positions in a low-vol regime. When bond vol spiked, all sold simultaneously — bonds fell while equities also fell. Diversification benefit reversed.

**How to detect:** Bonds and stocks both falling on the same day (broken correlation regime).

**Mitigation:** Don't assume historical -0.3 stock-bond correlation persists. Stress-test under +1 scenario.

**References:** Bridgewater 2020 letter

**Tags:** `crowding`, `risk_parity`

---

## Crypto winters
`l_045` · category=`regime_shift` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Crypto can drop 80-90% over 12-18 months (2014, 2018, 2022). Trend-following looks excellent in bull, brutal in bear.

**How to detect:** Trend-following crypto strategy with high Sharpe in 2020-2021 backtest.

**Mitigation:** Backtest across at least one crypto winter. Include 2014-2015 in any crypto backtest.

**References:** Cryptocurrency market history

**Tags:** `crypto`, `regime`

---

## Feature distribution drift
`l_048` · category=`data_quality` · severity 🔴🔴🔴🔴⚪ · source=`seed`

**What it is:** Features that were stationary in training become non-stationary live (e.g., VIX regime shift, structural change in spread). Model becomes uncalibrated.

**How to detect:** Live feature mean/std differs >2 sigma from training distribution.

**Mitigation:** Monitor feature drift weekly. Retrain when drift exceeds threshold.

**References:** Standard MLOps pattern

**Tags:** `data_quality`, `drift`

---

## Stop hunting in thin liquidity
`l_014` · category=`execution` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Market makers and HFTs can see stop clusters and push price through them, then revert. Most stops in extended-hours or thinly-traded names hit on noise.

**How to detect:** Stops hit at intraday lows that immediately reverse.

**Mitigation:** Use mental stops or wider hard stops in thin markets; avoid round-number stops.

**References:** Hasbrouck (2007)

**Tags:** `execution`, `stops`

---

## Published anomalies decay after publication
`l_019` · category=`crowding` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** McLean & Pontiff (2016): factor returns drop ~32% post-publication and ~58% post-implementation by funds.

**How to detect:** A strategy you read about in a paper underperforms its backtest in live trading.

**Mitigation:** Assume any public alpha is half-strength; require larger out-of-sample margin before trusting.

**References:** McLean & Pontiff (2016) Does Academic Research Destroy Stock Return Predictability?

**Tags:** `alpha_decay`, `factor`

---

## Yahoo Finance adjustment errors
`l_021` · category=`data_quality` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Free data providers occasionally misadjust for splits/dividends, producing spurious gaps and incorrect returns.

**How to detect:** Backtest shows occasional large one-day gains/losses that aren't in your broker's chart.

**Mitigation:** Cross-check with at least one other source; flag |daily return| > 30% for manual review.

**References:** Personal experience of every retail quant

**Tags:** `data_quality`, `free_data`

---

## Cron job in wrong timezone
`l_031` · category=`deployment` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Strategy scheduled in local time runs at the wrong UTC time after DST shift, missing or doubling executions.

**How to detect:** Trade execution times shift by 1h around DST changes.

**Mitigation:** Run cron in UTC; verify scheduled times in UTC explicitly.

**References:** Common SRE postmortem pattern

**Tags:** `operational`, `scheduling`

---

## Assignment risk on short ITM options
`l_034` · category=`execution` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Short ITM options can be assigned anytime before expiration. Surprise assignment over a weekend = directional exposure you didn't want.

**How to detect:** Holding short ITM options past pin date.

**Mitigation:** Close short ITM options 2+ days before expiration. Avoid Friday holding of short calls on ex-div names.

**References:** OCC Options Education

**Tags:** `options`, `assignment`

---

## Volatility risk premium evaporates around events
`l_035` · category=`execution` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Selling premium into earnings/FOMC ignores that IV is high for a reason. Realized vol often exceeds priced vol on the event.

**How to detect:** Short-vol strategies repeatedly lose around earnings.

**Mitigation:** Filter out trades within ±2 days of scheduled events. Read EarningsHub or Finnhub calendar.

**References:** Coval & Shumway (2001)

**Tags:** `options`, `events`

---

## Class imbalance in labels
`l_039` · category=`data_quality` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Long forward returns are 53% of samples but model predicts 80% long. Accuracy looks decent but precision/recall asymmetric.

**How to detect:** Most predictions are the majority class.

**Mitigation:** Class-weight balancing; sample weighting; or use AUC instead of accuracy.

**References:** Standard ML practice

**Tags:** `ml`, `imbalance`

---

## Pyramiding into winners without stop adjustment
`l_040` · category=`risk_sizing` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Adding to winners is fine, but if the original stop isn't tightened, the entire position is now exposed to a deeper loss than originally sized for.

**How to detect:** Position size growing while stop distance unchanged.

**Mitigation:** Trail stops as adds occur; re-compute position-level risk after every add.

**References:** Chan (2008)

**Tags:** `sizing`, `pyramiding`

---

## Drawdown psychology: cutting size mid-drawdown
`l_042` · category=`behavioral` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** When in drawdown, operator cuts size by 50%. When strategy recovers, they're under-sized for the rebound and never make it back.

**How to detect:** Position sizes shrink during DD but don't restore quickly.

**Mitigation:** Pre-commit to a vol-targeting rule. Don't manually intervene in sizing.

**References:** Kahneman & Tversky (1979)

**Tags:** `behavioral`, `drawdown`

---

## Over-monitoring causes intervention
`l_047` · category=`behavioral` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Watching every tick triggers emotional responses. Operator overrides systematic decisions based on noise.

**How to detect:** Operator checking P&L > 5x/hour during trading.

**Mitigation:** Look at dashboard 2x/day (morning + close). Set alerts for hard breaches only.

**References:** Tharp (2007)

**Tags:** `behavioral`, `psychology`

---

## Order types confused (market vs limit vs stop)
`l_049` · category=`execution` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Marketable limit confused with market; stop-limit not filled because limit was too tight in fast market.

**How to detect:** Stops don't fill when expected; limits cross book and become market.

**Mitigation:** Document order type semantics per broker; test each type explicitly before live.

**References:** Broker documentation, every retail forum

**Tags:** `execution`, `order_types`

---

## Partial fills on bracket orders
`l_051` · category=`execution` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Bracket order fills 50%; the take-profit and stop-loss apply to the full ordered qty, not the filled qty, on some brokers.

**How to detect:** Stop or TP attached to qty larger than actual position; broker error or unintended naked exposure.

**Mitigation:** Verify broker's bracket semantics; cancel and re-submit on partial fill if needed.

**References:** Alpaca docs

**Tags:** `execution`, `options`, `brackets`

---

## Free APIs returning stale data silently
`l_052` · category=`data_quality` · severity 🔴🔴🔴⚪⚪ · source=`seed`

**What it is:** Yahoo / Finnhub free tiers occasionally return delayed or stale prices without error. Your model thinks it has live data; it doesn't.

**How to detect:** Prices in your data feed lag the broker's quote by >5 minutes.

**Mitigation:** Cross-check timestamp of every quote vs system clock; flag stale data; never trade on data >5min old.

**References:** Personal experience

**Tags:** `data_quality`, `free_data`

---
