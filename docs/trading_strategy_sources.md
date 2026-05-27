# Trading Strategy Research Sources

A curated, deduplicated list of **120+ direct sources** for building, evaluating,
and risk-managing systematic trading strategies. Sources are grouped by
type so you can jump straight to the layer you need. Every entry is a
real, publicly findable work — no invented citations.

This file is a research index, not a manual. Use it to (a) learn the
mathematical foundations, (b) find specific strategy templates, (c) find
data sources, and (d) read about why most systematic strategies fail.

> **Honest framing.** The single most important reading on this list is the
> "Why most systematic strategies fail" section. Read those first. The rest
> of the list is only valuable if you internalize the failure modes first.

---

## Table of contents

1. [Foundational academic papers](#1-foundational-academic-papers)
2. [Modern empirical / factor papers](#2-modern-empirical--factor-papers)
3. [Machine learning for trading (papers)](#3-machine-learning-for-trading-papers)
4. [Microstructure & execution](#4-microstructure--execution)
5. [Volatility & options](#5-volatility--options)
6. [Risk management & portfolio construction](#6-risk-management--portfolio-construction)
7. [Why most systematic strategies fail (READ FIRST)](#7-why-most-systematic-strategies-fail-read-first)
8. [Canonical books](#8-canonical-books)
9. [Practitioner blogs](#9-practitioner-blogs)
10. [Open-source code & libraries](#10-open-source-code--libraries)
11. [Data sources](#11-data-sources)
12. [Online courses & curricula](#12-online-courses--curricula)
13. [Forums & communities](#13-forums--communities)
14. [Specific strategy references](#14-specific-strategy-references)
15. [Crypto / DeFi-specific](#15-crypto--defi-specific)
16. [Macro & event-driven](#16-macro--event-driven)

---

## 1. Foundational academic papers

1. **Markowitz, H. (1952).** *Portfolio Selection.* Journal of Finance, 7(1).
   The original mean-variance framework.
2. **Sharpe, W. F. (1964).** *Capital Asset Prices: A Theory of Market Equilibrium.*
   Journal of Finance, 19(3). CAPM.
3. **Fama, E. F. (1970).** *Efficient Capital Markets: A Review of Theory and Empirical Work.*
   Journal of Finance, 25(2). EMH baseline you must understand to argue against.
4. **Black, F., & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities.*
   Journal of Political Economy, 81(3).
5. **Merton, R. C. (1973).** *Theory of Rational Option Pricing.* Bell Journal of Economics and Management Science, 4(1).
6. **Granger, C. W. J. (1969).** *Investigating Causal Relations by Econometric Models and Cross-spectral Methods.*
   Econometrica, 37(3). Granger causality — used for feature selection.
7. **Engle, R. F. (1982).** *Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation.*
   Econometrica, 50(4). ARCH.
8. **Bollerslev, T. (1986).** *Generalized Autoregressive Conditional Heteroskedasticity.*
   Journal of Econometrics, 31(3). GARCH — used in `src/model/volatility_module.py`.
9. **Kyle, A. S. (1985).** *Continuous Auctions and Insider Trading.* Econometrica, 53(6).
10. **Hasbrouck, J. (1991).** *Measuring the Information Content of Stock Trades.*
    Journal of Finance, 46(1). Order flow toxicity foundations.

## 2. Modern empirical / factor papers

11. **Fama, E. F., & French, K. R. (1993).** *Common Risk Factors in the Returns on Stocks and Bonds.*
    Journal of Financial Economics, 33(1). 3-factor model.
12. **Jegadeesh, N., & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers.*
    Journal of Finance, 48(1). Cross-sectional momentum.
13. **Carhart, M. M. (1997).** *On Persistence in Mutual Fund Performance.*
    Journal of Finance, 52(1). 4-factor (adds momentum).
14. **Lo, A. W., & MacKinlay, A. C. (1988).** *Stock Market Prices do not Follow Random Walks.*
    Review of Financial Studies, 1(1).
15. **Asness, C., Moskowitz, T., & Pedersen, L. H. (2013).** *Value and Momentum Everywhere.*
    Journal of Finance, 68(3).
16. **Frazzini, A., & Pedersen, L. H. (2014).** *Betting Against Beta.* Journal of Financial Economics, 111(1).
17. **Asness, C., Frazzini, A., & Pedersen, L. H. (2019).** *Quality Minus Junk.* Review of Accounting Studies, 24(1).
18. **Novy-Marx, R. (2013).** *The Other Side of Value: The Gross Profitability Premium.*
    Journal of Financial Economics, 108(1).
19. **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).** *Time Series Momentum.*
    Journal of Financial Economics, 104(2).
20. **Daniel, K., & Moskowitz, T. J. (2016).** *Momentum Crashes.* Journal of Financial Economics, 122(2).
    The unavoidable downside of running momentum.

## 3. Machine learning for trading (papers)

21. **López de Prado, M. (2018).** *The 10 Reasons Most Machine Learning Funds Fail.*
    Journal of Portfolio Management, 44(6).
22. **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).**
    *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.*
    Notices of the AMS, 61(5).
23. **Krauss, C., Do, X. A., & Huck, N. (2017).** *Deep Neural Networks, Gradient-Boosted Trees, Random Forests:
    Statistical Arbitrage on the S&P 500.* European Journal of Operational Research, 259(2).
24. **Sirignano, J., & Cont, R. (2019).** *Universal Features of Price Formation in Financial Markets.*
    Quantitative Finance, 19(9).
25. **Heaton, J. B., Polson, N. G., & Witte, J. H. (2017).** *Deep Learning for Finance: Deep Portfolios.*
    Applied Stochastic Models in Business and Industry, 33(1).
26. **Borovykh, A., Bohte, S., & Oosterlee, C. W. (2017).** *Conditional Time Series Forecasting with Convolutional Neural Networks.* arXiv:1703.04691.
27. **Zhang, Z., Zohren, S., & Roberts, S. (2020).** *Deep Reinforcement Learning for Trading.* Journal of Financial Data Science.
28. **Vaswani et al. (2017).** *Attention Is All You Need.* NeurIPS.
    Backbone of `src/model/transformer_backbone.py`.
29. **Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).** *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting, 37(4).
30. **Araci, D. (2019).** *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
    Backbone of `src/model/sentiment_encoder.py`.

## 4. Microstructure & execution

31. **Bertsimas, D., & Lo, A. W. (1998).** *Optimal Control of Execution Costs.* Journal of Financial Markets, 1(1).
32. **Almgren, R., & Chriss, N. (2000).** *Optimal Execution of Portfolio Transactions.* Journal of Risk, 3(2).
    The TWAP/VWAP/implementation-shortfall foundation.
33. **Almgren, R. (2003).** *Optimal Execution with Nonlinear Impact Functions and Trading-enhanced Risk.* Applied Mathematical Finance, 10(1).
34. **Obizhaeva, A., & Wang, J. (2013).** *Optimal Trading Strategy and Supply/Demand Dynamics.* Journal of Financial Markets, 16(1).
35. **Cont, R., Stoikov, S., & Talreja, R. (2010).** *A Stochastic Model for Order Book Dynamics.* Operations Research, 58(3).
36. **Easley, D., López de Prado, M., & O'Hara, M. (2012).** *Flow Toxicity and Liquidity in a High Frequency World.* Review of Financial Studies, 25(5). VPIN.
37. **O'Hara, M. (2015).** *High Frequency Market Microstructure.* Journal of Financial Economics, 116(2).
38. **Gatheral, J., Schied, A., & Slynko, A. (2012).** *Transient Linear Price Impact and Fredholm Integral Equations.* Mathematical Finance, 22(3).

## 5. Volatility & options

39. **Heston, S. L. (1993).** *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* Review of Financial Studies, 6(2).
40. **Dupire, B. (1994).** *Pricing with a Smile.* Risk Magazine, 7(1). Local volatility.
41. **Carr, P., & Wu, L. (2003).** *Finite Moment Log Stable Process and Option Pricing.* Journal of Finance, 58(2).
42. **Bakshi, G., Cao, C., & Chen, Z. (1997).** *Empirical Performance of Alternative Option Pricing Models.* Journal of Finance, 52(5).
43. **Coval, J. D., & Shumway, T. (2001).** *Expected Option Returns.* Journal of Finance, 56(3).
    Why selling vol earns a premium — and why it kills accounts in crashes.
44. **Bondarenko, O. (2014).** *Why Are Put Options So Expensive?* Quarterly Journal of Finance, 4(3).
45. **Bollerslev, T., Tauchen, G., & Zhou, H. (2009).** *Expected Stock Returns and Variance Risk Premia.* Review of Financial Studies, 22(11).
46. **Engle, R. F., & Bollerslev, T. (1986).** *Modelling the Persistence of Conditional Variances.* Econometric Reviews, 5(1).

## 6. Risk management & portfolio construction

47. **Kelly, J. L. (1956).** *A New Interpretation of Information Rate.* Bell System Technical Journal, 35(4).
    The Kelly criterion — basis of position sizing in `src/risk/risk_manager.py`.
48. **Thorp, E. O. (1969).** *Optimal Gambling Systems for Favorable Games.* Review of the International Statistical Institute, 37(3).
49. **MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010).** *The Kelly Capital Growth Investment Criterion: Theory and Practice.* World Scientific.
50. **Black, F., & Litterman, R. (1992).** *Global Portfolio Optimization.* Financial Analysts Journal, 48(5).
51. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** *Optimal Versus Naïve Diversification: How Inefficient is the 1/N Portfolio Strategy?* Review of Financial Studies, 22(5).
52. **López de Prado, M. (2016).** *Building Diversified Portfolios that Outperform Out-of-Sample.* Journal of Portfolio Management, 42(4). HRP.
53. **Rockafellar, R. T., & Uryasev, S. (2000).** *Optimization of Conditional Value-at-Risk.* Journal of Risk, 2(3). CVaR.
54. **Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999).** *Coherent Measures of Risk.* Mathematical Finance, 9(3).

## 7. Why most systematic strategies fail (READ FIRST)

55. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *... and the Cross-Section of Expected Returns.* Review of Financial Studies, 29(1).
    Multiple-testing correction in factor research. Most "factors" are noise.
56. **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies.* Review of Financial Studies, 33(5).
    Most published anomalies don't replicate out-of-sample.
57. **Bailey, D. H., & López de Prado, M. (2014).** *The Deflated Sharpe Ratio.* Journal of Portfolio Management, 40(5).
    Why your in-sample Sharpe is wrong.
58. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
    Chapters 7 (cross-validation in finance) and 11 (the Triple Barrier Method) are essential.
59. **Bailey, D. H., et al. (2017).** *Mathematical Appendices to "The Probability of Backtest Overfitting".* Journal of Computational Finance, 20(4).
60. **Falck, A., Rej, A., & Thesmar, D. (2022).** *When Systematic Strategies Decay.* Quantitative Finance, 22(11).
61. **Arnott, R., Harvey, C. R., & Markowitz, H. (2019).** *A Backtesting Protocol in the Era of Machine Learning.* Journal of Financial Data Science, 1(1).
62. **Patton, A. J., & Timmermann, A. (2010).** *Monotonicity in Asset Returns: New Tests with Applications to the Term Structure, the CAPM, and Portfolio Sorts.* Journal of Financial Economics, 98(3).

## 8. Canonical books

63. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
64. **López de Prado, M. (2020).** *Machine Learning for Asset Managers.* Cambridge Elements.
65. **Chan, E. P. (2008).** *Quantitative Trading.* Wiley.
66. **Chan, E. P. (2013).** *Algorithmic Trading: Winning Strategies and Their Rationale.* Wiley.
67. **Chan, E. P. (2017).** *Machine Trading: Deploying Computer Algorithms to Conquer the Markets.* Wiley.
68. **Hull, J. C. (2017).** *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
69. **Sinclair, E. (2013).** *Volatility Trading* (2nd ed.). Wiley.
70. **Sinclair, E. (2010).** *Option Trading: Pricing and Volatility Strategies and Techniques.* Wiley.
71. **Natenberg, S. (2014).** *Option Volatility and Pricing* (2nd ed.). McGraw-Hill.
72. **McMillan, L. G. (2011).** *Options as a Strategic Investment* (5th ed.). Prentice Hall Press.
73. **Taleb, N. N. (1997).** *Dynamic Hedging.* Wiley.
74. **Aronson, D. R. (2007).** *Evidence-Based Technical Analysis.* Wiley.
75. **Shreve, S. E. (2004).** *Stochastic Calculus for Finance II: Continuous-Time Models.* Springer.
76. **Joshi, M. S. (2008).** *The Concepts and Practice of Mathematical Finance.* Cambridge.
77. **Tsay, R. S. (2010).** *Analysis of Financial Time Series* (3rd ed.). Wiley.
78. **Hamilton, J. D. (1994).** *Time Series Analysis.* Princeton.
79. **Cochrane, J. H. (2005).** *Asset Pricing.* Princeton.
80. **Pedersen, L. H. (2015).** *Efficiently Inefficient: How Smart Money Invests and Market Prices Are Determined.* Princeton.
81. **Schwager, J. D. (1989–2020).** *Market Wizards* series. HarperBusiness / Wiley. Discretionary trader interviews; cross-reference with quant material.
82. **Lewis, M. (2014).** *Flash Boys.* W. W. Norton. HFT context.
83. **Patterson, S. (2010).** *The Quants.* Crown Business. History.
84. **Jansen, S. (2020).** *Machine Learning for Algorithmic Trading* (2nd ed.). Packt. End-to-end ML pipelines.

## 9. Practitioner blogs

85. **Quantocracy** — https://quantocracy.com — aggregator of every active quant blog.
86. **AQR Insights** — https://www.aqr.com/Insights — institutional research, free.
87. **Two Sigma Insights** — https://www.twosigma.com/insights/
88. **Newfound Research** — https://blog.thinknewfound.com/ — tactical asset allocation.
89. **Alpha Architect** — https://alphaarchitect.com/blog/ — factor investing.
90. **Robot Wealth** — https://robotwealth.com/blog/ — practical retail quant.
91. **Quantpedia** — https://quantpedia.com/screener/ — strategy catalog (free + paid tiers).
92. **EP Chan blog** — https://epchan.blogspot.com — by the author of the Chan books above.
93. **Quantitative Research and Trading (Jonathan Kinlay)** — https://jonathankinlay.com/
94. **Sentiment Trader (Jason Goepfert)** — https://sentimentrader.com
95. **The Quants Hub (papers + community)** — https://quantshub.com
96. **Hudson and Thames (mlfinlab)** — https://hudsonthames.org/blog/

## 10. Open-source code & libraries

97. **awesome-quant** — https://github.com/wilsonfreitas/awesome-quant — curated meta-list.
98. **financial-machine-learning** — https://github.com/firmai/financial-machine-learning — curated ML-for-finance.
99. **vectorbt** — https://github.com/polakowo/vectorbt — fast vectorized backtester.
100. **backtrader** — https://github.com/mementum/backtrader — event-driven backtester.
101. **zipline-reloaded** — https://github.com/stefan-jansen/zipline-reloaded — fork of Quantopian's engine.
102. **qlib** — https://github.com/microsoft/qlib — Microsoft Research's quant platform.
103. **FinRL** — https://github.com/AI4Finance-Foundation/FinRL — RL for trading.
104. **mlfinlab** — https://github.com/hudson-and-thames/mlfinlab — López de Prado methods.
105. **quantstats** — https://github.com/ranaroussi/quantstats — performance metrics + tearsheets.
106. **pyfolio** — https://github.com/quantopian/pyfolio — same.
107. **alphalens** — https://github.com/quantopian/alphalens — factor analysis.
108. **yfinance** — https://github.com/ranaroussi/yfinance — already in this project.
109. **alpaca-py** — https://github.com/alpacahq/alpaca-py — already in this project.
110. **ibapi (Interactive Brokers Python)** — https://github.com/erdewit/ib_insync — IBKR Python client.
111. **tda-api** — https://github.com/alexgolec/tda-api — TD Ameritrade Python client.

## 11. Data sources

112. **Yahoo Finance** (free, scraping). Already used by `data_scraper.py`.
113. **Alpha Vantage** (free tier, 25/day). Already in `data_loader.py`.
114. **Finnhub** (free tier, 60/min). Already in `data_loader.py`.
115. **FRED — St. Louis Fed** (free). Macro indicators.
116. **Quandl / Nasdaq Data Link** — https://data.nasdaq.com — free + paid datasets.
117. **WRDS (Wharton Research Data Services)** — https://wrds-www.wharton.upenn.edu — gold standard but requires institutional access.
118. **CRSP** — https://www.crsp.org — survivorship-bias-free US equities. Paid, via WRDS.
119. **Compustat** — fundamentals, via WRDS.
120. **OptionMetrics IvyDB** — historical options + IV surfaces. Paid.
121. **EDI** — global market reference data. Paid.
122. **Refinitiv DataScope** — institutional. Paid.
123. **NYSE TAQ** — every quote and trade. Paid via WRDS.
124. **OpenBB** — https://openbb.co — open-source aggregator with free tier.

## 12. Online courses & curricula

125. **Coursera — Machine Learning for Trading (Georgia Tech)**.
126. **Coursera — Financial Engineering and Risk Management Specialization (Columbia)**.
127. **edX — MicroMasters in Finance (MIT)**.
128. **CQF — Certificate in Quantitative Finance** — https://www.cqf.com.
129. **CFA Curriculum** — https://www.cfainstitute.org — particularly Level II derivatives + portfolio management.
130. **Hudson and Thames "Machine Learning for Asset Management"** — paid course built on López de Prado material.
131. **Stefan Jansen's GitHub companion to "ML for Algorithmic Trading"** — https://github.com/stefan-jansen/machine-learning-for-trading

## 13. Forums & communities

132. **Quantitative Finance Stack Exchange** — https://quant.stackexchange.com
133. **r/algotrading** — https://www.reddit.com/r/algotrading
134. **r/quant** — https://www.reddit.com/r/quant
135. **r/options** — https://www.reddit.com/r/options
136. **Wilmott Forums** — https://forum.wilmott.com
137. **Elite Trader Forums** — https://www.elitetrader.com
138. **NuclearPhynance** — https://www.nuclearphynance.com — older but archives are gold

## 14. Specific strategy references

139. **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).** *Pairs Trading: Performance of a Relative-Value Arbitrage Rule.* Review of Financial Studies, 19(3).
140. **Avellaneda, M., & Lee, J. H. (2010).** *Statistical Arbitrage in the US Equities Market.* Quantitative Finance, 10(7).
141. **Caldeira, J. F., & Moura, G. V. (2013).** *Selection of a Portfolio of Pairs Based on Cointegration: A Statistical Arbitrage Strategy.* Brazilian Review of Finance, 11(1).
142. **Faber, M. T. (2007).** *A Quantitative Approach to Tactical Asset Allocation.* Journal of Wealth Management, 9(4).
    The famous 200-day moving average TAA rule.
143. **Wilcox, J., & Crittenden, E. (2005).** *Does Trend Following Work on Stocks?* Technical Analyst.
144. **Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017).** *A Century of Evidence on Trend-Following Investing.* Journal of Portfolio Management, 44(1).
145. **Asness, C., Liew, J. M., Pedersen, L. H., & Thapar, A. K. (2017).** *Deep Value.* AQR working paper.
146. **Israel, R., Jiang, S., & Ross, A. (2017).** *Craftsmanship Alpha: An Application to Style Investing.* Journal of Portfolio Management, 44(2).
147. **Goyal, A., Welch, I. (2008).** *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction.* Review of Financial Studies, 21(4).
    A warning: most equity-premium predictors fail out-of-sample.
148. **Tasty Trade Research** — https://www.tastylive.com/concepts-strategies — short premium options content (use with skepticism on tail risk).

## 15. Crypto / DeFi-specific

149. **Liu, Y., & Tsyvinski, A. (2021).** *Risks and Returns of Cryptocurrency.* Review of Financial Studies, 34(6).
150. **Makarov, I., & Schoar, A. (2020).** *Trading and Arbitrage in Cryptocurrency Markets.* Journal of Financial Economics, 135(2).
151. **Hougan, M., & Lawant, D. (2020).** *Cryptoassets: The Guide to Bitcoin, Blockchain, and Cryptocurrency for Investment Professionals.* CFA Institute.
152. **Glassnode** — https://glassnode.com — on-chain analytics.
153. **DeFiLlama** — https://defillama.com — TVL + protocol metrics, free API.
154. **The Graph** — https://thegraph.com — indexed on-chain data.

## 16. Macro & event-driven

155. **Ilmanen, A. (2011).** *Expected Returns: An Investor's Guide to Harvesting Market Rewards.* Wiley.
156. **Dalio, R. (2018).** *Principles for Navigating Big Debt Crises.* Bridgewater (free PDF).
157. **Dalio, R. (2021).** *Principles for Dealing with the Changing World Order.* Avid Reader Press.
158. **Antonacci, G. (2014).** *Dual Momentum Investing.* McGraw-Hill.
159. **Pesaran, M. H., & Timmermann, A. (1995).** *Predictability of Stock Returns: Robustness and Economic Significance.* Journal of Finance, 50(4).
160. **Fed FOMC Calendar + minutes** — https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm — read every minutes release.
161. **BEA, BLS, Treasury, EIA releases** — https://www.bea.gov, https://www.bls.gov, https://www.eia.gov — release calendars for event trading.

---

## Suggested reading order

If you read these in order, you'll have a much better foundation than 95% of
retail algo traders:

1. **Aronson — Evidence-Based Technical Analysis** (book 74) — sets your mental model.
2. **López de Prado — Advances in Financial Machine Learning** chapters 1, 2, 4, 7, 11, 14, 16 (book 63).
3. **Bailey & López de Prado — Pseudo-Mathematics and Financial Charlatanism** (paper 22).
4. **Harvey, Liu, Zhu — ... and the Cross-Section of Expected Returns** (paper 55).
5. **Chan — Algorithmic Trading** (book 66) for practical strategy implementation.
6. **Pedersen — Efficiently Inefficient** (book 80) for the institutional view.
7. **Asness et al. — Value and Momentum Everywhere** (paper 15).
8. **Hurst, Ooi, Pedersen — A Century of Evidence on Trend-Following** (paper 144).
9. Pick a strategy template from Quantpedia (blog 91) that matches your interests.
10. Implement it. Walk-forward backtest it using `mlfinlab` or `vectorbt` (libraries 99, 104).
11. Read Bailey & López de Prado paper 57 to deflate your Sharpe estimate.
12. If your strategy still looks viable, paper-trade it for 30+ sessions before going live.

The codebase in this repo gives you the **engineering substrate** for steps
9–12. The reading list is the **modeling substrate**. You need both.

---

*Last reviewed: this list mixes time-tested foundations (1950s–1990s) with
modern work (2010s–2020s). Update annually — academic finance moves slowly
but the ML-for-trading subfield is publishing new papers every quarter.*
