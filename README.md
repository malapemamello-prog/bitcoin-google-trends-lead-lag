# Lead-Lag Relationship Between Bitcoin and Internet Attention

Does internet attention (Google Trends) **lead or lag** Bitcoin market activity, and does it help predict Bitcoin **returns, trading volume and volatility**?

**Notebook:** `Lead_Lag_Relationship.ipynb` (Python, Google Colab)

---



## Overview

Retail investors tend to search for information before they trade. This project tests whether weekly Google Trends search interest gives an early signal for Bitcoin market behaviour, using four approaches: cross-correlation, Granger causality, an LSTM forecaster and an EGARCH volatility model.



## Data

| Item | Detail |
|------|--------|
| Bitcoin data | BTC-USD daily price, volume and log returns via `yfinance` |
| Attention data | Google Trends via `pytrends` (geo = US) |
| Search terms | `Buy Bitcoin`, `BTC`, `BTC price`, `Ripple`, `crypto` |
| Period | 1 Jan 2025 – 31 Dec 2025 |
| Frequency | Weekly (52 merged observations) |
| Weekly construction | Price = last value of week; log return = sum of daily log returns; volume = weekly sum; Trends = weekly mean |

## Methods

| Step | Method | Purpose / Detail |
|------|--------|------------------|
| 1. Returns | Log returns | Time-additive and standard in financial econometrics |
| 2. Alignment | Weekly resampling and inner join | Match Bitcoin and Trends data; forward/back-fill missing Trends values |
| 3. Smoothing | 4-week rolling mean | Visualised for noise reduction (not used as model input) |
| 4. Stationarity | ADF test, verified with KPSS | Check series before lead–lag and Granger tests |
| 5. Differencing | `diff()` on non-stationary series | `price_w` and `Ripple`: `diff(1)`; `Buy Bitcoin`: `diff(periods=2)`; `volume_w`: `diff(periods=3)` |
| 6. Decomposition | STL (period = 13) | Trend / seasonal / residual of Trends series |
| 7. Autocorrelation | ACF and PACF | Inspect dependence in returns and Trends |
| 8. Lead–lag | Cross-correlation, lags ±8 weeks | Direction and timing of relationship |
| 9. Causality | Granger causality, up to 12 lags, α = 0.05 | Do Trends help forecast returns and volume? |
| 10. Prediction | Stacked LSTM (32 → 16 units, dropout 0.3) | Next-week log return; 4-week input sequences, 80/20 split, MinMax scaling, Adam, max 30 epochs, early stopping (patience 5) |
| 11. Volatility | EGARCH(1,1), zero mean | Weekly log returns with Trends variables as `x` inputs |



## Key Findings

- Search interest tends to **lead** Bitcoin activity.
- The effect is **much stronger for trading volume** than for returns.
- Returns: `BTC`, `BTC price` and `crypto` matter at 1–4 week lags. `Buy Bitcoin` and `Ripple` do not.
- Volume: all five terms are significant, at 1–2 weeks (and 5–7 weeks for `Ripple`).
- The LSTM direction accuracy of 71.4% is promising but not reliable enough to use on its own.

## Limitations

- **Small sample:** one year of weekly data (52 observations, ~50 after differencing) and one asset (BTC), which limits statistical power and generalisability.
- **Google Trends data:** a relative 0-100 index, US-only and sample-based, so values can shift between downloads. Keyword choice is subjective, and `Ripple` is not Bitcoin-specific.
- **Weekly aggregation:** hides short-term (daily) dynamics.
- **Differencing:** `diff(periods=2)` and `diff(periods=3)` are lag differences, not 2nd/3rd-order differencing. Orders also differ across variables.
- **Rolling window:** the 4-week average is only visualised, not used in the models.
- **Lead-lag and Granger tests:** no significance bands on cross-correlations. The best lag is picked by smallest p-value across 12 lags and 10 pairs with no multiple-testing correction. Granger causality shows predictive precedence, not true causation.
- **LSTM:** the test set is tiny (~7 sequences), so 71.4% direction accuracy is only 5 of 7 correct. The test set is also used for early stopping, and there is no baseline without Google Trends, so the "Trends improve accuracy" claim is not directly tested.
- **EGARCH:** only ~50 observations, with β ≈ 1 and near-zero standard errors, which suggests optimiser instability. The output shows no coefficients for the attention variables, so the volatility claim is not directly supported.

## Recommendations

- Use Google Trends as a **supplementary** indicator, with more weight on volume signals.
- Focus on **1–3 week** lags.
- Combine attention data with technical and fundamental analysis, and monitor attention spikes alongside volatility.
- Future work: longer history, sentiment data, multiple cryptocurrencies, proper out-of-sample / walk-forward validation, a Trends-vs-no-Trends baseline, and attention variables placed in the EGARCH variance equation.


## Contributors

- **Mamello Malape**
- **Fumane Malangabe**
