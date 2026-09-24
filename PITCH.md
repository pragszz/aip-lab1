# Project Pitch: Short-Term Solar Power Forecasting (Own-Project, Lab 6)

Self-contained — single dataset, no external API — chosen specifically so the six hours go to building and measuring, not to fixing an unverifiable location assumption partway through.

## 1. What are you building?

A short-horizon (15-minutes-ahead) solar power forecaster for a real PV plant, built entirely from the plant's own generation and weather sensors. The deliverable isn't just an accuracy number — it's an operating recommendation: **when is the model actually worth running instead of the free, trivial baseline, and when isn't it?**

## 2. What data will you use, and do you have access to it?

**Dataset:** [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (Kaggle, Anikannal) — two independent Indian solar plants, 34 days at 15-minute resolution each.

**Files used (per plant):**
- `Plant_N_Generation_Data.csv` — `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`, per inverter (`SOURCE_KEY`)
- `Plant_N_Weather_Sensor_Data.csv` — `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`, one sensor per plant

**Access confirmed:** plain CSVs (~11 MB total), no API key, no scraping, no signup. Downloadable directly from Kaggle now, with public GitHub mirrors as a fallback. Both plants are in the same download — using Plant 2 as a second, independent check costs nothing extra in access risk.

## 3. Smallest end-to-end version (Hour 1)

1. Load `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv`, parsing `DATE_TIME`.
2. **Collapse generation to one row per timestamp.** Plant 1 has 22 inverters reporting per timestamp but only one weather sensor: `groupby("DATE_TIME")["AC_POWER"].sum()` before merging, or generation and weather won't line up 1:1.
3. **Merge** on `DATE_TIME`, sort chronologically, check for timestamp gaps (`.diff()`). This dataset has missing 15-minute readings, and a naive shift would silently pair non-adjacent rows as if they were 15 minutes apart.
4. **Build the lag feature.** `AC_POWER(t)` needs no transformation — it's already "current." The forecasting problem is created by shifting the *label*: `df["target"] = df["AC_POWER"].shift(-1)` pulls the next row's power backward onto the current row. Drop the final row (no future value).
5. **Feature set per row:** `IRRADIATION(t)`, `AMBIENT_TEMPERATURE(t)`, `MODULE_TEMPERATURE(t)`, `AC_POWER(t)` (lag-1) → **target:** `AC_POWER(t+1)`.
6. **Time-based train/test split** (first ~75% of days = train, last ~25% = test) — never random, since adjacent rows are highly correlated.
7. Fit a single `LinearRegression`; print MAE/RMSE/R² against the persistence baseline (`predicted = AC_POWER(t)`).

CSV in, number out. Crude, but a real end-to-end run — the Day-1 hard requirement.

## 4. What will you measure, and what's your baseline?

**Baseline:** persistence — `predicted AC_POWER(t+1) = AC_POWER(t)`. Standard, legitimately hard-to-beat for short-horizon time series — which is exactly why the *segmented* evaluation below matters more than one aggregate number.

**Metrics:** MAE, RMSE, R² on the held-out (future) time window, model vs. baseline.

**The recommendation-producing step:** split test-set error into two regimes — *stable irradiance* (low rolling variance in recent `IRRADIATION`) vs. *changing irradiance* (high variance — cloud transients). The expected, reportable pattern: the model **ties** persistence when conditions are stable and **wins** when conditions are changing. That split is what turns "the model has a lower RMSE" into an actual operating policy (Section on real-world output, below).

**Cross-plant check:** repeat the identical pipeline on Plant 2. If the tie/win pattern shows up on both independent plants, that's durability evidence — not an artifact of one site's data quirks.

**Models, in increasing complexity** (checked against this repo's `.venv` — scikit-learn 1.9.0 is already installed transitively, so everything below except the stretch item is a zero-install addition):

| Order | Model | Why it's on the list | Environment |
|---|---|---|---|
| 0 | Persistence (no model) | The baseline — literally the lag-1 feature used as the prediction | n/a |
| 1 | `LinearRegression` | Sanity check: is the power–irradiance relationship roughly linear? Fast, interpretable | ✅ already available |
| 2 | `RandomForestRegressor` | Captures nonlinearities/interactions (e.g. high temperature reducing panel efficiency at high irradiance) with no manual feature engineering | ✅ already available |
| 3 | `HistGradientBoostingRegressor` | Usually the strongest tabular-regression performer; sklearn's own boosting implementation | ✅ already available |
| optional | `KNeighborsRegressor` | "Find similar past weather conditions, average their outcomes" — cheap and interpretable | ✅ already available |
| stretch | `XGBoost` | Marginal gain over sklearn's own boosting at best | ⚠️ **not installed** — `pip install xgboost` first; skip unless time remains |

**Core question:** *Does a weather-informed model meaningfully reduce 15-minute-ahead forecasting error versus persistence — and specifically, does that win concentrate in changing-weather periods, consistently across two independent plants?*

## 5. Which course tools does this draw on?

- **Files/CSV I/O:** loading and merging raw CSVs, timestamp parsing.
- **Arrays/linear algebra:** feature matrix construction, lag features, rolling-variance regime labeling.
- **Pipelines & honest evaluation:** time-based (non-random) train/test split; baseline-vs-model comparison; segmented (regime-conditioned) error reporting instead of one blended number.
- **Training loops / model fitting:** scikit-learn `fit`/`predict` across models of increasing complexity.
- **Plots:** predicted vs. actual power over time; error by regime; feature importances/coefficients.
- **Error handling:** guarding against missing/NaN sensor readings and timestamp gaps between files.

No new framework or infrastructure — pandas + scikit-learn + matplotlib, already in the course stack.

## The real-world output this is aiming at

Not "always deploy the ML model" — a sharper, resource-aware recommendation:

> Run the ML forecaster selectively, triggered during high-variance irradiance windows (detectable from the plant's own recent sensor readings), where it delivers a measurable accuracy gain over persistence. Fall back to persistence — free, and statistically indistinguishable from the model — during stable conditions.

If feature importances also show `MODULE_TEMPERATURE` adding value beyond `IRRADIATION` alone, that's a second concrete finding worth stating: it's consistent with known PV temperature-derating behavior, not just a model artifact.

## Hour-by-hour (per Lab 6 guidelines)

| Hour | Plan |
|---|---|
| **1** | Steps above: merged table, lag feature, time split, `LinearRegression` vs. persistence, printed metrics. Must be running end-to-end by the end of Day 1. |
| **2–4** | Whatever measurably limits quality first — likely order: (a) add `RandomForestRegressor` / `HistGradientBoostingRegressor`, (b) build the stable-vs-changing regime split and confirm the win is concentrated where expected, (c) repeat the full pipeline on Plant 2, (d) check the temperature feature-importance finding. |
| **5** | Evaluation, plots (predicted-vs-actual, error-by-regime, feature importance), README. |
| **6** | Polish, wrap up, stop building. |

**Before leaving Day 1 (hour 3):** save the fitted Day-1 pipeline (`joblib.dump`) and write down the specific Day-2 next action (e.g., "add RF + HistGB, then build the regime split on the saved model's residuals").

**One sentence to have ready before hour 5** (write it as soon as it's true, not after): *"The model beats persistence by ~X% RMSE overall, concentrated in changing-irradiance periods, and this pattern holds on both plants."* — or whatever the real number turns out to be. If this sentence can't be written yet, that's the signal for what the remaining time is for.

## Submission checklist (from the guidelines)

- [ ] Runs end-to-end from a clean clone, not just locally.
- [ ] An actual baseline appears in the notebook itself, not just a final comparison number.
- [ ] What *didn't* work is reported too (e.g., if Random Forest overfits at small sample sizes, or the regime split doesn't show the expected pattern on Plant 2).
- [ ] README answers exactly these four questions: what does this do (2 sentences), how do I run it (exact commands + installs), what did you find (headline result), what would you do next.
