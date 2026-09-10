# Project Pitch: Short-Term Solar Power Forecasting

## 1. What are you building?

A model that forecasts a solar plant's power output 15 minutes ahead using current weather conditions (irradiation, ambient/module temperature) and recent generation history. The goal is to show that a simple ML model can beat the naive assumption that "the next reading equals the current reading."

## 2. What data will you use, and do you have access to it?

**Dataset:** [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (Kaggle, Anikannal) — two Indian solar plants, 34 days of data at 15-minute resolution.

**Files used:**
- `Plant_1_Generation_Data.csv` — `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`
- `Plant_1_Weather_Sensor_Data.csv` — `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`

**Access confirmed:** Files are plain CSVs (~11 MB total), no API key required. Downloadable directly from Kaggle (or via `kaggle datasets download`), with public GitHub mirrors of the same CSVs as a fallback. No scraping, auth, or infra setup needed before the clock starts.

## 3. Smallest end-to-end version in hour 1

1. Load `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv`, parsing `DATE_TIME`.
2. **Collapse generation to one row per timestamp.** Plant 1 has 22 inverters (`SOURCE_KEY`) reporting per timestamp but only one weather sensor, so `groupby("DATE_TIME")["AC_POWER"].sum()` before merging — otherwise generation and weather won't line up 1:1.
3. **Merge** the collapsed generation with weather on `DATE_TIME`, sort chronologically, and check for timestamp gaps (`.diff()` on `DATE_TIME`). This dataset is known to have missing 15-minute readings, and a naive shift would silently pair non-adjacent rows as if they were 15 minutes apart.
4. **Build the lag feature.** `AC_POWER(t)` needs no transformation — it's already "current" in each row. The forecasting problem is created by shifting the label instead: `df["target"] = df["AC_POWER"].shift(-1)` pulls the *next* row's power backward onto the current row, so each row reads as "here's now (weather + current power) → here's what happens next." Drop the final row, which has no future value to predict.
5. **Feature set per row:** `IRRADIATION(t)`, `AMBIENT_TEMPERATURE(t)`, `MODULE_TEMPERATURE(t)`, `AC_POWER(t)` (the lag-1 feature, relative to the target) → **target:** `AC_POWER(t+1)`.
6. **Time-based train/test split** (first ~75% of days = train, last ~25% = test) — never a random split, since adjacent rows are highly correlated.
7. Fit a single `LinearRegression` on this table and print MAE/RMSE/R² against the persistence baseline (`predicted = AC_POWER(t)`, i.e. the lag-1 feature used as the entire prediction).

That's a crude but fully working pipeline — CSV in, number out — to sanity-check the framing before adding models.

## 4. What will you measure, and what's your baseline?

**Baseline (persistence model):** predicted `AC_POWER(t+1) = AC_POWER(t)`. Standard, hard-to-beat baseline for short-horizon time series.

**Metrics:** MAE, RMSE, R² on the held-out (future) time window, compared model-vs-baseline. Optionally split error by "stable irradiance" vs. "changing irradiance" periods — that's where the model should earn its keep over persistence.

**Models considered, in increasing complexity** (checked against this repo's `.venv` — scikit-learn 1.9.0 is already installed via an existing dependency, so everything below is a zero-install addition except where noted):

| Order | Model | Why it's on the list | Environment |
|---|---|---|---|
| 0 | Persistence (no model) | The baseline itself — literally the lag-1 feature used as the prediction | n/a |
| 1 | `LinearRegression` | Sanity check: is the power–irradiance relationship even roughly linear? Fast, interpretable coefficients | ✅ already available |
| 2 | `RandomForestRegressor` | Captures nonlinearities and interactions (e.g. high temperature reducing panel efficiency at high irradiance) with no manual feature engineering | ✅ already available |
| 3 | `HistGradientBoostingRegressor` | Usually the strongest tabular-regression performer; sklearn's own boosting implementation | ✅ already available |
| optional | `KNeighborsRegressor` | Cheap, intuitive "find similar past weather conditions and average their outcomes" — good for an interpretability discussion | ✅ already available |
| stretch | `XGBoost` | Marginal gain over sklearn's own boosting at best | ⚠️ **not installed** — needs `pip install xgboost` before use; only pursue if time remains, since sklearn's boosting already covers this tier |

**Core question:** *Does adding weather features and a simple model meaningfully reduce 15-minute-ahead forecasting error versus assuming no change from the current reading?*

## 5. Which course tools does this draw on?

- **Files/CSV I/O:** loading and merging two raw CSVs, timestamp parsing.
- **Arrays/linear algebra:** feature matrix construction, lag features.
- **Pipelines & honest evaluation:** time-based (non-random) train/test split to avoid leakage; baseline-vs-model comparison.
- **Training loops / model fitting:** scikit-learn `fit`/`predict` for regression models of increasing complexity.
- **Plots:** predicted vs. actual power over time; error comparison across models.
- **Error handling:** guarding against missing/NaN sensor readings and timestamp mismatches between the two files.

No new frameworks or infrastructure required — everything runs on pandas + scikit-learn + matplotlib, already in the course stack.
