# 🏋️ Apple Health ML Analyzer — Behaviour Intelligence Edition

A Gradio-powered web dashboard that ingests your Apple Health export ZIP, parses every relevant biometric signal, engineers composite health scores, and runs four research-backed machine learning models — all in a single Jupyter notebook executed in Google Colab or locally.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How to Run](#how-to-run)
3. [Input: The Apple Health Export ZIP](#input-the-apple-health-export-zip)
   - [File Structure](#file-structure)
   - [export.xml Schema](#exportxml-schema)
4. [Dataset Variables — Every Field Explained](#dataset-variables--every-field-explained)
   - [Raw HealthKit Identifiers (SOURCE)](#raw-healthkit-identifiers-source)
   - [Sleep Stage Variables](#sleep-stage-variables)
   - [Workout Variables](#workout-variables)
   - [Engineered Scores](#engineered-scores)
   - [Rolling & Z-Score Features](#rolling--z-score-features)
   - [Training Load Variables (ACWR)](#training-load-variables-acwr)
   - [ML-Only Feature Columns](#ml-only-feature-columns)
   - [Clustering & PCA Columns](#clustering--pca-columns)
   - [Anomaly Detection Columns](#anomaly-detection-columns)
   - [Overtraining Detection Columns](#overtraining-detection-columns)
5. [Code Architecture](#code-architecture)
   - [Cell 1 — Installs](#cell-1--installs)
   - [Cell 2 — Imports](#cell-2--imports)
   - [Cell 3 — Constants & Parser (`parse_zip`)](#cell-3--constants--parser-parse_zip)
   - [Cell 4 — DataFrame Builder (`build_dataframe`)](#cell-4--dataframe-builder-build_dataframe)
   - [Cell 5 — Analysis Pipeline (`run_analysis`)](#cell-5--analysis-pipeline-run_analysis)
   - [Cell 6 — Machine Learning Engine (`run_all_ml`)](#cell-6--machine-learning-engine-run_all_ml)
   - [Cell 7 — ML Summary Card Builder](#cell-7--ml-summary-card-builder)
   - [Cell 8 — Chart Builder (`build_charts`)](#cell-8--chart-builder-build_charts)
   - [Cell 9 — Summary Builder (`build_summary`)](#cell-9--summary-builder-build_summary)
   - [Cell 10 — Gradio Handler (`analyze_health`)](#cell-10--gradio-handler-analyze_health)
   - [Cell 11 — Gradio UI Layout](#cell-11--gradio-ui-layout)
6. [Machine Learning Models](#machine-learning-models)
   - [Model 1 — Next-Day HRV Prediction (Random Forest)](#model-1--next-day-hrv-prediction-random-forest)
   - [Model 2 — Overtraining Detection (Isolation Forest)](#model-2--overtraining-detection-isolation-forest)
   - [Model 3 — Injury Risk Classification (Gradient Boosting)](#model-3--injury-risk-classification-gradient-boosting)
   - [Model 4 — Behavioural Fingerprint (Weekly PCA + Clustering)](#model-4--behavioural-fingerprint-weekly-pca--clustering)
7. [Dashboard Tabs & Charts](#dashboard-tabs--charts)
8. [Composite Score Formulas](#composite-score-formulas)
9. [Recovery Recommendation Logic](#recovery-recommendation-logic)
10. [Dependencies](#dependencies)
11. [Limitations & Notes](#limitations--notes)

---

## Project Overview

The notebook connects to your personal Apple Health export and transforms raw XML biometric data into actionable athletic intelligence. It calculates three daily composite scores (Recovery, Energy, Peak Performance), detects anomalies and overtraining signals, classifies injury risk, predicts tomorrow's HRV, and clusters your weekly training behaviour into personality archetypes — all visualised in an interactive web UI.

---

## How to Run

1. **Export your data** from iPhone: `Health app → your avatar → Export All Health Data`. This produces `apple_health_export.zip`.
2. Open the notebook in **Google Colab** (or a local Jupyter environment with Python 3.10+).
3. Run all cells **top to bottom** (Runtime → Run All).
4. The final cell prints a **public shareable URL** — open it in any browser.
5. Upload your ZIP file. The full dashboard appears automatically.

---

## Input: The Apple Health Export ZIP

### File Structure

```
apple_health_export.zip
├── apple_health_export/
│   ├── export.xml              ← PRIMARY DATA SOURCE (400 MB+)
│   ├── export_cda.xml          ← Clinical Document Architecture copy (not used)
│   ├── electrocardiograms/
│   │   └── ecg_YYYY-MM-DD.csv  ← Raw ECG waveform CSVs (not used by this notebook)
│   └── workout-routes/
│       └── route_YYYY-MM-DD_HH.MMam.gpx ← GPS tracks (not used by this notebook)
```

Only `apple_health_export/export.xml` is parsed by this notebook. Everything else is silently ignored.

### export.xml Schema

The XML file contains several element types. The notebook reads two of them:

#### `<Record>` elements

Each `<Record>` represents a single biometric measurement. Key attributes:

| Attribute | Type | Description |
|---|---|---|
| `type` | String | The HealthKit identifier, e.g. `HKQuantityTypeIdentifierHeartRate` |
| `value` | String (numeric) | The measured value as a string, cast to `float` in code |
| `unit` | String | Unit of the value (e.g. `ms`, `count/min`, `kcal`) |
| `sourceName` | String | Device or app that recorded the sample (e.g. `Apple Watch`) |
| `sourceVersion` | String | OS/app version at recording time |
| `device` | String | Hardware descriptor string |
| `creationDate` | DateTime string | When the record was created on the device |
| `startDate` | DateTime string | Start of the measurement window — used for date grouping |
| `endDate` | DateTime string | End of the measurement window — used for sleep duration math |

#### `<Workout>` elements

Each `<Workout>` represents a logged exercise session. Key attributes:

| Attribute | Type | Description |
|---|---|---|
| `workoutActivityType` | String | e.g. `HKWorkoutActivityTypeRunning`, `HKWorkoutActivityTypeCycling` |
| `duration` | Float | Total duration of the workout in minutes |
| `durationUnit` | String | Always `min` |
| `totalDistance` | Float | Total distance covered |
| `totalDistanceUnit` | String | Unit of distance |
| `totalEnergyBurned` | Float | Calories burned during the workout |
| `startDate` | DateTime string | Workout start time — used for date grouping |
| `endDate` | DateTime string | Workout end time |

Each `<Workout>` may contain child `<WorkoutStatistics>` elements. The notebook extracts the `average` attribute from any statistics record whose `type` contains `HeartRate`, yielding the average heart rate for the session.

---

## Dataset Variables — Every Field Explained

After parsing and processing, the notebook builds a single daily-granularity Pandas DataFrame (`dfw`). Each row is one calendar day. Here is every column:

### Raw HealthKit Identifiers (SOURCE)

These are the original Apple Health metric types that the parser extracts. The `TARGET` dictionary in the code maps each HealthKit identifier (left) to a short column name (right).

| Column Name | HealthKit Identifier | Unit | Aggregation Method | Description |
|---|---|---|---|---|
| `hrv` | `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` | ms | Mean of all readings that day | **Heart Rate Variability** — the standard deviation of beat-to-beat intervals (SDNN). Higher values generally indicate better recovery and parasympathetic nervous system activity. Apple Watch measures this during sleep. |
| `rhr` | `HKQuantityTypeIdentifierRestingHeartRate` | bpm | Last reading of the day | **Resting Heart Rate** — beats per minute measured at complete rest, typically morning. Lower values generally indicate better cardiovascular fitness. Uses the last recorded value because Apple typically logs one authoritative RHR per day. |
| `hr` | `HKQuantityTypeIdentifierHeartRate` | bpm | Mean of all readings that day | **Instantaneous Heart Rate** — all sampled heart rates across the day, averaged. Includes readings during activity, rest, and standing. Distinct from RHR. |
| `active_cal` | `HKQuantityTypeIdentifierActiveEnergyBurned` | kcal | Sum | **Active Calories** — energy burned through intentional movement above your baseline metabolism. This is the red "Move" ring value on Apple Watch. Summed across all readings per day. |
| `basal_cal` | `HKQuantityTypeIdentifierBasalEnergyBurned` | kcal | Sum | **Basal Metabolic Rate Calories** — estimated energy your body burns at complete rest to maintain basic functions (breathing, circulation, etc.). Determined by Apple using your age, height, weight, and sex. |
| `steps` | `HKQuantityTypeIdentifierStepCount` | count | Sum | **Step Count** — total number of steps taken, summed from all sources (iPhone, Apple Watch, other apps). Duplicate readings from multiple devices are all included in the sum; Apple Watch typically dominates. |
| `exercise_min` | `HKQuantityTypeIdentifierAppleExerciseTime` | min | Sum | **Exercise Minutes** — minutes during which your heart rate was elevated to an exercise-equivalent level, contributing to the green "Exercise" ring. |
| `vo2max` | `HKQuantityTypeIdentifierVO2Max` | mL/kg/min | Last reading of the day | **VO2 Max** — maximum oxygen uptake capacity, Apple's estimate of aerobic fitness. Measured periodically (not daily) during outdoor runs or walks with wrist heart rate. Values above 42 indicate good aerobic fitness for most adults. |
| `spo2` | `HKQuantityTypeIdentifierOxygenSaturation` | % | Mean | **Blood Oxygen Saturation** — percentage of haemoglobin carrying oxygen, measured by Apple Watch's SpO2 sensor during sleep. Normal range is 95–100%. |
| `resp_rate` | `HKQuantityTypeIdentifierRespiratoryRate` | breaths/min | Mean | **Respiratory Rate** — breathing rate measured during sleep. Elevation above your baseline can signal illness, stress, or overtraining. |
| `walk_speed` | `HKQuantityTypeIdentifierWalkingSpeed` | km/h or m/s | Mean | **Walking Speed** — average pace during walking sessions. Apple uses this as a mobility and fitness metric. |
| `weight` | `HKQuantityTypeIdentifierBodyMass` | kg | Last reading of the day | **Body Mass / Weight** — logged manually or from a connected smart scale. Uses the last reading of the day. |
| `wrist_temp` | `HKQuantityTypeIdentifierAppleSleepingWristTemperature` | °C | Last reading of the day | **Sleeping Wrist Temperature** — deviation from your personal baseline wrist skin temperature during sleep, measured by Apple Watch Series 8+. Elevation can indicate fever, illness, or alcohol consumption. |
| `distance` | `HKQuantityTypeIdentifierDistanceWalkingRunning` | km | Sum | **Distance Walked/Run** — total distance covered on foot during the day, summed from all sources. |
| `flights` | `HKQuantityTypeIdentifierFlightsClimbed` | count | Sum | **Flights of Stairs Climbed** — number of floor ascents detected by the barometer in iPhone or Apple Watch. |

### Sleep Stage Variables

Parsed from `HKCategoryTypeIdentifierSleepAnalysis` records. Each record has a `value` attribute with a sleep stage identifier, and the duration is computed as `(endDate - startDate)` in hours.

| Column Name | HKCategory Value | Description |
|---|---|---|
| `deep` | `HKCategoryValueSleepAnalysisAsleepDeep` | Hours of **deep (slow-wave) sleep** — the most physically restorative sleep stage. Critical for muscle repair, immune function, and growth hormone release. |
| `core` | `HKCategoryValueSleepAnalysisAsleepCore` | Hours of **core (light NREM) sleep** — stage 1 and 2 sleep. Makes up the largest portion of the night. Important for memory consolidation. |
| `rem` | `HKCategoryValueSleepAnalysisAsleepREM` | Hours of **REM sleep** — the stage associated with dreaming, emotional processing, and cognitive function. |
| `awake` | `HKCategoryValueSleepAnalysisAwake` | Hours spent **awake while in bed** — detected by movement and heart rate patterns. |
| `inbed` | `HKCategoryValueSleepAnalysisInBed` | Total hours in bed regardless of sleep stage. Superset of all other sleep categories. |
| `total_sleep` | Computed | **Total hours of actual sleep** = `deep + core + rem`. If this sum is zero (older Apple Watch firmware that doesn't break down stages), falls back to the `asleep` value. |

### Workout Variables

Parsed from `<Workout>` XML elements and aggregated per day.

| Column Name | Source | Description |
|---|---|---|
| `workout_count` | Workout records | Number of distinct workout sessions logged that day. |
| `total_duration` | `duration` attribute | Total minutes across all workouts logged that day. |
| `workout_load` | Computed | **Daily Training Load** — a composite intensity score calculated as: `Σ(duration_min × (avg_hr / 170))` for each workout. Workouts with no heart rate data assume a default average HR of 130 bpm. Higher values mean harder training sessions. This is the primary input to ACWR calculations. |
| `workout_types` | `workoutActivityType` | Comma-separated string of workout type names that day, e.g. `Running, Cycling`. The `HKWorkoutActivityType` prefix is stripped. |

### Engineered Scores

Three composite daily scores, each on a 0–100 scale, are computed from the raw metrics:

| Column Name | Range | Description |
|---|---|---|
| `recovery` | 0–100 | **Recovery Score** — how well your body has recovered from training. Formula: `50 + (hrv_z × 0.50 × 20) + (rhr_z × 0.30 × 20) + ((sleep_q - 0.5) × 0.20 × 20)`. Above 70 = Peak; 55–69 = Good; 40–54 = Moderate; below 40 = Low. |
| `energy` | 0–100 | **Energy Score** — daily physical output. Weighted combination of normalised `active_cal` (35%), `exercise_min` (35%), `steps` (15%), and `workout_load` (15%). Values are min-max scaled within the 99th percentile cap to suppress outliers. |
| `peak_score` | 0–100 | **Peak Performance Score** — readiness to perform at maximum intensity. Combines ACWR closeness to optimal (30 pts), recovery level (25 pts), recovery trend (20 pts), positive HRV slope (15 pts), and days since last heavy session (10 pts). |
| `rec_label` | Categorical | Human-readable recovery category: `'Peak'` (≥70), `'Good'` (≥55), `'Moderate'` (≥40), `'Low'` (<40). |
| `sleep_q` | 0–1 | **Sleep Quality Index** — intermediate score: `(total_sleep/8)×0.5 + (deep/1.5)×0.25 + (rem/2.0)×0.25`, clipped to [0,1]. Used internally in the recovery formula. |

### Rolling & Z-Score Features

| Column Name | Description |
|---|---|
| `hrv_roll_mean` | 28-day rolling mean of `hrv` (min 5 data points). Represents your personal HRV baseline. |
| `hrv_roll_std` | 28-day rolling standard deviation of `hrv`, clipped at minimum 1 to prevent division by zero. |
| `rhr_roll_mean` | 28-day rolling mean of `rhr`. Your personal resting HR baseline. |
| `rhr_roll_std` | 28-day rolling standard deviation of `rhr`, clipped at minimum 1. |
| `hrv_z` | **HRV Z-score**: `(hrv - hrv_roll_mean) / hrv_roll_std`. Positive values mean HRV is above your personal norm (good); negative means below baseline (fatigued). |
| `rhr_z` | **RHR Z-score**: `(rhr_roll_mean - rhr) / rhr_roll_std`. Note the sign inversion — higher RHR is worse, so the formula makes positive values correspond to better-than-baseline RHR. |
| `hrv_7d_slope` | Linear regression slope of HRV over the past 7 days. Positive = trending upward (improving recovery). Negative = HRV declining (accumulating fatigue). Computed via `numpy.polyfit`. |
| `rec_7d_avg` | 7-day rolling mean of `recovery`. Used as the trend baseline. |
| `rec_trend` | `(3-day rolling mean of recovery) - rec_7d_avg`. Positive values mean recovery is improving over the short term versus the week average. |
| `days_since_heavy` | Number of days since the last day where `workout_load ≥ 60`. Used in peak score calculation; 1–3 days after a hard session is considered optimal timing for a performance peak. |

### Training Load Variables (ACWR)

The Acute:Chronic Workload Ratio (ACWR) is a widely used sports science metric for injury risk monitoring, popularised by Gabbett (2016).

| Column Name | Description |
|---|---|
| `acute` | 7-day rolling mean of `workout_load`. Represents recent training stress. |
| `chronic` | 28-day rolling mean of `workout_load`, clipped at minimum 1. Represents your fitness base and adaptation to load. |
| `acwr` | `acute / chronic`. The ratio of recent to long-term load. Values between 0.8–1.3 are the "sweet spot". |
| `acwr_zone` | Categorical risk label: `'Undertraining'` (<0.8), `'Sweet Spot'` (0.8–1.3), `'Caution'` (1.3–1.5), `'High Risk'` (>1.5). |

### ML-Only Feature Columns

These additional rolling features are computed in `engineer_ml_features()` specifically for the machine learning models:

| Column Name | Description |
|---|---|
| `hrv_7d` | 7-day rolling mean of `hrv`. Smoothed short-term HRV trend. |
| `hrv_28d` | 28-day rolling mean of `hrv`. Long-term HRV baseline. |
| `rhr_7d` | 7-day rolling mean of `rhr`. Smoothed short-term RHR trend. |
| `rhr_28d` | 28-day rolling mean of `rhr`. Long-term RHR baseline. |
| `steps_7d` | 7-day rolling mean of `steps`. |
| `cal_7d` | 7-day rolling mean of `active_cal`. |
| `load_7d` | 7-day rolling mean of `workout_load`. Short-term load for ML features. |
| `load_28d` | 28-day rolling mean of `workout_load`. Long-term load for ML features. |
| `hrv_z_ml` | `(hrv - hrv_28d) / 28d_std`. HRV z-score computed specifically for the ML pipeline (same concept as `hrv_z` but slightly different rolling window handling). |
| `days_since_workout` | Number of days since the last day where `workout_load ≥ 60`. Recomputed within the ML module using a forward scan for precision. |
| `dow` | Day of the week as integer (0 = Monday, 6 = Sunday). |
| `is_weekend` | Binary flag: 1 if Saturday or Sunday, 0 otherwise. |
| `next_hrv` | One-day forward shift of `hrv`. This is the **target variable** for Model 1 (HRV prediction). The model learns to predict tomorrow's HRV from today's features. |
| `risk_class` | Integer label (0–3) for injury risk zone derived from `acwr`: 0 = Undertraining, 1 = Sweet Spot, 2 = Caution, 3 = High Risk. Target variable for Model 3. |
| `risk_pred` | Gradient Boosting model's predicted `risk_class` for each historical day. |
| `risk_name` | Human-readable version of `risk_pred`. |

### Clustering & PCA Columns

| Column Name | Description |
|---|---|
| `cal_n` | Min-max normalised `active_cal` (capped at 99th percentile). Intermediate feature for energy score. |
| `ex_n` | Min-max normalised `exercise_min`. |
| `step_n` | Min-max normalised `steps`. |
| `load_n` | Min-max normalised `workout_load`. |
| `cluster` | KMeans cluster ID (0–3) assigned to each day using features: `hrv`, `rhr`, `active_cal`, `steps`, `exercise_min`, `recovery`. |
| `day_type` | Human-readable cluster label: `'High Performance'`, `'Active Recovery'`, `'Moderate Training'`, or `'Fatigued'`. Assigned by comparing cluster-mean `recovery` and `active_cal` to thresholds. |
| `pc1` | First principal component score from PCA on the daily clustering features. The axis of maximum variance in your health data. |
| `pc2` | Second principal component score. Together with `pc1`, captures the majority of variance in the feature set. |
| `anomaly_score` | Mean of clipped positive z-scores across `hrv`, `rhr`, `active_cal`, and `steps` relative to 28-day baselines. Higher values indicate the day is unusually far from your norms. |
| `anomaly` | Boolean flag: `True` if `anomaly_score > 1.2`. Marks days where multiple metrics simultaneously deviated from baseline, suggesting illness, extreme fatigue, or unusual activity. |

### Overtraining Detection Columns

Added by Model 2 (Isolation Forest):

| Column Name | Description |
|---|---|
| `ot_flag` | Output from Isolation Forest: `1` = normal day, `-1` = anomalous / high-risk day. The model is trained with `contamination=0.10`, expecting ~10% of days to be flagged. |
| `ot_score` | Raw anomaly score from Isolation Forest (more negative = more anomalous). Used to distinguish Moderate Risk from Severe Risk. |
| `ot_severity` | Categorical: `'Normal'`, `'Moderate Risk'` (flagged, score ≥ -0.15), `'Severe Risk'` (flagged, score < -0.15). |

---

## Code Architecture

The notebook is structured as a pipeline of pure functions, each building on the previous. Here is a detailed walkthrough of every cell:

### Cell 1 — Installs

```python
!pip install -q gradio plotly scikit-learn scipy statsmodels xgboost
```

Installs all required packages in the Colab environment. `xgboost` is installed but not actively used in the current implementation (the gradient boosting uses sklearn's native `GradientBoostingClassifier`).

### Cell 2 — Imports

Imports are grouped by function:

- **Standard library**: `zipfile`, `xml.etree.ElementTree`, `collections.defaultdict`, `datetime`
- **Data**: `numpy`, `pandas`
- **Visualisation**: `plotly.graph_objects`, `plotly.express`, `plotly.subplots.make_subplots`
- **ML**: `sklearn` (preprocessing, decomposition, cluster, ensemble, model_selection, metrics), `scipy.stats`
- **UI**: `gradio`

### Cell 3 — Constants & Parser (`parse_zip`)

**`TARGET` dictionary**: Maps the 15 HealthKit quantity type identifiers to short snake_case column names. Only record types listed here are extracted; all others are ignored.

**`SLEEP_MAP` dictionary**: Maps the 6 sleep stage HealthKit category values to short stage names (`deep`, `core`, `rem`, `asleep`, `inbed`, `awake`).

**`to_date(s)`**: Helper that parses the first 10 characters of an ISO datetime string (`YYYY-MM-DD`) into a `datetime.date` object. Returns `None` on failure.

**`to_dt(s)`**: Helper that parses the first 19 characters of a datetime string into a full `datetime` object. Used for computing sleep duration.

**`parse_zip(zip_path)`**:
1. Opens the ZIP file as bytes, extracts `apple_health_export/export.xml`.
2. Parses the XML tree using `ElementTree`.
3. Iterates over every `<Record>` element:
   - If the `type` is in `TARGET`, appends the float `value` to `data[date][metric]`.
   - If the `type` is `HKCategoryTypeIdentifierSleepAnalysis`, looks up the stage in `SLEEP_MAP` and stores a dict with `date`, `stage`, and `hours` (computed from start/end timestamps).
4. Iterates over every `<Workout>` element, extracting `duration`, `workoutActivityType`, and optionally average heart rate from `<WorkoutStatistics>` children.
5. Returns three objects: `data` (nested defaultdict), `sleep_records` (list of dicts), `workout_records` (list of dicts).

### Cell 4 — DataFrame Builder (`build_dataframe`)

**`build_dataframe(data, sleep_records, workout_records)`**:

1. **Quantity metrics**: For each day in `data`, creates a row. Metrics are aggregated differently depending on their nature:
   - **Last-value** (`rhr`, `vo2max`, `wrist_temp`): Uses `v[-1]` — the final reading of the day, since Apple records one authoritative value.
   - **Sum** (`active_cal`, `basal_cal`, `steps`, `exercise_min`, `distance`, `flights`): Uses `sum(v)` — these are cumulative across the day.
   - **Mean** (all others including `hrv`, `hr`, `spo2`, etc.): Uses `np.mean(v)` — averaged across multiple sensor readings.

2. **Sleep data**: Pivots `sleep_records` into a date-indexed table by stage, computes `total_sleep`, and merges onto the main DataFrame.

3. **Workout data**: Groups `workout_records` by date, computes `workout_count`, `total_duration`, `workout_load` (weighted by heart rate intensity), and `workout_types`. Merges onto the main DataFrame.

4. **Date filtering**: Filters to `date >= 2025-01-01`. If fewer than 20 rows remain after filtering (insufficient recent data), falls back to the full historical dataset.

5. Returns the final daily DataFrame `dfw`.

### Cell 5 — Analysis Pipeline (`run_analysis`)

**`run_analysis(dfw)`**:

Computes all derived columns on the daily DataFrame. Executed in this order:

1. **HRV and RHR baselines**: 28-day rolling means and standard deviations.
2. **Z-scores**: `hrv_z` and `rhr_z` as deviations from personal baselines.
3. **Sleep quality** (`sleep_q`): Weighted index from `total_sleep`, `deep`, and `rem`.
4. **Recovery score** (0–100): Linear combination of HRV z-score, RHR z-score, and sleep quality.
5. **Recovery label** (`rec_label`): Thresholded from `recovery`.
6. **Energy score** (0–100): Normalised and weighted combination of activity metrics.
7. **ACWR**: `acute` (7d), `chronic` (28d), their ratio, and `acwr_zone`.
8. **HRV slope** (`hrv_7d_slope`): Linear regression gradient of HRV over the last 7 days.
9. **Recovery trend** (`rec_trend`): 3-day mean minus 7-day mean of recovery.
10. **Days since heavy session** (`days_since_heavy`): Backward scan for last day with `workout_load ≥ 60`.
11. **Peak score** (0–100): Multi-factor performance readiness score.
12. **Anomaly detection**: Z-score-based multivariate deviation flagging.
13. **KMeans clustering** (k=4): Groups days by health profile.
14. **Day-type labelling**: Translates cluster IDs to human-readable archetypes.
15. **PCA** (2 components): Dimensionality reduction for visualisation.
16. **Spearman correlation matrix**: Computed on 10 key metrics.

Returns `(dfw, corr_m, pca_var)`.

### Cell 6 — Machine Learning Engine (`run_all_ml`)

Orchestrates four ML models. See [Machine Learning Models](#machine-learning-models) for detailed descriptions. The top-level function `run_all_ml(dfw)` calls:

1. `engineer_ml_features(dfw)` — adds ML-specific rolling features.
2. `run_model1_hrv_prediction(df)` — Random Forest HRV forecasting.
3. `run_model2_overtraining_detection(df)` — Isolation Forest.
4. `run_model3_injury_risk(df)` — Gradient Boosting classifier.
5. `run_model4_behavioural_fingerprint(df)` — Weekly PCA + KMeans.

Each model function returns a `result` dict and the updated DataFrame.

### Cell 7 — ML Summary Card Builder

**`build_ml_summary_html(m1, m2, m3, m4)`**: Generates an HTML string containing a 2×2 grid of styled cards, one per ML model. Each card shows:

- Model name and status badge (`ACTIVE` or `LOW DATA`)
- Research citation
- Key output values (predicted HRV, anomaly count, risk zone, archetype)
- Supporting metrics (CV R², sample count, probabilities)

### Cell 8 — Chart Builder (`build_charts`)

**`build_charts(dfw, corr_m, pca_var)`**: Generates 8 interactive Plotly figures and returns them in a `charts` dictionary.

| Key | Chart | Description |
|---|---|---|
| `timeline` | 3-panel line chart | 90-day history of Recovery, Energy, and Peak Score with workout day markers |
| `hrv` | Scatter plot | All-time HRV readings coloured by recovery label, with 7-day moving average |
| `acwr` | 2-panel chart | 60-day training load (bars + lines) and ACWR with coloured risk zones |
| `correlation` | Heatmap | Spearman correlation matrix of 10 key metrics |
| `clustering` | PCA scatter | Daily data points in PC1/PC2 space, coloured by day type |
| `anomaly` | Line + markers | Recovery score over time with anomaly days marked as red X |
| `calendar` | Heatmap | Recovery values in a week × day-of-week grid |
| `vo2max` | Line + markers | VO2 Max trend over time (only generated if VO2 data exists) |

### Cell 9 — Summary Builder (`build_summary`)

**`build_summary(dfw)`**: Extracts the most recent day's values and computes period summaries. Returns a nested dict with:

- `today`: Current values for recovery, energy, peak_score, hrv, rhr, steps, active_cal, exercise_min, acwr, acwr_zone.
- `trends`: avg_recovery_7d, avg_hrv_7d, workout_days_30d, peak_days_total, low_days_total, anomaly_days.
- `recommendation`: The next training window recommendation and colour code.
- `total_days`: Total number of days in the dataset.

### Cell 10 — Gradio Handler (`analyze_health`)

**`analyze_health(zip_file)`**: The main entry point called by Gradio on every file upload. Orchestrates the full pipeline:

1. Calls `parse_zip` → `build_dataframe` → `run_analysis` → `run_all_ml` → `build_charts` → `build_summary` → `build_ml_summary_html`.
2. Renders the top summary HTML card with today's scores, stats, recommendation, and ML insights.
3. Returns a tuple of 13 outputs: `(summary_html, chart1, ..., chart12)`.
4. Catches all exceptions and returns a formatted error HTML with full traceback.

### Cell 11 — Gradio UI Layout

Builds the web interface using `gr.Blocks` with a dark GitHub-inspired theme. The layout:

- **Header**: Markdown with instructions and model descriptions.
- **Row**: File upload widget + Analyze button.
- **Summary HTML block**: Full-width HTML output panel.
- **Tabbed chart area**: 12 tabs, each containing a `gr.Plot` widget.

Both `file_input.change` and `analyze_btn.click` trigger `analyze_health` automatically.

---

## Machine Learning Models

### Model 1 — Next-Day HRV Prediction (Random Forest)

**Research basis**: Flatt et al. 2021, *International Journal of Sports Physiology and Performance* — HRV-guided training improves adaptation when daily HRV is predicted from rolling physiological context.

**Task**: Regression. Predict tomorrow's HRV value from today's physiological state.

**Features** (9 inputs):

| Feature | Description |
|---|---|
| `hrv_7d` | 7-day rolling mean HRV |
| `hrv_28d` | 28-day rolling mean HRV (baseline) |
| `hrv_z_ml` | Current HRV deviation from 28-day baseline |
| `rhr_7d` | 7-day rolling mean RHR |
| `steps_7d` | 7-day rolling mean step count |
| `cal_7d` | 7-day rolling mean active calories |
| `load_7d` | 7-day rolling mean workout load |
| `acwr` | Acute:Chronic workload ratio |
| `days_since_workout` | Days since last heavy session |

**Target**: `next_hrv` (tomorrow's HRV, via `.shift(-1)`)

**Model**: `RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=3)`

**Evaluation**: 5-fold cross-validation R² (when ≥30 samples). Requires minimum 15 samples.

**Outputs**: Predicted next-day HRV in ms, CV R², training MAE, top 5 feature importances, forecast chart.

---

### Model 2 — Overtraining Detection (Isolation Forest)

**Research basis**: Meeusen et al. 2023, *European Journal of Sport Science* — early overtraining detection requires multivariate monitoring, not single-metric thresholds.

**Task**: Unsupervised anomaly detection. Identify days where the combination of fatigue signals is statistically unusual.

**Features** (5 inputs):

| Feature | Description |
|---|---|
| `hrv_z_ml` | HRV deviation from personal baseline |
| `rhr_7d` | Smoothed resting heart rate |
| `load_7d` | Recent training load |
| `acwr` | Workload ratio |
| `steps_7d` | Recent daily activity level |

**Model**: `IsolationForest(contamination=0.10, n_estimators=200)` — expects ~10% of days to be anomalous.

**Severity classification**:
- `ot_score ≥ -0.15` → Moderate Risk
- `ot_score < -0.15` → Severe Risk

**Outputs**: Count of high-risk days, top 5 most anomalous dates with their metrics, colour-coded recovery timeline chart.

---

### Model 3 — Injury Risk Classification (Gradient Boosting)

**Research basis**: Gabbett 2016, *British Journal of Sports Medicine* — ACWR between 0.8–1.3 is the sweet spot; values above 1.5 dramatically increase injury probability. A classifier outperforms simple threshold rules.

**Task**: Multi-class classification. Predict which ACWR risk zone each day falls into (4 classes).

**Features** (7 inputs):

| Feature | Description |
|---|---|
| `load_7d` | 7-day rolling mean workout load |
| `load_28d` | 28-day rolling mean workout load |
| `hrv_z_ml` | HRV deviation from baseline |
| `steps_7d` | Recent activity level |
| `cal_7d` | Recent active calorie burn |
| `acwr` | Acute:Chronic workload ratio |
| `days_since_workout` | Recovery days since last heavy session |

**Target**: `risk_class` (0 = Undertraining, 1 = Sweet Spot, 2 = Caution, 3 = High Risk)

**Model**: `GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3)`

**Outputs**: Today's predicted risk zone, probability distribution across all 4 classes, top 5 feature importances, 60-day load + ACWR chart with coloured risk zones. Requires minimum 20 samples.

---

### Model 4 — Behavioural Fingerprint (Weekly PCA + Clustering)

**Research basis**: Koskela et al. 2022, *Sensors* — wearable data reveals stable weekly behavioural patterns; PCA identifies training personality archetypes useful for load management personalisation.

**Task**: Unsupervised weekly pattern clustering. Identify which type of training week you are in.

**Weekly features** (mean and std per feature per week):
`steps`, `active_cal`, `workout_load`, `exercise_min`, `distance`, `recovery`

**Pipeline**:
1. Aggregate daily data to ISO calendar weeks.
2. Compute mean and std for each feature per week.
3. Min-max scale the feature matrix.
4. PCA reduces to 3 components (or fewer if data is limited).
5. KMeans (k=4) clusters weeks in PCA space.

**Archetype labelling logic**:

| Archetype | Condition |
|---|---|
| High Performance Week | avg_load > 150 AND avg_recovery > 60 |
| Training Block Week | avg_load > 80 |
| Active Rest Week | avg_steps > 7000 AND avg_load < 40 |
| Recovery Week | All others |

**Outputs**: Current week's archetype, distribution of archetypes across all weeks, % variance explained by PC1+PC2, interactive PCA scatter plot. Requires minimum 8 weeks of data.

---

## Dashboard Tabs & Charts

| Tab | Chart Key | Description |
|---|---|---|
| Timeline | `timeline` | 90-day stacked area chart: Recovery (green), Energy (yellow), Peak Score (red) |
| HRV | `hrv` | Full HRV history coloured by recovery level; normal range band; 7-day rolling average |
| Training Load | `acwr` | 60-day bar + line chart for daily load, acute (7d), chronic (28d) loads; ACWR with risk zone colour bands |
| Correlations | `correlation` | Spearman ρ heatmap across 10 key metrics: HRV, RHR, active calories, steps, exercise minutes, recovery, energy, workout load, ACWR, peak score |
| Day Clusters | `clustering` | KMeans + PCA scatter: each dot is a day, coloured by day type (High Performance / Active Recovery / Moderate Training / Fatigued) |
| Anomalies | `anomaly` | Recovery timeline with multivariate anomaly days marked by red X symbols |
| Calendar | `calendar` | Recovery heatmap organised by day-of-week (rows) × calendar week (columns) |
| VO2 Max | `vo2max` | VO2 Max trend with aerobic fitness zone annotations (only visible if Apple Watch has recorded VO2) |
| ML1 HRV Forecast | `m1['fig']` | Actual vs. predicted HRV timeline with tomorrow's forecast point |
| ML2 Overtraining | `m2['fig']` | Recovery scatter coloured by Isolation Forest severity (Normal / Moderate / Severe Risk) |
| ML3 Injury Risk | `m3['fig']` | 2-panel: training load history + ACWR scatter coloured by Gradient Boosting predicted risk zone |
| ML4 Fingerprint | `m4['fig']` | Weekly archetypes in PCA space, each week labelled by year and ISO week number |

---

## Composite Score Formulas

### Recovery Score (0–100)

```
recovery = clip(
    50
    + hrv_z  × 0.50 × 20
    + rhr_z  × 0.30 × 20
    + (sleep_q - 0.5) × 0.20 × 20
, 0, 100)
```

Where:
- `hrv_z = (hrv - 28d_mean_hrv) / 28d_std_hrv`
- `rhr_z = (28d_mean_rhr - rhr) / 28d_std_rhr` ← note: inverted (lower RHR is better)
- `sleep_q = (total_sleep/8)×0.5 + (deep/1.5)×0.25 + (rem/2.0)×0.25`

### Energy Score (0–100)

```
energy = (cal_n×0.35 + ex_n×0.35 + step_n×0.15 + load_n×0.15) × 100
```

All inputs are min-max scaled within the 99th percentile cap.

### Peak Performance Score (0–100)

```
peak_score = clip(
    acwr_component  (max 30)
  + recovery_component (max 25)
  + trend_component (max 20)
  + hrv_slope_component (max 15)
  + timing_component (max 10)
, 0, 100)
```

Where:
- `acwr_component = 30 × max(0, 1 - |acwr - 1.05| / 0.5)` — peaks at ACWR = 1.05
- `recovery_component = recovery / 100 × 25`
- `trend_component = max(0, min(20, rec_trend × 2))`
- `hrv_slope_component = max(0, min(15, hrv_7d_slope × 3))`
- `timing_component = 10 if 1 ≤ days_since_heavy ≤ 3 else 0`

### Sleep Quality Index (0–1)

```
sleep_q = clip(total_sleep/8, 0,1) × 0.5
        + clip(deep/1.5, 0,1) × 0.25
        + clip(rem/2.0, 0,1) × 0.25
```

### Workout Load (per session)

```
workout_load = Σ [ duration_min × (avg_hr / 170) ]
```

Missing HR defaults to 130 bpm. A 60-minute workout at HR = 170 produces a load of 60.

---

## Recovery Recommendation Logic

The `build_summary` function determines a training recommendation based on three conditions evaluated in priority order:

| Condition | Window Label | Action |
|---|---|---|
| `recovery ≥ 70` AND `acwr ≤ 1.3` AND `hrv_slope ≥ 0` | TODAY — Peak condition | GO HARD: Max intensity session, push for PRs |
| `recovery ≥ 55` AND `acwr ≤ 1.4` | 1-2 DAYS — Building phase | TRAIN HARD: High volume at 80–90% effort |
| `recovery < 40` OR `consecutive workout days ≥ 4` | 2-4 DAYS — After recovery | REST: Light walk or yoga only |
| All others | 2-3 DAYS — After moderate build | MODERATE: Zone 2 cardio, build the base |

---

## Dependencies

| Package | Version (tested) | Purpose |
|---|---|---|
| `gradio` | ≥4.0 | Web UI framework |
| `plotly` | ≥5.0 | Interactive charts |
| `scikit-learn` | ≥1.3 | All ML models and preprocessing |
| `scipy` | ≥1.10 | Statistical utilities |
| `statsmodels` | ≥0.14 | Installed but not directly called in current version |
| `xgboost` | ≥2.0 | Installed but not directly called in current version |
| `numpy` | ≥1.24 | Numerical operations |
| `pandas` | ≥2.0 | DataFrame operations |

---

## Limitations & Notes

- **Data quality depends on Apple Watch wear consistency.** Days without a watch will have `NaN` for most metrics. Rolling calculations use `min_periods` to handle gaps gracefully.
- **The notebook filters to 2025-01-01 onwards by default.** If you have fewer than 20 days of recent data it falls back to the full dataset automatically.
- **ML models require minimum data thresholds:** Model 1 needs 15+ samples, Model 3 needs 20+, Model 4 needs 8+ weeks. Below these thresholds, the models report `status='insufficient_data'` and display `LOW DATA` badges.
- **Workout load uses a simplified heart rate intensity formula.** It is not equivalent to Training Stress Score (TSS) or other formalised systems that require lactate threshold calibration.
- **Recovery score is a relative measure**, not an absolute physiological standard. It compares you to your own rolling baseline, so it adapts to your fitness level over time.
- **All processing happens locally** — no data is sent to external servers beyond what Gradio requires for the shareable link tunnel.
- **The export XML file is typically 400–500 MB.** Parsing can take 30–90 seconds depending on the runtime environment.
- **ECG CSV files and GPX workout route files** in the ZIP are not read by this notebook.
