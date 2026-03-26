# 🚦 Traffic Accident Severity Prediction

> A complete end-to-end machine learning pipeline — from synthetic data generation, preprocessing & GridSearchCV-driven feature selection, through to multi-model evaluation using Logistic Regression and SVM.

---

## 📁 Project Structure

```
traffic-accident-severity/
│
├── 📓 01_data_generation.ipynb              ← Synthetic dataset creation
├── 📓 02_preprocess_traffic_accidents.ipynb ← Preprocessing + Feature Selection
├── 📓 03_modelling_gridsearchcv.ipynb       ← Modelling & Evaluation
│
├── 📄 traffic_accident_dataset.csv          ← Raw synthetic dataset (~8 400 rows)
├── 📄 train_preprocessed.csv                ← Clean training split (6 400 rows)
├── 📄 test_preprocessed.csv                 ← Clean test split   (1 600 rows)
│
└── 📄 README.md                             ← You are here
```

---

## 🎯 Problem Statement

Road traffic accidents are a leading cause of injury and death worldwide. This project builds a **multi-class classification system** that predicts the severity of a traffic accident as:

| Label | Class | Meaning |
|-------|-------|---------|
| `0` | **Minor** | Low-impact, no serious injury |
| `1` | **Moderate** | Injury requiring medical attention |
| `2` | **Severe** | Life-threatening or fatal |

Key input signals include vehicle speed, weather condition, road surface, driver profile, light conditions, and more.

---

## 🗂️ Notebook 1 — Data Generation

**File:** `01_data_generation.ipynb`

A synthetic dataset of **8 000 base rows** (8 400 after duplicates) is generated using NumPy and Pandas with realistic statistical relationships baked in.

### Schema — 26 Columns

| Column | Type | Description |
|--------|------|-------------|
| `accident_id` | String | Unique identifier |
| `date` / `time` | String | Accident timestamp |
| `day_of_week` | Categorical | Monday – Sunday |
| `hour` | Integer | Hour of day (0–23) |
| `state` | Categorical | US state (with inconsistencies) |
| `road_type` | Categorical | Highway / Urban / Rural / Expressway / One-way |
| `junction_type` | Categorical | None / T-Junction / Roundabout / Crossroads / Slip Road |
| `weather_condition` | Categorical | Clear / Rainy / Foggy / Snowy / Windy / Stormy |
| `road_condition` | Categorical | Dry / Wet / Icy / Under Construction / Potholed |
| `light_condition` | Categorical | Daylight / Dusk / Dawn / Night-Lit / Night-Unlit |
| `speed_kmh` | Float | Vehicle speed — correlated with severity |
| `speed_limit_kmh` | Integer | Posted speed limit |
| `visibility_m` | Float | Visibility in metres — inversely correlated with severity |
| `num_vehicles` | Integer | Number of vehicles involved |
| `driver_age` | Integer | Driver age (16–80) |
| `driver_experience_yrs` | Float | Years of driving experience |
| `vehicle_type` | Categorical | Car / Truck / Motorcycle / Bus / SUV / Van |
| `alcohol_involved` | Binary | 1 = alcohol detected |
| `seatbelt_worn` | Binary | 1 = seatbelt worn |
| `num_casualties` | Integer | Number of casualties |
| `redundant_record_id` ⛔ | Integer | **Noise** — plain row counter |
| `system_flag` ⛔ | String | **Noise** — constant `"PROCESSED"` |
| `random_noise_code` ⛔ | Integer | **Noise** — random 6-digit integer |
| `useless_ratio` ⛔ | Float | **Noise** — random float [0, 1] |
| `accident_severity` | Categorical | **Target** — Minor / Moderate / Severe |

### Intentional Data Quality Issues

| Issue | Amount | Purpose |
|-------|--------|---------|
| Missing values | **10 %** per key column | Realistic null handling practice |
| Duplicate rows | **5 %** (~400 rows) | Deduplication practice |
| Categorical inconsistencies | **~15 %** of weather & state | Standardisation practice |
| Gaussian noise on numerics | σ = 5–12 per feature | Makes patterns less trivially learnable |
| 4 noise columns | Constant / random | Feature selection challenge |

### Class Balance

```
Minor     ≈ 2 667   (33.3 %)
Moderate  ≈ 2 666   (33.3 %)
Severe    ≈ 2 667   (33.3 %)
```

---

## 🧹 Notebook 2 — Preprocessing & Feature Selection

**File:** `02_preprocess_traffic_accidents.ipynb`

A six-cell pipeline that transforms the raw dataset into model-ready splits **and** identifies the optimal feature subset via GridSearchCV.

### Cell 1 — Data Inspection

- Load dataset and display shape, dtypes
- Null-value audit table (count + %)
- Descriptive statistics for all numerical columns
- Target class distribution bar chart
- Categorical cardinality check

### Cell 2 — Data Cleaning

| Step | Action | Why |
|------|--------|-----|
| **Remove duplicates** | `drop_duplicates()` | Prevents overfitting on repeated rows; avoids leakage across splits |
| **Standardise weather** | Map 24 variants → 6 canonical labels | `"Clear"`, `"CLEAR"`, `"Sunny"`, `"Fair"` are the same category |
| **Standardise state** | Map 20 variants → 7 two-letter codes | `"New York"`, `"N.Y."`, `"ny"` → `"NY"` |
| **Impute numerics** | Fill with **median** | Robust to outliers — unaffected by extreme values unlike mean |
| **Impute categoricals** | Fill with **mode** | Most frequent class is the safest neutral guess |

### Cell 3 — Outlier Management (IQR)

Tukey fences applied to 6 continuous columns:

```
Lower fence = Q1 − 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

Values outside the fence are **clipped** (Winsorised), not dropped — preserving row count while removing extreme influence.

Columns clipped: `speed_kmh`, `visibility_m`, `driver_age`, `driver_experience_yrs`, `num_casualties`, `num_vehicles`

### Cell 4 — Feature Engineering

| Action | Detail |
|--------|--------|
| Drop 7 noise/ID columns | `accident_id`, `redundant_record_id`, `system_flag`, `random_noise_code`, `useless_ratio`, `date`, `time` |
| New feature: `speed_excess_kmh` | `max(speed_kmh − speed_limit_kmh, 0)` — captures recklessness better than raw speed |
| New feature: `is_night` | `1` if hour ∈ {21–23, 0–5} — makes non-linear risk visible to linear models |
| Encode target | `Minor → 0`, `Moderate → 1`, `Severe → 2` |
| Label-encode 8 categoricals | `day_of_week`, `state`, `road_type`, `junction_type`, `weather_condition`, `road_condition`, `light_condition`, `vehicle_type` |

### Cell 5 — Feature Scaling

> ⚠️ **The scaler is fit on training data only, then applied to test data using training statistics — fitting on the full dataset leaks test-set information (data leakage).**

```
StandardScaler → mean = 0, std = 1
Applied to 9 continuous columns only
(Binary and label-encoded columns are left unchanged)
```

### Cell 6 — Feature Selection via GridSearchCV

Instead of applying a manual threshold, **GridSearchCV treats `k` (the number of features) as a hyperparameter** and finds the value that maximises recall via cross-validation.

#### How it works

```
Pipeline(
  Step 1 : SelectKBest(f_classif, k=?)   ← k is the search parameter
  Step 2 : LogisticRegression            ← fast scorer inside CV folds
)

GridSearchCV
  param_grid : selector__k ∈ {1, 2, 3, … 20}
  scoring    : 'recall_macro'
  cv         : StratifiedKFold(n_splits=5, shuffle=True)
  n_jobs     : -1  (parallel)
```

#### Why each design choice?

| Choice | Reasoning |
|--------|-----------|
| `SelectKBest(f_classif)` | ANOVA F-test ranks features by class-discriminability — efficient for multi-class problems |
| Search all k values | Only 20 features × 5 folds = 100 fits — negligible compute cost |
| `scoring='recall_macro'` | Weights all 3 severity classes equally — missing *Severe* is as costly as missing *Minor* |
| `StratifiedKFold(5)` | Preserves Minor/Moderate/Severe ratio in every fold — prevents a fold with no Severe samples |
| LR as CV scorer | Fast convergence; gives a clean recall signal without the training cost of SVM |

#### Outputs

- **Recall vs k line chart** — with ±1 std band showing where adding features stops helping
- **ANOVA F-score bar chart** — selected features highlighted, cutoff line shown
- Best `k` value and CV macro recall printed; selected feature list saved for Notebook 3

**Final output of Notebook 2:**

| File | Rows | Columns |
|------|------|---------|
| `train_preprocessed.csv` | 6 400 | 21 |
| `test_preprocessed.csv` | 1 600 | 21 |
| `selected_features` | — | best_k columns (passed to Notebook 3) |

---

## 🤖 Notebook 3 — Modelling & Evaluation

**File:** `03_modelling_gridsearchcv.ipynb`

Imports all libraries, loads `train_preprocessed.csv` / `test_preprocessed.csv`, applies the GridSearchCV-selected feature columns, and confirms class balance.

### Cells 3–6 — Four Models

#### Logistic Regression

```python
LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
```

- Models the log-odds of each class as a weighted sum of features
- Fast, interpretable baseline
- Coefficient plot shows which features push predictions toward each severity class

#### SVM — Linear Kernel

```python
SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
```

- Finds the **maximum-margin hyperplane** separating classes
- Best when classes are approximately linearly separable after feature selection
- Sensitive to outliers on the margin — hence the importance of IQR clipping in preprocessing

#### SVM — Polynomial Kernel

```python
SVC(kernel='poly', degree=3, coef0=1, C=1.0, decision_function_shape='ovr')
```

- Maps features into degree-3 space, enabling **curved decision boundaries**
- Useful when severity depends on *feature interactions* (e.g., high speed AND icy road AND night)
- `degree=3` is cubic — expressive but not prone to extreme overfitting

#### SVM — RBF Kernel

```python
SVC(kernel='rbf', gamma='scale', C=1.0, decision_function_shape='ovr')
```

- Gaussian kernel creates **arbitrarily complex, non-linear boundaries**
- `gamma='scale'` adapts automatically to feature variance
- Typically the strongest but least interpretable kernel

---

### Evaluation: Confusion Matrix + Recall

Every model produces two outputs:

**1. Dual Confusion Matrix**

```
Left panel  : Raw counts        Right panel : Normalised (row = recall)
               Predicted                         Predicted
           Minor  Mod  Severe               Minor  Mod  Severe
Actual Min [  TP   FP   FP  ]    Actual Min [ 0.xx  ...  ...  ]
       Mod [  FP   TP   FP  ]           Mod [ ...   0.xx ...  ]
       Sev [  FP   FP   TP  ]           Sev [ ...   ...  0.xx ]
```

**2. Per-Class Recall Bar Chart**

Shows recall for Minor, Moderate, and Severe with the macro recall line overlaid.

#### Why Recall over Accuracy?

> **Missing a *Severe* accident (false negative) carries far greater real-world cost than a false alarm.**
> A model that labels every accident as "Minor" would achieve ~33 % accuracy but 0 % recall on Severe — catastrophic in practice.
> **Recall on the Severe class is the single most important number to watch.**

---

### Cell 7 — Model Comparison Dashboard

| Visual | Description |
|--------|-------------|
| Results table | Colour-coded (green = best per column) with per-class and macro recall |
| Grouped bar chart | All 4 models × 3 classes side by side |
| Macro recall leaderboard | Horizontal bar, models ranked worst → best |
| Full recall heatmap | `YlGn` heatmap of model × metric matrix |
| Kernel summary table | When to prefer each model + trade-offs |

---

## ⚙️ Requirements

```txt
python       >= 3.9
pandas       >= 1.5
numpy        >= 1.23
scikit-learn >= 1.2
matplotlib   >= 3.6
seaborn      >= 0.12
jupyter      >= 1.0
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 🚀 How to Run

```bash
# 1. Clone or download the project folder
cd traffic-accident-severity/

# 2. Launch Jupyter
jupyter notebook

# 3. Run notebooks in order:
#    01_data_generation.ipynb              → produces traffic_accident_dataset.csv
#    02_preprocess_traffic_accidents.ipynb → produces train/test_preprocessed.csv + selected features
#    03_modelling_gridsearchcv.ipynb       → produces model results + plots

# Or run all at once from the command line:
jupyter nbconvert --to notebook --execute 01_data_generation.ipynb
jupyter nbconvert --to notebook --execute 02_preprocess_traffic_accidents.ipynb
jupyter nbconvert --to notebook --execute 03_modelling_gridsearchcv.ipynb
```

---

## 🔄 Full Pipeline at a Glance

```
01_data_generation.ipynb
  │
  │  8 000 rows · 26 columns · balanced 3-class target
  │  + 10 % nulls · 5 % duplicates · categorical inconsistencies · 4 noise cols
  ▼
traffic_accident_dataset.csv
  │
02_preprocess_traffic_accidents.ipynb
  │
  ├─ Cell 1 : Inspect        — dtypes · null audit · descriptive stats
  ├─ Cell 2 : Clean          — dedup · standardise categories · median/mode impute
  ├─ Cell 3 : Outliers       — IQR clip (Winsorise) on 6 columns
  ├─ Cell 4 : Engineer       — drop noise · speed_excess · is_night · encode
  ├─ Cell 5 : Scale          — train/test split · StandardScaler (train-fit only)
  └─ Cell 6 : Feature Select — Pipeline(SelectKBest → LR)
                               GridSearchCV: k ∈ {1 … 20}
                               scoring = recall_macro
                               cv = StratifiedKFold(5)
                               → best_k features identified
  │
  ▼
train_preprocessed.csv (6 400 × 21)   test_preprocessed.csv (1 600 × 21)
+ selected_features list (best_k columns)
  │
03_modelling_gridsearchcv.ipynb
  │
  ├─ Cell 1–2 : Setup & Load  — apply selected feature columns to train/test
  ├─ Cell 3   : Logistic Regression      → Confusion Matrix + Recall
  ├─ Cell 4   : SVM — Linear kernel      → Confusion Matrix + Recall
  ├─ Cell 5   : SVM — Polynomial kernel  → Confusion Matrix + Recall
  ├─ Cell 6   : SVM — RBF kernel         → Confusion Matrix + Recall
  │
  └─ Cell 7   : Comparison Dashboard
                 ├─ Results table (colour-coded)
                 ├─ Grouped bar chart
                 ├─ Macro recall leaderboard
                 └─ Recall heatmap
```

---

## 📊 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Synthetic data with intentional noise | Allows controlled study of preprocessing impact |
| Median imputation for numerics | Robust to the outliers identified in Cell 3 |
| Clip outliers, don't drop | Preserves sample size — important for SVM margin estimation |
| GridSearchCV for feature selection | Removes arbitrary thresholds; `k` is validated directly against recall |
| `recall_macro` as CV scoring | Equal penalty for missing any severity class |
| Fit scaler on train only | Prevents data leakage from test statistics into training |
| Same feature set for all 4 models | Ensures a fair apples-to-apples model comparison |
| Recall as primary metric | False negatives on Severe accidents carry the highest real-world cost |

---

*Built with Python · Pandas · Scikit-Learn · Matplotlib · Seaborn*
