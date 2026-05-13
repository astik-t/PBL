# 🚦 Traffic Accident Severity Prediction

> A complete end-to-end machine learning pipeline — from synthetic data generation, preprocessing & GridSearchCV-driven feature selection, through to multi-model evaluation using Logistic Regression, SVM, and tree-based baselines, with Random Forest selected as the best model in Phase 3.

---

## 📁 Project Structure

```
PBL/
│
├── Phase_1/
│   ├── 01_data_generation.ipynb              ← Synthetic dataset creation
│   ├── 02_preprocess_traffic_accidents.ipynb ← Preprocessing + feature engineering
│   ├── traffic_accident_dataset.csv          ← Raw synthetic dataset (~8 400 rows)
│   ├── train_preprocessed.csv                ← Clean training split (6 400 rows)
│   ├── test_preprocessed.csv                 ← Clean test split   (1 600 rows)
│   ├── preprocessed.csv                      ← Full cleaned dataset
│   ├── boxplots_after_clipping.png           ← Outlier clipping summary plot
│   └── feature_correlation_heatmap.png       ← Feature correlation heatmap
│
├── Phase_2/
│   ├── 03_model_training.ipynb               ← Train models and find evaluation metrics
│   └── 03_model_training.py                  ← Script export of Phase 2 training
│
├── Phase_3/
│   ├── 04_evaluate_models.ipynb              ← Full pipeline + evaluation
│   ├── 04_evaluate_models.py                 ← Python export of the notebook
│   ├── predict.py                            ← CLI prediction helper
│   ├── model_lr.pkl
│   ├── model_svm_linear.pkl
│   ├── model_svm_poly.pkl
│   ├── model_svm_rbf.pkl
│   ├── model_rf.pkl
│   └── random_forest_model.pkl
│
├── requirements.txt
└── README.md                                 ← You are here
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

**File:** `Phase_1/01_data_generation.ipynb`

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

**File:** `Phase_1/02_preprocess_traffic_accidents.ipynb`

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
- Best `k` value and CV macro recall printed; selected feature list used in Phase 2/Phase 3 modelling

**Final output of Notebook 2:**

| File | Rows | Columns |
|------|------|---------|
| `train_preprocessed.csv` | 6 400 | 21 |
| `test_preprocessed.csv` | 1 600 | 21 |
| `selected_features` | — | best_k columns (used in Phase 2/Phase 3) |

---

## 🤖 Phase 2 — Model Training & Evaluation

**Files:** `Phase_2/03_model_training.ipynb`, `Phase_2/03_model_training.py`

Loads `train_preprocessed.csv` / `test_preprocessed.csv`, runs GridSearchCV to pick the best `k`, then trains Logistic Regression and SVM (linear, polynomial, RBF). It prints classification reports (accuracy, precision, recall, F1), confusion matrices, and per-class + macro recall, then compares models by macro recall.

### Cells 3–6 — Four Models

#### Logistic Regression

```python
LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
```

- Models the log-odds of each class as a weighted sum of features
- Fast, interpretable baseline
- Coefficients can be inspected to understand feature influence per class

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

Every model prints a classification report and shows a confusion matrix heatmap. Per-class recall and macro recall are also reported to keep focus on severe cases.

#### Why Recall over Accuracy?

> **Missing a *Severe* accident (false negative) carries far greater real-world cost than a false alarm.**
> A model that labels every accident as "Minor" would achieve ~33 % accuracy but 0 % recall on Severe — catastrophic in practice.
> **Recall on the Severe class is the single most important number to watch.**

---

### Model Comparison

Outputs include a results table with per-class recall and macro recall, plus a best-model summary based on macro recall. Phase 3 extends this comparison and selects Random Forest as the best model.

## 🧪 Phase 3 — Full Evaluation Notebook

**Files:** `Phase_3/04_evaluate_models.ipynb`, `Phase_3/04_evaluate_models.py`

Runs the full preprocessing pipeline (inspection, cleaning, outlier clipping, feature engineering, scaling), performs GridSearchCV feature selection, and evaluates a wider set of models. The notebook adds tree-based baselines (Random Forest, Decision Tree) and selects the best model based on macro recall. In this project, Random Forest is the best performer.

**Prediction CLI:** `Phase_3/predict.py` loads saved model files and expects `selected_features.json` generated by the training steps for interactive predictions (single model or majority vote).

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

Install dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter (for notebooks)
jupyter notebook

# 3. Run notebooks in order (Phase_1):
#    Phase_1/01_data_generation.ipynb              → produces traffic_accident_dataset.csv
#    Phase_1/02_preprocess_traffic_accidents.ipynb → produces train/test_preprocessed.csv

# 4. Model evaluation options:
#    Phase_2/03_model_training.ipynb               → Train models and find evaluation metrics
#    Phase_2/03_model_training.py                  → Script export of Phase 2 training
#    Phase_3/04_evaluate_models.ipynb              → Full pipeline + evaluation
#    Phase_3/04_evaluate_models.py                 → Python export of the notebook

# 5. Prediction CLI (from Phase_3):
#    python predict.py
```

Note: Phase_2 and Phase_3 scripts/notebooks read CSVs using relative paths. Run them with the working directory set to Phase_1, or update the paths in the scripts to point at Phase_1/.

---

## 🔄 Full Pipeline at a Glance

```
Phase_1/01_data_generation.ipynb
  │
  │  8 000 rows · 26 columns · balanced 3-class target
  │  + 10 % nulls · 5 % duplicates · categorical inconsistencies · 4 noise cols
  ▼
Phase_1/traffic_accident_dataset.csv
  │
Phase_1/02_preprocess_traffic_accidents.ipynb
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
Phase_1/train_preprocessed.csv (6 400 × 21)   Phase_1/test_preprocessed.csv (1 600 × 21)
+ selected_features list (best_k columns)
  │
Phase_2/03_model_training.ipynb
  │
  ├─ Cell 1–2 : Setup & Load  — apply selected feature columns to train/test
  ├─ Cell 3   : Logistic Regression      → Confusion Matrix + Recall
  ├─ Cell 4   : SVM — Linear kernel      → Confusion Matrix + Recall
  ├─ Cell 5   : SVM — Polynomial kernel  → Confusion Matrix + Recall
  ├─ Cell 6   : SVM — RBF kernel         → Confusion Matrix + Recall
  │
  └─ Cell 7   : Model comparison table + best macro recall
  │
Phase_3/04_evaluate_models.ipynb
  │
  ├─ Full preprocessing + feature selection
  ├─ Extra models: Random Forest, Decision Tree
  └─ Best model: Random Forest (macro recall)
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
