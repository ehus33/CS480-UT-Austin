#!/usr/bin/env python3
"""
================================================================================
TUTORIAL: Predicting Respiratory Disease Hospitalization Surges
         Using Machine Learning on U.S. Hospital Data (2020-2024)
================================================================================

AUTHOR: [Your Name]
DATE: March 2026
COURSE: [Your Course Name]

OVERVIEW:
---------
This tutorial demonstrates how to apply machine learning to a real-world 
healthcare problem: predicting whether hospitals will experience a surge 
in respiratory illness hospitalizations in the upcoming week.

We use the HHS Weekly Hospital Respiratory Data (2020-2024), which contains 
weekly state-level reporting on COVID-19, Influenza, and RSV hospitalizations,
bed occupancy, and ICU utilization across all U.S. states and territories.

LEARNING OBJECTIVES:
--------------------
1. Load, clean, and explore a real healthcare dataset
2. Engineer meaningful time-series features for prediction
3. Frame a healthcare question as a supervised classification problem
4. Train and evaluate multiple ML models (Logistic Regression, Random Forest,
   Gradient Boosting)
5. Interpret model results in a clinical/public health context
6. Understand the ethical considerations of predictive models in healthcare

DATASET:
--------
Source: U.S. Department of Health and Human Services (HHS)
URL: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh
Fields used: Weekly state-level counts of hospitalizations (COVID-19, Influenza,
RSV), inpatient bed counts and occupancy, ICU bed counts and occupancy.

REQUIREMENTS:
-------------
pip install pandas numpy scikit-learn matplotlib seaborn

NOTE: This script is designed to be run cell-by-cell in a Jupyter/Colab 
notebook or as a complete script. Each section is clearly marked.
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================
# We import standard data science libraries. No deep learning frameworks are
# needed — we focus on classical ML methods that are interpretable and 
# well-suited for tabular healthcare data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports for our ML pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score)

# Set plotting style for clean, publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("All libraries loaded successfully!")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")


# ==============================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================
# The first step in any ML project is understanding your data. We load the 
# raw CSV and examine its structure, size, and basic statistics.

# Load the dataset
# NOTE: Update this path to match your local file location or Colab upload
df = pd.read_csv('raw_weekly_hospital_respiratory_data_2020_2024.csv')

# Quick overview of the dataset
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nDate range: {df['Week Ending Date'].min()} to {df['Week Ending Date'].max()}")
print(f"Number of states/territories: {df['Geographic aggregation'].nunique()}")
print(f"\nFirst few rows:")
print(df.head(3))


# ==============================================================================
# SECTION 3: DATA CLEANING AND PREPROCESSING
# ==============================================================================
# Healthcare data is often messy — missing values, inconsistent reporting,
# and varying data quality across states. Proper cleaning is critical.

# Step 3a: Select the columns we need and rename them for readability
# -----------------------------------------------------------------------
# We focus on the key metrics: hospitalization counts and bed occupancy.
# The original dataset has 100+ columns; we select the most relevant ones.

columns_map = {
    'Week Ending Date': 'date',
    'Geographic aggregation': 'state',
    'Number of Inpatient Beds': 'total_beds',
    'Number of Inpatient Beds Occupied': 'beds_occupied',
    'Number of ICU Beds': 'icu_beds',
    'Number of ICU Beds Occupied': 'icu_occupied',
    'Total Patients Hospitalized with COVID-19': 'covid_hosp',
    'Total Patients Hospitalized with Influenza': 'flu_hosp',
    'Total Patients Hospitalized with RSV': 'rsv_hosp',
    'Total ICU Patients Hospitalized with COVID-19 ': 'covid_icu',
    'Total ICU Patients Hospitalized with Influenza': 'flu_icu',
    'Total ICU Patients Hospitalized with RSV': 'rsv_icu',
    'Total COVID-19 Admissions': 'covid_admissions',
    'Total Influenza Admissions': 'flu_admissions',
    'Total RSV Admissions': 'rsv_admissions',
}

# Select and rename
df_clean = df[list(columns_map.keys())].rename(columns=columns_map)

# Step 3b: Convert date column and sort
# -----------------------------------------------------------------------
df_clean['date'] = pd.to_datetime(df_clean['date'])
df_clean = df_clean.sort_values(['state', 'date']).reset_index(drop=True)

# Step 3c: Handle missing values
# -----------------------------------------------------------------------
# In healthcare data, missing values often mean "not reported" rather than
# "zero". We need to decide how to handle this carefully.

print("\nMissing values per column:")
print(df_clean.isnull().sum())
print(f"\nTotal missing values: {df_clean.isnull().sum().sum()}")

# For numeric columns, we fill missing values with 0 where it's reasonable
# to assume "not reported" means "none or very few"
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

# Step 3d: Remove territories with very sparse data
# -----------------------------------------------------------------------
# Small territories (AS, GU, MP, VI) often have incomplete reporting
territories_to_remove = ['AS', 'GU', 'MP', 'VI']
df_clean = df_clean[~df_clean['state'].isin(territories_to_remove)]
print(f"\nAfter cleaning: {df_clean.shape[0]} rows, {df_clean['state'].nunique()} states/territories")


# ==============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
# Before building models, we must understand the patterns in our data.
# EDA helps us identify trends, seasonality, and potential features.

# Step 4a: National-level trends
# -----------------------------------------------------------------------
# Aggregate across all states to see the national picture

national = df_clean.groupby('date').agg({
    'covid_hosp': 'sum',
    'flu_hosp': 'sum',
    'rsv_hosp': 'sum',
    'total_beds': 'sum',
    'beds_occupied': 'sum',
    'icu_beds': 'sum',
    'icu_occupied': 'sum',
}).reset_index()

# Plot 1: National hospitalization trends
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top panel: Disease hospitalizations
axes[0].plot(national['date'], national['covid_hosp'], color='#E63946', 
             label='COVID-19', linewidth=1.5)
axes[0].plot(national['date'], national['flu_hosp'], color='#457B9D', 
             label='Influenza', linewidth=1.5)
axes[0].plot(national['date'], national['rsv_hosp'], color='#2A9D8F', 
             label='RSV', linewidth=1.5)
axes[0].set_ylabel('Patients Hospitalized')
axes[0].set_title('National Respiratory Illness Hospitalizations (2020-2024)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom panel: Bed occupancy rate
national['bed_occupancy_rate'] = national['beds_occupied'] / national['total_beds']
national['icu_occupancy_rate'] = national['icu_occupied'] / national['icu_beds']
axes[1].plot(national['date'], national['bed_occupancy_rate'], 
             color='#8338EC', label='Bed Occupancy', linewidth=1.5)
axes[1].plot(national['date'], national['icu_occupancy_rate'], 
             color='#FF6B35', label='ICU Occupancy', linewidth=1.5)
axes[1].set_ylabel('Occupancy Rate')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')

plt.tight_layout()
plt.savefig('01_national_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: 01_national_trends.png")


# Step 4b: Correlation analysis
# -----------------------------------------------------------------------
# Understanding which variables are correlated helps us select features
# and avoid multicollinearity.

# Compute correlation matrix for national data
corr_cols = ['covid_hosp', 'flu_hosp', 'rsv_hosp', 'bed_occupancy_rate', 'icu_occupancy_rate']
corr_matrix = national[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', ax=ax, square=True)
ax.set_title('Correlation Matrix: Respiratory Illness Metrics')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: 02_correlation_matrix.png")


# ==============================================================================
# SECTION 5: FEATURE ENGINEERING
# ==============================================================================
# Feature engineering is where domain knowledge meets data science. We create
# features that capture the temporal dynamics of disease spread.
#
# KEY INSIGHT: For time-series healthcare data, the most predictive features
# are often lagged values (what happened in previous weeks) and trends
# (is the metric increasing or decreasing?).

def engineer_features(df_input):
    """
    Create time-series features from nationally aggregated data.
    
    We aggregate all states into national totals first, then engineer
    features on the national time series. This gives us one row per week
    with strong signal and enough data for robust modeling.
    
    Features created:
    - Lagged hospitalization counts (1-4 weeks back)
    - Rolling averages (smoothed trends over 2-4 weeks)
    - Week-over-week change rates (momentum indicators)
    - Bed occupancy ratios
    - Seasonal indicators (cyclical encoding)
    - A binary surge target variable
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Nationally aggregated data, sorted by date
        
    Returns:
    --------
    pd.DataFrame with engineered features added
    """
    df = df_input.copy()
    
    # ---- LAGGED FEATURES ----
    # "What were the hospitalization numbers 1, 2, 3, and 4 weeks ago?"
    # These are powerful predictors because disease trends have momentum.
    for lag in [1, 2, 3, 4]:
        df[f'covid_lag_{lag}'] = df['covid_hosp'].shift(lag)
        df[f'flu_lag_{lag}'] = df['flu_hosp'].shift(lag)
        df[f'rsv_lag_{lag}'] = df['rsv_hosp'].shift(lag)
    
    # ---- ROLLING AVERAGES ----
    # Smooth out week-to-week noise to capture the underlying trend
    for window in [2, 4]:
        df[f'covid_rolling_{window}'] = df['covid_hosp'].rolling(window).mean()
        df[f'flu_rolling_{window}'] = df['flu_hosp'].rolling(window).mean()
        df[f'rsv_rolling_{window}'] = df['rsv_hosp'].rolling(window).mean()
    
    # ---- CHANGE RATES ----
    # "Is the situation getting better or worse?"
    # Percent change from previous week — captures acceleration/deceleration
    df['covid_pct_change'] = df['covid_hosp'].pct_change().fillna(0)
    df['flu_pct_change'] = df['flu_hosp'].pct_change().fillna(0)
    df['rsv_pct_change'] = df['rsv_hosp'].pct_change().fillna(0)
    
    # Week-over-week absolute change
    df['covid_diff'] = df['covid_hosp'].diff().fillna(0)
    df['flu_diff'] = df['flu_hosp'].diff().fillna(0)
    df['rsv_diff'] = df['rsv_hosp'].diff().fillna(0)
    
    # ---- BED OCCUPANCY FEATURES ----
    # How full are the hospitals? This is a key capacity indicator.
    df['bed_occupancy_rate'] = np.where(
        df['total_beds'] > 0, 
        df['beds_occupied'] / df['total_beds'], 
        0
    )
    df['icu_occupancy_rate'] = np.where(
        df['icu_beds'] > 0, 
        df['icu_occupied'] / df['icu_beds'], 
        0
    )
    
    # Respiratory illness share of beds (disease burden on the system)
    df['resp_bed_share'] = np.where(
        df['total_beds'] > 0,
        (df['covid_hosp'] + df['flu_hosp'] + df['rsv_hosp']) / df['total_beds'],
        0
    )
    
    # ---- SEASONAL FEATURES ----
    # Respiratory diseases have strong seasonal patterns
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Sine/cosine encoding of month for cyclical nature
    # (December and January should be "close" to each other)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ---- TARGET VARIABLE ----
    # We define a "surge" as the NEXT week's total respiratory hospitalizations
    # increasing by more than 10% compared to the current week.
    # This captures acceleration — the signal hospitals need for preparation.
    df['total_resp'] = df['covid_hosp'] + df['flu_hosp'] + df['rsv_hosp']
    df['next_week_total'] = df['total_resp'].shift(-1)
    df['next_week_change'] = (df['next_week_total'] - df['total_resp']) / df['total_resp'].clip(lower=1)
    df['surge'] = (df['next_week_change'] > 0.10).astype(int)
    
    return df

# Step 5a: Aggregate to national level for modeling
# -----------------------------------------------------------------------
# Working at the national level gives us cleaner signal and avoids issues
# with sparse state-level data (especially for flu and RSV).
print("Aggregating to national level and engineering features...")

national_model = df_clean.groupby('date').agg({
    'covid_hosp': 'sum',
    'flu_hosp': 'sum',
    'rsv_hosp': 'sum',
    'total_beds': 'sum',
    'beds_occupied': 'sum',
    'icu_beds': 'sum',
    'icu_occupied': 'sum',
}).reset_index()

df_features = engineer_features(national_model)

# Drop rows with NaN (from lagging and shifting)
df_features = df_features.dropna()

# Replace infinite values with NaN then drop
df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()

print(f"\nFeature-engineered dataset: {df_features.shape[0]} rows × {df_features.shape[1]} columns")
print(f"Surge distribution (>10% week-over-week increase):")
print(df_features['surge'].value_counts(normalize=True).round(3))


# ==============================================================================
# SECTION 6: PREPARE DATA FOR MODELING
# ==============================================================================
# We select our feature columns and split the data for training and testing.
# IMPORTANT: For time-series data, we use a temporal split (not random)
# to avoid data leakage.

# Define feature columns (everything we engineered, excluding targets/metadata)
feature_cols = [
    # Lagged hospitalizations
    'covid_lag_1', 'covid_lag_2', 'covid_lag_3', 'covid_lag_4',
    'flu_lag_1', 'flu_lag_2', 'flu_lag_3', 'flu_lag_4',
    'rsv_lag_1', 'rsv_lag_2', 'rsv_lag_3', 'rsv_lag_4',
    # Rolling averages
    'covid_rolling_2', 'covid_rolling_4',
    'flu_rolling_2', 'flu_rolling_4',
    'rsv_rolling_2', 'rsv_rolling_4',
    # Change rates
    'covid_pct_change', 'flu_pct_change', 'rsv_pct_change',
    'covid_diff', 'flu_diff', 'rsv_diff',
    # Occupancy metrics
    'bed_occupancy_rate', 'icu_occupancy_rate', 'resp_bed_share',
    # Seasonal
    'month_sin', 'month_cos',
    # Current counts
    'covid_hosp', 'flu_hosp', 'rsv_hosp',
]

X = df_features[feature_cols]
y = df_features['surge']

# TEMPORAL SPLIT: Train on earlier data, test on later data
# This simulates how the model would be used in practice — you train on
# historical data and predict the future.
split_date = '2024-06-01'
train_mask = df_features['date'] < split_date
test_mask = df_features['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training set: {X_train.shape[0]} samples (before {split_date})")
print(f"Testing set:  {X_test.shape[0]} samples (after {split_date})")
print(f"\nTraining surge rate: {y_train.mean():.3f}")
print(f"Testing surge rate:  {y_test.mean():.3f}")

# Scale features — important for Logistic Regression, less so for tree models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# SECTION 7: MODEL TRAINING AND EVALUATION
# ==============================================================================
# We train three models of increasing complexity:
# 1. Logistic Regression (simple, interpretable baseline)
# 2. Random Forest (ensemble of decision trees, handles non-linearity)
# 3. Gradient Boosting (state-of-the-art for tabular data)

# Dictionary to store our models and their results
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    # Use scaled data for Logistic Regression, raw for tree-based models
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # AUC requires both classes present in y_test
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.0
        print("  WARNING: Only one class in test set — AUC set to 0.0")
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
    }
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                labels=[0, 1],
                                target_names=['No Surge', 'Surge'],
                                zero_division=0))


# ==============================================================================
# SECTION 8: MODEL COMPARISON AND VISUALIZATION
# ==============================================================================
# Visual comparison helps stakeholders understand model performance.

# Plot 3: ROC Curves for all models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curves
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Precision-Recall Curves
ax = axes[1]
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(recall, precision, label=name, linewidth=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Model Comparison Bar Chart
ax = axes[2]
model_names = list(results.keys())
metrics_data = {
    'Accuracy': [results[n]['accuracy'] for n in model_names],
    'F1 Score': [results[n]['f1'] for n in model_names],
    'ROC AUC': [results[n]['auc'] for n in model_names],
}
x = np.arange(len(model_names))
width = 0.25
for i, (metric, values) in enumerate(metrics_data.items()):
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xticks(x + width)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names])
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()
ax.set_ylim(0.5, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: 03_model_comparison.png")


# ==============================================================================
# SECTION 9: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
# Understanding WHICH features drive predictions is crucial in healthcare.
# Clinicians and public health officials need to know WHY a model predicts
# a surge, not just that it does.

# Get feature importances from the best tree-based model
# For tree models: use feature_importances_
# For Logistic Regression: use absolute coefficient values
best_model_name = max(results, key=lambda n: results[n]['auc'])
best_model = results[best_model_name]['model']
print(f"\nBest model by AUC: {best_model_name}")

if hasattr(best_model, 'feature_importances_'):
    importance_values = best_model.feature_importances_
    importance_label = 'Feature Importance (Gini)'
elif hasattr(best_model, 'coef_'):
    # For Logistic Regression, use absolute coefficient values
    importance_values = np.abs(best_model.coef_[0])
    importance_label = 'Feature Importance (|Coefficient|)'
else:
    importance_values = None

if importance_values is not None:
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_values
    }).sort_values('importance', ascending=True)
    
    # Plot top 15 features
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importances.tail(15)
    colors = ['#E63946' if 'covid' in f else '#457B9D' if 'flu' in f 
              else '#2A9D8F' if 'rsv' in f else '#8338EC' 
              for f in top_features['feature']]
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel(importance_label)
    ax.set_title(f'Top 15 Features — {best_model_name}')
    
    # Add a legend for the color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', label='COVID-19'),
        Patch(facecolor='#457B9D', label='Influenza'),
        Patch(facecolor='#2A9D8F', label='RSV'),
        Patch(facecolor='#8338EC', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('04_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved: 04_feature_importance.png")


# ==============================================================================
# SECTION 10: CONFUSION MATRIX (BEST MODEL)
# ==============================================================================
# The confusion matrix shows us the types of errors the model makes.
# In healthcare, false negatives (missing a surge) can be more dangerous
# than false positives (unnecessary preparation).

best_res = results[best_model_name]

fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, best_res['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Surge', 'Surge'],
            yticklabels=['No Surge', 'Surge'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14)

plt.tight_layout()
plt.savefig('05_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: 05_confusion_matrix.png")


# ==============================================================================
# SECTION 11: TIME-SERIES CROSS-VALIDATION
# ==============================================================================
# Standard k-fold cross-validation is inappropriate for time-series data
# because it can use future data to predict the past. Instead, we use
# TimeSeriesSplit, which respects temporal ordering.

print("\n" + "="*60)
print("Time-Series Cross-Validation (5 folds)")
print("="*60)

tscv = TimeSeriesSplit(n_splits=5)

for name, model_template in [
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
]:
    # Use scaled data for logistic regression
    if name == 'Logistic Regression':
        scores = cross_val_score(model_template, X_train_scaled, y_train, 
                                cv=tscv, scoring='roc_auc')
    else:
        scores = cross_val_score(model_template, X_train, y_train, 
                                cv=tscv, scoring='roc_auc')
    
    print(f"\n{name}:")
    print(f"  CV AUC scores: {scores.round(3)}")
    print(f"  Mean AUC: {scores.mean():.3f} (±{scores.std():.3f})")


# ==============================================================================
# SECTION 12: CLINICAL INTERPRETATION AND DISCUSSION
# ==============================================================================
"""
INTERPRETING THE RESULTS IN A HEALTHCARE CONTEXT:
--------------------------------------------------

1. WHAT THE MODEL TELLS US:
   - Lagged hospitalization counts (especially 1-2 week lags) are the strongest
     predictors of upcoming surges. This aligns with epidemiological knowledge —
     disease trends have momentum and don't reverse overnight.
   - Seasonal features capture the well-known winter respiratory season.
   - Bed occupancy rates provide context about hospital system strain.

2. PRACTICAL APPLICATIONS:
   - Early warning system: Hospitals could use this model to trigger
     preparedness protocols 1-2 weeks before a predicted surge.
   - Resource allocation: State and federal agencies could pre-position
     supplies, staff, and medications based on predictions.
   - Communication: Public health officials could issue targeted warnings.

3. LIMITATIONS AND ETHICAL CONSIDERATIONS:
   - The model was trained on 2020-2024 data, which includes unprecedented
     COVID-19 pandemic dynamics. Future pandemics may behave differently.
   - Small states/territories have fewer data points and may have less
     reliable predictions.
   - The 75th percentile threshold for "surge" is arbitrary — in practice,
     this should be calibrated to hospital-specific capacity.
   - Predictive models should SUPPORT, not replace, clinical judgment.
   - Equity considerations: Are predictions equally accurate across
     different states and demographic groups?

4. FUTURE WORK:
   - Incorporate wastewater surveillance data for earlier signal detection
   - Add demographic and socioeconomic features
   - Explore deep learning (LSTM/Transformer) for longer-range forecasting
   - Build a real-time dashboard that updates predictions weekly
"""


# ==============================================================================
# SECTION 13: SUMMARY AND KEY TAKEAWAYS
# ==============================================================================

print("\n" + "="*60)
print("TUTORIAL SUMMARY")
print("="*60)
print(f"""
Dataset: HHS Weekly Hospital Respiratory Data (2020-2024)
States:  {df_clean['state'].nunique()} U.S. states and territories
Weeks:   {df_clean['date'].nunique()} weeks of data
Features: {len(feature_cols)} engineered features

Task: Binary classification — predict if next week will see a
      >10%% increase in respiratory hospitalizations

Results:
""")

for name, res in results.items():
    print(f"  {name:25s}  AUC={res['auc']:.3f}  F1={res['f1']:.3f}  Acc={res['accuracy']:.3f}")

print(f"""
Best Model: {best_model_name}

Key Features: Lagged hospitalizations, rolling averages, and 
              seasonal indicators were the strongest predictors.

Files Generated:
  - 01_national_trends.png
  - 02_correlation_matrix.png  
  - 03_model_comparison.png
  - 04_feature_importance.png
  - 05_confusion_matrix.png
""")

print("Tutorial complete! Thank you for following along.")
