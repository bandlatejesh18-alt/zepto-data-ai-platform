# Analytics Module

## Overview

The Analytics module performs **Exploratory Data Analysis (EDA)** on the Titanic dataset to understand the factors influencing passenger survival. The workflow includes dataset profiling, data cleaning, statistical analysis, visualization, correlation analysis, multivariate exploration, and feature standardization.

This module serves as the analytical foundation for the machine learning pipeline implemented in Part B.

---

# Objectives

- Load and profile the Titanic dataset.
- Handle missing values using predefined threshold-based rules.
- Perform univariate, bivariate, and multivariate analysis.
- Identify outliers using the IQR method.
- Analyze feature relationships through correlation analysis.
- Standardize numerical features as an exploratory preprocessing step.
- Generate visualizations to explain the data and support observations.

---

# Dataset

| Property | Value |
|----------|-------|
| Dataset | Titanic |
| Source | Seaborn (`sns.load_dataset("titanic")`) |
| Original Shape | **891 Rows × 15 Columns** |

The dataset is loaded only once using the Seaborn library and immediately saved locally as:

```text
analytics/data/raw/titanic.csv
```

This offline copy ensures that the project remains reproducible even if the Seaborn dataset is unavailable during grading.

---

# Dataset Profiling

Immediately after loading, the dataset was profiled using:

- `df.shape`
- `df.info()`
- `df.describe()`

The profiling step provides:

- Dataset dimensions
- Feature data types
- Missing values
- Statistical summary of numerical features

---

# Missing Value Analysis

The percentage of missing values was computed for every column containing missing data.

| Column | Missing Values | Missing Percentage | Strategy |
|---------|---------------:|-------------------:|----------|
| deck | 688 | 77.22% | Drop Column |
| age | 177 | 19.87% | Median Imputation |
| embarked | 2 | 0.22% | Drop Rows |
| embark_town | 2 | 0.22% | Drop Rows |

---

## Missing Value Handling Decisions

### deck (77.22%)

The `deck` column contains **77.22%** missing values, making imputation unreliable. Since more than three-quarters of the observations are unavailable, retaining this feature would introduce substantial uncertainty into the analysis. Therefore, the column was removed from the cleaned dataset.

### age (19.87%)

The `age` column contains **19.87%** missing values, which falls within the assignment's **5%–30%** threshold. Median imputation was selected because the median is less sensitive to extreme values than the mean and provides a robust estimate for numerical data.

### embarked (0.22%)

Only **0.22%** of the observations are missing. Since the missing percentage is below **5%**, the affected rows were removed without significantly impacting the dataset.

### embark_town (0.22%)

The `embark_town` column also contains only **0.22%** missing values. Removing these rows has a negligible effect on the dataset while satisfying the assignment's threshold rule.

---

# Cleaned Dataset

After applying all preprocessing steps:

| Property | Value |
|----------|-------|
| Rows | **889** |
| Columns | **14** |

Cleaning operations performed:

- Removed the `deck` column.
- Imputed missing values in `age` using the median.
- Removed rows containing missing values in `embarked`.
- Removed rows containing missing values in `embark_town`.

The cleaned dataset is stored as:

```text
analytics/data/processed/titanic_cleaned.csv
```

---

# Univariate Analysis

Histograms and box plots were generated for the following numerical features:

- Age
- Fare

---

## Outlier Detection (IQR Rule)

Outliers were identified using the Interquartile Range (IQR) rule.

```text
Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

### Outlier Summary

| Feature | Outliers |
|----------|----------:|
| Age | **65** |
| Fare | **114** |

---

### Age Distribution

<p align="center">
<img src="outputs/plots/age_histogram.png" width="700">
</p>

The histogram shows that most passengers were between 20 and 40 years of age. The distribution is moderately right-skewed due to a smaller number of elderly passengers. The accompanying box plot confirms the presence of several age-related outliers identified using the IQR rule.

---

### Fare Distribution

<p align="center">
<img src="outputs/plots/fare_histogram.png" width="700">
</p>

The fare distribution is highly right-skewed, with most passengers paying relatively low ticket prices while a small number paid exceptionally high fares. These expensive tickets increase the mean substantially compared to the median. The box plot also highlights several high-fare outliers.

---

## Fare Statistics

| Statistic | Value |
|-----------|------:|
| Mean | **32.10** |
| Median | **14.45** |
| Mode | **8.05** |

### Distribution Interpretation

Since:

```text
Mean > Median > Mode
```

the fare distribution is **right-skewed**. A small number of extremely expensive tickets pull the mean upward while the majority of passengers paid considerably lower fares.

---

# Bivariate Analysis

Boolean masking using `&` operators was used to compute survival rates for different passenger groups.

---

## Survival Rate by Sex

| Sex | Survival Rate |
|------|--------------:|
| Male | **18.89%** |
| Female | **74.04%** |

<p align="center">
<img src="outputs/plots/survival_by_sex.png" width="700">
</p>

Female passengers experienced a substantially higher survival rate than male passengers. This indicates that gender was one of the strongest factors associated with survival on the Titanic. The results support the historical observation that women were generally given higher priority during evacuation, leading to significantly better survival outcomes.

---

## Survival Rate by Passenger Class

| Passenger Class | Survival Rate |
|-----------------|--------------:|
| First Class | **62.62%** |
| Second Class | **47.28%** |
| Third Class | **24.24%** |

<p align="center">
<img src="outputs/plots/survival_by_pclass.png" width="700">
</p>

Passengers travelling in first class had the highest survival rate, while passengers in third class had the lowest. This suggests that passenger class had a significant influence on survival probability. Better cabin locations, easier access to lifeboats, and socioeconomic advantages may have contributed to the higher survival rates of first-class passengers.

---

## Survival Rate by Sex and Passenger Class

| Sex | Passenger Class | Survival Rate |
|------|-----------------|--------------:|
| Female | First | **96.74%** |
| Female | Second | **92.11%** |
| Female | Third | **50.00%** |
| Male | First | **36.89%** |
| Male | Second | **15.74%** |
| Male | Third | **13.54%** |

<p align="center">
<img src="outputs/plots/survival_by_sex_pclass.png" width="700">
</p>

Combining gender and passenger class reveals an even clearer survival pattern. Women travelling in first and second class experienced exceptionally high survival rates, whereas men travelling in third class experienced the lowest survival rates. This suggests that survival was influenced by the combined effects of both gender and passenger class rather than either factor alone.

---

# Correlation Analysis

A correlation matrix was computed using only the following numerical features:

- survived
- pclass
- age
- sibsp
- parch
- fare

The boolean columns **adult_male** and **alone** were intentionally excluded because they are derived attributes rather than independent measured features.

<p align="center">
<img src="outputs/plots/correlation_heatmap.png" width="700">
</p>

---

## Strongest Correlations

### 1. Passenger Class vs Fare

**Correlation Coefficient:** **-0.548**

Passenger class and fare exhibit the strongest negative correlation in the dataset. Passengers travelling in first class generally paid significantly higher fares than passengers travelling in lower classes. Since passenger class is numerically encoded as 1 (First Class), 2 (Second Class), and 3 (Third Class), higher class numbers correspond to lower ticket prices, resulting in a negative correlation.

---

### 2. SibSp vs Parch

**Correlation Coefficient:** **0.415**

The number of siblings/spouses (`sibsp`) is moderately positively correlated with the number of parents/children (`parch`) travelling together. Families often travelled as groups, making it common for passengers to have both sibling/spouse and parent/child relationships onboard. Consequently, increases in one family-related feature are often accompanied by increases in the other.

---

# Multivariate Data Story

To better understand the factors influencing passenger survival, four multivariate visualizations were created. Together, these charts demonstrate how gender, passenger class, age, and fare contributed to survival outcomes.

---

## Chart 1 — Survival by Sex

<p align="center">
<img src="outputs/plots/multivariate_chart_1.png" width="700">
</p>

Female passengers experienced substantially higher survival rates than male passengers. This suggests that gender was one of the most influential factors affecting survival during the disaster. The visualization supports the historical evacuation policy that prioritized women, resulting in significantly higher survival probabilities.

---

## Chart 2 — Survival by Passenger Class

<p align="center">
<img src="outputs/plots/multivariate_chart_2.png" width="700">
</p>

Passengers travelling in first class were considerably more likely to survive than passengers travelling in third class. This indicates that socioeconomic status and cabin location likely influenced access to lifeboats and emergency assistance. The chart clearly demonstrates that survival probability decreased as passenger class decreased.

---

## Chart 3 — Age vs Fare (Colored by Survival)

<p align="center">
<img src="outputs/plots/multivariate_chart_3.png" width="700">
</p>

The scatter plot illustrates the relationship between passenger age, ticket fare, and survival. Passengers paying higher fares were generally associated with higher survival rates, reflecting the relationship between fare and passenger class. Although survivors appear across all age groups, fare shows a stronger association with survival than age alone.

---

## Chart 4 — Age Distribution by Sex and Survival

<p align="center">
<img src="outputs/plots/multivariate_chart_4.png" width="700">
</p>

This visualization combines age, gender, and survival into a single chart. Female passengers maintained relatively high survival rates across different age groups, whereas male passengers experienced substantially lower survival rates regardless of age. These observations suggest that gender had a stronger influence on survival than age, although age still contributed to differences within each group.

---

# Exploratory Standardization Check

As an exploratory preprocessing step, the **Age** and **Fare** features were standardized using the Z-score formula:

```text
z = (x − mean) / std
```

This standardization was performed **only for exploratory analysis** and was **not used** in the machine learning pipeline. During model training, standardization will be performed separately using only the training dataset to avoid data leakage.

---

## Before Standardization

| Feature | Mean | Standard Deviation |
|----------|-----:|-------------------:|
| Age | 29.32 | 12.98 |
| Fare | 32.10 | 49.70 |

---

## After Standardization

| Feature | Mean | Standard Deviation |
|----------|-----:|-------------------:|
| Age | ≈ 0 | ≈ 1 |
| Fare | ≈ 0 | ≈ 1 |

The standardized features have means approximately equal to zero and standard deviations approximately equal to one, confirming that Z-score standardization was successfully applied.

---

### Age Distribution: Before vs After Standardization

<p align="center">
<img src="outputs/plots/age_standardization_comparison.png" width="700">
</p>

The standardized age distribution preserves the overall shape of the original distribution while shifting the mean to approximately zero and scaling the standard deviation to one. This confirms that standardization changes the scale of the data without altering its underlying distribution.

---

### Fare Distribution: Before vs After Standardization

<p align="center">
<img src="outputs/plots/fare_standardization_comparison.png" width="700">
</p>

The standardized fare distribution also maintains the original right-skewed shape while rescaling the values around a mean of zero. Although the distribution remains skewed due to the presence of high-fare passengers, the transformed values are now suitable for algorithms that are sensitive to feature scaling.

---

# Generated Outputs

Running the module automatically generates the following outputs.

## Raw Dataset

```text
analytics/data/raw/titanic.csv
```

---

## Cleaned Dataset

```text
analytics/data/processed/titanic_cleaned.csv
```

---

## Generated Visualizations

```text
analytics/outputs/plots/
```

The plots directory contains:

- Age Histogram
- Fare Histogram
- Age Box Plot
- Fare Box Plot
- Survival by Sex
- Survival by Passenger Class
- Survival by Sex and Passenger Class
- Correlation Heatmap
- Multivariate Chart 1
- Multivariate Chart 2
- Multivariate Chart 3
- Multivariate Chart 4
- Age Standardization Comparison
- Fare Standardization Comparison

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Module Status

| Task | Status |
|------|:------:|
| Dataset Loading | ✅ |
| Dataset Profiling | ✅ |
| Missing Value Handling | ✅ |
| Univariate Analysis | ✅ |
| Bivariate Analysis | ✅ |
| Correlation Analysis | ✅ |
| Multivariate Analysis | ✅ |
| Standardization Check | ✅ |

---

# Next Step

The exploratory data analysis is now complete. The cleaned dataset and observations from this module will be used to build and evaluate machine learning models in **Part B**, where preprocessing, model training, hyperparameter tuning, and performance evaluation will be performed.

---