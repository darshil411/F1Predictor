# 🏎️ Formula 1 Race Finish Prediction

This project predicts whether an F1 driver will **finish in the top position (target_finish = 1)** based on historical race, driver, and constructor data.  
It involves **EDA, feature engineering, model training, and deployment using Streamlit**.

---

## 🚀 Project Overview

**Goal:**  
Predict if a driver finishes first (`target_finish = 1`) using historical race metrics and categorical data such as team, driver, and circuit.

**Dataset Summary (10,000 rows × 31 columns):**
- Race details: `year`, `round`, `grid`, `laps`, `milliseconds`
- Driver info: `driverRef`, `surname`, `dob`, `nationality_x`
- Constructor info: `constructorRef`, `nationality_y`
- Circuit details: `circuitRef`, `country`, `lat`, `lng`, `alt`
- Target variable: `target_finish` (1 = top finish, 0 = otherwise)

---

## 📊 Exploratory Data Analysis (EDA)

### Key Steps:
1. **Data Understanding**
   - Replaced all `\N` and blanks with `NaN`.
   - Checked missing values and data types.
   - Dropped irrelevant or duplicate columns.

2. **Missing Value Handling**
   - `points`, `laps`, `milliseconds`, `fastestLap`, `rank`, `fastestLapTime`, `fastestLapSpeed` contained missing values.
   - Numeric columns (`points`, `laps`, `milliseconds`) filled with **mean/median** based on distribution.
   - Non-numeric columns (`fastestLap`, `rank`, etc.) filled using **mode**.

3. **Distribution & Outlier Check**
   - `milliseconds` and `fastestLapSpeed` ~ normally distributed.
   - Outliers handled by capping using IQR method where necessary.

4. **Correlation Insights**
   - `milliseconds` → **0.88** correlation with `target_finish`
   - `points` → **0.59**
   - `driver_avg_points` → **0.44**
   - `constructor_avg_points` → **0.42**
   - `laps` → **0.29**

5. **Categorical Analysis**
   - Encoded categorical features: `constructorRef`, `circuitRef`
   - Grouped top constructors by average `target_finish` for insight.

---

## ⚙️ Feature Engineering

Final feature set used for modeling:

```python
final_features = [
    'points',
    'laps',
    'grid',
    'driver_avg_points',
    'constructor_avg_points',
    'driver_median_grid',
    'constructor_median_grid',
    'constructorRef_enc',
    'circuitRef_enc',
    'target_finish'
]
