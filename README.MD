# 🏎️ Formula 1 Anomaly Detection

<div align="center">
<img src="images/pexels-jonathanborba-29406740.jpg" width="60%">
</div>

---

## Project Overview

Formula 1 is a sport driven by precision, strategy, and data. Most races follow predictable patterns — consistent lap times, planned pit strategies, and performance trends aligned with expectations.

However, some races break these patterns.

This project applies **unsupervised machine learning** to historical Formula 1 data to automatically detect unusual driver performances and abnormal race events — without relying on predefined anomaly labels.

Instead of predicting outcomes, the goal is to understand:

What does *normal race behavior* look like?  
Which performances deviate significantly from this baseline?

By transforming raw race data into aggregated performance features and applying anomaly detection techniques, the analysis identifies meaningful irregularities such as:

- DNFs and incomplete races
- Pace instability or performance drops
- Unusual pit strategies
- Chaotic race conditions or unexpected outcomes

The full workflow — data preparation, feature engineering, modeling, and validation — is documented inside the notebook.

---

## Objectives

- Understand structure and behavior patterns in Formula 1 race data
- Build **driver × race** performance features
- Explore feature distributions using Exploratory Data Analysis (EDA)
- Apply unsupervised anomaly detection models
- Compare PCA-based anomaly detection and Isolation Forest
- Detect abnormal performances without labeled anomaly data
- Validate detected anomalies against real-world race events

---

## Methodology

The project follows a complete unsupervised ML workflow:

1. **Data Cleaning & Preprocessing**
   - Filtering seasons (2015–2024)
   - Handling inconsistent timing data
   - Aligning datasets by raceId

2. **Feature Engineering**
   - Driver × race aggregation
   - Lap time statistics
   - Pace evolution ("pace drop")
   - Pit stop behavior features

3. **Exploratory Data Analysis**
   - Feature distributions
   - Heavy-tail analysis
   - Transformation validation

4. **Feature Transformation**
   - Log transformations for skewed features
   - Signed log for pace changes
   - Standard scaling

5. **Anomaly Detection**
   - Principal Component Analysis (PCA)
   - Isolation Forest
   - Percentile-based thresholding

6. **Post-hoc Validation**
   - Compare anomalies with race results and known events

---

## Key Insights

- Formula 1 race behavior shows strong structured patterns rather than random variation.
- Most races cluster around stable performance characteristics.
- A small subset of performances deviate significantly and align with real-world incidents.

Main drivers of anomalies include:

- Pace instability across laps
- Unusual pit stop strategies
- Incomplete races or DNFs
- Large performance shifts during a race

Model comparison reveals:

- **PCA** captures global deviations from normal race structure.
- **Isolation Forest** detects locally rare combinations of features.
- Overlap between models indicates meaningful anomaly detection rather than noise.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Dataset

Data is sourced from the **Ergast Developer API**, which provides structured historical Formula 1 data:

- Race metadata
- Driver information
- Lap times
- Pit stops
- Race results

Although the dataset spans multiple decades, modeling focuses on seasons **2015–2024** to maintain consistency with modern Formula 1 regulations and race dynamics.

---

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
