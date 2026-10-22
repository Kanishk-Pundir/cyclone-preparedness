# AI-Based Disaster Preparedness Assistant: Coastal Karnataka Cyclone Resilience

[![MLOps Pipeline](https://img.shields.io/badge/MLOps-DVC%20%7C%20Git%20LFS%20%7C%20W%26B-blue)](#mlops-architecture--reproducibility)
[![Dataset](https://img.shields.io/badge/Dataset-1%2C879%20Villages%20%7C%2049%20Features-green)](#dataset-architecture--source-catalogue)
[![Champion Regression](https://img.shields.io/badge/Champion%20Regression-Ridge%20(R%C2%B2%200.976)-orange)](#8-machine-learning-baseline-experiments)
[![Champion Classification](https://img.shields.io/badge/Champion%20Classification-Logistic%20Reg%20(94.3%25)-red)](#8-machine-learning-baseline-experiments)

An end-to-end MLOps platform for granular disaster risk reduction, vulnerability profiling, and cyclone preparedness evaluation across the coastal belt of Karnataka, India.

---

## Table of Contents
1. [Executive Summary & Project Mission](#1-executive-summary--project-mission)
2. [Geographic Scope & Problem Context](#2-geographic-scope--problem-context)
3. [Dataset Architecture & Source Catalogue](#3-dataset-architecture--source-catalogue)
4. [Key Architectural Decisions & Trade-Off Justifications](#4-key-architectural-decisions--trade-off-justifications)
5. [Spatial Join Pipeline & Data Engineering](#5-spatial-join-pipeline--data-engineering)
6. [The Preparedness Index — Formulation & AHP Justification](#6-the-preparedness-index--formulation--ahp-justification)
7. [Exploratory Data Analysis (EDA) & Key Insights](#7-exploratory-data-analysis-eda--key-insights)
8. [Comprehensive Feature Engineering](#8-comprehensive-feature-engineering)
9. [Machine Learning Baseline Experiments](#9-machine-learning-baseline-experiments)
10. [MLOps Architecture & Reproducibility (Git LFS, DVC, W&B)](#10-mlops-architecture--reproducibility)
11. [Long-Term Vision: Risk Score & Actionable Recommendations](#11-long-term-vision-risk-score--actionable-recommendations)
12. [Defense & Presentation FAQ (Oral Exam Prep)](#12-defense--presentation-faq-oral-exam-prep)

---

## 1. Executive Summary & Project Mission

Cyclones along the west coast of India (such as Cyclone Tauktae in May 2021) have intensified in frequency and surge height due to warming surface waters in the Arabian Sea. While macroscopic early warnings are issued at state or meteorological subdivision scales, disaster response happens at the **Gram Panchayat (village)** level. 

Local administrations face three fundamental challenges:
1. **Data Fragmentation**: Administrative demographics (Census) lack spatial coordinates, while spatial facilities (hospitals, rain gauges, multi-purpose cyclone shelters) lack demographic denominators.
2. **Subjective Resource Allocation**: Disaster relief and infrastructure hardening are often reactive rather than prioritized by multidimensional capacity deficits.
3. **Absence of Actionable Prescriptions**: Existing vulnerability maps tell authorities *where* danger exists, but not *what specific intervention* (e.g., upgrading an access road vs. expanding shelter capacity) yields the highest risk reduction per rupee.

**Our Solution**: An end-to-end reproducible MLOps platform that unifies 6 disparate national and state datasets into a single, complete village-level master dataset ($N = 1,879$ villages), formulates a mathematically defensible Preparedness Index ($0\text{–}100$), trains baseline ML models with complete experiment tracking, and establishes a prescriptive framework for automated, personalized mitigation recommendations.

---

## 2. Geographic Scope & Problem Context

The project targets the three coastal districts of Karnataka that constitute the state's entire $\approx 320\text{ km}$ coastline:
* **Uttara Kannada (District Code 563)**: 1,257 village polygons. Characterized by a narrow coastal strip abruptly meeting the dense, elevated Western Ghats; high forest cover and isolated hamlets.
* **Udupi (District Code 569)**: 253 village polygons. Densely populated coastal belt with extensive river estuaries, fisheries infrastructure, and low-lying coastal plains.
* **Dakshina Kannada (District Code 575)**: 369 village polygons. Highly urbanized around Mangaluru with heavy infrastructure, port activity, and steep coastal river basins.

### Ground-Truth Cyclone Tauktae (May 2021) Validation
To validate our indices against reality, we extracted damage data from Karnataka State Disaster Management Authority (KSDMA) situation reports during Cyclone Tauktae:
* **194 coastal villages ($10.3\%$)** were severely impacted by tidal surges ($2.5\text{–}4.0\text{ m}$), massive beachfront erosion (e.g., the $12\text{ km}$ Kulai–Sasihithlu stretch), and road washouts.
* These 194 villages serve as our empirical validation cohort to ensure our constructed index aligns with real physical damage patterns.

---

## 3. Dataset Architecture & Source Catalogue

The master dataset contains **1,879 rows $\times$ 49 columns** with **0 missing values**. Every row represents a unique spatial Census village unit (`shrid2`).

| Source Name | Raw File & Format | Entity Level | Key Features Contributed | Provenance / Authority |
| :--- | :--- | :--- | :--- | :--- |
| **Census 2011 Village Directory (DCHB)** | `DH_2011_DCHB_Village_Release_2900.xlsx` (58.1 MB) | Village | Population, households, SC/ST share, tap water, pucca roads, bus service, HQ distances | Office of the Registrar General & Census Commissioner, India |
| **SHRUG 2.1 Spatial Boundaries** | `shrug-shrid-poly-shp/shrid2_open.shp` (645 MB) | Polygon | Polygon boundary geometry, metric area ($\text{km}^2$), UTM and WGS84 centroids | Development Data Lab (Dartmouth / World Bank) |
| **National Health Facility Directory (HMIS)** | `geocode_health_centre.csv` (20.9 MB) | Point (GPS) | Distances to nearest Health Centre, PHC/CHC, and District Hospital; facility density within 10 km | Ministry of Health & Family Welfare (MoHFW) / data.gov.in |
| **KSNDMC Telemetric Rain Gauges** | `70578d25-7ede-426c-8e8e-5c679274a55a.kml` (3.5 MB) | Point (GPS) | Nearest rain gauge distance, station name, telemetric monitoring density within 20 km | Karnataka State Natural Disaster Monitoring Centre / OpenCity |
| **Multi-Purpose Cyclone Shelters (NCRMP-II)** | `shelters_manual.csv` compiled from official DPRs | Point (GPS) | Nearest shelter distance, shelter count within 10 km, total capacity, capacity per capita | National Cyclone Risk Mitigation Project (World Bank / NDMA) |
| **Coastal Vulnerability Index (CVI/SVI)** | `IntegratedCoastalVulnerabilityAssessment.pdf` | Taluka | Physical-Geological CVI, Socioeconomic CVI, Integrated CVI, vulnerability tier | Naik & Kunte (2023), CSIR-NIO & Symbiosis |
| **KSDMA Cyclone Tauktae Situational Reports** | Situational damage archives | Village | Binary empirical validation flag (`tauktae_affected`: 0 or 1) | Karnataka State Disaster Management Authority |

---

## 4. Key Architectural Decisions & Trade-Off Justifications

### Decision 1: Choosing the Census Village Unit over Alternatives
* **Alternatives Considered**: District-level, Taluka-level, or a uniform $1\text{ km} \times 1\text{ km}$ spatial raster grid.
* **Why Village-Level Won**:
  1. *Administrative Actionability*: Disaster response budgets, Gram Panchayat development plans, and emergency relief shelters are administered at the village level. District/Taluka aggregates mask extreme coastal vs. inland micro-disparities.
  2. *Granularity without Spatial Disconnection*: A $1\text{ km}$ grid creates artificial boundary cuts and decouples data from demographic registries (Census), making policy implementation impossible.

### Decision 2: Using SHRUG 2.1 Polygons as the Spatial Backbone
* **Alternatives Considered**: Reverse-geocoding village names via Google Maps API or OpenStreetMap; KGIS portal WFS layers.
* **Why SHRUG Won**:
  1. *Spelling Ambiguity in Geocoding*: Indian village names suffer from severe phonetic divergence across Kannada and English records (e.g., "Coondapoor" vs. "Kundapura", "Paduvari" vs. "Paduvare"). Reverse-geocoding introduces an estimated $15\text{–}25\%$ mismatch or misplacement error.
  2. *Direct Census Key Linkage*: SHRUG provides verified geographic polygons where `pc11_id` matches the 6-digit Census 2011 village code directly.
* **Schema Discovery**: We identified that SHRUG's `shrid2` follows a strict hierarchical schema:
  $$\underbrace{\text{11}}_{\text{Census 2011}}-\underbrace{\text{29}}_{\text{Karnataka}}-\underbrace{\text{DDD}}_{\text{District Code}}-\underbrace{\text{SSSSS}}_{\text{Sub-District}}-\underbrace{\text{VVVVVV}}_{\text{Village Code}}$$
  Filtering prefixes `11-29-563-` (Uttara Kannada), `11-29-569-` (Udupi), and `11-29-575-` (Dakshina Kannada) extracted all 1,879 coastal village polygons directly with **97.36% direct primary key overlap** with Census 2011 records.

### Decision 3: Reprojecting to EPSG:32643 (UTM Zone 43N) Metric CRS
* **Alternatives Considered**: Calculating distances in degrees using EPSG:4326 (WGS84).
* **Why EPSG:32643 Won**: EPSG:4326 uses angular decimal degrees. Computing Euclidean distances or buffers in degrees distorts distances significantly as longitude lines converge away from the equator ($\Delta \text{lon} \neq \Delta \text{lat}$). EPSG:32643 is a conformal projection in meters designed specifically for Western India ($72^\circ\text{E}$ to $78^\circ\text{E}$), guaranteeing metric accuracy for all hospital, shelter, and gauge radii.

---

## 5. Spatial Join Pipeline & Data Engineering

The pipeline executes deterministically through five standalone, DVC-tracked stages in `src/data/`:

```
data/raw/ 
   │
   ├── 02_prepare_boundaries.py ────────> data/interim/villages_coastal_utm.parquet
   │                                      data/interim/census_coastal.parquet
   ├── 03_join_health_centres.py ───────> data/interim/features_health_centres.parquet
   │
   ├── 04_join_rain_gauges.py ──────────> data/interim/features_rain_gauges.parquet
   │
   ├── 05_join_shelters_and_vuln.py ───> data/interim/features_shelters_vulnerability.parquet
   │
   └── 06_build_master_table.py ────────> data/processed/master_village_preparedness.parquet (.csv)
```

1. **Geometry Sanity Checking**: Cleaned all 1,879 SHRUG polygons. Used Shapely's `buffer(0)` to resolve self-intersections. **Result: 0 invalid, 0 empty, 0 null geometries**.
2. **Nearest-Facility Joins (`gpd.sjoin_nearest`)**: Performed Euclidean nearest-neighbor distance searches in UTM meters for health facilities, hospitals, rain gauges, and cyclone shelters.
3. **Buffered Density Joins (`gpd.sjoin(..., predicate="contains")`)**:
   - $10\text{ km}$ centroid buffer for local healthcare access and evacuation shelter catchments.
   - $20\text{ km}$ centroid buffer for KSNDMC telemetric rain gauge density.
   - Fixed a common GeoPandas artifact where left-joins on empty buffers artificially count as 1 by explicitly filtering on `dropna(subset=['index_right'])` before aggregation.
4. **Coastline Derivation**: Formulated the Arabian Sea coastline polyline from Talapady ($12.75^\circ\text{N}$) to Majali ($14.92^\circ\text{N}$) in EPSG:32643 and computed the minimum perpendicular distance (`dist_coast_km`) for all villages.

---

## 6. The Preparedness Index — Formulation & AHP Justification

The composite Preparedness Score ($0\text{–}100$) operationalizes **Coping Capacity** under the **UN Sendai Framework for Disaster Risk Reduction (2015–2030)** and **NDMA Guidelines**:

$$\text{Disaster Risk} = \frac{\text{Hazard Exposure} \times \text{Vulnerability}}{\text{Coping \& Adaptive Capacity}}$$

$$\text{Preparedness Score} = 0.25 S_{\text{medical}} + 0.25 S_{\text{shelters}} + 0.20 S_{\text{infra}} + 0.15 S_{\text{early\_warning}} + 0.15 S_{\text{socioeconomic}}$$

### A. Sub-Score Dimensions
1. **Medical Infrastructure ($S_{\text{medical}}, 25\%$)**: Triage capability for storm-surge trauma, drowning, and electrocution during the $0\text{–}72\text{ hr}$ landfall window. Proximity to nearest health centre ($35\%$), nearest hospital ($25\%$), health centre density within 10 km ($25\%$), and local village clinic presence ($15\%$).
2. **Evacuation & Shelters ($S_{\text{shelters}}, 25\%$)**: Dedicated life-safety barrier against $120+\text{ km/h}$ winds and $2\text{–}4\text{ m}$ storm surges. Proximity to nearest shelter ($35\%$), shelter count within 10 km ($25\%$), aggregate shelter capacity ($25\%$), and shelter capacity per capita ($15\%$).
3. **Roads & Lifeline Utilities ($S_{\text{infra}}, 20\%$)**: All-weather pucca roads enable mechanized evacuation convoys; treated drinking water prevents post-disaster cholera and leptospirosis outbreaks. Road connectivity score ($60\%$) + drinking water reliability ($40\%$).
4. **Early Warning & Meteorological Monitoring ($S_{\text{early\_warning}}, 15\%$)**: Proximity to KSNDMC telemetric rain gauges ($60\%$) and 20 km gauge density ($40\%$) dictate the lead-time window ($24\text{–}48\text{ hr}$) for issuing localized village evacuation alerts.
5. **Socioeconomic Resilience ($S_{\text{socioeconomic}}, 15\%$)**: Baseline private coping reserves. Communities with high SC/ST shares historically face disproportionate vulnerability due to kutcha housing materials on low-lying tidal fringes.

### B. Mathematical Justification via Saaty's Analytic Hierarchy Process (AHP)
To eliminate arbitrary heuristic weighting, weights were derived from Saaty's pairwise comparison matrix:

$$\mathbf{A} = \begin{pmatrix} 1.00 & 1.00 & 1.25 & 1.67 & 1.67 \\ 1.00 & 1.00 & 1.25 & 1.67 & 1.67 \\ 0.80 & 0.80 & 1.00 & 1.33 & 1.33 \\ 0.60 & 0.60 & 0.75 & 1.00 & 1.00 \\ 0.60 & 0.60 & 0.75 & 1.00 & 1.00 \end{pmatrix} \quad \longrightarrow \quad \mathbf{w} = \begin{pmatrix} 0.250 \\ 0.250 \\ 0.200 \\ 0.150 \\ 0.150 \end{pmatrix}$$

* **Eigenvalue & Consistency**: $\lambda_{\max} = 5.000$, Consistency Index $CI = 0.000$, **Consistency Ratio $CR = 0.000 < 0.10$** (mathematically consistent under Saaty's theorem).
* **Monte Carlo Sensitivity Analysis**: We simulated $N = 2,000$ random weight perturbations ($\pm 20\%$). The resulting Spearman rank correlation between perturbed rankings and our baseline was **$\text{mean } \rho = 0.992$ (minimum $\rho = 0.961$)**, proving village rankings are remarkably stable and not an artifact of parameter selection.

---

## 7. Exploratory Data Analysis (EDA) & Key Insights

Executed in [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb); all 10 figures are saved in [`notebooks/`](notebooks/):

1. **District Preparedness Disparity ([`fig_01_overview.png`](notebooks/fig_01_overview.png))**:
   - **Dakshina Kannada**: Highest preparedness (mean $67.4$), driven by dense road networks and Mangaluru healthcare infrastructure.
   - **Udupi**: Moderate preparedness (mean $58.1$), dense coastal population, high reliance on coastal cyclone shelters.
   - **Uttara Kannada**: Lowest preparedness (mean $48.2$), geographically isolated forest villages with long distances to hospitals.
2. **Univariate Skewness ([`fig_02_distributions.png`](notebooks/fig_02_distributions.png))**:
   - Variables like `dist_nearest_hospital_km` ($\text{skew} = +2.84$) and `shelter_capacity_nearby` ($\text{skew} = +4.12$) exhibit extreme power-law distributions.
3. **Correlation Matrix ([`fig_03_correlation_full.png`](notebooks/fig_03_correlation_full.png))**:
   - Strong collinearity between `dist_coast_km` and `elevation_mean` ($r = +0.89$), reflecting the abrupt orographic rise of the Western Ghats.
4. **Validation against Tauktae Ground Truth ([`fig_04_target_correlations.png`](notebooks/fig_04_target_correlations.png) & [`fig_05_bivariate.png`](notebooks/fig_05_bivariate.png))**:
   - Villages impacted by Cyclone Tauktae had a **mean distance to coast of $0.66\text{ km}$** vs. **$38.79\text{ km}$** for unaffected villages.
   - Tauktae impact strongly correlates with shelter presence ($r = +0.547$), confirming that government cyclone shelters were accurately placed in the historically vulnerable coastal inundation strip.

---

## 8. Comprehensive Feature Engineering

In `src/features/generate_eda_notebook.py` and `notebooks/01_EDA.ipynb`, we engineered 7 specialized domain features:

| Feature Name | Mathematical Formulation | Rationale & Domain Purpose |
| :--- | :--- | :--- |
| **`log1p_*` Transforms** | $\ln(1 + x)$ applied to population, density, shelter capacity, hospital distance | Compresses extreme right-tailed distributions (e.g. population skew reduced from $+2.12 \rightarrow -0.18$), stabilizing linear model gradients. |
| **`coastal_risk_score`** | $\exp(-\frac{\text{dist\_coast}}{10}) \times \exp(-\frac{\text{elevation}}{30})$ | Models compound non-linear storm surge inundation decay. Surges attenuate exponentially inland and cannot crest high elevations. |
| **`healthcare_access_gap`**| $\frac{\text{dist\_hospital}}{100} - \frac{n\_hc\_10km}{\max(n\_hc)}$ | Captures coverage divergence: identifies villages that may have minor clinics nearby but are dangerously isolated from tertiary emergency hospitals. |
| **`evacuation_sufficiency`**| $\frac{\text{shelter\_capacity\_10km}}{\text{population}}$ (clipped to 10) | Normalizes shelter capacity by village population, highlighting areas with severe evacuation deficits per capita. |
| **`vulnerability_exposure`**| $0.5 \frac{\text{CVI}}{\max} + 0.3 (1 - \frac{\text{dist\_coast}}{\max}) + 0.2 \text{vuln\_pop\_share}$ | Integrates taluka physical-geological hazard exposure with village-level socioeconomic demographic vulnerability. |
| **`infra_composite`** | $\frac{1}{3} (\frac{\text{road\_score}}{3} + \frac{\text{water\_score}}{2} + \text{bus\_conn})$ | Unified multidimensional infrastructure lifeline metric. |
| **`monitoring_coverage`**| $0.6 (1 - \frac{\text{dist\_gauge}}{\max}) + 0.4 \frac{n\_gauges\_20km}{\max}$ | Telemetric early-warning density index. |

All features are preprocessed using **`RobustScaler`** (scaling via median and IQR) rather than `StandardScaler` to prevent leverage distortion from extreme coastal outliers while preserving genuine physical variance.

---

## 9. Machine Learning Baseline Experiments

Executed in [`notebooks/02_Baseline_Models.ipynb`](notebooks/02_Baseline_Models.ipynb). We evaluated **3 distinct model families** across **3 stratified split ratios** (60/20/20, 70/15/15, 80/10/10) to assess stability and generalization:

### A. Continuous Regression (Target: `preparedness_score`, 0–100)
| Split Ratio | Model Architecture | Train $R^2$ | Test $R^2$ ($\uparrow$) | Test RMSE ($\downarrow$) | Test MAE ($\downarrow$) | Generalization Gap |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **60 / 20 / 20** | **Ridge Regression (Champion)** | $0.976$ | **$0.977$** | **$1.64$** | **$1.21$** | **$0.000$ (Optimal)** |
| 60 / 20 / 20 | HistGradientBoosting | $0.991$ | $0.970$ | $1.86$ | $1.40$ | $+0.021$ |
| 60 / 20 / 20 | Random Forest | $0.974$ | $0.938$ | $2.67$ | $1.88$ | $+0.036$ |
| **70 / 15 / 15** | **Ridge Regression (Champion)** | $0.976$ | **$0.976$** | **$1.65$** | **$1.23$** | **$0.000$ (Optimal)** |
| 70 / 15 / 15 | HistGradientBoosting | $0.990$ | $0.967$ | $1.93$ | $1.44$ | $+0.023$ |
| 70 / 15 / 15 | Random Forest | $0.977$ | $0.946$ | $2.47$ | $1.73$ | $+0.031$ |
| **80 / 10 / 10** | **Ridge Regression (Champion)** | $0.977$ | **$0.975$** | **$1.71$** | **$1.30$** | **$+0.002$** |
| 80 / 10 / 10 | HistGradientBoosting | $0.991$ | $0.968$ | $1.94$ | $1.48$ | $+0.023$ |
| 80 / 10 / 10 | Random Forest | $0.979$ | $0.949$ | $2.45$ | $1.74$ | $+0.030$ |

### B. Multi-Class Classification (Target: `preparedness_tier`, Low / Medium / High)
| Split Ratio | Model Architecture | Train Acc | Test Accuracy ($\uparrow$) | Test F1 Macro ($\uparrow$) | Test F1 Weighted ($\uparrow$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **60 / 20 / 20** | **Logistic Regression (Champion)** | $95.5\%$ | **$92.0\%$** | **$0.889$** | **$0.920$** |
| 60 / 20 / 20 | HistGradientBoosting | $100\%$ | $90.2\%$ | $0.838$ | $0.899$ |
| 60 / 20 / 20 | Random Forest | $96.7\%$ | $90.2\%$ | $0.838$ | $0.899$ |
| **70 / 15 / 15** | **Logistic Regression (Champion)** | $95.9\%$ | **$94.3\%$** | **$0.906$** | **$0.942$** |
| 70 / 15 / 15 | HistGradientBoosting | $100\%$ | $92.6\%$ | $0.871$ | $0.923$ |
| 70 / 15 / 15 | Random Forest | $96.8\%$ | $90.1\%$ | $0.806$ | $0.894$ |
| **80 / 10 / 10** | **Logistic Regression (Champion)** | $95.7\%$ | **$94.7\%$** | **$0.921$** | **$0.946$** |
| 80 / 10 / 10 | HistGradientBoosting | $100\%$ | $92.0\%$ | $0.889$ | $0.919$ |
| 80 / 10 / 10 | Random Forest | $96.7\%$ | $90.4\%$ | $0.850$ | $0.903$ |

### C. Why Ridge and Logistic Regression are the Champion Models
1. **Occam's Razor & Generalization**: Ridge Regression matches or beats complex gradient-boosted trees on every test split ($R^2 = 0.976$ vs. $0.967$) with **zero variance gap** between train and test. Complex ensembles showed mild memorization ($100\%$ train accuracy vs. $92.6\%$ test).
2. **Policy Transparency**: Government and disaster authorities require explainable linear coefficients over black-box decision splits. Ridge regression allows exact marginal attribution: a village administration can see that paving $5\text{ km}$ of gravel road directly adds $+3.2$ points to their preparedness score.

---

## 10. MLOps Architecture & Reproducibility

### A. Dataset Versioning with Git LFS
Tracked in [`.gitattributes`](.gitattributes). All large spatial files, Parquet datasets, and shapefiles are managed via Git LFS:
```powershell
# Verify tracked files and SHA-256 cryptographic hashes
git lfs ls-files

# Inspect the lightweight 130-byte pointer file committed to Git
git show HEAD:data/processed/master_village_preparedness.parquet
```

### B. Pipeline Automation with DVC (`dvc.yaml`)
Every processing step from raw downloads to interim spatial joins and master table compilation is codified in [`dvc.yaml`](dvc.yaml) with explicit dependencies (`deps`) and outputs (`outs`):
```powershell
# Reproduce the entire data engineering and feature pipeline end-to-end
dvc repro
```

### C. Experiment Tracking with Weights & Biases (W&B)
All 18 model training runs are logged in W&B project `cyclone-preparedness-karnataka`.
* Stored locally in [`notebooks/wandb/`](notebooks/wandb/).
* To sync with your cloud dashboard at [wandb.ai](https://wandb.ai):
  ```powershell
  wandb login
  Get-ChildItem -Path "notebooks/wandb/offline-run-*" | ForEach-Object { wandb sync $_.FullName }
  ```

---

## 11. Long-Term Vision: Risk Score & Actionable Recommendations

### A. The 0–10 Cyclone Risk Score Formulation
Risk is the product of **Hazard Exposure** and **Preparedness Deficit**:

$$\text{Hazard Exposure} = 0.50 \cdot \text{coastal\_risk\_score} + 0.35 \cdot \left(1 - \frac{\text{dist\_coast}}{\max}\right) + 0.15 \cdot \text{CVI}_{\text{integrated\_norm}}$$

$$\text{Risk Score (0–10)} = 10 \times \left[ 0.60 \cdot \text{Hazard Exposure} + 0.40 \cdot \left(1 - \frac{\text{Preparedness Score}}{100}\right) \right]$$

### B. The Recommendation Engine (Counterfactual Prescriptions)
The engine separates features into **Immutable** (`dist_coast_km`, `elevation_mean`) vs. **Actionable** (`pucca_road`, `n_shelters_10km`, `drinking_water_score`). 

For any high-risk village, it calculates the marginal risk reduction ($\Delta \text{Risk}$) for simulated policy interventions:

```
VILLAGE INTERVENTION REPORT: Sasihithlu (Mangaluru Taluk, Dakshina Kannada)
Baseline Metrics: Distance to coast: 0.3 km | Current Risk Score: 8.4 / 10 (Critical Tier)

Actionable Interventions Ranked by ROI:
1. Priority 1 (Connectivity): Convert coastal access track to All-Weather Pucca Road.
   -> Simulated Risk Impact: 8.4 -> 7.1  (-1.3 Risk Points | 15.5% reduction)
2. Priority 2 (Shelter): Retrofit local Higher Primary School as an auxiliary cyclone shelter (+500 cap).
   -> Simulated Risk Impact: 7.1 -> 6.0  (-1.1 Risk Points | 13.1% reduction)
3. Priority 3 (Warning): Install automated early-warning telemetry gauge & siren in Gram Panchayat.
   -> Simulated Risk Impact: 6.0 -> 5.5  (-0.5 Risk Points | 6.0% reduction)

Cumulative Impact: Combined interventions drop Risk from 8.4/10 to 5.5/10 (Shifts tier from Critical to Moderate).
```

---

## 12. Defense & Presentation FAQ (Oral Exam Prep)

### Q1: *"Why did you use Ridge Regression instead of a Deep Neural Network or XGBoost?"*
> **Answer**: *"For tabular geospatial data with 1,879 rows and collinear spatial predictors, neural networks are prone to severe overfitting without offering inductive spatial bias. While HistGradientBoosting achieved a strong $R^2$ of $0.970$, it exhibited an overfit gap ($100\%$ train vs. $92.6\%$ test on classification). Ridge Regression achieved an $R^2$ of $0.976$ on the test set with zero generalization gap. More importantly, in disaster management, linear models provide transparent, ungameable coefficients that local administrations can directly audit."*

### Q2: *"How do you prove your Preparedness Index weights aren't arbitrary?"*
> **Answer**: *"We used Saaty's Analytic Hierarchy Process (AHP), comparing dimensions based on mortality drivers during the 0–72 hr landfall window per NDMA and Sendai frameworks. The comparison matrix achieved a Consistency Ratio of $CR = 0.000$, well within the $<0.10$ threshold. Furthermore, we conducted a Monte Carlo sensitivity analysis with 2,000 random $\pm 20\%$ weight perturbations; the Spearman rank correlation of village rankings remained above $\rho = 0.961$ (mean $\rho = 0.992$), proving the rankings are structurally robust."*

### Q3: *"How do you know the index reflects real-world cyclone risk?"*
> **Answer**: *"We validated against ground-truth damage from Cyclone Tauktae (May 2021). The 194 villages documented with severe surge inundation, road breach, and destruction had an average distance to the coast of just $0.66\text{ km}$, compared to $38.79\text{ km}$ for unaffected villages. The empirical damage strongly aligns with our hazard exposure and coastal proximity metrics."*

### Q4: *"How does Git LFS fit into your MLOps pipeline alongside DVC?"*
> **Answer**: *"Git LFS handles binary and dataset versioning directly within GitHub, replacing multi-megabyte Parquet and shapefile blobs with 130-byte SHA-256 text pointers so developers can clone repositories quickly. DVC complements Git LFS by tracking the execution DAG and caching intermediate stage outputs (`dvc repro`), creating an auditable chain from raw government files to final model metrics."*

---

## Contributors & Acknowledgements
* **Author**: Kanishk
* **Project**: AI-Based Disaster Preparedness Assistant (Phase 1)
* **Data Sources**: Census of India, SHRUG (Development Data Lab), MoHFW (HMIS), KSNDMC, NDMA/NCRMP-II, and CSIR-NIO.
