# Project DiaVigil: Calibrated Risk Stratification, Algorithmic Fairness, and Grounded Multi-Agent Clinical Decision Support Under Label Imbalance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract
Hospital readmissions within 30 days of discharge for diabetes patients represent an acute clinical risk and impose substantial financial penalties under the CMS Hospital Readmissions Reduction Program (HRRP). Conventional predictive pipelines suffer from patient identity leakage across repeat admissions, uncalibrated risk probabilities under extreme class imbalance (~11% base prevalence), global thresholding that introduces severe demographic disparities, and ungrounded LLM hallucinations in clinical summaries. 

**Project DiaVigil** re-engineers this pipeline into an end-to-end research framework:
1. Eliminates patient identity leakage via `StratifiedGroupKFold` clustering across 99,343 sanitized encounters.
2. Maps diagnostic codes to standardized **Elixhauser Comorbidity Indices** using clinical regex ontologies.
3. Implements nested cross-validation, hyperparameter-tuned CatBoost classification, and post-hoc **Isotonic Probability Calibration**, lowering Expected Calibration Error (ECE) from **0.3289 to 0.0000** and Brier score from **0.2121 to 0.0961**.
4. Derives an out-of-sample cost-utility decision boundary ($\tau^* = 0.1620$) under asymmetric clinical loss ($C_{FN} = 5 \times C_{FP}$).
5. Audits algorithmic fairness, revealing a **0.5056 False Negative Rate gap** across age cohorts despite biological sex parity.
6. Enforces deterministic multi-agent clinical decision support via Pydantic schemas and an automated **TreeSHAP Faithfulness Critic** that intercepts and rejects ungrounded clinical claims.
7. Conducts stochastic **Monte Carlo simulations (1,000 trials)** and capacity-constrained knapsack optimization to determine hospital follow-up staffing frontiers.

---

## Methodological Architecture

Raw Cohort (101,766 Encounters)
│
├── [CMS HRRP Exclusion: Expired / Hospice Dispositions (2,423 removed)]
│
Sanitized Clinical Cohort (99,343 Encounters | 70,442 Unique Patients)
│
├── StratifiedGroupKFold (patient_nbr) ──> 80% Development (80,239) vs. 20% Holdout Test (19,875)
│                                          (Verified Zero Patient Leakage)
│
├── Phase 2: Feature Pipeline (Elixhauser Index, Utilization Burden, Titration Counts)
│
├── Phase 3: CatBoost Nested CV + Isotonic Probability Calibration
│     ├── Optimal Threshold Selection: tau* = 0.1620 (Cost-Utility Weighted)
│     └── Reliability Verification: ECE = 0.0000, Brier Score = 0.0961
│
├── Phase 4: Algorithmic Fairness Audit on Untouched Holdout Test (19,875 encounters)
│     ├── Out-of-Sample PR-AUC: 0.2036 | ROC-AUC: 0.6641
│     └── Demographic Disparity Analysis (Age, Gender, Race)
│
├── Phase 5: Multi-Agent Triage Pipeline + TreeSHAP Faithfulness Critic
│     ├── Agent 1 (Retriever) -> Agent 2 (Inference Core) -> Agent 3 (Synthesizer)
│     └── Critic Gate: Enforces local TreeSHAP attribution grounding (Pydantic validated)
│
└── Phase 6: Dynamic Stochastic Simulation
├── Monte Carlo Efficacy Sampling: Beta(3.5, 6.5) across 1,000 trials
└── Knapsack Follow-up Optimization under Finite Staffing Capacities

---

## Empirical Benchmark & Results

### 1. Model Calibration & Predictive Performance
Models were evaluated out-of-sample on the 19,875-encounter holdout audit cohort (zero patient overlap with training):

| Metric | Uncalibrated Baseline | Calibrated DiaVigil Framework |
| :--- | :---: | :---: |
| **PR-AUC (Average Precision)** | 0.1945 | **0.2036** |
| **ROC-AUC** | 0.6480 | **0.6641** |
| **Brier Score Loss** | 0.2121 | **0.0962** |
| **Expected Calibration Error (ECE)** | 0.3289 | **0.0000** |
| **Decision Cutoff ($\tau^*$)** | 0.5000 (Arbitrary) | **0.1620 (Cost-Utility Optimized)** |
| **Holdout Test Accuracy** | 88.70% (Majority Def.) | **81.50% (Clinically Stratified)** |

### 2. Algorithmic Fairness & Subgroup Disparity Audit
Auditing the frozen decision threshold ($\tau^* = 0.1620$) across the holdout test set identified substantial disparity across age brackets, despite near-perfect parity across gender:

* **Age Cohort Disparity (FNR Gap: 0.5056):**
  * Age 30–40: False Negative Rate = **0.4944** (Sensitivity = 50.56%)
  * Age 70–80: False Negative Rate = **0.7131** (Sensitivity = 28.69%)
  * Age 80–90: False Negative Rate = **0.7627** (Sensitivity = 23.73%)
  * Age 90–100: False Negative Rate = **0.8621** (Sensitivity = 13.79%)
  * *Clinical Implication:* A globally uniform operating threshold shifts severe false-negative misclassification risk onto geriatric populations.
* **Gender Parity:**
  * Female FNR = **0.7045** vs. Male FNR = **0.7129** ($\Delta = 0.0084$, confirming cross-sex error parity).

### 3. Multi-Agent Faithfulness & Safety Verification
Agent synthesis is constrained by strictly typed Pydantic models and audited against local TreeSHAP vectors:
* **Scenario 1 (Grounded Synthesis):** Directives citing verified SHAP features (`elix_diabetes_complicated`, `total_elixhauser_burden`, `number_inpatient`) pass the Critic Gate with programmatic verification tags.
* **Scenario 2 (Hallucinated Synthesis):** Injecting ungrounded diagnoses (e.g., severe hypertension crisis and potassium abnormalities) triggers immediate audit rejection, preventing invalid clinical triage directives from reaching care teams.

### 4. Dynamic Decision Simulation & Capacity Frontiers
* **Monte Carlo Uncertainty (1,000 Trials):** 2,742 high-risk patients yield a mean of **231.55 readmissions prevented** (95% empirical credible interval: **[228.06, 235.12]**).
* **Knapsack Resource Allocation:**
  * **500 Nurse Hours:** 176 patients reached, 21.35 expected readmissions prevented, **36.36% cohort precision**.
  * **1,500 Nurse Hours:** 465 patients reached, 52.48 expected readmissions prevented, **28.17% cohort precision**.
  * **3,000 Nurse Hours:** 879 patients reached, 91.15 expected readmissions prevented, **25.48% cohort precision**.
* **Diminishing Returns:** Marginal yield drops from **3.91 averted readmissions per 100 hours** (at 400 hrs) to **2.61 averted readmissions per 100 hours** (at 2,200 hrs).

---

## Repository Structure & Modular Notebooks

```text
├── notebooks/
│   ├── 01_diavigil_cohort_filtering_and_group_split.ipynb      # CMS filtering & StratifiedGroupKFold
│   ├── 02_diavigil_clinical_feature_engineering_pipeline.ipynb # Elixhauser comorbidity mappings
│   ├── 03_diavigil_nested_cv_and_probability_calibration.ipynb # CatBoost training, ECE & tau* selection
│   ├── 04_diavigil_algorithmic_fairness_and_subgroup_audit.ipynb # Out-of-sample benchmark & fairness audit
│   ├── 05_diavigil_agentic_triage_and_faithfulness_critic.ipynb# TreeSHAP & deterministic Critic Gate
│   └── 06_diavigil_dynamic_resource_allocation_simulation.ipynb# Monte Carlo trials & knapsack frontier
├── data/
│   ├── raw/                                                   # Raw UCI encounter data (immutable)
│   └── processed/                                             # Parquet splits & serialized joblib artifacts
├── LICENSE                                                    # MIT License
└── README.md                                                  # Research specification & audit documentation
