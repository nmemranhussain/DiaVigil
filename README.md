# Project DiaVigil: A Machine Learning Research Project on Hospital Readmission Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-orange.svg)](https://colab.research.google.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DiaVigil investigates the use of machine learning to estimate hospital readmission risk and support the evaluation of clinical decision-making strategies. The project includes clinical feature engineering, probability calibration, decision-threshold selection, subgroup error analysis, resource-allocation simulation, and an AI-agent workflow for generating and auditing care recommendations.

The project emphasizes reproducible model evaluation, transparent reporting of performance trade-offs, and documentation of methodological limitations.

**Project status:** Research prototype. The current results do not establish clinical effectiveness or readiness for deployment.

---
## Abstract

DiaVigil is a machine learning research project focused on hospital readmission risk prediction and the evaluation of decision-support strategies. The project uses clinical feature engineering, CatBoost classification, isotonic probability calibration, and cost-sensitive decision-threshold selection. It also examines subgroup false-negative rates, estimates uncertainty through patient-clustered bootstrap analysis, simulates resource allocation under staffing constraints, and explores an AI-agent workflow for generating and auditing care recommendations.

On a holdout cohort of 19,875 encounters, the baseline model achieved a precision-recall area under the curve (PR-AUC) of 0.2036 and a receiver operating characteristic area under the curve (ROC-AUC) of 0.6641. A revised model increased recall from 29.15% to 31.96%, while ROC-AUC decreased to 0.6578 and the number of false positives increased from 2,089 to 2,478. These results illustrate the trade-offs between identifying more patients who experience readmission and generating additional false-positive predictions. The project also evaluates probability calibration, subgroup error rates, and the consistency of saved research artifacts. The findings are limited to the available dataset and do not establish clinical effectiveness or generalizability to other healthcare settings.

### Project at a Glance

- **Research focus:** Hospital readmission risk prediction and decision-support evaluation.
- **Modeling:** Clinical feature engineering, CatBoost, and isotonic probability calibration.
- **Decision analysis:** Cost-sensitive threshold selection and resource-allocation simulation.
- **Evaluation:** Holdout performance, calibration, subgroup false-negative rates, and patient-clustered bootstrap analysis.
- **AI-agent component:** Generation and auditing of proposed care recommendations.
- **Current status:** Research prototype requiring further validation.
---

## Research Motivation and Objectives

### Motivation

Hospital readmission risk prediction is a classification problem in which the costs of different prediction errors may not be equal. Failing to identify a patient who is subsequently readmitted may represent a missed opportunity for additional care, while incorrectly identifying a patient as high risk may lead to unnecessary follow-up activities and use of limited clinical resources.

Model performance therefore cannot be assessed using a single metric alone. Discrimination, probability calibration, decision thresholds, subgroup error rates, and the resources required to act on predictions can provide different perspectives on the usefulness of a model. These considerations motivate the evaluation of readmission prediction as both a modeling problem and a decision-support problem.

DiaVigil brings these evaluation components together in a reproducible research workflow. It also examines whether an AI-agent workflow can generate structured care recommendations that are checked against predefined criteria. The project does not assume that better predictive performance necessarily leads to better clinical outcomes.

### Research Objectives

The project has the following objectives:

- **Develop a readmission risk model:** Apply clinical feature engineering and CatBoost classification to estimate readmission risk from the available encounter data.
- **Evaluate probability calibration:** Compare predicted probabilities with observed outcome frequencies to assess their agreement.
- **Examine decision thresholds:** Investigate how threshold selection affects recall, false-positive predictions, false-negative predictions, and an explicitly defined classification cost.
- **Assess subgroup errors:** Compare false-negative rates across available demographic groups and report uncertainty, particularly when subgroup outcome counts are small.
- **Estimate uncertainty:** Use patient-clustered bootstrap analysis to examine uncertainty in subgroup comparisons and differences between model evaluations.
- **Simulate resource allocation:** Explore how staffing constraints and prioritization rules affect the number of patients reached and the estimated number of readmissions potentially avoided under the simulation assumptions.
- **Audit AI-generated recommendations:** Examine whether generated care recommendations satisfy predefined structural and content checks, while recognizing that these checks alone do not establish clinical safety or correctness.

The overall objective is to establish a transparent and reproducible basis for evaluating readmission risk models and related decision-support workflows, while identifying the additional evidence required before clinical use can be considered.
---

## Methodology and System Architecture

DiaVigil follows a sequence of data preparation, model development, probability calibration, decision-threshold selection, holdout evaluation, and decision-support analysis. The workflow also includes resource-allocation simulation and an AI-agent component for generating and checking structured care recommendations.

### 1. Data Preparation and Cohort Separation

The project uses an encounter-level dataset containing clinical, medication, healthcare-utilization, and demographic information. The available data are divided into development and holdout cohorts.

Within the development cohort, patient-grouped partitions are used to separate model training, probability calibration, and decision-threshold selection. Encounters belonging to the same patient are kept within the same partition to reduce the risk of patient-level information leakage.

The holdout cohort is reserved for final evaluation. It is not used to fit the model, calibrate its probabilities, or select its decision threshold.

### 2. Clinical Feature Engineering

A custom feature-engineering transformer derives additional variables from the available encounter data. These include:

- Comorbidity indicators derived from selected ICD-9 diagnosis codes.
- A summary measure of Elixhauser comorbidity burden.
- A healthcare-utilization index based on outpatient, emergency, and inpatient encounters.
- Diabetes medication counts and indicators of medication dosage changes.
- An insulin-use indicator.
- An encounter-severity measure based on length of stay, medication count, and laboratory procedures.
- A categorical code derived from the recorded HbA1c result.

The transformed features are passed to a preprocessing pipeline before model training and evaluation.

### 3. Model Training

DiaVigil uses CatBoost as its classification model. The current model configuration uses 400 boosting iterations, a learning rate of 0.05, a tree depth of 6, and balanced class weights.

Model training is performed using the training partition of the development cohort. The project also retains an earlier baseline model to support comparison with the revised workflow.

### 4. Probability Calibration

The model's predicted probabilities are calibrated using isotonic regression. Calibration is fitted using a separate calibration partition and a model fitted on the training partition.

Calibration is evaluated by comparing predicted risk with observed outcome frequencies across probability bins and by calculating the Brier score and expected calibration error (ECE).

These measures assess the agreement between predicted probabilities and observed outcomes; they do not, by themselves, establish that using the predictions improves clinical decisions.

### 5. Decision-Threshold Selection

A decision threshold is selected on a separate threshold-selection partition. The current implementation evaluates classification thresholds using an asymmetric cost function that assigns a higher cost to false negatives than to false positives.

The configured costs are:

- False-positive cost (\(C_{FP}\)): 1
- False-negative cost (\(C_{FN}\)): 5

The selected threshold determines which encounters are classified as positive. Changing this threshold affects recall, precision, false-positive counts, and false-negative counts.

These costs represent modeling assumptions and should not be interpreted as validated estimates of actual clinical or financial costs.

### 6. Holdout Evaluation

The selected model, calibrator, and threshold are evaluated on the reserved holdout cohort. The evaluation includes:

- Precision-recall area under the curve (PR-AUC).
- Receiver operating characteristic area under the curve (ROC-AUC).
- Brier score and calibration analysis.
- Confusion matrix, precision, recall, and false-negative rate.
- Subgroup error analysis across available age, gender, and race categories.
- Patient-clustered bootstrap estimates of uncertainty in selected comparisons.

The baseline and revised model are compared using the same holdout cohort. This comparison makes their measured performance differences easier to interpret, but it does not establish that either model will generalize to other healthcare settings.

### 7. Resource-Allocation Simulation

A Monte Carlo simulation examines how assumed intervention efficacy and available nursing hours affect resource allocation among high-risk encounters. The workflow estimates patients reached, expected readmissions potentially avoided, and uncertainty around the simulated estimates.

A capacity-constrained allocation procedure also ranks candidates using an estimated efficiency measure based on expected readmissions potentially avoided per required nursing hour.

The results depend on the simulation assumptions, intervention-efficacy distribution, and resource constraints. They are estimates from a model, not observed clinical outcomes or demonstrated cost savings.

### 8. AI-Agent Recommendation and Audit Workflow

DiaVigil includes an AI-agent workflow that generates structured care-recommendation briefs from model-related information. The briefs contain fields for estimated readmission risk, risk tier, potential risk drivers, protective factors, and proposed care directives.

A separate audit component checks generated outputs against predefined criteria. The current demonstration includes an example that passes the checks and an example that is rejected.

These checks provide an initial assessment of output structure and specified requirements. Further evaluation is needed to establish factual accuracy, clinical appropriateness, consistency, and safety. The generated recommendations are not intended to replace clinician judgment.

### 9. Reproducibility and Artifact Management

The workflow saves model artifacts, evaluation results, simulation outputs, and research audit reports for subsequent inspection. The project also includes consistency checks to verify selected relationships between saved labels, predictions, thresholds, calibration results, and evaluation artifacts.

A reproducibility manifest records the presence, file sizes, and SHA-256 hashes of key artifacts, along with the software environment versions recorded during the audit.

These checks support reproducibility and traceability within the tested workflow. They do not independently establish the correctness of every modeling decision or guarantee identical results across all software environments.

## Architectural Workflow

```mermaid
flowchart TD
    %% Styling
    classDef data fill:#f8f9fa,stroke:#6c757d,color:#212529
    classDef process fill:#e7f5ff,stroke:#1971c2,color:#1864ab
    classDef model fill:#e6fcf5,stroke:#0ca678,color:#087f5b
    classDef audit fill:#fff3bf,stroke:#f08c00,color:#a65d03
    classDef agent fill:#f3f0ff,stroke:#7048e8,color:#5f3dc4
    classDef gate fill:#ffe3e3,stroke:#e03131,color:#c92a2a
    classDef ops fill:#ebfbee,stroke:#2b8a3e,color:#2b8a3e

    subgraph DATA["1. Cohort Preparation"]
        direction TB
        A["Clinical Encounter Dataset"]:::data
        B["Patient-Grouped Data Partitions"]:::process
        C["Development Data"]:::process
        D["Untouched Holdout Cohort"]:::audit
        A --> B
        B --> C
        B --> D
    end

    subgraph ML["2. Model Training and Evaluation"]
        direction TB
        E["Clinical Feature Engineering"]:::process
        F["CatBoost Classifier"]:::model
        G["Isotonic Probability Calibration"]:::model
        H["Cost-Based Threshold Selection"]:::model
        I["Freeze Pipeline, Model and Threshold"]:::model
        J["Holdout Performance and Calibration"]:::audit
        K["Subgroup Error Analysis"]:::audit
        L["Paired Patient-Clustered Bootstrap"]:::audit

        C --> E --> F --> G --> H --> I
        I --> J
        D --> J
        J --> K
        J --> L
    end

    subgraph AGENTS["3. Three-Agent Recommendation Workflow"]
        direction TB
        M["Agent 1: Structured Data Retriever<br/>Retrieve encounter information"]:::agent
        N["Agent 2: ML Inference and TreeSHAP<br/>Estimate risk and identify feature contributions"]:::agent
        O["Agent 3: Recommendation Synthesis<br/>Draft structured care recommendations<br/>using a Pydantic contract"]:::agent
        P{"Deterministic Critic Gate<br/>Check structure and attribution grounding"}:::gate
        Q["Accepted Triage Brief<br/>Critic checks passed"]:::model
        R["Rejected Draft<br/>Record audit notes and revise"]:::gate

        M --> N --> O --> P
        P -->|Checks passed| Q
        P -->|Checks failed| R
        R -.->|Retry| O
    end

    I -.->|Saved model artifacts| N
    A -.->|Encounter records| M

    subgraph RESOURCE["4. Resource-Allocation Simulation"]
        direction TB
        S["Risk-Scored Cohort and Staffing Assumptions"]:::ops
        T["Monte Carlo Simulation"]:::ops
        U["Capacity-Constrained Allocation"]:::ops
        V["Estimated Outcomes and Uncertainty"]:::ops

        S --> T --> U --> V
    end

    I -.-> S
    Q -.-> S

    subgraph REPRO["5. Audit and Reproducibility"]
        direction TB
        W["Saved Predictions, Thresholds<br/>and Evaluation Results"]:::data
        X["Artifact Consistency Checks<br/>and Reproducibility Manifest"]:::audit
        W --> X
    end

    J -.-> W
    K -.-> W
    L -.-> W
    P -.-> W
```

## Data Provenance & Access

The underlying cohort is derived from the **Diabetes 130-US Hospitals (1999–2008)** clinical database, encompassing 101,766 inpatient encounters across 130 hospital facilities:

* **Canonical Academic Archive:** [UCI Machine Learning Repository (DOI: 10.24432/C5230J)](https://doi.org/10.24432/C5230J)
* **Foundational Publication:** [Strack et al. (2014), *BioMed Research International* (DOI: 10.1155/2014/781670)](https://doi.org/10.1155/2014/781670)
* **Distribution Mirror:** [Kaggle Dataset Mirror (Brandão, Ph.D.)](https://www.kaggle.com/datasets/brandao/diabetes)

For local replication, place the source files (`diabetic_data.csv` and `IDs_mapping.csv`) into the `data/raw/` directory prior to running `01_diavigil_cohort_filtering_and_group_split.ipynb`.

---

## Experimental Results

### Evaluation Setup

The baseline and revised models were evaluated on the same untouched holdout cohort of **19,875 encounters**. The cohort was kept separate from model training, probability calibration, and decision-threshold selection. Both models were evaluated using their saved predictions and operating thresholds.

The evaluation examined discrimination, probability error, classification performance, and the trade-off between false-positive and false-negative predictions.

### Holdout Performance

| Metric | Original baseline | Revised model |
|---|---:|---:|
| Holdout encounters | 19,875 | 19,875 |
| PR-AUC | 0.2036 | 0.2013 |
| ROC-AUC | 0.6641 | 0.6578 |
| Brier score | 0.0962 | 0.0961 |
| Precision | 23.81% | 22.42% |
| Recall | 29.15% | 31.96% |
| False-positive rate | 11.85% | 14.05% |
| False positives | 2,089 | 2,478 |
| False negatives | 1,587 | 1,524 |
| False-negative rate | 70.85% | 68.04% |

\*False-positive rate (FPR) = FP / (FP + TN), representing the proportion of actual negative cases incorrectly classified as positive.
The revised model used a selected decision threshold of approximately **0.1620** and an assumed classification cost of 1 for a false positive and 5 for a false negative.

### Interpretation of the Results

The revised model increased recall from 29.15% to 31.96% and reduced false negatives from 1,587 to 1,524. This represents 63 fewer missed positive cases in the evaluated cohort.

However, false positives increased from 2,089 to 2,478, while precision decreased from 23.81% to 22.42%. ROC-AUC also decreased from 0.6641 to 0.6578, and PR-AUC decreased from 0.2036 to 0.2013. The Brier score changed only slightly, from 0.096154 to 0.096093.

These results indicate a change in the operating-point trade-off rather than a consistent improvement across all metrics.

### Paired Patient-Clustered Bootstrap Analysis

A paired patient-clustered bootstrap with 1,000 valid replicates was used to estimate uncertainty in the differences between the two models. Encounters belonging to the same patient were sampled together to account for repeated encounters within patients.

| Metric difference (revised − baseline) | Estimate | 95% bootstrap confidence interval |
|---|---:|---:|
| PR-AUC | −0.00227 | [−0.00807, 0.00362] |
| ROC-AUC | −0.00627 | [−0.01091, −0.00136] |
| Brier score | −0.000061 | [−0.000411, 0.000276] |
| Recall | +0.02813 | [+0.01543, +0.04162] |
| False-negative rate | −0.02813 | [−0.04162, −0.01543] |
| Cost per 1,000 encounters* | +3.72 | [−4.50, +11.75] |

\*Calculated using the project's assumed false-positive and false-negative costs.

The paired analysis indicates that the 95% bootstrap confidence intervals for the recall increase and false-negative-rate reduction excluded zero. The ROC-AUC decrease was also distinguishable from zero. The confidence intervals for PR-AUC, Brier score, and estimated cost per 1,000 encounters included zero.

These comparisons do not establish that the revised model is universally preferable. Model selection depends on the intended use, the relative consequences of each error type, and the validity of the assumed cost structure.

### Calibration and Subgroup Evaluation

The audit also examined agreement between predicted probabilities and observed outcome frequencies, as well as false-negative rates across available age, gender, and race categories. Patient-clustered bootstrap analysis was used to quantify uncertainty in selected subgroup comparisons.

Subgroup estimates require particular caution when positive-outcome counts are small. Differences in observed error rates do not, by themselves, establish the cause of a disparity or demonstrate that a model is fair or unfair in every relevant sense.

Calibration and subgroup analyses are reported as evaluation results for this dataset. They do not establish clinical usefulness, fairness in other populations, or improved patient outcomes.

### Summary

On the evaluated holdout cohort, the revised model traded some discrimination and precision for higher recall and fewer false negatives. *The paired bootstrap confidence interval for the recall difference excluded zero*, but did not establish a reduction in the project's estimated classification cost.

DiaVigil therefore treats model performance as a set of related measures rather than relying on a single score. Further validation on an independent cohort, review of the assumed error costs, and assessment of clinical and operational requirements are necessary before drawing conclusions about clinical use.

---

## Calibration and Subgroup Fairness Analysis

### Probability Calibration

Probability calibration examines whether predicted readmission risks agree with observed outcome frequencies. For example, among encounters assigned a predicted risk of 10%, the observed outcome rate should be close to 10% if the probabilities are well calibrated.

DiaVigil evaluates calibration using an equal-frequency, 10-bin analysis. Within each bin, the mean predicted risk is compared with the observed outcome rate. Expected calibration error (ECE) summarizes the weighted average absolute difference between these quantities.

On the holdout cohort of 19,875 encounters, the revised model achieved an ECE of **0.00656**. This corresponds to a weighted average absolute difference of approximately 0.656 percentage points across the evaluated bins.

Calibration describes the agreement between predicted probabilities and observed outcomes. It does not establish that the model identifies the patients who would benefit most from intervention or that using its predictions improves clinical outcomes.

### False-Negative Rate by Age Group

The subgroup audit examines false-negative rates (FNRs) across age groups. FNR measures the proportion of actual positive outcomes that the model fails to identify:

`FNR = False Negatives / (False Negatives + True Positives)`

The audit uses a minimum of 50 positive outcomes when calculating the observed FNR gap between groups. The following table presents the age groups that meet this requirement.

| Age group | Positive outcomes | False negatives | FNR |
|---|---:|---:|---:|
| 30–40 | 89 | 51 | 57.30% |
| 40–50 | 191 | 133 | 69.63% |
| 50–60 | 341 | 220 | 64.52% |
| 60–70 | 486 | 343 | 70.58% |
| 70–80 | 603 | 407 | 67.50% |
| 80–90 | 413 | 286 | 69.25% |
| 90–100 | 58 | 48 | 82.76% |

Among groups meeting the minimum-count requirement, the observed FNR ranged from 57.30% for ages 30–40 to 82.76% for ages 90–100, a difference of 25.46 percentage points.

This difference identifies an area for further investigation. The oldest group's estimate is based on 58 positive outcomes, so uncertainty remains important. Groups with fewer than 50 positive outcomes were excluded from this particular gap calculation because their estimates may be less stable.

### False-Negative Rate by Gender

The audit compared observed FNRs across the two reported gender categories.

| Group | Positive outcomes | False negatives | FNR |
|---|---:|---:|---:|
| Female | 1,188 | 802 | 67.51% |
| Male | 1,052 | 722 | 68.63% |

The observed FNR was 1.12 percentage points higher for the male group. This is a descriptive difference within the evaluated cohort. It does not establish equivalent model performance across all populations or prove that the model is fair across gender categories.

### False-Negative Rate by Race

The audit also examined FNRs across the available race categories. The primary gap calculation included only groups with at least 50 positive outcomes.

| Group | Positive outcomes | False negatives | FNR |
|---|---:|---:|---:|
| African American | 417 | 285 | 68.35% |
| Caucasian | 1,731 | 1,177 | 68.00% |

The observed difference between these groups was approximately 0.35 percentage points, with a higher FNR for the African American group.

Other race categories had fewer than 50 positive outcomes and were excluded from this minimum-count comparison. Their exclusion from the gap calculation does not mean that their results are unimportant; rather, estimates based on small numbers of positive outcomes require particular caution.

The small observed difference between the two groups included in the comparison does not establish fairness across race categories. It also does not identify the causes of any differences in model performance.

### Limitations and Interpretation

The calibration and subgroup analyses have several limitations:

- **Dataset dependence:** The results describe the evaluated holdout cohort and may differ in other healthcare settings.
- **Small subgroup counts:** FNR estimates based on relatively few positive outcomes may be uncertain.
- **Metric selection:** Similar FNRs do not establish fairness across other error measures, demographic groups, or clinical outcomes.
- **Calibration versus clinical usefulness:** Agreement between predicted risks and observed outcome frequencies does not demonstrate that the predictions improve care.
- **No causal conclusions:** Differences in observed error rates do not, by themselves, explain why the differences occur.

DiaVigil uses these analyses to document model behavior, identify subgroup differences for further investigation, and report calibration and error patterns in the available data. Independent validation, additional uncertainty analysis, and clinical review would be required before drawing conclusions about clinical use.

---

## Resource-Allocation Simulation

DiaVigil includes a resource-allocation simulation to examine how different nursing-hour budgets may affect patient outreach and the estimated number of hospital readmissions potentially averted. The simulation uses model-derived risk estimates, assumed intervention effectiveness, and staffing constraints. Its outputs are estimates under the specified assumptions, not measurements of actual clinical outcomes.

### Simulation Setup

The workflow evaluates allocation scenarios with different nursing-hour budgets and estimates the number of patients reached, nursing hours utilized, and readmissions potentially averted. It also examines the marginal estimated yield associated with additional nursing hours.

The saved simulation artifact contains three outputs:

- **Allocation summary:** Compares selected nursing-hour budgets and reports patients reached, hours utilized, expected readmissions averted, true readmissions caught, and cohort precision.
- **Capacity frontier:** Evaluates 25 nursing-hour budgets, ranging from 200 to 5,000 hours, to examine how patient outreach and estimated readmissions averted change as capacity increases.
- **Monte Carlo bounds:** Records the simulated mean estimate and its associated uncertainty bounds under the configured assumptions.

### Selected Resource-Allocation Results

| Nurse-hour budget | Patients reached | Hours utilized | Expected readmissions averted | True readmissions caught | Cohort precision |
|---:|---:|---:|---:|---:|---:|
| 500 | 176 | 499.8 | 21.35 | 64 | 36.36% |
| 1,500 | 465 | 1,498.0 | 52.48 | 131 | 28.17% |
| 3,000 | 879 | 2,997.5 | 91.15 | 224 | 25.48% |

### Capacity Frontier and Marginal Yield

The capacity-frontier analysis estimates how outreach and potential readmission prevention change across a wider range of staffing budgets. In the saved results, estimated readmissions averted increase from approximately 9.91 at 200 nurse-hours to 135.08 at 5,000 nurse-hours.

The estimated marginal yield declines as the budget increases. For example, the saved output reports approximately 3.91 additional estimated readmissions averted per 100 nurse-hours at the 400-hour point, compared with approximately 2.07 at the 5,000-hour point. This pattern suggests diminishing marginal yield within the simulation.

These estimates describe the behavior of the configured allocation model. They do not establish that increasing nursing hours in a real hospital would produce the same outcomes.

### Monte Carlo Results

The saved simulation artifact reports the following summary:

- **Mean estimated readmissions averted:** 231.55
- **Lower uncertainty bound:** 228.06
- **Upper uncertainty bound:** 235.12

These bounds reflect the simulation's assumptions and implementation. They should not be interpreted as a clinical confidence interval or as evidence of realized reductions in readmissions.

### Limitations

The resource-allocation results depend on the assumptions used to estimate intervention effectiveness, the risk model's predictions, and the available staffing capacity. The simulation does not independently demonstrate that an intervention prevents readmissions, that the allocation strategy improves patient outcomes, or that a hospital would achieve financial savings.

Further evaluation would require defensible intervention-effectiveness estimates, operational validation of staffing assumptions, and evidence from an appropriate external or prospective study.

### Summary  
The simulation provides a way to compare hypothetical staffing budgets and examine the relationship between outreach capacity and estimated readmission prevention. It supports resource-allocation analysis with the DiaVigil research prototype but does not establish clinical effectiveness or real world- cost savings.

---

## AI-Agent Recommendation and Audit Workflow

DiaVigil includes an AI-agent workflow that generates structured care-recommendation briefs using encounter data, readmission-risk predictions, and model-derived feature attributions. The workflow is intended to demonstrate how model outputs can be organized into recommendations and checked against predefined criteria.

### Agent 1: Encounter Data Retrieval

The first component identifies an encounter for review and supplies the information needed by the subsequent workflow. The demonstration selects a high-risk encounter from the evaluation data, falling back to the encounter with the highest predicted risk when no encounter exceeds the configured threshold.

The available implementation uses saved evaluation predictions when the corresponding audit artifact exists. Otherwise, it computes predictions from the saved feature pipeline and probability calibrator. The encounter is then passed to the agent workflow for further processing.

### Agent 2: Risk Inference and Feature Attribution

The second component obtains the encounter's estimated readmission risk and identifies features contributing to the model's prediction. The workflow uses the saved feature-engineering pipeline, probability calibrator, and decision threshold. Local TreeSHAP attributions are used to identify risk-related and protective features.

These attributions provide information about the model's prediction; they do not establish causal relationships between individual features and readmission outcomes.

### Agent 3: Structured Recommendation Generation

The third component generates a structured brief containing the encounter identifier, estimated readmission risk, risk category, contributing and protective features, and proposed care directives.

The brief follows a defined data structure so that its fields can be inspected and evaluated by the audit component. The generated directives are proposed recommendations, not independently validated clinical instructions.

### Critic Audit and Pass/Reject Evaluation

A separate critic evaluates whether the generated directives are supported by the model-derived risk information. The saved demonstration contains two examples:

- **Accepted example:** The audit passed, with three care directives reported as grounded in verified risk drivers.
- **Rejected example:** The audit failed because the directives referenced clinical factors that were unsupported by the local TreeSHAP attributions. The saved audit notes indicate that regeneration was required.

These examples demonstrate the intended pass/reject behavior for the tested cases. They do not establish that every unsupported statement will be detected or that rejected recommendations are automatically regenerated.

### Limitations

The agent workflow is a research demonstration. Passing the critic's checks does not establish that a recommendation is clinically correct, safe, complete, or appropriate for a particular patient. TreeSHAP attributions explain model behavior rather than clinical causation, and the audit depends on the criteria implemented in the critic.

Further evaluation would be needed to measure the reliability of recommendation generation, the critic's ability to detect unsupported claims, and the effectiveness of any regeneration process. Clinical review and human oversight would be required before considering use in patient care.

### Summary

DiaVigil demonstrates a workflow that connects encounter-level risk estimation, local feature attribution, structured recommendation generation, and a pass/reject audit. The saved examples provide evidence of both accepted and rejected outputs, while the system's clinical accuracy and safety remain unvalidated.

---
## Clinical Data Dictionary

The DiaVigil pipeline ingests, filters, and transforms 50 raw attributes into structured clinical representations. Below is the specification of all primary cohort identifiers, administrative features, clinical biomarkers, and engineered ontologies.

### 1. Cohort Identifiers & Ground Truth Target

| Feature Name | Source / Stage | Data Type | Permissible Values / Domain | Clinical & Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `encounter_id` | Raw (`diabetic_data.csv`) | Integer | Unique identifier | Primary key for distinct hospital inpatient admissions. |
| `patient_nbr` | Raw (`diabetic_data.csv`) | Integer | Unique identifier | **Group Splitting Key:** Used by `StratifiedGroupKFold` to isolate patient longitudinal trajectories and prevent identity leakage across train/test splits. |
| `readmitted` | Raw (`diabetic_data.csv`) | String | `<30`, `>30`, `NO` | Raw multi-class administrative discharge readmission indicator. |
| `readmitted_30d` | Notebook `01` | Binary (`int64`) | `1` ($<30$ days), `0` (else) | **Primary Target:** Binary label indicating unplanned readmission within the 30-day CMS HRRP penalty window. Baseline prevalence: **11.28%**. |

---

### 2. Patient Demographics & Fairness Audit Attributes

| Feature Name | Source / Stage | Data Type | Permissible Values / Domain | Clinical & Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Raw (`diabetic_data.csv`) | Categorical | `[0-10)`, `[10-20)`, ..., `[90-100)` | 10-year age bins. Evaluated in Notebook `04` to audit algorithmic disparity (revealed a **0.5056 FNR gap**). |
| `gender` | Raw (`diabetic_data.csv`) | Categorical | `Female`, `Male` | Biological sex indicator (records with `Unknown/Invalid` dropped during initial filtering). Audited for sex parity (**0.84% FNR delta**). |
| `race` | Raw (`diabetic_data.csv`) | Categorical | `Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other`, `?` | Self-reported racial demographic identifier. Missing values preserved as explicit category `?`. |

---

### 3. Healthcare Utilization & Inpatient Dynamics

| Feature Name | Source / Stage | Data Type | Permissible Values / Domain | Clinical & Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `time_in_hospital` | Raw (`diabetic_data.csv`) | Integer | $1 - 14$ days | Total length of stay (LOS) during index hospital admission. |
| `num_lab_procedures` | Raw (`diabetic_data.csv`) | Integer | $\ge 1$ | Total count of diagnostic laboratory tests performed during encounter. |
| `num_procedures` | Raw (`diabetic_data.csv`) | Integer | $\ge 0$ | Total count of non-lab operational/clinical procedures performed. |
| `num_medications` | Raw (`diabetic_data.csv`) | Integer | $\ge 1$ | Total distinct pharmaceutical agents administered during hospitalization. |
| `number_outpatient` | Raw (`diabetic_data.csv`) | Integer | $\ge 0$ | Outpatient clinic visits in the 12-month period preceding the index encounter. |
| `number_emergency` | Raw (`diabetic_data.csv`) | Integer | $\ge 0$ | Emergency department encounters in the 12-month period preceding admission. |
| `number_inpatient` | Raw (`diabetic_data.csv`) | Integer | $\ge 0$ | Inpatient admissions in the preceding 12 months (identified as a top TreeSHAP risk driver). |
| `number_diagnoses` | Raw (`diabetic_data.csv`) | Integer | $1 - 16$ | Total clinical diagnostic billing codes entered into the encounter chart. |

---

### 4. Diagnostic Ontologies: Quan's Elixhauser Comorbidity Vectors

Extracted in Notebook `02` using vectorized regular expressions applied across primary and secondary diagnostic fields (`diag_1`, `diag_2`, `diag_3`):

| Feature Name | Clinical Entity | ICD-9 Mapping Logic | Architectural Significance |
| :--- | :--- | :--- | :--- |
| `elix_chf` | Congestive Heart Failure | `398.91`, `402.01-402.91`, `404.01-404.93`, `428.*` | Indicator of underlying cardiac compromise. |
| `elix_cardiac_arrhythmias` | Cardiac Arrhythmias | `426.0-426.13`, `426.7`, `427.0-427.9` | Electrical conduction dysfunction marker. |
| `elix_valvular_disease` | Valvular Disease | `093.2`, `394.0-397.9`, `424.0-424.99` | Structural cardiac pathology indicator. |
| `elix_pulmonary_circ` | Pulmonary Circulation Disorders | `415.0-415.19`, `416.0-417.9` | Pulmonary arterial hypertension index. |
| `elix_peripheral_vascular` | Peripheral Vascular Disease | `093.0`, `437.3`, `440.0-441.9`, `443.1-443.9` | Macrovascular complication risk vector. |
| `elix_hypertension` | Hypertension (Combined) | `401.1-405.99` | Essential and secondary hypertensive disease. |
| `elix_paralysis` | Paralysis / Plegia | `334.1`, `342.0-344.9`, `438.2-438.5` | Neurological immobility comorbidity marker. |
| `elix_neuro_other` | Other Neurological Disorders | `331.9-333.9`, `340.*`, `345.0-345.9` | Cognitive/degenerative neurological conditions. |
| `elix_chronic_pulmonary` | Chronic Pulmonary Disease | `416.8-416.9`, `490.*-505.*`, `506.4` | COPD, asthma, and chronic restrictive lung disease. |
| `elix_diabetes_uncomplicated` | Diabetes without Complication | `250.00 - 250.33` | Baseline diabetic endocrine pathology. |
| `elix_diabetes_complicated` | Diabetes with End-Organ Damage | `250.40 - 250.93` | Nephropathy, retinopathy, and neuropathy (top TreeSHAP risk driver). |
| `elix_renal_failure` | Renal Failure / CKD | `403.01-404.93`, `585.0-586.*`, `V42.0` | Chronic kidney disease and end-stage renal illness. |
| `total_elixhauser_burden` | Cumulative Comorbidity Index | $\sum (\text{Elixhauser Binary Flags})$ | Multi-morbid disease burden score ($0 - 12$). |

---

### 5. Glycemic Biomarkers & Pharmacotherapy Dynamics

| Feature Name | Source / Stage | Data Type | Permissible Values / Domain | Clinical & Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `A1Cresult` | Raw (`diabetic_data.csv`) | Categorical | `None`, `Norm`, `>7`, `>8` | Categorical glycated hemoglobin ($HbA1c$) lab monitoring indicator. |
| `max_glu_serum` | Raw (`diabetic_data.csv`) | Categorical | `None`, `Norm`, `>200`, `>300` | Point-of-care or laboratory glucose serum reading. |
| `insulin` | Raw (`diabetic_data.csv`) | Categorical | `No`, `Steady`, `Up`, `Down` | Subcutaneous insulin administration and acute hospital titration pattern. |
| `change` | Raw (`diabetic_data.csv`) | Categorical | `Ch`, `No` | Indicates whether any diabetic medication regimen was modified during admission. |
| `diabetesMed` | Raw (`diabetic_data.csv`) | Categorical | `Yes`, `No` | Indicates whether patient was discharged on any antidiabetic pharmacotherapy. |
| `active_med_count` | Notebook `02` | Integer | $0 - 23$ | Count of distinct concurrent oral and injectable antidiabetic agents prescribed. |

---

## Methodological Contributions & Empirical Findings

### 1. Leak-Free Validation & Probability Calibration

Tree-based ensembles trained on imbalanced clinical cohorts produce distorted probability outputs. DiaVigil calibrates CatBoost predictions using Isotonic Regression fitted strictly on inner validation folds, establishing calibrated risk probabilities prior to evaluation on the untouched audit test cohort:

| Metric | Uncalibrated CatBoost | Calibrated DiaVigil Framework | Improvement / Status |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | 0.1945 | **0.2036** | +4.68% gain out-of-sample |
| **ROC-AUC** | 0.6480 | **0.6641** | Out-of-sample discrimination |
| **Brier Score Loss** | 0.2121 | **0.0961** | **-54.69% error reduction** |
| **Expected Calibration Error (ECE)** | 0.3289 | **0.0000** | **100.00% calibration alignment** |
| **Operating Threshold ($\tau^*$)** | 0.5000 (Arbitrary) | **0.1620 (Cost-Utility Derived)** | Loss-weighted ($C_{FN} = 5 \times C_{FP}$) |
| **Holdout Test Accuracy** | 88.70% (Majority Baseline) | **81.50%** | Balanced clinical stratification |

### 2. Algorithmic Fairness & Subgroup Disparity Audit

Auditing the frozen threshold ($\tau^* = 0.1620$) across 19,875 out-of-sample test encounters revealed substantial disparities in clinical miss rates across age brackets, while demonstrating balanced performance across biological sex:

#### Age Bracket Disparity (Clinical Vulnerability Risk)
* **Age False Negative Rate Gap:** **0.5056** (Max FNR: 1.0000 in pediatric/adolescent vs. Min FNR: 0.4944 in ages 30–40).
* **Geriatric Risk Shift:** As patient age increases, the false-negative rate escalates:
  * **Ages 30–40:** FNR = **0.4944** (Sensitivity = 50.56%)
  * **Ages 50–60:** FNR = **0.6569** (Sensitivity = 34.31%)
  * **Ages 70–80:** FNR = **0.7131** (Sensitivity = 28.69%)
  * **Ages 80–90:** FNR = **0.7627** (Sensitivity = 23.73%)
  * **Ages 90–100:** FNR = **0.8621** (Sensitivity = 13.79%)
* *Academic Takeaway:* A globally uniform operating cutoff shifts disproportionate false-negative misclassification risk onto elderly patients who account for the largest proportion of total hospital admissions.

#### Biological Sex Parity
* **Female:** 10,792 encounters | Sensitivity = **0.2955** | False Negative Rate = **0.7045** | FPR = 0.1210
* **Male:** 9,083 encounters | Sensitivity = **0.2871** | False Negative Rate = **0.7129** | FPR = 0.1154
* *Cross-Sex FNR Difference:* **0.0084 (0.84%)**, demonstrating performance parity across sex.

### 3. Multi-Agent Decision Support & Faithfulness Critic

To prevent generative clinical hallucinations, DiaVigil structures inference through three agents governed by Pydantic schemas and a deterministic **Critic Gate**:
```text
[Encounter ID] ──> Agent 1: Parameterized SQL Retriever
                          │
                          ▼
                   Agent 2: ML Inference & Local TreeSHAP
                          │ (Calibrated Probability + Top Signed SHAP Drivers)
                          ▼
                   Agent 3: Synthesis Engine (Pydantic Structured Contract)
                          │ (Draft Clinical Care Directives)
                          ▼
                   Critic Gate: Local SHAP Attribution Audit
                     ├── Grounded Directives? ──> PASSED ──> Verified Triage Brief
                     └── Hallucinated Claims? ──> REJECTED ──> Force Regeneration
```
#### Empirical Demonstration (Encounter #1157574)
* **Model Parameters:** Calibrated Risk = **17.79%** (Tier: High Risk - Enhanced Follow-up).
* **Local TreeSHAP Drivers:** `elix_diabetes_complicated` (+0.091), `total_elixhauser_burden` (+0.084), `number_inpatient` (+0.076).
* **Scenario 1 (Grounded Synthesis):** Directives citing verified SHAP features pass the Critic Gate with verification flags (`Passed = True`).
* **Scenario 2 (Hallucinated Synthesis):** Injecting ungrounded claims (e.g., severe hypertension crisis and potassium abnormalities) triggers an immediate audit rejection (`Passed = False`), preventing misleading recommendations from reaching care teams.

### 4. Dynamic Stochastic Simulation & Capacity Optimization

Moving beyond static financial spreadsheets, DiaVigil evaluates care coordination interventions using dynamic operational modeling:

#### Monte Carlo Uncertainty Quantification (1,000 Trials)
* **High-Risk Flagged Cohort:** 2,742 encounters (13.80% of test population).
* **Efficacy Distribution:** Sampled stochastically from $\text{Beta}(\alpha=3.5, \beta=6.5)$ ($\mu \approx 35\%$).
* **Mean Expected Readmissions Prevented:** **231.55 encounters**
* **95% Empirical Credible Interval:** **[228.06, 235.12]**

#### Knapsack Follow-up Staffing Frontier
Using an efficiency ratio ($\mathbb{E}[\text{Readmissions Averted}] / \text{Nurse Contact Hours Required}$), DiaVigil solves capacity-constrained care allocation across operational staffing tiers:

| Monthly Staffing Budget | Patients Reached | Contact Hours Used | Exp. Readmissions Averted | Actual Readmissions Caught | Cohort Precision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **500 Nurse Hours** | 176 | 499.8 | 21.35 | 64 | **36.36%** |
| **1,500 Nurse Hours** | 465 | 1,498.0 | 52.48 | 131 | **28.17%** |
| **3,000 Nurse Hours** | 879 | 2,997.5 | 91.15 | 224 | **25.48%** |

* **Staffing Frontier Dynamics:** Marginal yield drops from **3.91 averted readmissions per 100 hours** at low budgets to **2.61 averted readmissions per 100 hours** past 2,200 contact hours, providing an empirical resource allocation curve for hospital administrators.

---

## Repository Structure & Reproducibility Guide

The codebase is organized into modular Google Colab notebooks that execute sequentially without intermediate data loss:

```text
├── notebooks/
│   ├── 01_diavigil_cohort_filtering_and_group_split.ipynb
│   │   └── CMS HRRP filtering, target definition, and leak-free StratifiedGroupKFold partitioning.
│   │
│   ├── 02_diavigil_clinical_feature_engineering_pipeline.ipynb
│   │   └── Elixhauser comorbidity mappings, utilization indices, and ColumnTransformer assembly.
│   │
│   ├── 03_diavigil_nested_cv_and_probability_calibration.ipynb
│   │   └── Inner group cross-validation, CatBoost training, Isotonic calibration, and tau* optimization.
│   │
│   ├── 04_diavigil_algorithmic_fairness_and_subgroup_audit.ipynb
│   │   └── Frozen out-of-sample evaluation on 19,875 encounters with Age, Gender, and Race audits.
│   │
│   ├── 05_diavigil_agentic_triage_and_faithfulness_critic.ipynb
│   │   └── Local TreeSHAP attribution, Pydantic contracts, and automated Critic Gate verification.
│   │
│   └── 06_diavigil_dynamic_resource_allocation_simulation.ipynb
│       └── 1,000-trial Monte Carlo simulation and bounded knapsack capacity frontiers.
│
├── data/
│   ├── raw/                  # UCI 130-US Hospitals Diabetes dataset (immutable)
│   └── processed/            # Parquet cohorts (dev/audit test) and serialized joblib artifacts
│
├── LICENSE                   # MIT Open-Source Academic License
└── README.md                 # System specification and empirical research findings
```
---
## Quickstart & Execution Protocol

### 1. Repository Setup & Dependencies
Clone the repository and install the verified core dependencies for local environments:

```bash
git clone https://github.com/nmemranhussain/DiaVigil.git
cd DiaVigil
pip install pandas numpy scikit-learn catboost optuna netcal shap pydantic joblib pyarrow
```
### 2. Sequential Pipeline Replication

Execute the notebooks in strict numerical order to preserve feature artifact states and prevent data leakage:

1. **`notebooks/01_diavigil_cohort_filtering_and_group_split.ipynb`**
   - Ingests raw UCI 130-US Hospitals data, excludes CMS HRRP discharge dispositions (codes 11, 19, 20, 21), and creates group-partitioned Parquet files (`dev_cohort.parquet` and `audit_test_cohort.parquet`) clustered on `patient_nbr`.
2. **`notebooks/02_diavigil_clinical_feature_engineering_pipeline.ipynb`**
   - Extracts Quan's Elixhauser Comorbidity Index patterns via vectorized regex, constructs pharmacotherapy complexity metrics, and serializes `feature_engineering_pipeline.joblib`.
3. **`notebooks/03_diavigil_nested_cv_and_probability_calibration.ipynb`**
   - Fits `CatBoostClassifier` via nested group cross-validation, executes post-hoc Isotonic Calibration (reducing ECE to 0.0000), optimizes $\tau^* = 0.1620$ via asymmetric cost-loss matrices, and serializes `model_calibration_artifacts.joblib`.
4. **`notebooks/04_diavigil_algorithmic_fairness_and_subgroup_audit.ipynb`**
   - Performs frozen evaluation on the 19,875-encounter holdout test partition, compiling demographic audit tables and quantifying the 0.5056 False Negative Rate gap across age bands.
5. **`notebooks/05_diavigil_agentic_triage_and_faithfulness_critic.ipynb`**
   - Initializes the 3-agent clinical triage pipeline, registers Pydantic schemas, and validates the deterministic TreeSHAP Critic Gate against grounded vs. hallucinated clinical summaries.
6. **`notebooks/06_diavigil_dynamic_resource_allocation_simulation.ipynb`**
   - Executes 1,000-trial Monte Carlo stochastic efficacy sampling and runs bounded knapsack optimization to generate hospital nurse-hour allocation frontiers.

---

## Technical & Data Limitations
- **Historical EHR Distribution:** The UCI Diabetes dataset spans hospital encounters from 1999 to 2008. Practice patterns, clinical guidelines (such as SGLT2i/GLP-1 RA adoption), and billing coding practices have evolved significantly since this cohort was collected.
- **Censoring and Out-of-Hospital Mortality:** The dataset tracks inpatient readmissions to participating clinical centers but does not record out-of-hospital mortality or admissions to non-participating hospital systems, introducing potential informative right-censoring.
- **Unmeasured Social Determinants of Health (SDOH):** Clinical readmission risk is heavily driven by post-discharge environmental factors (e.g., medication affordability, food security, transportation access, and health literacy) that are not captured in structured administrative discharge summaries.
- **Coarse Glycemic Monitoring:** Glycated hemoglobin (`A1Cresult`) is recorded categorically (`None`, `Norm`, `>7`, `>8`) rather than as continuous clinical laboratory values, limiting the model's ability to capture granular glycemic volatility.

---

## AI Disclosure & Academic Integrity

In alignment with emerging guidelines for ethical AI in scientific computing and biomedical research:
* **System Architecture:** Large Language Models utilized within DiaVigil are strictly restricted to downstream decision support and summary synthesis; all risk scores, probability calibrations, and feature attributions are computed deterministically via Scikit-Learn, CatBoost, and TreeSHAP.
* **Development Assistance:** Generative AI tools (e.g., Google Gemini) were utilized for code refactoring, script boilerplate generation, and Markdown syntax formatting.
* **Intellectual Ownership:** All experimental designs, cohort partitioning logic (`StratifiedGroupKFold`), comorbidity ontology mappings, calibration experiments, fairness audit analyses, and research hypotheses were conceptualized, directed, and verified by the author.

---

## Future Research Directions

1. **Multiobjective Demographic-Specific Threshold Optimization ($\tau_g^*$):**
   - Formulate a constrained optimization framework (e.g., Pareto-frontier optimization) to derive subgroup-specific decision cutoffs ($\tau_{\text{age}}^*$). The objective is to enforce Equal Opportunity by equalizing False Negative Rates between younger and geriatric cohorts without exceeding hospital care-management bed and staffing capacities.
2. **Longitudinal State-Space Modeling on EHR Trajectories:**
   - Transition from cross-sectional static encounter aggregations to continuous-time, structured state-space models (e.g., Mamba / S4) to model irregular temporal trajectories of inpatient glucose readings, vital sign instability, and medication titrations over sequential admissions.
3. **Formal Verification of Agentic Clinical Synthesis:**
   - Extend the deterministic Critic Gate beyond local TreeSHAP attribution matching to formal constraint-satisfaction solvers and automated theorem proving (e.g., SMT solvers). This will mathematically verify that LLM-generated clinical triage plans comply strictly with established clinical practice guidelines (e.g., ADA Standards of Medical Care in Diabetes).
4. **Translational Scalability & FHIR Interoperability:**
   - Transition the parameterized retriever into an asynchronous HL7 FHIR (Fast Healthcare Interoperability Resources) endpoint capable of subscribing to discharge notifications (`Encounter/discharged`), streaming inference in real time, and depositing calibrated risk estimates directly back into clinical EHR workflows.

---

## Citation & Attribution

If you utilize the DiaVigil architecture or build upon these findings, please cite both the project repository and the primary data publication:

```bibtex
@misc{hussain2026diavigil,
  author = {Hussain, N M Emran},
  title = {Project DiaVigil: Calibrated Machine Learning, Algorithmic Fairness Auditing, and Grounded Multi-Agent Decision Support for Diabetic Readmission Risk},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/nmemranhussain/DiaVigil}
}

@article{strack2014diabetes,
  author = {Strack, Beata and DeShazo, Jonathan P. and Gennings, Chris and Olmo, Juan L. and Ventura, Sebasti{\'a}n and Cios, Krzysztof J. and Clore, John N.},
  title = {Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records},
  journal = {BioMed Research International},
  volume = {2014},
  pages = {781670},
  year = {2014},
  publisher = {Hindawi},
  doi = {10.1155/2014/781670}
}
```
