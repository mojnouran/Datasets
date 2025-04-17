# Open Datasets for Reproducible Experiments
This repository hosts every dataset underpinning my current (and in‑review) papers so reviewers and readers can grab the exact files behind each result table.

---

## At‑a‑Glance Statistics

| Dataset | Source | Samples | Raw feat. | Used feat. |
|---------|--------|---------|-----------|------------|
| ECG‑3 | ISLab | 3 747 | 3 | 3 |
| ECG‑5 | ISLab | 3 745 | 5 | 5 |
| Wine‑Red | UCI | 1 599 | 11 | 5 |
| Wine‑White | UCI | 4 898 | 11 | 5 |
| ETTh1 | Zhou et al. (2021) | 17 420 | 6 | 5 |
| ETTh2 | Zhou et al. (2021) | 17 420 | 6 | 5 |
| Concrete | UCI | 1 030 | 8 | 5 |
| PowerPlant | UCI | 9 568 | 4 | 4 |
| Protein | UCI | 45 730 | 9 | 5 |
| Mackey‑Glass | MATLAB | 3 995 | 5 | 5 |

---

## Repository Contents

### ECG‑3  (`ECG_3.csv`)
- **Source:** ISLab — internal benchmark (link coming)  
- **Task:** 3‑class heartbeat classification  
- **Rows × Cols:** 3 747 × 3 (+ label)  
- **Download:** `[URL‑to‑be‑added]`

### ECG‑5  (`ECG_5.csv`)
- **Source:** ISLab — internal benchmark  
- **Task:** 5‑class heartbeat classification  
- **Rows × Cols:** 3 745 × 5 (+ label)  
- **Download:** `[URL‑to‑be‑added]`

### ETTh1 / ETTh2  (`ETTh1.csv`, `ETTh2.csv`)
Long‑horizon transformer‑oil‑temperature forecasting sets introduced by Zhou et al. (2021). Each file contains 17 420 hourly records of six electrical variables; the prediction target is **OT** (oil temperature).

### Concrete Compressive Strength  (`Concrete_Data.csv`)
1 030 mixes, eight material proportions, one target (**strength**). Classic UCI regression set.

### Power Plant  (`PowerPlant.csv`)
Combined‑cycle plant data – 9 568 hourly records of four ambient variables predicting net electrical output.

### Protein Tertiary Structure  (`Protein.csv`)
45 730 proteins with nine physicochemical descriptors and RMSD target. Useful for non‑linear regression testing.

### Wine Quality (Red & White)
Two CSVs (`winequality-red.csv`, `winequality-white.csv`) with eleven physicochemical features and an integer quality score.

### Mackey‑Glass Chaotic Series  (`mackey_glass_dataset.csv`)
3 995 samples from the Mackey‑Glass differential equation; good for chaotic time‑series prediction or anomaly detection.

---

## Directory Layout

