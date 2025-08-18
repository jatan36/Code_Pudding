# Code_Pudding
# August Code Pudding — Predicting Spotify Track Popularity with PyTorch

This repository contains the notebook(s) and assets for a regression project that predicts Spotify track **popularity** (0–100) using **audio features only** from the *Ultimate Spotify Tracks Database* (~232k tracks). The work combines EDA, preprocessing, and multiple PyTorch MLP architectures to benchmark the extent to which audio characteristics alone can explain popularity.

> **Key result:** Best validation performance achieved by **MLP3LayerDropout** with **Val MSE ≈ 211.11**; held-out test set achieved **RMSE ≈ 14.65** and **MAE ≈ 11.55**.

---

## Contributors
- Jatan Bhatt  
- Tirso Paneque  
- Eric  
- Rawaa Yousseif  

---

## 1) Introduction
Music streaming at scale enables data-driven insights into what makes a song popular. Spotify popularity (0–100) is influenced by many factors, but here we deliberately restrict inputs to **numeric audio features** (e.g., *danceability, energy, tempo, acousticness, valence, loudness*) to test how far these signals alone can go.

**Objective:** Build and evaluate neural networks in PyTorch to predict track popularity from audio features only, and assess modeling choices (scaling, regularization, architecture depth) that improve generalization.

---

## 2) Data
- **Source:** Kaggle — *Ultimate Spotify Tracks Database*  
- **Rows:** ~232,725  
- **Key columns:** `popularity` (target) and numeric features: `acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence` (plus categorical/meta fields not used in modeling).

---

## 3) Methodology

### 3.1 Preprocessing
- Remove duplicates and handle missing values.  
- Select numeric audio features only; **target = popularity**.  
- Scale features using **RobustScaler** (less sensitive to outliers than MinMax).  
- Split: **80/10/10** into train (186,180), validation (23,272), test (23,273).  
- Custom `Dataset` and `DataLoader` with `batch_size=64`.  

### 3.2 Exploratory Data Analysis (EDA)
- **Distributions:** Several features highly skewed (*instrumentalness, speechiness, liveness*). Outliers in `duration_ms` and `tempo`.  
- **Correlations with popularity:**  
  - Positive: **loudness (≈0.36)**, **danceability (≈0.26)**, **energy (≈0.25)**  
  - Negative: **acousticness (≈-0.38)**, **instrumentalness (≈-0.21)**, **speechiness (≈-0.15)**  
- **Inter-feature relationships:** strong `energy ↔ loudness` (≈0.82), `acousticness ↔ energy` (≈-0.73), moderate `danceability ↔ valence` (≈0.55).  
- **Implication:** No single feature is decisive; non-linear models are appropriate.  

### 3.3 Models
All models are feed-forward MLPs trained with **Adam (lr=1e-3)** and **MSELoss** for **10 epochs**.

- **MLP1Layer:** `input → 64 → 1` (ReLU)  
- **MLP2Layer:** `input → 64 → 32 → 1` (ReLU)  
- **MLP3LayerDropout:** `input → 128 → 64 → 32 → 1` with **Dropout(0.2)** between hidden layers (ReLU)  
- **MLP2LayerBatchNorm:** `input → 64 → 32 → 1` with **BatchNorm1d** on hidden layers (ReLU)  

**Model selection:** Best by validation loss.  

---

## 4) Results

| Model                | Val MSE (↓) | Notes                                  |
|----------------------|-------------|----------------------------------------|
| MLP1Layer            | ~220–224    | Simple baseline                        |
| MLP2Layer            | ~212–217    | Deeper than baseline, modest gains     |
| **MLP3LayerDropout** | **≈211.11** | **Best val loss; benefits from dropout** |
| MLP2LayerBatchNorm   | ~212–216    | Competitive; stable convergence        |

**Test performance (best model):**  
- **RMSE ≈ 14.65**  
- **MAE ≈ 11.55**  

---

## 5) Recommendations & Next Steps
- Integrate richer features (user behavior, lyrics sentiment, social data).  
- Explore hyperparameter tuning, deeper networks, and ensemble models.  
- Compare against tree-based baselines (LightGBM/XGBoost).  
- Build an artist/label dashboard to interpret how features (danceability, energy, loudness) affect popularity.  

---

## License
Research/educational use. Verify Kaggle dataset terms before redistribution.
