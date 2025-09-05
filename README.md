#  Bitcoin Price Prediction Using LSTM & Transformer-Based Deep Learning Models

##  **Aim**
To investigate the efficacy of LSTM and Transformer architectures in
accurately forecasting Bitcoin prices using historical time series data
from Yahoo Finance, and to determine the most reliable model for
cryptocurrency forecasting.  

---

##  **Research Questions**
-  Can LSTM and Transformer-based deep learning models correctly
forecast Bitcoin prices using historical time series data?  
-  Do architectural variations within LSTM (Unidirectional,
Bi-directional, Stacked) significantly impact prediction accuracy?  
-  How does the number of attention blocks in Transformer models
affect predictive accuracy and robustness?  

---

## **Objectives**
1. Gather & preprocess historical Bitcoin price data from Yahoo
Finance using `yfinance`.  
2. Develop multiple LSTM variants (Unidirectional, Bi-directional,
Stacked).  
3. Build Transformer-based models with varying attention blocks.  
4. Train models under different scenarios (no stationarity, with
stationarity, train-validation-test split).  
5. Compare accuracy & robustness of LSTM and Transformer models.  

---

##  **Tools Required**
- Python 3.8+  
- Jupyter Notebook / Google Colab  
- Internet connection (for data fetching)  
- GPU-enabled environment *(optional for faster training)*  

---

##  **Packages Used**
| Package | Purpose |
|---------|---------|
| `yfinance` | Download historical BTC-USD data |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing & metrics |
| `tensorflow`, `keras` | Model building & training |
| `statsmodels` | Stationarity testing (ADF) |

---

##  **Research Workflow**
1. **Data Collection** — Fetch BTC-USD data (2014–2025) from Yahoo Finance.  
2. **Preprocessing** — Handle missing values, normalize, reset index, prepare sequences.  
3. **Scenario 1** — Train models without stationary check.  
4. **Scenario 2** — Apply ADF test, make data stationary, retrain models.  
5. **Scenario 3** — Train-validation-test split for real-world evaluation.  
6. **Model Development** — Implement LSTM variants & Transformer models (1–5 attention blocks).  
7. **Training & Tuning** — Optimize parameters, monitor validation loss.  
8. **Evaluation** — Compare models using MAE, MSE, RMSE, R², MAPE.  
9. **Results Analysis** — Identify best-performing architecture.  
10. **Conclusion & Future Work** — Summarize findings & suggest improvements.  

---

### **Project Files**
-  **`1. EDA and LSTM models (Uni, Bi and Stacked).ipynb`** — Data exploration and implementation of LSTM variants (Unidirectional, Bidirectional, Stacked).  
-  **`2. Time Series Transformer Model with varying Attention Blocks.ipynb`** — Implementation of Transformer models with different attention block configurations.  
-  **`3. LSTM models with stationary data.ipynb`** — LSTM experiments using differenced (stationary) Bitcoin data.  
-  **`4. Transformer Model with stationary data.ipynb`** — Transformer experiments with stationary Bitcoin data.  
-  **`5. Included Validation set Evaluation - LSTM models (Uni, Bi and Stacked).ipynb`** — LSTM models evaluated with train-validation-test split.  
-  **`6. Included Validation set Evaluation - Transformer Models.ipynb`** — Transformer models evaluated with train-validation-test split.  
  
   
###  **Datasets**
-  **`btc_usd.csv`** — Full historical BTC-USD dataset (2014–2025).  
-  **`btc_usd(date&Close).csv`** — Simplified dataset with only Date and Close columns.  
- **`btc_test_3months_scaled.npz`** — Preprocessed and scaled BTC test dataset (3 months).  

---

##  **Results Summary**

### **Scenario 1: Without Stationary Check**
| Model               | MAE     | MSE          | RMSE   | R²     | MAPE  |
|---------------------|---------|--------------|--------|--------|-------|
| **Stacked LSTM**    | 1977.81 | 6,948,578.99 | 2636.02| 0.8710 | 2.16% |
| Transformer (2 blk) | 2400.80 | 10,843,260.24| 3292.91| 0.7988 | 2.62% |

---

### **Scenario 2: With Stationary Check**
| Model                 | MAE     | MSE          | RMSE   | R²     | MAPE   |
|-----------------------|---------|--------------|--------|--------|--------|
| Transformer (1 block) | 6071.08 | 47,848,637.71| 6917.27| 0.1120 | 95.67% |

---

### **Scenario 3: Train-Validation-Test Split**
| Model                 | Validation R² | Validation MAPE | Testing R² | Testing MAPE |
|-----------------------|---------------|-----------------|------------|--------------|
| **Unidirectional LSTM** | 0.9664      | 2.52%           | 0.8677     | 2.21%        |
| Transformer (3 blocks) | 0.9429      | 3.32%           | 0.7692     | 2.73%        |

---

##  **Conclusion**
- **LSTM models**, especially Stacked LSTM, excel without stationarity checks.  
- Stationarity preprocessing degraded LSTM performance but slightly improved Transformers.  
- In real-world testing, **Unidirectional LSTM** achieved the highest R² (0.8677) with lowest MAPE (2.21%).  
- Transformer performance depends on attention block count — moderate complexity is optimal.  

---

##  **Future Work**
- Integrate alternative data sources (social media sentiment, macroeconomic factors, blockchain analytics).  
- Explore hybrid LSTM-Transformer models.  
- Experiment with attention-augmented LSTMs.  
