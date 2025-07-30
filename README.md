# 📈 Transformers_BTC_price_predictoin

This repository explores various **deep learning** and **statistical models** to forecast **Bitcoin (BTC)** prices using high-frequency time series data.

It includes implementations of:

- 🧠 **Transformer networks**
- 🔁 **Bidirectional LSTM (BiLSTM)**
- 📉 **GARCH** for volatility modeling
- 🌲 **XGBoost** for tree-based forecasting

---

## 🗂️ Project Structure

| File/Folder                 | Description |
|----------------------------|-------------|
| `volatility/`              | Contains additional scripts and notebooks for **BTC volatility analysis**. |
| `BiLSTM.ipynb`             | Trains and evaluates a **Bidirectional LSTM** model for BTC price prediction. Focuses on capturing temporal dependencies. |
| `btc-transformer.py`       | Python script defining the **Transformer architecture** and forward pass for price prediction. |
| `btc_bilstm_model.pth`     | Pretrained weights of the **BiLSTM model** for direct inference or fine-tuning. |
| `data_15m.csv`             | Raw BTC price data sampled at **15-minute intervals**, used for model training. |
| `data_15m_actual.csv`      | Likely contains **ground-truth BTC prices** used for validating model predictions. |
| `data_hr.csv`              | BTC data sampled at **hourly frequency** for longer-horizon modeling or comparison. |
| `garch_model.ipynb`        | Implements a **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** model to forecast BTC volatility. |
| `model_transformers.pth`   | Pretrained weights of the **Transformer model** used in BTC prediction. |
| `transformer_train.ipynb`  | Handles full training pipeline for the Transformer model — including **data prep, training, evaluation**, and visualization. |
| `xgb.ipynb`                | Uses **XGBoost**, a tree-based model, for BTC forecasting or **feature importance** analysis. |

---

## 🎯 Project Goals

The objective is to benchmark and compare multiple forecasting techniques for BTC price prediction using high-resolution time series data. This includes:

- 🧮 **Sequence models:** Transformer, BiLSTM  
- 📊 **Volatility modeling:** GARCH  
- 🌲 **Gradient boosting:** XGBoost

Each model is evaluated on both **15-minute** and **hourly BTC price datasets**, with pretrained weights included for rapid experimentation.

---

> _This project demonstrates a hybrid modeling approach by combining time series deep learning with classic statistical and ensemble learning methods._
