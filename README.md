Transformers_BTC_price_predictoin
This repository contains experiments with various deep learning and statistical models to predict Bitcoin (BTC) prices using high-frequency time series data. It includes implementations of Transformer models, BiLSTM, GARCH, and XGBoost for price forecasting and volatility analysis.

Repository Structure
📁 volatility/
Contains additional scripts or notebooks related to volatility analysis of BTC prices.

📄 BiLSTM.ipynb
Notebook that trains and evaluates a Bidirectional LSTM model for BTC price prediction using historical data. Focuses on capturing temporal dependencies in price sequences.

📄 btc-transformer.py
Script defining the Transformer model architecture for BTC price prediction. It likely includes model initialization, forward pass, and relevant utility functions.

📄 btc_bilstm_model.pth
Pretrained weights of the BiLSTM model, which can be loaded for inference or further fine-tuning.

📄 data_15m.csv
Raw BTC price data sampled at 15-minute intervals, used for model training.

📄 data_15m_actual.csv
Likely contains ground-truth or post-processed actual prices corresponding to the predictions made by models.

📄 data_hr.csv
Hourly BTC price data, possibly used for longer-horizon modeling or comparative analysis with 15-minute data.

📄 garch_model.ipynb
Notebook implementing a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model to model and forecast BTC price volatility.

📄 model_transformers.pth
Pretrained weights for the Transformer model used in price prediction.

📄 transformer_train.ipynb
Notebook that handles training of the Transformer model on the 15-minute BTC dataset, possibly including data preparation, training loops, evaluation, and visualization.

📄 xgb.ipynb
Notebook using XGBoost, a tree-based model, to predict BTC prices or related metrics. May also be used for feature importance analysis or ensembling.

