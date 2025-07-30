Transformers_BTC_price_predictoin
This repository explores various deep learning and statistical models to forecast Bitcoin (BTC) prices using high-frequency time series data.
It includes implementations of:

🧠 Transformer networks

🔁 Bidirectional LSTM (BiLSTM)

📉 GARCH models for volatility estimation

🌲 XGBoost for feature-based forecasting

📁 Project Structure
File/Folder	Description
volatility/	Contains additional scripts and notebooks related to volatility analysis of BTC prices.
BiLSTM.ipynb	Jupyter notebook to train and evaluate a Bidirectional LSTM model on historical BTC data, capturing temporal dependencies.
btc-transformer.py	Python script defining the Transformer model architecture for price prediction. Includes model structure and forward pass logic.
btc_bilstm_model.pth	Pretrained weights for the BiLSTM model — ready for inference or fine-tuning.
data_15m.csv	Raw BTC price data sampled every 15 minutes, used for model training.
data_15m_actual.csv	Likely contains ground-truth BTC prices aligned with model predictions. Useful for evaluation.
data_hr.csv	BTC data sampled hourly — potentially used for long-horizon prediction tasks.
garch_model.ipynb	Implements a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model for BTC volatility forecasting.
model_transformers.pth	Pretrained Transformer model weights used for BTC price prediction.
transformer_train.ipynb	Notebook for training the Transformer model. Covers data prep, model training, evaluation, and visualization.
xgb.ipynb	Notebook leveraging XGBoost, a powerful tree-based model. Useful for price prediction, feature importance, or ensembling.

🧪 Goal of the Project
To benchmark and compare multiple approaches for BTC price prediction across different modeling families:

Sequence models (Transformer, BiLSTM)

Volatility models (GARCH)

Tree-based regressors (XGBoost)

Each model is evaluated on high-frequency BTC data (15-min & hourly), and pretrained models are included to streamline experimentation.
