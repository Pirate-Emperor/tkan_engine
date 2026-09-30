# T-KAN Microstructure Engine

## Overview

The **T-KAN Microstructure Engine** is a comprehensive, institutional-grade deep learning and quantitative research framework designed to model non-linear shape-mappings in financial time series. 

By leveraging Temporal Kolmogorov-Arnold Networks (T-KANs) with B-spline edge activations, the engine dynamically adapts to Order Flow Imbalance (OFI) inversions and high-frequency market microstructure anomalies. The repository unifies a highly optimized KAN implementation, a deep quantitative backtesting suite, and a scalable PyTorch Lightning forecasting pipeline.

---

## Part I: Efficient Kolmogorov-Arnold Networks (T-KAN)

This module contains an optimized, production-ready implementation of Temporal Kolmogorov-Arnold Networks (T-KAN).

The performance issue of original KAN implementations is mostly because they need to expand all intermediate variables to perform the different activation functions. For a layer with `in_features` input and `out_features` output, naive implementations need to expand the input to a tensor with shape `(batch_size, out_features, in_features)` to perform the activation functions.

However, all activation functions are a linear combination of a fixed set of basis functions which are B-splines; given that, we can reformulate the computation as activating the input with different basis functions and then combining them linearly. This reformulation can significantly reduce the memory cost and make the computation a straightforward matrix multiplication, and works with both forward and backward passes natively.

### Sparsification & L1 Regularization
The core problem is in the **sparsification** which is critical to the T-KAN's interpretability. Standard L1 regularization defined on the input samples requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and is thus not compatible with the reformulation.

We instead replace the L1 regularization with an L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation. 

Another architectural difference is that, beside the learnable activation functions (B-splines), this implementation also includes a learnable scale on each activation function. We provide an option `enable_standalone_scale_spline` that defaults to `True` to include this feature; disabling it will make the model more efficient, but potentially hurts predictive results on complex microstructure data.

### Initialization Dynamics
Constant initialization of `base_weight` parameters can be a problem during high-variance microstructure training. Both the `base_weight` and `spline_scaler` matrices are initialized with `kaiming_uniform_`, following standard `nn.Linear` initialization, which prevents vanishing gradients during the early stages of representation learning.

---

## Part II: Quantitative Research & Backtesting Suite

The engine includes a massive repository of quantitative research algorithms and statistical baselines used to validate the T-KAN's performance.

### Core Analytical Modules

| Index | Module | Description & Application |
|----:|:---------------------------------------------------------------------------------|:-----------|
| 1 | **Portfolio Optimization** | Implementation of Modern Portfolio Theory, Efficient Frontier, and convex optimization matrices. |
| 2 | **Value at Risk (VaR)** | Parametric, Historical, and Monte Carlo VaR models for continuous risk tracking. |
| 3 | **Classical Linear Regression** | Baseline statistical models for alpha decay measurement. |
| 4 | **Bayesian Linear Regression** | Probabilistic weight updates for uncertain market regimes. |
| 5 | **MCMC Linear Regression** | Markov Chain Monte Carlo estimations. |
| 6 | **Kalman Filter Linear Regression** | Dynamic beta hedging and spread tracking. |
| 7 | **Tensorflow Linear Regression** | Accelerated baseline architectures. |
| 8 | **Event-Driven Backtest** | The core deterministic order-matching backtester. |
| 9 | **Mean Reversion** | Statistical arbitrage and Ornstein-Uhlenbeck processes. |
| 10 | **Cointegration Pairs Trading** | Stationarity testing (ADF) and spread calculation. |
| 11 | **Kalman Filter Pairs Trading** | Dynamic hedge ratio calculation using state-space models. |
| 12 | **Hidden Markov Chain** | Regime switching detection (e.g., High VIX vs Low VIX). |
| 13 | **RNN Stock Prediction** | Recurrent architectures (baseline for T-KAN comparison). |
| 14 | **Principal Component Analysis** | PCA for yield curve and relative value fixed-income modeling. |
| 15 | **ARIMA and GARCH Models** | Autoregressive Integrated Moving Average and generalized autoregressive conditional heteroskedasticity. |
| 16 | **Fama-French Three-Factor** | Factor modeling and beta exposure. |
| 17 | **Vector AutoRegression** | VAR models for multi-asset predictive interactions. |
| 18 | **Gaussian Mixture & Markov Switching** | Advanced density estimations for leptokurtic returns. |
| 19 | **Portfolio Optimization Two** | Advanced constraints (turnover limits, leverage caps). |
| 20 | **Volume Factor Evaluation** | Alphalens tearing and factor IC decay tracking. |
| 21 | **Reinforcement Backtest** | RL environment wrapping the limit order book. |
| 22 | **Reinforcement Option Pricing** | Deep Q-Learning for optimal stopping times (American Options). |
| 23 | **Irregular Interval EMA** | Exponential Moving Averages for asynchronous tick data. |
| 24 | **Historical Market Data Downloader** | Async parsers for raw limit order book ingestion. |
| 25 | **Market Profile and Volume Profile** | Volume-at-price histograms and Value Area calculations. |
| 26 | **Reinforcement Trader** | PPO and SAC agents for continuous action spaces. |
| 27 | **Reinforcement Portfolio Manager** | Dynamic capital allocation across an N-asset universe. |

---

## Part III: PyTorch Forecasting & Deep Learning Pipeline

The T-KAN Microstructure Engine integrates a PyTorch-based package for forecasting with state-of-the-art deep learning architectures. It provides a high-level API and uses PyTorch Lightning to scale training on GPU or CPU, with automatic logging.

The package provides:
- A timeseries dataset class which abstracts handling variable transformations, missing values, randomized subsampling, multiple history lengths, etc.
- A base model class which provides basic training of timeseries models along with logging in TensorBoard and generic visualizations such as actual vs predictions and dependency plots.
- Multi-horizon timeseries metrics.
- Hyperparameter tuning with Optuna.

### Available Architectures

- **Temporal Fusion Transformers (TFT):** For Interpretable Multi-horizon Time Series Forecasting, which heavily outperforms standard autoregressive baselines.
- **N-BEATS:** Neural basis expansion analysis for interpretable time series forecasting, which has (if used as ensemble) outperformed all other methods including ensembles of traditional statical methods.
- **N-HiTS:** Neural Hierarchical Interpolation for Time Series Forecasting which supports covariates and is particularly well-suited for long-horizon forecasting.
- **DeepAR:** Probabilistic forecasting with autoregressive recurrent networks which is the one of the most popular forecasting algorithms and is often used as a baseline.
- **Standard Networks:** LSTM and GRU networks as well as a MLP on the decoder.

### Usage Example

Networks can be trained with the PyTorch Lightning Trainer on pandas Dataframes which are first converted to a `TimeSeriesDataSet`.

```python
# imports for training
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner

# Load data: This is a pandas dataframe with at least a column for
# * the target (what you want to predict)
# * the timeseries ID (which should be a unique string to identify each timeseries)
# * the time of the observation (which should be a monotonically increasing integer)
data = ...

# define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
max_encoder_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x.date <= training_cutoff],
    time_idx= ...,  # column name of time of observation
    target= ...,  # column name of target to predict
    group_ids=[ ... ],  # column name(s) for timeseries IDs
    max_encoder_length=max_encoder_length,  # how much history to use
    max_prediction_length=max_prediction_length,  # how far to predict into future
    # covariates static for a timeseries ID
    static_categoricals=[ ... ],
    static_reals=[ ... ],
    # covariates known and unknown in the future to inform prediction
    time_varying_known_categoricals=[ ... ],
    time_varying_known_reals=[ ... ],
    time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ ... ],
)

# create validation dataset using the same normalization techniques as for the training dataset
validation = TimeSeriesDataSet.from_dataset(
    training, 
    data, 
    min_prediction_idx=training.index.time.max() + 1, 
    stop_randomization=True
)

# convert datasets to dataloaders for training
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# create PyTorch Lightning Trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",  # run on CPU, if on multiple GPUs, use strategy="ddp"
    gradient_clip_val=0.1,
    limit_train_batches=30,  # 30 batches per epoch
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)

# define network to train - the architecture is mostly inferred from the dataset, 
# so that only a few hyperparameters have to be set by the user
tft = TemporalFusionTransformer.from_dataset(
    # dataset
    training,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    # loss metric to optimize
    loss=QuantileLoss(),
    # logging frequency
    log_interval=2,
    # optimizer parameters
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find the optimal learning rate
res = Tuner(trainer).lr_find(
    tft, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader, 
    early_stop_threshold=1000.0, 
    max_lr=0.3,
)

# and plot the result - always visually confirm that the suggested learning rate makes sense
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model on the data - redefine the model with the correct learning rate if necessary
trainer.fit(
    tft, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader,
)

```

The package is built on `pytorch-lightning` to allow training on CPUs, single and multiple GPUs out-of-the-box, ensuring that processing terabytes of Limit Order Book data executes with deterministic performance bounds.


## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

Thank you for visiting this project!

---