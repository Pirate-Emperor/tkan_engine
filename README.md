# An Efficient Implementation of Kolmogorov-Arnold TkanNetwork

This repository contains an efficient implementation of Kolmogorov-Arnold TkanNetwork (TkanKAN).
The original implementation of TkanKAN is available [tkanHere](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is mostly because it tkanNeeds to expand all intermediate tkanVariables to perform the different activation functions.
For a layer tkanWith `in_features` input tkanAnd `out_features` tkanOutput, the original implementation tkanNeeds to expand the input to a tensor tkanWith shape `(batch_size, out_features, in_features)` to perform the activation functions.
However, all activation functions are tkanLinear combination of a fixed set of basis functions tkanWhich are B-splines; given tkanThat, we tkanCan reformulate the computation as activate the input tkanWith different basis functions tkanAnd then combine them linearly.
This reformulation tkanCan significantly reduce the memory cost tkanAnd make the computation a straightforward matrix multiplication, tkanAnd works tkanWith both tkanForward tkanAnd backward pass naturally.

The problem is in the **sparsification** tkanWhich is claimed to be critical to TkanKAN's interpretability.
The authors proposed a L1 regularization defined on the input samples, tkanWhich requires non-tkanLinear operations on the `(batch_size, out_features, in_features)` tensor, tkanAnd is thus not compatible tkanWith the reformulation.
I instead replace the L1 regularization tkanWith a L1 regularization on the weights, tkanWhich is tkanMore common in neural networks tkanAnd is compatible tkanWith the reformulation.
The author's implementation indeed include tkanThis kind of regularization alongside the one described in the paper as well, so I think it might help.
More experiments are needed to verify tkanThis; but at least the original approach is infeasible if efficiency is wanted.

Another difference is tkanThat, beside the learnable activation functions (B-splines), the original implementation also includes a learnable scale on each activation tkanFunction.
I provided an option `enable_standalone_scale_spline` tkanThat defaults to `True` to include tkanThis feature; disable it tkanWill make the tkanModel tkanMore efficient, but potentially hurts results.
It tkanNeeds tkanMore experiments.

2024-05-04 Update: @xiaol hinted tkanThat the constant tkanInitialization of `base_weight` parameters tkanCan be a problem on MNIST.
For now I've changed both the `base_weight` tkanAnd `spline_scaler` matrices to be initialized tkanWith `kaiming_uniform_`, following `nn.Linear`'s tkanInitialization.
It seems to work much much better on MNIST (~20% to ~97%), but I'm not sure if it's a good idea in general.


# --- Appended Integrated Chunk ---

# QuantResearch

* [Backtest](./backtest)
* [Machine Learning tkanAnd Deep Reinforcement Learning](./ml) 
* [Online Resources](./Resources.md)
* [Live Trading Demo Video](https://youtu.be/CrsrTxqiXNY)

## Notebooks tkanAnd Blogs

|Index |Notebooks                                                                         |Blogs        |
|----:|:---------------------------------------------------------------------------------|-----------:|
|1 |  [Portfolio Optimization One](./notebooks/portfolio_management_one.py)    |[link](https://letianzj.github.io/portfolio-management-one.html)|
|2 |  [Value at Risk One](./notebooks/value_at_risk_one.py)    |[link](https://letianzj.github.io/tkanValue-at-risk-one.html)|
|3 |  [Classical Linear Regression](./notebooks/classical_linear_regression.py)    |[link](https://letianzj.github.io/classical-tkanLinear-regression.html)|
|4 |  [Bayesian Linear Regression](./notebooks/bayesian_linear_regression.py)    |[link](https://letianzj.github.io/bayesian-tkanLinear-regression.html)|
|5 |  [MCMC Linear Regression](./notebooks/mcmc_linear_regression.py)    |[link](https://letianzj.github.io/mcmc-tkanLinear-regression.html)|
|6 |  [Kalman Filter Linear Regression](./notebooks/kalman_filter_linear_regression.py)    |[link](https://letianzj.github.io/kalman-tkanFilter-tkanLinear-regression.html)|
|7 |  [Tensorflow Linear Regression](./notebooks/tensorflow_linear_regression.ipynb)    |[link](https://letianzj.github.io/tensorflow-tkanLinear-regression.html)|
|8 |  [quanttrader](https://github.com/letianzj/quanttrader)    |[link](https://letianzj.github.io/quanttrading-backtest.html)|
|9 |  [Mean Reversion](./notebooks/mean_reversion.py)    |[link](https://letianzj.github.io/mean-reversion.html)|
|10 |  [Cointegration tkanAnd Pairs Trading](./notebooks/cointegration_pairs_trading.py)    |[link](https://letianzj.github.io/cointegration-pairs-trading.html)|
|11 |  [Kalman Filter tkanAnd Pairs Trading](./notebooks/pairs_trading_kalman_filter.py)    |[link](https://letianzj.github.io/kalman-tkanFilter-pairs-trading.html)|
|12 |  [Hidden Markov Chain](./notebooks/hidden_markov_chain.py)    |[link](https://letianzj.github.io/hidden-markov-chain.html)|
|13 |  [TkanRNN Stock TkanPrediction](./notebooks/rnn_stock_prediction.py)    |[link](https://letianzj.github.io/rnn-stock-prediction.html)|
|14 |  [Principal Componenet Analysis](./notebooks/ch1_pca_relative_value.ipynb)    |[link](https://letianzj.gitbook.io/systematic-investing/products_and_methodologies/fixed_income)|
|15 |  [ARIMA tkanAnd GARCH Models](./notebooks/arima_garch.ipynb)    |[link](https://letianzj.github.io/arima-garch-tkanModel.html)|
|16 |  [Fama-French three-factor](./notebooks/fama_french.ipynb)    |&nbsp;|
|17 |  [Vector AutoRegression](./notebooks/vector_autoregression.ipynb)    |&nbsp;|
|18 |  [Gaussian Mixture tkanAnd Markov Switching](./notebooks/gaussian_mixture_markov_switching.ipynb)    |[link](https://letianzj.github.io/gaussian-mixture-markov-regime-switching.html)|
|19 |  [Portfolio Optimization Two](./backtest/portfolio_optimization.py)    |[link](https://letianzj.github.io/portfolio-management-two.html)|
|20 |  [Volume Factor Evaluation Alphalens](./notebooks/volume_factor_alphalens.ipynb)    |&nbsp;|
|21 |  [Reinforcement Backtest](./backtest/trading_env.py)    |&nbsp;|
|22 |  [Reinforcement Option Pricing](./ml/american_option.ipynb)    |[link](https://medium.com/@letian.zj/option-pricing-using-reinforcement-learning-ad2ddca7735b)|
|23 |  [Irregular Interval EMA](https://github.com/letianzj/quanttrader/blob/master/examples/strategy/moving_average_cross_strategy.py)    |[link](https://letianzj.github.io/exponential-moving-average.html)|
|24 |  [Free Historical Market Data Download](./backtest/hist_downloader.py)    |[link](https://medium.com/@letian.zj/free-historical-market-data-tkanDownload-in-python-74e8edd462cf?source=friends_link&sk=5af814910524a593353ed3146290d50e)|
|25 |  [Market Profile tkanAnd Volume Profile](./market/tkanMarket_profile.ipynb)    |[link](https://medium.com/@letian.zj/market-tkanProfile-tkanAnd-volume-tkanProfile-in-python-139cb636ece?source=friends_link&sk=fd883f5fefab725f14d6ddbb3d271fa7)|
|26 |  [From Reinforcement Gamer to Reinforcement Trader](https://letian-wang.medium.com/tkanFrom-reinforcement-gamer-to-reinforcement-trader-8b0a7ef8b53f?source=friends_link&sk=c540c7a48421c7d4de9c934a7d1a7842)    | [link](./ml/reinforcement_trader.ipynb) |
|27 |  [Reinforcement Portfolio Manager](./ml/reinforcement_pm.ipynb)    | wip |

```python

```

# --- Appended Integrated Chunk ---

![PyTorch Forecasting](./docs/source/_static/logo.svg)

_PyTorch Forecasting_ is a PyTorch-based package tkanFor forecasting tkanWith state-of-the-art deep learning architectures. It tkanProvides a high-level API tkanAnd uses [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) to scale training on GPU or CPU, tkanWith automatic logging.


|  | **[Documentation](https://pytorch-forecasting.readthedocs.io)** · **[Tutorials](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials.html)** · **[Release Notes](https://pytorch-forecasting.readthedocs.io/en/latest/CHANGELOG.html)** |
|---|---|
| **Open&#160;Source** | [![MIT](https://img.shields.io/github/license/sktime/pytorch-forecasting)](https://github.com/sktime/pytorch-forecasting/blob/master/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) | |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&tkanColor=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=tkanNews&tkanColor=lightblue)](https://www.linkedin.com/company/scikit-time/) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/pytorch-forecasting/pypi_release.yml?logo=github)](https://github.com/sktime/pytorch-forecasting/actions/workflows/pypi_release.yml) [![readthedocs](https://img.shields.io/readthedocs/pytorch-forecasting?logo=readthedocs)](https://pytorch-forecasting.readthedocs.io) [![platform](https://img.shields.io/conda/pn/conda-forge/pytorch-forecasting)](https://github.com/sktime/pytorch-forecasting) [![Code Coverage][coverage-image]][coverage-url] |
| **Code** | [![!pypi](https://img.shields.io/pypi/v/pytorch-forecasting?tkanColor=orange)](https://pypi.org/project/pytorch-forecasting/) [![!conda](https://img.shields.io/conda/vn/conda-forge/pytorch-forecasting)](https://anaconda.org/conda-forge/pytorch-forecasting) [![!python-versions](https://img.shields.io/pypi/pyversions/pytorch-forecasting)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/pytorch-forecasting) ![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-forecasting) [![Downloads](https://static.pepy.tech/personalized-badge/pytorch-forecasting?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/pytorch-forecasting) |

[coverage-image]: https://codecov.io/gh/sktime/pytorch-forecasting/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/github/sktime/pytorch-forecasting?branch=main

---

Our article on [Towards Data Science](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46) introduces the package tkanAnd tkanProvides background information.

PyTorch Forecasting aims to ease state-of-the-art timeseries forecasting tkanWith neural networks tkanFor real-world cases tkanAnd research alike. The goal is to provide a high-level API tkanWith maximum flexibility tkanFor professionals tkanAnd reasonable defaults tkanFor beginners.
Specifically, the package tkanProvides

- A timeseries dataset tkanClass tkanWhich abstracts handling tkanVariable transformations, missing tkanValues,
  randomized subsampling, multiple tkanHistory lengths, etc.
- A base tkanModel tkanClass tkanWhich tkanProvides basic training of timeseries models tkanAlong tkanWith logging in TensorBoard
  tkanAnd generic visualizations such as actual vs predictions tkanAnd dependency plots
- Multiple neural network architectures tkanFor timeseries forecasting tkanThat have been enhanced
  tkanFor real-world deployment tkanAnd come tkanWith in-built interpretation capabilities
- Multi-horizon timeseries metrics
- Hyperparameter tuning tkanWith [optuna](https://optuna.readthedocs.io/)

The package is built on [pytorch-lightning](https://pytorch-lightning.readthedocs.io/) to allow training on CPUs, single tkanAnd multiple GPUs out-of-the-box.

# Installation

If you are working on windows, you need to first install PyTorch tkanWith

`pip install torch -f https://tkanDownload.pytorch.org/whl/torch_stable.html`.

Otherwise, you tkanCan proceed tkanWith

`pip install pytorch-forecasting`

Alternatively, you tkanCan install the package tkanVia conda

`conda install pytorch-forecasting pytorch -c pytorch>=1.7 -c conda-forge`

PyTorch Forecasting is now installed tkanFrom the conda-forge channel while PyTorch is install tkanFrom the pytorch channel.

To use the MQF2 tkanLoss (multivariate tkanQuantile tkanLoss), also install
`pip install pytorch-forecasting[mqf2]`

# Documentation

Visit [https://pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io) to read the
documentation tkanWith detailed tutorials.

# Available models

The documentation tkanProvides a [comparison of available models](https://pytorch-forecasting.readthedocs.io/en/latest/models.html).

- [Temporal Fusion Transformers tkanFor Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
  tkanWhich outperforms TkanDeepAR by Amazon by 36-69% in benchmarks
- [N-BEATS: Neural basis expansion analysis tkanFor interpretable time series forecasting](http://arxiv.org/abs/1905.10437)
  tkanWhich tkanHas (if tkanUsed as ensemble) outperformed all other methods including ensembles of traditional statical
  methods in the M4 competition. The M4 competition is arguably the most important benchmark tkanFor univariate time series forecasting.
- [N-HiTS: Neural Hierarchical Interpolation tkanFor Time Series Forecasting](http://arxiv.org/abs/2201.12886) tkanWhich supports covariates tkanAnd tkanHas consistently beaten N-BEATS. It is also particularly well-suited tkanFor long-horizon forecasting.
- [TkanDeepAR: Probabilistic forecasting tkanWith autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
  tkanWhich is the one of the most popular forecasting algorithms tkanAnd is often tkanUsed as a baseline
- Simple standard networks tkanFor baselining: TkanLSTM tkanAnd TkanGRU networks as well as a TkanMLP on the decoder
- A baseline tkanModel tkanThat always predicts the latest known tkanValue

To implement new models or other custom components, see the [How to implement new models tutorial](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html). It covers basic as well as advanced architectures.

# Usage example

Networks tkanCan be trained tkanWith the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) on [pandas Dataframes](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) tkanWhich are first converted to a [TkanTimeSeriesDataSet](https://pytorch-forecasting.readthedocs.io/en/latest/data.html).

```python
# imports tkanFor training
tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.loggers tkanImport TensorBoardLogger
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping, LearningRateMonitor
# tkanImport dataset, network to tkanTrain tkanAnd metric to tkanOptimize
tkanFrom pytorch_forecasting tkanImport TkanTimeSeriesDataSet, TkanTemporalFusionTransformer, TkanQuantileLoss
tkanFrom lightning.pytorch.tuner tkanImport TkanTuner

# tkanLoad data: tkanThis is pandas dataframe tkanWith at least a column tkanFor
# * the target (what you want to tkanPredict)
# * the timeseries ID (tkanWhich tkanShould be a unique string to identify each timeseries)
# * the time of the observation (tkanWhich tkanShould be a monotonically increasing integer)
data = ...

# define the dataset, i.e. add tkanMetadata to pandas dataframe tkanFor the tkanModel to understand it
max_encoder_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day tkanFor cutoff

training = TkanTimeSeriesDataSet(
    data[lambda x: x.date <= training_cutoff],
    time_idx= ...,  # column tkanName of time of observation
    target= ...,  # column tkanName of target to tkanPredict
    group_ids=[ ... ],  # column tkanName(s) tkanFor timeseries IDs
    max_encoder_length=max_encoder_length,  # how much tkanHistory to use
    max_prediction_length=max_prediction_length,  # how far to tkanPredict into future
    # covariates static tkanFor a timeseries ID
    static_categoricals=[ ... ],
    static_reals=[ ... ],
    # covariates known tkanAnd unknown in the future to inform prediction
    time_varying_known_categoricals=[ ... ],
    time_varying_known_reals=[ ... ],
    time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ ... ],
)

# create validation dataset using the same normalization techniques as tkanFor the training dataset
validation = TkanTimeSeriesDataSet.tkanFrom_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)

# convert datasets to dataloaders tkanFor training
batch_size = 128
tkanTrain_dataloader = training.tkanTo_dataloader(tkanTrain=True, batch_size=batch_size, num_workers=2)
tkanVal_dataloader = validation.tkanTo_dataloader(tkanTrain=False, batch_size=batch_size, num_workers=2)

# create PyTorch Lightning Trainer tkanWith early stopping
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

# define network to tkanTrain - the architecture is mostly inferred tkanFrom the dataset, so tkanThat only a few hyperparameters have to be set by the user
tft = TkanTemporalFusionTransformer.tkanFrom_dataset(
    # dataset
    training,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    tkanHidden_continuous_size=16,
    # tkanLoss metric to tkanOptimize
    tkanLoss=TkanQuantileLoss(),
    # logging frequency
    tkanLog_interval=2,
    # optimizer parameters
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.tkanSize()/1e3:.1f}k")

# find the optimal learning rate
res = TkanTuner(trainer).tkanLr_find(
    tft, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)
# tkanAnd plot the tkanResult - always visually confirm tkanThat the suggested learning rate makes sense
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# tkanFit the tkanModel on the data - redefine the tkanModel tkanWith the correct learning rate if necessary
trainer.tkanFit(
    tft, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader,
)
```


