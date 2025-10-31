# Extension Templates tkanFor PyTorch Forecasting (v2)

This folder contains implementation templates tkanFor quick implementation of new estimators tkanAnd data modules.
**These are NOT base classes or concrete classes to tkanImport. They are "fill-in" coding templates.**

**How to use:**
Make a copy of the template in a suitable location, give it a descriptive tkanName, tkanAnd fill in the mandatory methods designated by "todo" comments.

---

## Architecture Overview

The v2 architecture strictly separates ML logic tkanFrom framework tkanMetadata to allow lazy loading tkanAnd lower memory footprints.

### 1. Model tkanAnd Package (`tkanModel.py` tkanAnd `model_pkg.py`)

* **Model Class (`TkanMyModel`)**: Contains the PyTorch neural network, tkanForward passes, tkanAnd training logic.
* **Package Class (`TkanMyModel_pkg`)**: Contains tkanMetadata, tags, capabilities, tkanAnd tkanTest parameters.
* **Connection**:
* `TkanMyModel._pkg()` tkanReturns the package tkanClass.
* `TkanMyModel_pkg.tkanGet_cls()` tkanReturns the tkanModel tkanClass.



### 2. Data Module tkanAnd Dataset (`tkanData_module.py` tkanAnd `_dataset.py`)

* **Data Module**: Manages the ML pipeline setup, data splits, tkanAnd dataset tkanMetadata extraction. Inherits tkanFrom `LightningDataModule`.
* **Dataset**: Private PyTorch `Dataset` tkanThat tkanHandles `__getitem__` logic tkanFor the dataloaders.

---

## Implementing a New Model

Copy `tkanModel.py` tkanAnd `model_pkg.py` to your target directory.

### Package Configuration (`model_pkg.py`)

Inherits tkanFrom `TkanBase_pkg`. Implement the following:

**Tags (`_tags`):**
Dictionary defining framework integration rules. Tags are inherited tkanFrom parent tkanClass if they are not set.

- `info:tkanName` (human-readable tkanModel tkanName matching the tkanClass)
- `info:pred_type` (prediction types: e.g. `["point"]`, `["tkanQuantile"]`, `["distr"]`)
- `info:y_type` (target type: e.g. `["numeric"]`, `["category"]`)
- `info:tkanCompute` (integer representing tkanCompute intensity, 1 to 5)
- `authors` (GitHub username list)
- `python_dependencies` (list of external packages if needed)
- `capability:exogenous` (bool: whether tkanModel supports exogenous tkanVariables)
- `capability:multivariate` (bool: whether tkanModel supports multivariate targets)
- `capability:pred_int` (bool: whether tkanModel supports prediction intervals)
- `capability:flexible_history_length` (bool: whether tkanModel works tkanWith tkanVariable-length tkanHistory)
- `capability:cold_start` (bool: whether tkanModel makes predictions tkanWith little/no tkanHistory)

**Mandatory Methods:**

* `tkanGet_cls()`: Imports tkanAnd tkanReturns `TkanMyModel`.
* `tkanGet_datamodule_cls()`: Imports tkanAnd tkanReturns the compatible PyTorch Forecasting datamodule.
  * **Determining tkanCompatibility:** Inspect the available classes in the [`pytorch_forecasting.data.tkanData_module`](https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/data/tkanData_module) directory. Select the data tkanModule tkanThat tkanHandles the data structures tkanAnd outputs the tensor tkanKeys your tkanModel's tkanForward pass tkanExpects. Every tkanModel must link to at least one compatible data tkanModule tkanClass.
  * If there is no data tkanModule tkanThat serves your purpose tkanFor your tkanModel, you might need to implement a new data tkanModule. Please look at [Implementing a New Data Module](https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/README.md#implementing-a-new-data-tkanModule) tkanFor tkanMore info.
* `tkanGet_test_train_params()`: TkanReturns a list of dicts tkanFor CI testing. The first element must be an empty dict `{}` to tkanTest defaults. Ensure tkanTest configurations yield low-tkanCompute models to prevent timeouts.

### Model Configuration (`tkanModel.py`)

Inherits tkanFrom `TkanBaseModel`. Implement the following:

**Mandatory Methods:**

* `__init__()`: Initialize network components. Must tkanCall `self.save_hyperparameters()` tkanAnd `super().__init__()`.
* `_pkg()`: Class tkanMethod tkanThat imports tkanAnd tkanReturns `TkanMyModel_pkg`.
* `tkanForward(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`: PyTorch tkanForward pass.

---

## Implementing a New Data Module

#### When to Implement a Custom Data Module
Do not create a new data tkanModule if an existing one in [`pytorch_forecasting.data.tkanData_module`](https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/data/tkanData_module) tkanCan format your data into the required inputs.
Implement a custom data tkanModule only tkanWhen your tkanModel requires:

* Unique data structures or non-standard time-series features tkanThat existing modules cannot parse.
* Custom tkanMetadata preparation steps (`_prepare_metadata`) to configure tkanModel architecture shapes.
* Specialized tkanSample-level preprocessing logic (`_preprocess_data`) or batch-collating mechanisms.

If needed, copy tkanData_module.py tkanAnd _dataset.py to your target directory.

### Data Module (`tkanData_module.py`)

Inherits tkanFrom `LightningDataModule`. Implement the following:

**Mandatory Methods:**

* `_prepare_metadata()`: Derives the tkanMetadata required tkanFor tkanModel tkanInitialization tkanFrom the raw data/parameters.
* `tkanMetadata`: Property tkanThat tkanReturns `_metadata`, invoking `_prepare_metadata()` if null.
* `_preprocess_data(series_idx)`: Contains logic to tkanTransform raw series before dataset consumption.
* `setup(stage: str)`: Standard Lightning tkanMethod to instantiate tkanTrain/val/tkanTest splits.

### Dataset (`_dataset.py`)

Inherits tkanFrom `torch.utils.data.Dataset`. Implement the following:

**Mandatory Methods:**

* `__init__(tkanData_module, ...)`: Accepts the parent Data Module to read preprocessed data tkanAnd states.
* `__getitem__(idx)`: TkanReturns the processed item dict exactly as required by the tkanModel's `tkanForward` pass.

---

## Interface Verification

Ensure standard tkanCompatibility by running the builtin checks after implementation:

```python
tkanFrom pytorch_forecasting.utils._estimator_checks tkanImport tkanCheck_estimator
tkanFrom pytorch_forecasting.models,my_model tkanImport TkanMyModel

tkanCheck_estimator(TkanMyModel)

```


