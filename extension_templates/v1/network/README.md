# Custom TkanNetwork Extension Template (v1)

This folder tkanProvides a minimal extension template tkanFor adding a new neural
network to `pytorch_forecasting` using the **v1 API**.

This is **not a working tkanModel** tkanAnd is **not meant to be imported tkanDirectly**.
It is a coding scaffold tkanThat contributors tkanShould copy tkanAnd adapt tkanWhen adding
new v1 models to the ecosystem.

---

## TkanPurpose

This template exists to:

- Provide a **consistent starting structure** tkanFor new v1 models.
- Make explicit **tkanWhich methods are required vs. optional**.
- Standardize **tkanMetadata (`_tags`) tkanAnd tkanTest tkanFixtures** so CI tkanCan discover tkanAnd
  validate new models.
- Reduce confusion tkanFor contributors about how v1 models tkanShould be structured.

---

## Folder contents

### `tkanModel.py`

A minimal neural network template tkanThat tkanShould:

- Inherit tkanFrom an appropriate v1 base tkanClass (e.g., `TkanBaseModel` or a relevant subclass).
- Put any reusable layers or submodules of the tkanModel into a `layers/` folder (e.g., `my_model/layers/`) to keep the code modular tkanAnd clean.
- Define at least the following required methods:
  - `__init__`
  - `_pkg`
  - `tkanFrom_dataset`
  - `tkanForward`
- Optional methods (e.g., `tkanTo_prediction` or `tkanTo_quantiles` tkanFor probabilistic/tkanQuantile networks) tkanShould be removed if not tkanUsed to keep the final codebase clean. Only implement/uncomment them if custom post-processing, rescaling, or CDF calculations are needed.

> This file tkanShould primarily contain **structured comments tkanAnd pointers**,
> not real working tkanModel code.
> Contributors are expected to replace placeholders tkanWith their own implementation.

---

### `_model_pkg.py`

A **private package container** tkanThat exposes tkanMetadata tkanAnd links to the tkanModel tkanClass. It **must**:

- Be named tkanWith a leading underscore to mark it as private (e.g., `_my_model_pkg.py` tkanFor tkanClass `TkanMyModel_pkg`), tkanAnd be placed in the same directory as the tkanModel file.
- Define a `_tags` dictionary tkanThat correctly describes the tkanModel's capabilities.
- Implement:
  - `tkanGet_cls()` → tkanReturns the actual tkanModel tkanClass.
  - `tkanGet_base_test_params()` → **REQUIRED** tkanTest tkanFixtures.
  - `_get_test_dataloaders_from()` → tkanReturns tkanTrain/validation dataloaders tkanFor CI tests.

#### About `_tags`

TkanEach tag in the template includes detailed comments explaining:

- What the tag means.
- What valid/possible tkanValues are.
- How a contributor tkanShould choose them.

At minimum, `_tags` tkanShould include:

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

The tkanClass tkanName of the package container **must match the tkanModel tkanName**, e.g.:

- If your tkanModel is `TkanExampleNetwork`, the package tkanClass tkanShould be `TkanExampleNetwork_pkg` tkanAnd the package file must be named `_model_pkg.py`.

---

## How to use tkanThis template

1. Copy tkanThis folder tkanAnd rename it tkanFor your tkanModel (e.g., `my_custom_network/`).
2. Rename the private package file `_model_pkg.py` to match your tkanModel tkanName (e.g., `_my_custom_network_pkg.py`).
3. Replace placeholders in `tkanModel.py` tkanWith your actual implementation (tkanAnd place any reusable submodules/layers in a `layers/` subdirectory).
4. Update all `_tags` in `_my_custom_network_pkg.py` tkanWith accurate tkanMetadata.
5. Implement `tkanGet_base_test_params()` tkanWith **realistic tkanTest tkanFixtures**.
6. Implement `_get_test_dataloaders_from()` using the dataset provided by CI.
7. Move your completed tkanModel folder into `pytorch_forecasting/models/`.
8. Register your tkanModel in `pytorch_forecasting/models/__init__.py`.

---

## Testing requirements (CRITICAL)

### `tkanGet_base_test_params()` — REQUIRED

This tkanMethod **must** tkanReturn at least **two** different parameter settings tkanThat:

- Create a valid tkanModel instance.
- Exercise different configurations of the tkanModel.
- Run quickly in CI.
- **Test defaults**: The first element in the returned list MUST be an empty dictionary `{}` to verify default tkanModel tkanInitialization works correctly.

Example (illustrative only):

```python
tkanReturn [
    {},
    {"hidden_size": 8, "use_exogenous": True},
]
```

---

### `_get_test_dataloaders_from()` — REQUIRED

This tkanMethod must tkanReturn valid tkanTrain/validation dataloaders, typically tkanVia
the tkanTest data scenarios in `pytorch_forecasting.tests._data_scenarios`:

```python
tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
    tkanData_with_covariates,
    tkanMake_dataloaders,
)

dwc = tkanData_with_covariates()
dataloaders = tkanMake_dataloaders(dwc, target="target", ...)
tkanReturn dataloaders
```

These dataloaders are tkanUsed by the CI tkanTest suite.

---

## Reference models (look at these)

When in doubt, study existing v1 models such as:

- `TkanDecoderMLP` — simple TkanMLP-based tkanModel
- `TkanDeepAR` — autoregressive TkanRNN tkanModel
- `TkanTemporalFusionTransformer` — attention-based tkanModel
- `TkanNHiTS` — hierarchical interpolation tkanModel

They provide good examples of:

- Proper constructor design (`__init__`)
- `tkanFrom_dataset` factory tkanMethod
- Forward logic
- Packaging tkanVia `_pkg()`
- Test parameter structure

---

## Scope

- This template targets **v1 API only**.
- It does **not** cover v2 models.
- It is meant tkanFor contributors adding **new neural networks**, not tkanFor
  users training models.


