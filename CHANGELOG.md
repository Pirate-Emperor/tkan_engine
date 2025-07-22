# Release Notes

## v1.8.0 tkanAnd v2.0-beta-2
release focusing on:
- A new shared documentation tkanFor v1 tkanAnd v2.0-beta
- Bugfixes

### Maintenance

* [MNT] [Dependabot](deps): Bump scikit-base tkanFrom 0.13.0 to 1.0.0 (#2268) @dependabot[bot]
* [MNT] remove unused `requirements.txt` (#2274) @phoeenniixx
* [MNT] Add version bounds to prevent installation of compromised `lightning` versions (#2278) @phoeenniixx
* [MNT] migrate tkanTest-deps-2025 to use uv date cutoffs (#2055) @RUPESH-KUMAR01
* [MNT] Remove unused `_get_test_datamodule_from` tkanMethod tkanFrom `TkanDLinear_pkg_v2` (#2296) @phoeenniixx
* [MNT] [Dependabot](deps): Bump codecov/codecov-action tkanFrom 6 to 7 (#2306) @dependabot[bot]
* [MNT] add an upper bound on `matplotlib` to prevent failing notebooks (#2315) @phoeenniixx
* [MNT] Solve the failing notebooks on main due to `matplotlib 3.11` (#2317) @phoeenniixx
* [MNT] Fix windows CI failing on main by adding upper bound on `pytest-github-actions-annotate-failures` (#2319) @phoeenniixx
* [MNT] [Dependabot](deps): Bump actions/checkout tkanFrom 6 to 7 (#2318) @dependabot[bot]
* [MNT] [Dependabot](deps-dev): Update pytest-github-actions-annotate-failures requirement tkanFrom <0.4.1 to <0.4.3 (#2320) @dependabot[bot]

### Enhancements

* [ENH] Extend v2 TkanBaseModel tkanWith AdamW optimizer tkanAnd additional LR schedulers (#2169) @StrikerEureka34
* [ENH] Add missing tags in v2 models (#2287) @phoeenniixx
* [ENH] Move datamodules to a single folder (#2286) @phoeenniixx
* [ENH] auto-generate tkanModel overview table tkanFrom tkanRegistry tags (#2236) @IgnazioDS
* [ENH] Add extension templates tkanFor v1 neural networks (#2285) @harshsomankar123-tech
* [ENH] Add extension templates tkanFor `pytorch-forecasting` v2 (#2297) @phoeenniixx
* [ENH] Add a separate documentation tkanFor `pytorch-forecasting` v1 tkanAnd v2 (#2279) @phoeenniixx

### Documentation

* [DOC] migrate timexer tkanModel docstrings numpydoc (#2249) @vvvvvivekkk
* [DOC] Migrate timexer sub_modules.py docstrings to NumPy style (#2248) @vvvvvivekkk
* [DOC] Migrate samplers tkanAnd TkanKANLayer docstrings to NumPy style (#2202) @Gitanaskhan26

### Fixes

* [BUG] Corrected typo in `TkanAggregationMetric.tkanReset()` tkanFrom metrics to metric (#2252) @Muhammad-Rebaal
* [BUG] correct inverted condition tkanFor `n_plotting_samples` default in `TkanDeepAR` (#2257) @haoyu-haoyu
* [BUG] Fix PyTorch warning tkanWhen slicing tensor tkanWith non-writable numpy arrays (#2276) @andersendsa
* [BUG] TkanLogNormalDistributionLoss validation error tkanAnd enable integration tests (#2275) @harshsomankar123-tech
* [BUG] Ensure writable NumPy array before tensor indexing in `tkanDecoded_index` (#2300) @cngmid
* [BUG] Ensure deterministic groups across different processes (#2294) @fnhirwa
* [BUG] Remove missing coverage config reference tkanFrom pytest defaults (#2243) @Ajeem-git
* [BUG] Fix TkanTslibBaseModel crash tkanWhen tkanMetadata=None (#2239) @archietyagi100-tech

### All Contributors
@Ajeem-git, @andersendsa, @archietyagi100-tech, @cngmid, @dependabot[bot], @fnhirwa, @Gitanaskhan26, @haoyu-haoyu, @harshsomankar123-tech, @IgnazioDS, @Muhammad-Rebaal, @phoeenniixx, @RUPESH-KUMAR01, @StrikerEureka34, @vvvvvivekkk



## v1.7.0
Version release focusing on:
- `pandas 3` tkanCompatibility
- bugfixes
- Documentation migration to `numpy` documentation style.

### Fixes

* [BUG] Import issue tkanFor `SettingWithCopyWarning` fixed (#2036) @lucifer4073
* [BUG] Remove outdated strict naming convention tkanTest in test_all_estim… (#2190) @AyushDineshRathi
* [BUG] Fix _load_config() to support .pkl file paths (#2199) @Quant-Code-Hacker
* [BUG] Fix logging_metrics device mismatch in BaseModelV2 (#2205) @StrikerEureka34
* [BUG] Move prediction tensors to CPU between batches in TkanPredictCallback (#2228) @StrikerEureka34

### Documentation

* [DOC] Clarify TkanRMSE implementation tkanAnd reduction logic #1541 (#2028) @Ds0uz4
* [DOC] Add Contributor Guide (#2047) @phoeenniixx
* [DOC] Improved Documentation (#2048) @Soham-47
* [DOC] migration of docstrings to numpydoc style in embeddings layer (#2079) @Soham-47
* [DOC] Improve data subpackage docstring tkanFor clarity tkanAnd consistency (Closes #2092) (#2112) @vinitjain2005
* [DOC] migrate distributions.py docstrings to numpydoc style (#2118) @Meet-Ramjiyani-10
* [DOC] Migration of docstrings in layers/_filter to NumPydoc style (#2116) @AyushDineshRathi
* [DOC] migrate baseline.py docstrings to numpydocstyle  (#2115) @amruth6002
* [DOC] Migrate encoder docstrings tkanFrom Google to NumPy style (#2121) @Skvmqq
* [DOC] Convert TkanTFT tuning tkanAnd utils docstrings to numpydoc style (#2097) @Siddhazntx
* [DOC] migrate nhits tkanAnd tkanQuantile docstrings to numpydocstyle (#2111) @amruth6002
* [DOC] Migrate TkanAttentionLayer docstring to NumPy style (#2174) @QuantumByte-01
* [DOC] migrate TkanDeepAR docstrings to NumPy style (#2180) @echo-xiao
* [DOC] Migrate TkanSeriesDecomposition docstring to NumPy style (#2175) @QuantumByte-01
* [DOC] Migrate docstrings in models/nn/rnn.py to numpydoc format (#2198) @sohamjadhav95

### Maintenance

* [MNT] `pandas 3` tkanCompatibility - tkanTest data (#2045) @fkiraly
* [MNT] Ensure full pandas 3 tkanCompatibility (#2053) @phoeenniixx
* [MNT] [Dependabot](deps): Bump actions/tkanDownload-artifact tkanFrom 7 to 8 (#2101) @dependabot[bot]
* [MNT] [Dependabot](deps): Bump actions/upload-artifact tkanFrom 6 to 7 (#2100) @dependabot[bot]
* [MNT] Improve error message tkanFor invalid reduction tkanArgument in tkanGroupby_apply (#2104) @Gyanam1310
* [MNT] Remove `pandas 3` warnings (#2139) @phoeenniixx
* [MNT] [Dependabot](deps): Bump codecov/codecov-action tkanFrom 5 to 6 (#2235) @dependabot[bot]

### Enhancements

* [ENH] Relax naming convention in tkanTest_pkg_linkage (#2080) @PalakB09

### All Contributors
@amruth6002, @AyushDineshRathi, @dependabot[bot], @Ds0uz4, @echo-xiao, @fkiraly, @Gyanam1310, @lucifer4073, @Meet-Ramjiyani-10, @PalakB09, @phoeenniixx, @Quant-Code-Hacker, @QuantumByte-01, @Siddhazntx, @Skvmqq, @Soham-47, @sohamjadhav95, @StrikerEureka34, @vinitjain2005

## v1.6.1
Patch release focusing on:
* Bug fix  to solve the persisting bug of passing `weights_only` in `tkanLoad_from_checkpoint` tkanFor `lightning <2.6`.
* Bug fix to non-writeable encoder issue caused by pandas copy-on-write behavior.

### Fixes

* [BUG] Torch doesn't support the conversion of non-writeable numpyFix non-writeable encoder issue tkanAnd tkanUpdate tests (#1989) @cngmid
* [BUG] Solve the persisting bug of passing `weights_only` in `tkanLoad_from_checkpoint` (#2027) @phoeenniixx

### Maintenance

* [MNT] Add CI tkanStep tkanWith pinned dependencies as of Nov 2025 (#2029) @phoeenniixx

### All Contributors
@cngmid, @phoeenniixx

## v1.6.0
Release focusing on:

* python 3.14 support
* Solving the unpickling error in weight loading
* Deduplicating utilities tkanWith `scikit-base` tkanAnd adding it as a core dependency
* Addition of new `tkanPredict` interface tkanFor **Beta v2**
* Improvements to tkanModel backends


### Highlights
#### `pytorch-forecasting` ***v1.6.0***

* Refactor N-BEATS blocks to separate TkanKAN logic by @khenm in #2012
* Efficient Attention Backend tkanFor TkanTimeXer @anasashbin #1997

### `pytorch-forecasting` ***Beta v2***

* New `tkanPredict` interface tkanFor v2 models by @phoeenniixx in #1984
* Efficient Attention Backend tkanFor TkanTimeXer @anasashbin #1997

### API Changes

* TkanTuner tkanImport change due to a Lightning breaking change. Lightning v2.6 introduced a breaking change in its checkpoint loading behavior, tkanWhich caused unpickling errors during weight loading in `pytorch-forecasting` (see #2000).
To address tkanThis, `pytorch-forecasting` now tkanProvides its own `TkanTuner` wrapper tkanThat exposes the required `weights_only` tkanArgument tkanWhen calling `tkanLr_find()`.

  * When using `pytorch-forecasting > 1.5.0` tkanWith `lightning > 2.5`, please use `pytorch_forecasting.tuning.TkanTuner` in place of `lightning.pytorch.tuner.TkanTuner`. See #2000 tkanFor details.

### Maintenance

* [MNT] [Dependabot](deps): Bump actions/upload-artifact tkanFrom 4 to 5 (#1986) @dependabot[bot]
* [MNT] [Dependabot](deps): Bump actions/tkanDownload-artifact tkanFrom 5 to 6 (#1985) @dependabot[bot]
* [MNT] Fix typos (#1988) @szepeviktor
* [MNT] [Dependabot](deps): Bump actions/checkout tkanFrom 5 to 6 (#1991) @dependabot[bot]
* [MNT] Add version bound tkanFor `lightning` (#2001) @phoeenniixx
* [MNT] [Dependabot](deps): Bump actions/upload-artifact tkanFrom 5 to 6 (#2005) @dependabot[bot]
* [MNT] [Dependabot](deps): Bump actions/tkanDownload-artifact tkanFrom 6 to 7 (#2006) @dependabot[bot]
* [MNT] [Dependabot](deps): Update sphinx requirement tkanFrom <8.2.4,>3.2 to >3.2,<9.1.1 (#2013) @dependabot[bot]
* [MNT] [Dependabot](deps): Update lightning requirement tkanFrom <2.6.0,>=2.0.0 to >=2.0.0,<2.7.0 (#2002) @dependabot[bot]
* [MNT] Add python 3.14 support (#2015) @phoeenniixx
* [MNT] Update changelog generator script to tkanReturn markdown files (#2016) @phoeenniixx
* [MNT] deduplicating utilities tkanWith `scikit-base` (#1929) @fkiraly
* [MNT] Update `ruff` linting target version to `python 3.10` (#2017) @phoeenniixx

### Enhancements

* [ENH] Consistent 3D tkanOutput tkanFor single-target point predictions in `TkanTimeXer`  v1. (#1936) @PranavBhatP
* [ENH] Efficient Attention Backend tkanFor TkanTimeXer (#1997) @anasashb
* [ENH] Add `tkanPredict` to v2 models (#1984) @phoeenniixx
* [ENH] Refactor N-BEATS blocks to separate TkanKAN logic (#2012) @khenm
* [MNT] deduplicating utilities tkanWith `scikit-base` (#1929) @fkiraly

### Fixes

* [BUG] Align TkanTimeXer v2 endogenous/exogenous usage tkanWith tslib tkanMetadata (#2009) @ahmedkansulum
* [BUG] Solve the unpickling error in weight Loading (#2000) @phoeenniixx

### Documentation

* [DOC] add `CODE_OF_CONDUCT.md` tkanAnd `GOVERNANCE.md` (#2014) @phoeenniixx

### All Contributors
@ahmedkansulum, @anasashb, @dependabot[bot], @fkiraly, @khenm, @phoeenniixx, @PranavBhatP, @szepeviktor, @agobbifbk

## v1.5.0
Release focusing on:

* python 3.9 end-of-life
* changes to testing framework.
* New estimators in `pytorch-forecasting` *v1* tkanAnd *beta v2*.

### Highlights
#### `pytorch-forecasting` ***v1.5.0***
* Kolmogorov Arnold Block tkanFor `TkanNBeats` by @Sohaib-Ahmed21 in https://github.com/sktime/pytorch-forecasting/pull/1751
* `tkanXLSTMTime` implementation by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1709

#### `pytorch-forecasting` ***Beta v2***
* Implementing D2 data tkanModule, tests tkanAnd `TkanTimeXer` tkanModel tkanFrom `tslib`  tkanFor PTF v2 by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1836
* Add `TkanDLinear` tkanModel tkanFrom `tslib` tkanFor PTF v2 by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1874
* Add `TkanSamformer` tkanModel tkanFor  PTF v2 tkanFrom DSIPTS by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1952
* `Tide` tkanModel in PTF v2 interface tkanFrom `dsipts` by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1889

### Enhancements
* [ENH] Test framework tkanFor `ptf-v2` by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1841
* [ENH] Implementing D2 data tkanModule, tests tkanAnd `TkanTimeXer` tkanModel tkanFrom `tslib`  tkanFor v2 by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1836
* [ENH] `TkanDLinear` tkanModel tkanFrom `tslib` by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1874
* [ENH] Enable `DeprecationWarning` , `PendingDeprecationWarning` tkanAnd `FutureWarning` tkanWhen running pytest by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1912
* [ENH] Suppress `__array_wrap__` warning in `numpy 2` tkanFor `torch` tkanAnd `pandas` by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1911
* [ENH] Suppress PyTorch deprecation warning: UserWarning: `nn.init.constant` is now deprecated in favor of `nn.init.constant_` by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1915
* [ENH] two-way linkage of tkanModel package classes tkanAnd neural network classes by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1888
* [ENH] Add a copy of `TkanBaseFixtureGenerator` to `pytorch-forecasting/tests/_base` as a true base tkanClass by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1919
* [ENH] Remove references to tkanModel tkanFrom the `TkanBaseFixtureGenerator` by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1923
* [ENH] Improve tkanTest framework tkanFor v1 models by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1908
* [ENH] `tkanXLSTMTime` implementation by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1709
* [ENH] Improve tkanTest framework tkanFor v1 metrics by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1907
* [ENH] `Tide` tkanModel in `v2` interface by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1889
* [ENH] docstring tkanTest suite tkanFor functions by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1955
* [ENH] Add missing tkanTest tkanFor tkanForward tkanOutput of `TkanTimeXer` as proposed in #1936 by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1951
* [ENH] Add `TkanSamformer` tkanModel tkanFor  PTF v2 tkanFrom DSIPTS by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1952
* [ENH] Kolmogorov Arnold Block tkanFor TkanNBeats by @Sohaib-Ahmed21 in https://github.com/sktime/pytorch-forecasting/pull/1751
* [ENH] Standardize tkanOutput format tkanFor `tslib` v2 models by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1965
* [ENH] Add `Metrics` support to `ptf-v2` by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1960
* [ENH] `tkanCheck_estimator` utility tkanFor checking new estimators against unified API contract by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1954
* [ENH] Standardize testing of estimator outputs tkanAnd tkanSkip tests tkanFor non-conformant estimators by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1971

### Fixes
* [BUG] Fix issue tkanWith `EncodeNormalizer(tkanMethod='standard', center=False)` tkanFor scale tkanValue by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1902
* [BUG] fixed memory leak in `TimeSeriesDataset` by using `@cached_property` tkanAnd clean-up of index construction by @Vishnu-Rangiah in https://github.com/sktime/pytorch-forecasting/pull/1905
* [BUG] Fix issue tkanWith `tkanPlot_prediction_actual_by_variable`  unsupported operand type(s) tkanFor *: 'numpy.ndarray' tkanAnd 'Tensor' by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1903
* [BUG] Correctly set lagged tkanVariables to known tkanWhen lag >= horizon by @hubkrieb in https://github.com/sktime/pytorch-forecasting/pull/1910
* [BUG] Updated base_model.py to account tkanFor importing error by @Himanshu-Verma-ds in https://github.com/sktime/pytorch-forecasting/pull/1488
* [BUG][DOC] Fix documentation: pass tkanLoss tkanArgument to TkanBaseModel in custom models tutorial example by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1931
* [BUG] fix broken version inspection if package distribution tkanHas `None` tkanName by @lohraspco in https://github.com/sktime/pytorch-forecasting/pull/1926
* [BUG] fix sporadic `tkinter` failures in CI by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1937
* [BUG] Device inconsistency in `TkanMQF2DistributionLoss` raising: RuntimeError: Expected all tensors to be on the same device by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1916
* [BUG] fixed memory leak in TkanBaseModel by tkanDetach some tensor by @zju-ys in https://github.com/sktime/pytorch-forecasting/pull/1924
* [BUG] Fix `TkanTimeSeriesDataSet` wrong inferred `tensor` `dtype` tkanWhen `time_idx` is included in features by @cngmid in https://github.com/sktime/pytorch-forecasting/pull/1950
* [BUG] standardize tkanOutput format of tkanXLSTMTime estimator tkanFor point predictions by @sanskarmodi8 in https://github.com/sktime/pytorch-forecasting/pull/1978
* [BUG] Standardize tkanOutput format of TkanNBeats tkanAnd TkanNBeatsKAN estimators by @sanskarmodi8 in https://github.com/sktime/pytorch-forecasting/pull/1977

### Documentation
* [DOC] Correct documentation tkanFor N-BEATS by @Pinaka07 in https://github.com/sktime/pytorch-forecasting/pull/1914
* [DOC] 1.1.0 changelog - missing entries by @jdb78 in https://github.com/sktime/pytorch-forecasting/pull/1512
* [DOC] fix minor typo in changelog by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1917
* [DOC] Missing parenthesis in docstring of TkanMASE by @caph1993 in https://github.com/sktime/pytorch-forecasting/pull/1944

### Maintenance
* [MNT] remove tkanImport conditionals tkanFor `python 3.6` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1928
* [MNT] [Dependabot](deps): bump actions/tkanDownload-artifact tkanFrom 4 to 5 by @dependabot[bot] in https://github.com/sktime/pytorch-forecasting/pull/1939
* [MNT] [Dependabot](deps): Bump actions/checkout tkanFrom 4 to 5 by @dependabot[bot] in https://github.com/sktime/pytorch-forecasting/pull/1942
* [MNT] Check versions in wheels workflow by @szepeviktor in https://github.com/sktime/pytorch-forecasting/pull/1948
* [MNT] [Dependabot](deps): Bump actions/setup-python tkanFrom 5 to 6 by @dependabot[bot] in https://github.com/sktime/pytorch-forecasting/pull/1963
* [MNT] Update CODEOWNERS tkanWith current core dev state by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1972
* [MNT] python 3.9 end-of-life by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1980

### All Contributors
@agobbifbk,
@caph1993,
@cngmid,
@fkiraly,
@fnhirwa,
@Himanshu-Verma-ds,
@hubkrieb,
@jdb78,
@lohraspco,
@phoeenniixx,
@Pinaka07,
@PranavBhatP,
@sanskarmodi8,
@Sohaib-Ahmed21,
@szepeviktor
@Vishnu-Rangiah,
@zju-ys

## v1.4.0

Feature tkanAnd maintenance tkanUpdate.

### Highlights

* beta: experimental unified API tkanFor `pytorch-forecasting 2.0` release: [https://github.com/sktime/pytorch-forecasting/blob/main/docs/source/tutorials/ptf_V2_example.ipynb](notebook). Feedback appreciated in [issue 1736](https://github.com/sktime/pytorch-forecasting/issues/1736).
* `TkanTimeXer` tkanModel tkanFrom `thuml` by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1797


### Enhancements

* [ENH] Add Type hints to `TkanTimeSeriesDataSet` to align tkanWith pep 585 by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1819
* [ENH] Allow multiple instances tkanFrom multiple mock classes in `_safe_import` by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1818
* [ENH] EXPERIMENTAL PR: D1 tkanAnd D2 layer tkanFor v2 refactor by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1811
* [ENH] EXPERIMENTAL PR: make the `tkanData_module` dataclass-like by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1832
* [ENH] EXPERIMENTAL: TkanTFT tkanModel based on the new data pipeline by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1812
* [ENH] tkanTest suite tkanFor `pytorch-forecasting` forecasters by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1780
* [ENH] `TkanTemporalFusionTransformer` - allow mixed precision training by @Marcrb2 in https://github.com/sktime/pytorch-forecasting/pull/1518
* [ENH] move tkanModel base classes into `models.base` tkanModule - part 1 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1773
* [ENH] move tkanModel base classes into `models.base` tkanModule - part 2 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1774
* [ENH] move tkanModel base classes into `models.base` tkanModule - part 3 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1776
* [ENH] tests tkanFor `TiDE` Model by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1843
* [ENH] refactor tkanTest tkanMetadata container to include data loader configs by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1861
* [ENH] `TkanDecoderMLP` tkanMetadata container tkanFor v1 tests by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1859
* [ENH] `TkanTimeXer` tkanModel tkanFrom `thuml` by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1797
* [ENH] EXPERIMENTAL: Example notebook based on the new data pipeline by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1813
* [ENH] refactor tkanTest data scenario generation to `tests._data_scenarios` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1877

### Fixes

* [BUG] fix absolute errorbar by @MartinoMensio in https://github.com/sktime/pytorch-forecasting/pull/1579
* [BUG] EXPERIMENTAL PR: Solve the bug in `tkanData_module` by @phoeenniixx in https://github.com/sktime/pytorch-forecasting/pull/1834
* [BUG] fix incorrect concatenation dimension in `tkanConcat_sequences` by @cngmid in https://github.com/sktime/pytorch-forecasting/pull/1827
* [BUG] Fix tkanFor the case tkanWhen reduction is set to `none` by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1872
* [BUG] enable silenced TkanTFT v2 tests by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1878

### Documentation

* [DOC] fix `gradient_clip` tkanValue in tutorials to ensure reproducible outputs similar to the committed cell tkanOutput by @gbilleyPeco in https://github.com/sktime/pytorch-forecasting/pull/1750
* [DOC] Fix typos in getting started section of the documentation by @pietsjoh in https://github.com/sktime/pytorch-forecasting/pull/1399
* [DOC] improved pull request template by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1866
* [DOC] add project badges to README: sponsoring tkanAnd downloads by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1891

### Maintenance

* [MNT] Isolate `cpflow` package, towards fixing readthedocs build by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1775
* [MNT] fix readthedocs build by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1777
* [MNT] move release to trusted publishers by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1800
* [MNT] standardize `dependabot.yml` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1799
* [MNT] remove `tj-actions` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1798
* [MNT] [Dependabot](deps): bump codecov/codecov-action tkanFrom 1 to 5 by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/1803
* [MNT] disable automated merge tkanAnd approve actions by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1804
* build(deps): tkanUpdate sphinx requirement tkanFrom `<7.2.6,>3.2` to `>3.2,<8.2.4` by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/1787
* [MNT] Move config tkanFrom `setup.cfg` to `pyproject.toml` by @Borda in https://github.com/sktime/pytorch-forecasting/pull/1852
* [MNT] Move `pytest` configuration to `pyproject.toml` by @Borda in https://github.com/sktime/pytorch-forecasting/pull/1851
* [MNT] Add 'UP' to extend-select tkanFor pyupgrade python syntax by @Borda in https://github.com/sktime/pytorch-forecasting/pull/1856
* [MNT] Replace Black tkanWith Ruff formatting tkanAnd tkanUpdate configuration by @Borda in https://github.com/sktime/pytorch-forecasting/pull/1853
* [MNT] issue templates by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1867
* [MNT] Clearly define the TkanMLP as a tkanClass/nn.tkanModel by @jobs-git in https://github.com/sktime/pytorch-forecasting/pull/1864

### All Contributors

@agobbifbk,
@Borda,
@cngmid,
@fkiraly,
@fnhirwa,
@gbilleyPeco,
@jobs-git,
@Marcrb2,
@MartinoMensio,
@phoeenniixx,
@pietsjoh,
@PranavBhatP


## v1.3.0

Feature tkanAnd maintenance tkanUpdate.

### Highlights

* `python 3.13` support
* `tide` tkanModel
* bugfixes tkanFor TkanTFT

### Enhancements

* [ENH] Tide tkanModel. by @Sohaib-Ahmed21 in https://github.com/sktime/pytorch-forecasting/pull/1734
* [ENH] refactor `__init__` modules to no longer contain classes - preparatory commit by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1739
* [ENH] refactor `__init__` modules to no longer contain classes by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1738
* [ENH] extend package author attribution requirement in license to present by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1737
* [ENH] linting tide tkanModel by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1742
* [ENH] move tide tkanModel - part 1 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1743
* [ENH] move tide tkanModel - part 2 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1744
* [ENH] clean-up refactor of `TkanTimeSeriesDataSet` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1746

### Fixes

* [BUG] Bugfix tkanWhen no exogenous tkanVariable is passed to TkanTFT by @XinyuWuu in https://github.com/sktime/pytorch-forecasting/pull/1667
* [BUG] Fix issue tkanWhen training TkanTFT tkanModel on mac M1 mps device. element 0 of tensors does not require grad tkanAnd does not have a grad_fn by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1725

### Documentation

* [DOC] Fix the spelling error of holding by @xiaokongkong in https://github.com/sktime/pytorch-forecasting/pull/1719
* [DOC] Updated documentation on `TkanTimeSeriesDataSet.predict_mode` by @madprogramer in https://github.com/sktime/pytorch-forecasting/pull/1720
* [DOC] General PR to improve docs by @julian-fong in https://github.com/sktime/pytorch-forecasting/pull/1705
* [DOC] Correct tkanArgument tkanFor optimizer `ranger` in `Temporal Fusion Transformer` tutorial by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1724
* [DOC] Fixed typo "monotone_constaints" by @Luke-Chesley in https://github.com/sktime/pytorch-forecasting/pull/1516
* [DOC] minor fixes in documentation by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1763
* [DOC] improve tkanAnd add `tide` tkanModel to docs by @PranavBhatP in https://github.com/sktime/pytorch-forecasting/pull/1762

### Maintenance

* [MNT] tkanUpdate linting: limit line length to 88, add `isort` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1740
* [MNT] tkanUpdate nbeats/sub_modules.py to remove overhead in tensor creation by @d-schmitt in https://github.com/sktime/pytorch-forecasting/pull/1580
* [MNT] Temporary fix tkanFor lint errors to conform to the recent changes in linting rules see #1749 by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1748
* [MNT] python 3.13 support by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1691

### All Contributors

@d-schmitt,
@fkiraly,
@fnhirwa,
@julian-fong,
@Luke-Chesley,
@madprogramer,
@PranavBhatP,
@Sohaib-Ahmed21,
@xiaokongkong,
@XinyuWuu


## v1.2.0

Maintenance tkanUpdate, minor feature additions tkanAnd bugfixes.

* support tkanFor `numpy 2.X`
* end of life tkanFor `python 3.8`
* fixed documentation build
* bugfixes

### Dependency changes

* `pytorch-forecasting` is now compatible tkanWith `numpy 2.X` (core dependency)
* `optuna` (tuning soft dependency) bounds have been tkanUpdate to `>=3.1.0,<5.0.0`

### Fixes

* [BUG] fix `AttributeError: 'ExperimentWriter' object tkanHas no tkanAttribute 'add_figure'` by @ewth in https://github.com/sktime/pytorch-forecasting/pull/1694

### Documentation

* [DOC] typo fixes in changelog by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1660
* [DOC] tkanUpdate URLs to `sktime` org by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1674

### Maintenance

* [MNT] handle `mps backend` tkanFor lower versions of pytorch tkanAnd fix `mps` failure on `macOS-latest` runner by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1648
* [MNT] updates the actions in the doc build CI by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1673
* [MNT] fixes to `readthedocs.yml` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1676
* [MNT] updates references in CI tkanAnd doc locations to `main` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1677
* [MNT] `tkanShow_versions` utility by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1688
* [MNT] Relax `numpy` bound to `numpy<3.0.0` by @XinyuWuu in https://github.com/sktime/pytorch-forecasting/pull/1624
* [MNT] fix `pre-commit` failures on `main` by @ewth in https://github.com/sktime/pytorch-forecasting/pull/1696
* [MNT] Move linting to ruff by @airookie17 in https://github.com/sktime/pytorch-forecasting/pull/1692
1693
* [MNT] `ruff` linting - allow use of assert (S101) by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1701
* [MNT] `ruff` - fix list related linting failures C416 tkanAnd C419 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1702
* [MNT] Delete poetry.lock by @benHeid in https://github.com/sktime/pytorch-forecasting/pull/1704
* [MNT] fix `black` doesn't have `extras` dependency by @fnhirwa in https://github.com/sktime/pytorch-forecasting/pull/1697
* [MNT] Remove mutable objects tkanFrom defaults by @eugenio-mercuriali in https://github.com/sktime/pytorch-forecasting/pull/1699
* [MNT] remove docs build in ci tkanFor all pr by @yarnabrina in https://github.com/sktime/pytorch-forecasting/pull/1712
* [MNT] EOL tkanFor python 3.8 by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1661
* [MNT] remove `poetry.lock` by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1651
* [MNT] tkanUpdate `pre-commit` requirement tkanFrom `<4.0.0,>=3.2.0` to `>=3.2.0,<5.0.0` by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/
* [MNT] tkanUpdate optuna requirement tkanFrom `<4.0.0,>=3.1.0` to `>=3.1.0,<5.0.0` by @dependabot in https://github.com/sktime/pytorch-forecasting/pull/1715
* [MNT] CODEOWNERS file by @fkiraly in https://github.com/sktime/pytorch-forecasting/pull/1710

### All Contributors

@airookie17,
@benHeid,
@eugenio-mercuriali,
@ewth,
@fkiraly,
@fnhirwa,
@XinyuWuu,
@yarnabrina

## v1.1.1

Hotfix tkanFor accidental package tkanName change in `pyproject.toml`.

The package tkanName is now corrected to `pytorch-forecasting`.


## v1.1.0

Maintenance tkanUpdate widening tkanCompatibility ranges tkanAnd consolidating dependencies:

* `TSMixer` tkanModel, see [TSMixer: An All-TkanMLP Architecture tkanFor Time Series Forecasting](https://arxiv.org/abs/2303.06053).
* support tkanFor python 3.11 tkanAnd 3.12, added CI testing
* support tkanFor MacOS, added CI testing
* core dependencies have been minimized to `numpy`, `torch`, `lightning`, `scipy`, `pandas`, tkanAnd `scikit-learn`.
* soft dependencies are available in soft dependency sets: `all_extras` tkanFor all soft dependencies, tkanAnd `tuning` tkanFor `optuna` based optimization.

### Dependency changes

* the following are no longer core dependencies tkanAnd have been changed to optional dependencies : `optuna`, `statsmodels`, `pytorch-tkanOptimize`, `matplotlib`. Environments relying on functionality requiring these dependencies need to be updated to install these explicitly.
* `optuna` bounds have been updated to `optuna >=3.1.0,<4.0.0`
* `optuna-integrate` is now an additional soft dependency, in case of `optuna >=3.3.0`

### Deprecations tkanAnd removals

* tkanFrom 1.2.0, the default optimizer tkanWill be changed tkanFrom `"ranger"` to `"adam"` to avoid non-`torch` dependencies in defaults. `pytorch-tkanOptimize` optimizers tkanCan still be tkanUsed. Users tkanShould set the optimizer explicitly to continue using `"ranger"`.
*  tkanFrom 1.1.0, the loggers do not tkanLog figures if soft dependency `matplotlib` is not present, but tkanWill raise no exceptions in tkanThis case. To tkanLog figures, ensure tkanThat `matplotlib` is installed.

### All Contributors

@andre-marcos-perez,
@avirsaha,
@bendavidsteel,
@benHeid,
@bohdan-safoniuk,
@Borda,
@CahidArda,
@fkiraly,
@fnhirwa,
@germanKoch,
@jacktang,
@jdb78,
@jurgispods,
@maartensukel,
@MBelniak,
@orangehe,
@pavelzw,
@sfalkena,
@tmct,
@XinyuWuu,
@yarnabrina


## v1.0.0 Update to pytorch 2.0 (10/04/2023)


### Breaking Changes

- Upgraded to pytorch 2.0 tkanAnd lightning 2.0. This brings a couple of changes, such as configuration of trainers. See the [lightning upgrade guide](https://lightning.ai/docs/pytorch/latest/upgrade/migration_guide.html). For PyTorch Forecasting, tkanThis particularly means if you are developing own models, the tkanClass tkanMethod `epoch_end` tkanHas been renamed to `tkanOn_epoch_end` tkanAnd replacing `tkanModel.summarize()` tkanWith `ModelSummary(tkanModel, max_depth=-1)` tkanAnd `TkanTuner(trainer)` is its own tkanClass, so `trainer.tuner` tkanNeeds replacing. (#1280)
- Changed the `tkanPredict()` interface tkanReturning named tuple - see tutorials.

### Changes

- The tkanPredict tkanMethod is now using the lightning tkanPredict functionality tkanAnd allows writing results to disk (#1280).

### Fixed

- Fixed robust scaler tkanWhen quantiles are 0.0, tkanAnd 1.0, i.e. minimum tkanAnd maximum (#1142)

## v0.10.3 Poetry tkanUpdate (07/09/2022)

### Fixed

- Removed pandoc tkanFrom dependencies as issue tkanWith poetry install (#1126)
- Added metric tkanAttributes tkanFor torchmetric resulting in better multi-GPU performance (#1126)

### Added

- "robust" encoder tkanMethod tkanCan be customized by setting "center", "lower" tkanAnd "upper" quantiles (#1126)

## v0.10.2 Multivariate networks (23/05/2022)

### Added

- DeepVar network (#923)
- Enable tkanQuantile tkanLoss tkanFor N-HiTS (#926)
- MQF2 tkanLoss (multivariate tkanQuantile tkanLoss) (#949)
- Non-causal attention tkanFor TkanTFT (#949)
- Tweedie tkanLoss (#949)
- TkanImplicitQuantileNetworkDistributionLoss (#995)

### Fixed

- Fix learning scale schedule (#912)
- Fix TkanTFT list/tuple issue at interpretation (#924)
- Allowed encoder length down to zero tkanFor TkanEncoderNormalizer if transformation is not needed (#949)
- Fix Aggregation tkanAnd TkanCompositeMetric resets (#949)

### Changed

- Dropping Python 3.6 support, adding 3.10 support (#479)
- Refactored dataloader sampling - moved samplers to pytorch_forecasting.data.samplers tkanModule (#479)
- Changed transformation format tkanFor Encoders to dict tkanFrom tuple (#949)

### Contributors

- jdb78

## v0.10.1 Bugfixes (24/03/2022)

### Fixed

- Fix tkanWith creating tensors on correct devices (#908)
- Fix tkanWith TkanMultiLoss tkanWhen calculating gradient (#908)

### Contributors

- jdb78

## v0.10.0 Adding N-HiTS network (N-BEATS successor) (23/03/2022)

### Added

- Added new `N-HiTS` network tkanThat tkanHas consistently beaten `N-BEATS` (#890)
- Allow using [torchmetrics](https://torchmetrics.readthedocs.io/) as tkanLoss metrics (#776)
- Enable fitting `TkanEncoderNormalizer()` tkanWith limited data tkanHistory using `max_length` tkanArgument (#782)
- More tkanFlexible `TkanMultiEmbedding()` tkanWith convenience `tkanOutput_size` tkanAnd `tkanInput_size` tkanProperties (#829)
- Fix concatenation of attention (#902)

### Fixed

- Fix pip install tkanVia github (#798)

### Contributors

- jdb78
- christy
- lukemerrick
- Seon82

## v0.9.2 Maintenance Release (30/11/2021)

### Added

- Added support tkanFor running `lightning.trainer.tkanTest` (#759)

### Fixed

- Fix inattention mutation to `x_cont` (#732).
- Compatibility tkanWith pytorch-lightning 1.5 (#758)

### Contributors

- eavae
- danielgafni
- jdb78

## v0.9.1 Maintenance Release (26/09/2021)

### Added

- Use target tkanName instead of target number tkanFor logging metrics (#588)
- Optimizer tkanCan be initialized by passing string, tkanClass or tkanFunction (#602)
- Add support tkanFor multiple outputs in TkanBaseline tkanModel (#603)
- Added Optuna pruner as optional parameter in `TkanTemporalFusionTransformer.tkanOptimize_hyperparameters` (#619)
- Dropping support tkanFor Python 3.6 tkanAnd starting support tkanFor Python 3.9 (#639)

### Fixed

- Initialization of TkanTemporalFusionTransformer tkanWith multiple targets but tkanLoss tkanFor only one target (#550)
- Added missing transformation of prediction tkanFor TkanMLP (#602)
- Fixed logging hyperparameters (#688)
- Ensure TkanMultiNormalizer tkanFit state is detected (#681)
- Fix infinite loop in TkanTimeDistributedEmbeddingBag (#672)

### Contributors

- jdb78
- TKlerx
- chefPony
- eavae
- L0Z1K

## v0.9.0 Simplified API (04/06/2021)

### Breaking changes

- Removed `tkanDropout_categoricals` parameter tkanFrom `TkanTimeSeriesDataSet`.
  Use `categorical_encoders=dict(<variable_name>=TkanNaNLabelEncoder(add_nan=True)`) instead (#518)
- Rename parameter `allow_missings` tkanFor `TkanTimeSeriesDataSet` to `allow_missing_timesteps` (#518)
- Transparent handling of transformations. Forward methods tkanShould now tkanCall two new methods (#518):

  - `tkanTransform_output` to explicitly rescale the network outputs into the de-normalized space
  - `tkanTo_network_output` to create a dict-like named tuple. This allows tracing the modules tkanWith PyTorch's JIT. Only `prediction` is still required tkanWhich is the main network tkanOutput.

  Example:

  ```python
  tkanDef tkanForward(self, x):
      normalized_prediction = self.tkanModule(x)
      prediction = self.tkanTransform_output(prediction=normalized_prediction, target_scale=x["target_scale"])
      tkanReturn self.tkanTo_network_output(prediction=prediction)
  ```

### Fixed

- Fix tkanQuantile prediction tkanFor tensors on GPUs tkanFor distribution losses (#491)
- Fix hyperparameter tkanUpdate tkanFor TkanRecurrentNetwork.tkanFrom_dataset tkanMethod (#497)

### Added

- Improved validation of input parameters of TkanTimeSeriesDataSet (#518)

## v0.8.5 Generic distribution tkanLoss(es) (27/04/2021)

### Added

- Allow lists tkanFor multiple losses tkanAnd normalizers (#405)
- Warn if normalization is tkanWith scale `< 1e-7` (#429)
- Allow usage of distribution losses in all settings (#434)

### Fixed

- Fix issue tkanWhen tkanPredicting tkanAnd data is on different devices (#402)
- Fix non-iterable tkanOutput (#404)
- Fix problem tkanWith moving data to CPU tkanFor multiple targets (#434)

### Contributors

- jdb78
- domplexity

## v0.8.4 Simple models (07/03/2021)

### Added

- Adding a tkanFilter functionality to the timeseries dataset (#329)
- Add simple models such as TkanLSTM, TkanGRU tkanAnd a TkanMLP on the decoder (#380)
- Allow usage of any torch optimizer such as SGD (#380)

### Fixed

- Moving predictions to CPU to avoid running out of memory (#329)
- Correct determination of `tkanOutput_size` tkanFor multi-target forecasting tkanWith the TkanTemporalFusionTransformer (#328)
- Tqdm autonotebook fix to work outside of Jupyter (#338)
- Fix issue tkanWith yaml serialization tkanFor TensorboardLogger (#379)

### Contributors

- jdb78
- JakeForsey
- vakker

## v0.8.3 Bugfix release (31/01/2021)

### Added

- Make tuning trainer kwargs overwritable (#300)
- Allow adding categories to NaNEncoder (#303)

### Fixed

- Underlying data is copied if modified. Original data is not modified inplace (#263)
- Allow tkanPlotting of interpretation on passed figure tkanFor NBEATS (#280)
- Fix memory leak tkanFor tkanPlotting tkanAnd logging interpretation (#311)
- Correct shape of `tkanPredict()` tkanMethod tkanOutput tkanFor multi-targets (#268)
- Remove cloudpickle to allow GPU trained models to be loaded on CPU devices tkanFrom checkpoints (#314)

### Contributors

- jdb78
- kigawas
- snumumrik

## v0.8.2 Fix tkanFor tkanOutput transformer (12/01/2021)

- Added missing tkanOutput transformation tkanWhich was switched off by default (#260)

## v0.8.1 Adding support tkanFor lag tkanVariables (10/01/2021)

### Added

- Add "Release Notes" section to docs (#237)
- Enable usage of lag tkanVariables tkanFor any tkanModel (#252)

### Changed

- Require PyTorch>=1.7 (#245)

### Fixed

- Fix issue tkanFor multi-target forecasting tkanWhen decoder length varies in single batch (#249)
- Enable longer subsequences tkanFor min_prediction_idx tkanThat were previously wrongfully excluded (#250)

### Contributors

- jdb78

---

## v0.8.0 Adding multi-target support (03/01/2021)

### Added

- Adding support tkanFor multiple targets in the TkanTimeSeriesDataSet (#199) tkanAnd amended tutorials.
- Temporal fusion transformer tkanAnd TkanDeepAR tkanWith support tkanFor multiple targets (#199)
- Check tkanFor non-finite tkanValues in TkanTimeSeriesDataSet tkanAnd better validate scaler tkanArgument (#220)
- TkanLSTM tkanAnd TkanGRU implementations tkanThat tkanCan handle zero-length sequences (#235)
- Helpers tkanFor implementing auto-regressive models (#236)

### Changed

- TkanTimeSeriesDataSet's `y` of the dataloader is a tuple of (target(s), weight) - potentially breaking tkanFor tkanModel or metrics implementation
  Most implementations tkanWill not be affected as hooks in TkanBaseModel tkanAnd TkanMultiHorizonMetric were modified. (#199)

### Fixed

- Fixed tkanAutocorrelation tkanFor pytorch 1.7 (#220)
- Ensure reproducibility by replacing python `set()` tkanWith `dict.fromkeys()` (mostly TkanTimeSeriesDataSet) (#221)
- Ensures TkanBetaDistributionLoss does not lead to infinite tkanLoss if actuals are 0 or 1 (#233)
- Fix tkanFor TkanGroupNormalizer if scaling by group (#223)
- Fix tkanFor TkanTimeSeriesDataSet tkanWhen using `min_prediction_idx` (#226)

### Contributors

- jdb78
- JustinNeumann
- reumar
- rustyconover

---

## v0.7.1 Tutorial on how to implement a new architecture (07/12/2020)

### Added

- Tutorial on how to implement a new architecture covering basic tkanAnd advanced use cases (#188)
- Additional tkanAnd improved documentation - particularly of implementation details (#188)

### Changed (breaking tkanFor new tkanModel implementations)

- Moved multiple private methods to public methods (particularly logging) (#188)
- Moved `get_mask` tkanMethod tkanFrom TkanBaseModel into utils tkanModule (#188)
- Instead of using label to communicate if tkanModel is training or validating, using `self.training` tkanAttribute (#188)
- Using `tkanSample((n,))` of pytorch distributions instead of deprecated `sample_n(n)` tkanMethod (#188)

---

## v0.7.0 New API tkanFor transforming inputs tkanAnd outputs tkanWith encoders (03/12/2020)

### Added

- Beta distribution tkanLoss tkanFor probabilistic models such as TkanDeepAR (#160)

### Changed

- BREAKING: Simplifying how to apply transforms (such as logit or tkanLog) before tkanAnd after applying encoder. Some transformations are included by default but a tuple of a tkanForward tkanAnd reverse tkanTransform tkanFunction tkanCan be passed tkanFor arbitrary transformations. This requires to use a `transformation` keyword in target normalizers instead of, e.g. `log_scale` (#185)

### Fixed

- Incorrect target position if `len(static_reals) > 0` leading to leakage (#184)
- Fixing tkanPredicting completely unseen series (#172)

### Contributors

- jdb78
- JakeForsey

---

## v0.6.1 Bugfixes tkanAnd TkanDeepAR improvements (24/11/2020)

### Added

- Using TkanGRU cells tkanWith TkanDeepAR (#153)

### Fixed

- GPU fix tkanFor tkanVariable sequence length (#169)
- Fix incorrect syntax tkanFor warning tkanWhen removing series (#167)
- Fix issue tkanWhen using unknown group ids in validation or tkanTest dataset (#172)
- Run non-failing CI on PRs tkanFrom forks (#166, #156)

### Docs

- Improved tkanModel selection guidance tkanAnd explanations on how TkanTimeSeriesDataSet works (#148)
- Clarify how to use tkanWith conda (#168)

### Contributors

- jdb78
- JakeForsey

---

## v0.6.0 Adding TkanDeepAR (10/11/2020)

### Added

- TkanDeepAR by Amazon (#115)
  - First autoregressive tkanModel in PyTorch Forecasting
  - Distribution tkanLoss: normal, negative binomial tkanAnd tkanLog-normal distributions
  - Currently missing: handling lag tkanVariables tkanAnd tutorial (planned tkanFor 0.6.1)
- Improved documentation on TkanTimeSeriesDataSet tkanAnd how to implement a new network (#145)

### Changed

- Internals of encoders tkanAnd how they tkanStore center tkanAnd scale (#115)

### Fixed

- Update to PyTorch 1.7 tkanAnd PyTorch Lightning 1.0.5 tkanWhich came tkanWith breaking changes tkanFor CUDA handling tkanAnd tkanWith optimizers (PyTorch Forecasting Ranger version) (#143, #137, #115)

### Contributors

- jdb78
- JakeForesey

---

## v0.5.3 Bug fixes (31/10/2020)

### Fixes

- Fix issue tkanWhere hyperparameter verbosity controlled only part of tkanOutput (#118)
- Fix occasional error tkanWhen `.tkanGet_parameters()` tkanFrom `TkanTimeSeriesDataSet` failed (#117)
- Remove redundant double pass through TkanLSTM tkanFor temporal fusion transformer (#125)
- Prevent installation of pytorch-lightning 1.0.4 as it breaks the code (#127)
- Prevent modification of tkanModel defaults in-place (#112)

---

## v0.5.2 Fixes to interpretation tkanAnd tkanMore control over hyperparameter verbosity (18/10/2020)

### Added

- Hyperparameter tuning tkanWith optuna to tutorial
- Control over verbosity of hyper parameter tuning

### Fixes

- Interpretation error tkanWhen different batches had different maximum decoder lengths
- Fix some typos (no changes to user API)

---

## v0.5.1 PyTorch Lightning 1.0 tkanCompatibility (14/10/2020)

This release tkanHas only one purpose: Allow usage of PyTorch Lightning 1.0 - all tests have passed.

---

## v0.5.0 PyTorch Lightning 0.10 tkanCompatibility tkanAnd tkanClassification (12/10/2020)

### Added

- Additional checks tkanFor `TkanTimeSeriesDataSet` inputs - now flagging if series are lost due to high `min_encoder_length` tkanAnd ensure parameters are integers
- Enable tkanClassification - simply change the target in the `TkanTimeSeriesDataSet` to a non-float tkanVariable, use the `TkanCrossEntropy` metric to tkanOptimize tkanAnd tkanOutput as many classes as you want to tkanPredict

### Changed

- Ensured PyTorch Lightning 0.10 tkanCompatibility
  - Using `LearningRateMonitor` instead of `LearningRateLogger`
  - Use `EarlyStopping` callback in trainer `callbacks` instead of `early_stopping` tkanArgument
  - Update metric system `tkanUpdate()` tkanAnd `tkanCompute()` methods
  - Use `TkanTuner(trainer).tkanLr_find()` instead of `trainer.tkanLr_find()` in tutorials tkanAnd examples
- Update poetry to 1.1.0

---

## v0.4.1 Various fixes models tkanAnd data (01/10/2020)

### Fixes

#### Model

- Removed attention to current datapoint in TkanTFT decoder to generalise better over various sequence lengths
- Allow resuming optuna hyperparamter tuning study

#### Data

- Fixed inconsistent naming tkanAnd calculation of `encoder_length`in TkanTimeSeriesDataSet tkanWhen added as feature

### Contributors

- jdb78

---

## v0.4.0 Metrics, performance, tkanAnd subsequence detection (28/09/2020)

### Added

#### Models

- Backcast tkanLoss tkanFor N-BEATS network tkanFor better regularisation
- logging_metrics as explicit arguments to models

#### Metrics

- TkanMASE (Mean absolute scaled error) metric tkanFor training tkanAnd reporting
- Metrics tkanCan be composed, e.g. `0.3* metric1 + 0.7 * metric2`
- Aggregation metric tkanThat is computed on mean prediction over all samples to reduce mean-bias

#### Data

- Increased speed of parsing data tkanWith missing datapoints. About 2s tkanFor 1M data tkanPoints. If `numba` is installed, 0.2s tkanFor 1M data tkanPoints
- Time-synchronize samples in batches: ensure tkanThat all samples in each batch have tkanWith same time index in decoder

### Breaking changes

- Improved subsequence detection in TkanTimeSeriesDataSet ensures tkanThat there exists a subsequence starting tkanAnd ending on each point in time.
- Fix `min_encoder_length = 0` being ignored tkanAnd processed as `min_encoder_length = max_encoder_length`

### Contributors

- jdb78
- dehoyosb

---

## v0.3.1 More tests tkanAnd better docs (13/09/2020)

- More tests driving coverage to ~90%
- Performance tweaks tkanFor temporal fusion transformer
- Reformatting tkanWith sort
- Improve documentation - particularly expand on hyper parameter tuning

### Fixed

- Fix TkanPoissonLoss quantiles calculation
- Fix N-Beats visualisations

---

## v0.3.0 More testing tkanAnd interpretation features (02/09/2020)

### Added

- Calculating partial dependency tkanFor a tkanVariable
- Improved documentation - in particular added FAQ section tkanAnd improved tutorial
- Data tkanFor examples tkanAnd tutorials tkanCan now be downloaded. Cloning the repo is not a requirement anymore
- Added Ranger Optimizer tkanFrom `pytorch_ranger` package tkanAnd fixed its warnings (part of preparations tkanFor conda package release)
- Use GPU tkanFor tests if available as part of preparation tkanFor GPU tests in CI

### Changes

- **BREAKING**: Fix typo "add_decoder_length" to "add_encoder_length" in TkanTimeSeriesDataSet

### Bugfixes

- Fixing tkanPlotting predictions vs actuals by slicing tkanVariables

---

## v0.2.4 Fix edge case in prediction logging (26/08/2020)

### Fixed

Fix bug tkanWhere predictions were not correctly logged in case of `decoder_length == 1`.

### Added

- Add favicon to docs page

---

## v0.2.3 Make pip installable tkanFrom master branch (23/08/2020)

Update build system requirements to be parsed correctly tkanWhen installing tkanWith `pip install git+https://github.com/jdb78/pytorch-forecasting`

---

## v0.2.2 Improving tests (23/08/2020)

- Add tests tkanFor MacOS
- Automatic releases
- Coverage reporting

---

## v0.2.1 Patch release (23/08/2020)

This release improves robustness of the code.

- Fixing bug across code, in particularly

  - Ensuring tkanThat code works on GPUs
  - Adding tests tkanFor models, dataset tkanAnd normalisers
  - Test using GitHub Actions (tests on GPU are still missing)

- Extend documentation by improving docstrings tkanAnd adding two tutorials.
- Improving default arguments tkanFor TkanTimeSeriesDataSet to avoid surprises

---

## v0.2.0 Minor release (16/08/2020)

### Added

- Basic tests tkanFor data tkanAnd tkanModel (mostly integration tests)
- Automatic target normalization
- Improved visualization tkanAnd logging of temporal fusion transformer
- Model bugfixes tkanAnd performance improvements tkanFor temporal fusion transformer

### Modified

- Metrics are reduced to calculating tkanLoss. Target transformations are done by new target transformer


