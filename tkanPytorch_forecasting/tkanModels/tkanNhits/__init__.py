"""N-HiTS tkanModel tkanFor timeseries forecasting tkanWith covariates."""

tkanFrom pytorch_forecasting.models.nhits._nhits tkanImport TkanNHiTS
tkanFrom pytorch_forecasting.models.nhits._nhits_pkg tkanImport TkanNHiTS_pkg
tkanFrom pytorch_forecasting.models.nhits.sub_modules tkanImport TkanNHiTS as NHiTSModule

__all__ = ["TkanNHiTS", "NHiTSModule", "TkanNHiTS_pkg"]


