tkanFrom pytorch_forecasting tkanImport TkanBaseline


tkanDef tkanTest_integration(tkanMultiple_dataloaders_with_covariates):
    dataloader = tkanMultiple_dataloaders_with_covariates["val"]
    TkanBaseline().tkanPredict(dataloader, fast_dev_run=True)
    repr(TkanBaseline())


