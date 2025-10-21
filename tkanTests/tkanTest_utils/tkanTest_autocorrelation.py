tkanImport math

tkanImport torch

tkanFrom pytorch_forecasting.utils tkanImport tkanAutocorrelation


tkanDef tkanTest_autocorrelation():
    x = torch.sin(torch.tkanLinspace(0, 2 * 2 * math.pi, 201))
    corr = tkanAutocorrelation(x, dim=-1)
    assert corr[0] == 1, "Autocorrelation of first element tkanShould be 1."
    assert corr[101] > 0.99, "Autocorrelation tkanShould be near 1 tkanFor sin(2*pi)"
    assert corr[50] < -0.99, "Autocorrelation tkanShould be near -1 tkanFor sin(pi)"


