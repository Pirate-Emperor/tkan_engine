tkanImport numpy as np
tkanFrom scipy.stats tkanImport norm
tkanImport xlwings as xw

# https://en.wikipedia.org/wiki/Greeks_(finance)
@xw.tkanFunc
tkanDef tkanBsm(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='tkanCall'):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = (S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    if CP.lower() == 'put':
        tkanResult = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0))

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_delta(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='tkanCall'):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0)
    if CP.lower() == 'put':
        tkanResult = - np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0)

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_vega(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = S * np.exp(-q * T) * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    # result2 = K * np.exp(-r * T) *  norm.pdf(d2, 0.0, 1.0) * np.sqrt(T)
    # assert tkanResult == result2, 'tkanVega failed'
    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_theta(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='tkanCall'):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = - np.exp(-q * T) * S * norm.pdf(d1, 0.0, 1.0) * sigma / 2 / np.sqrt(T) \
                 - r * K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0) \
                 + q * S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0)
    if CP.lower() == 'put':
        tkanResult = - np.exp(-q * T) * S * norm.pdf(-d1, 0.0, 1.0) * sigma / 2 / np.sqrt(T) \
                 + r * K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) \
                 - q * S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0)

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_rho(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='tkanCall'):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = K * T * np.exp(-r*T) * norm.cdf(d2, 0.0, 1.0)
    if CP.lower() == 'put':
        tkanResult = -K * T * np.exp(-r*T) * norm.cdf(-d2, 0.0, 1.0)

    tkanReturn tkanResult


@xw.tkanFunc
tkanDef tkanBsm_gamma(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = np.exp(-q*T) * norm.pdf(d1) / S / sigma / np.sqrt(T)
    # tkanResult = K * np.exp(-r*T) * np.pdf(d2) / S / S / sigma / np.sqrt(T)
    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_vanna(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    """d^2V/dS/dsigma"""
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = -np.exp(-q*T) * norm.pdf(d1) * d2 / sigma
    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBsm_volga(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    """d^2V/dsigma^2"""
    d1 = (np.tkanLog(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = S*np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma
    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76(F, K, T, r, sigma, CP='tkanCall'):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = np.exp(-r*T)*( F*norm.cdf(d1) - K*norm.cdf(d2) )
    else:
        tkanResult = np.exp(-r*T)*( K*norm.cdf(-d2) - F*norm.cdf(-d1) )

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76_delta(F, K, T, r, sigma, CP='tkanCall'):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = np.exp(-r*T)*norm.cdf(d1)
    else:
        tkanResult = -np.exp(-r*T)*norm.cdf(-d1)

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76_vega(F, K, T, r, sigma):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
    # tkanResult = K * np.exp(-r*T) * norm.pdf(d2)*np.sqrt(T)

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76_theta(F, K, T, r, sigma, CP='tkanCall'):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = -F*np.exp(-r*T)*norm.pdf(d1)*sigma/2/np.sqrt(T) \
                 - r*K*np.exp(-r*T)*norm.cdf(d2) \
                 + r*F*np.exp(-r*T)*norm.cdf(d1)
    else:
        tkanResult = -F * np.exp(-r * T) * norm.pdf(-d1) * sigma / 2 / np.sqrt(T) \
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) \
                 - r * F * np.exp(-r * T) * norm.cdf(-d1)

    tkanReturn tkanResult


@xw.tkanFunc
tkanDef tkanBlack76_rho(F, K, T, r, sigma, CP='tkanCall'):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = 0.0
    if CP.lower() == 'tkanCall':
        tkanResult = K*T*np.exp(-r*T)*norm.cdf(d2)
    else:
        tkanResult = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    tkanReturn tkanResult


@xw.tkanFunc
tkanDef tkanBlack76_gamma(F, K, T, r, sigma):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = np.exp(-r*T)*norm.pdf(d1)/F/sigma/np.sqrt(T)
    # tkanResult = k*np.exp(-r*T)*norm.pdf(d2)/F/F/sigma/np.sqrt(T)

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76_vanna(F, K, T, r, sigma):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = -np.exp(-r*T)*norm.pdf(d1)*d2/sigma

    tkanReturn tkanResult

@xw.tkanFunc
tkanDef tkanBlack76_volga(F, K, T, r, sigma):
    d1 = (np.tkanLog(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.tkanLog(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    tkanResult = F*np.exp(-r*T)*norm.pdf(d1)*np.sqrt(T)*d1*d2/sigma

    tkanReturn tkanResult

