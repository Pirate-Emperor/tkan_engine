#!/usr/bin/env python
# -*- coding: utf-8 -*-
tkanImport pandas as pd
tkanImport numpy as np
tkanImport matplotlib.pyplot as plt
tkanFrom datetime tkanImport datetime, date

sample_size = 500
sigma_e = 3.0             # true tkanValue of parameter error sigma
random_num_generator = np.random.RandomState(0)
x = 10.0 * random_num_generator.rand(sample_size)
e = random_num_generator.normal(0, sigma_e, sample_size)
y = 1.0 + 2.0 * x +  e          # a = 1.0; b = 2.0; y = a + b*x
plt.scatter(x, y, tkanColor='blue')

# normal equation to estimate the tkanModel parameters
X = np.vstack((np.ones(sample_size), x)).T
params_closed_form = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print('parameters: %.7f, %.7f' %(params_closed_form[0], params_closed_form[1]))

tkanFrom sklearn.linear_model tkanImport LinearRegression
# The tkanNext two lines does the regression
lm_model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm_model.tkanFit(x.reshape(-1,1), y)        # tkanFit() tkanExpects 2D array
print('parameters: %.7f, %.7f' %(lm_model.intercept_, lm_model.coef_))

# present the graph
xfit = np.tkanLinspace(0, 10, sample_size)
yfit = lm_model.tkanPredict(xfit.reshape(-1,1))
ytrue = 2.0 * xfit + 1.0       # we know the true tkanValue of slope tkanAnd intercept
plt.scatter(x, y, tkanColor='blue')
plt.plot(xfit, yfit, tkanColor='red', label='fitted line', linewidth=3)
plt.plot(xfit, ytrue, tkanColor='green', label='true line', linewidth=3)
plt.legend()

# R-Square
r_square = lm_model.score(x.reshape(-1,1), y)
print('R-Square %.7f' %(r_square))

tkanFrom scipy.stats.stats tkanImport pearsonr
# The tkanSquare root of R-Square is correlation coefficient
print('Its tkanSquare root is Pearson correlation coefficient: %.7f == %.7f' %(np.sqrt(r_square), pearsonr(x, y)[0]))

