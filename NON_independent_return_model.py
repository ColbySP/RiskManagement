# imports
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import pandas as pd
import numpy as np

plt.style.use('ggplot')


class ARGG:
    __slots__ = ('prices', 'returns', 'error', 'slope', 'intercept', 'mu', 'std', 'beta', 'loc', 'scale')

    def __init__(self, prices):
        # store original data
        self.prices = prices.to_numpy()
        self.returns = prices.pct_change(1).to_numpy()[1:]

        # model returns using an AR-1
        intercept, slope = AutoReg(self.returns, 1).fit().params
        self.intercept = intercept
        self.slope = slope

        # calculate error term
        error = (self.intercept + (self.slope * self.returns[:-1])) - self.returns[1:]
        self.error = error

        # calculate typical statistics for error term
        self.mu = self.error.mean()
        self.std = self.error.std()

        # model error term using a generalized gaussian
        beta, loc, scale = stats.gennorm.fit(self.error, method='MM')
        self.beta = beta
        self.loc = loc
        self.scale = scale

    def error_cdf(self, x):
        return stats.gennorm.cdf(x, self.beta, self.loc, self.scale)

    def error_pdf(self, x):
        return stats.gennorm.pdf(x, self.beta, self.loc, self.scale)

    def inverse_error_pdf(self, x):
        return stats.gennorm.ppf(x, self.beta, self.loc, self.scale)

    def gen_error(self, n):
        return stats.gennorm.rvs(self.beta, self.loc, self.scale, size=n)

    def gen_walk(self, n):
        walk_arr = [self.returns[-1]]
        while len(walk_arr) < n + 1:
            new_return = float((self.intercept + (self.slope * walk_arr[-1])) + self.gen_error(1))
            walk_arr.append(new_return)
        return np.array(walk_arr[1:])

    def error_conf(self, conf):
        return stats.gennorm.interval(conf, self.beta, self.loc, self.scale)

    def __str__(self):
        string = 'Distribution Details:\n'
        string += f'Intercept: {self.intercept:.3f}\n'
        string += f'Lag Weight: {self.slope:.3f}\n'
        string += f'Error Mu: {self.mu:.3f}\n'
        string += f'Error STD: {self.std:.3f}\n'
        string += f'Error Loc: {self.loc:.3f}\n'
        string += f'Error Scale: {self.scale:.3f}\n'
        string += f'Error Beta: {self.beta:.3f}'
        return string


# load in stock price data
tickers = 'AAPL'
start = '2018-01-06'
end = '2023-01-06'

# fetch stock data
prices = yf.download(tickers, start=start, end=end, group_by='column', progress=True, threads=True)['Adj Close']
prices.fillna(method='ffill', inplace=True)
prices.dropna(axis=0, inplace=True)
returns = prices.pct_change(1)[1:].values

# check the first lag auto-correlation
plt.scatter(returns[0:-1], returns[1:], alpha=0.25)
plt.title("Apple Daily Returns vs. Apple 1-Lag Daily Returns")
plt.xlabel("Apple 1-Lag Daily Returns (%)")
plt.ylabel("Apple Daily Returns (%)")
lagged_corr = pearsonr(returns[1:], returns[0:-1])
print(f'Lagged Correlation is {lagged_corr} Explaining {lagged_corr[0] ** 2 * 100 :.2f}% Of Tomorrow\'s Variance')

# investigate the auto-correlation effects
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(returns, alpha=0.05, zero=False, method='ywm')
plt.figure()

# model the stock returns as an ARGG
model = ARGG(prices)

# plot the different distributions to visualize their fits
x = np.linspace(min(model.error), max(model.error), 101)
plt.style.use('ggplot')
plt.title('Comparing Fitted CDFs')
plt.plot(ECDF(model.error).x, ECDF(model.error).y, label='Empirical CDF')
plt.plot(x, stats.norm.cdf(x, loc=model.mu, scale=model.std), label='Normal Gaussian CDF')
plt.plot(x, model.error_cdf(x), label='Generalized Gaussian CDF')
plt.ylabel('Cumulative Probability (%)')
plt.xlabel('Daily Return (%)')
plt.legend()
plt.figure()

plt.title('Comparing Fitted PDFs')
plt.hist(model.error, density=True, bins=50, label='Empirical PDF')
plt.plot(x, stats.norm.pdf(x, loc=model.mu, scale=model.std), label='Normal Gaussian PDF')
plt.plot(x, model.error_pdf(x), label='Generalized Gaussian PDF')
plt.ylabel('Probability Density')
plt.xlabel('Daily Return (%)')
plt.legend()
plt.figure()


def simulate(stock_dist, n_days, n_times, plot=None):

    final_arr = []
    for i in range(n_times):
        rand_walk = stock_dist.prices[-1] * np.cumprod(1 + stock_dist.gen_walk(n_days))
        rand_walk = np.insert(rand_walk, 0, stock_dist.prices[-1])

        if plot is not None:
            if i % plot == 0 or i == 0:
                plt.plot(rand_walk, c='dimgray')
            else:
                pass

        final_arr.append(rand_walk[-1])

    if plot is not None:
        plt.title("Autocorrelated Hypothetical Stock Price Paths")
        plt.xlabel('Number Of Trading Days (#)')
        plt.ylabel('Stock Price ($)')

    return final_arr


projection_dist = simulate(stock_dist=model, n_days=21, n_times=10_000, plot=100)
lower = np.percentile(projection_dist, 1)
mid = np.percentile(projection_dist, 50)
upper = np.percentile(projection_dist, 99)
print(f'1st Percentile Price Projection: {lower:.2f} ({((lower / model.prices[-1]) - 1) * 100:.2f}%)')
print(f'50th Percentile Price Projection: {mid:.2f} ({((mid / model.prices[-1]) - 1) * 100:.2f}%)')
print(f'99th Percentile Price Projection: {upper:.2f} ({((upper / model.prices[-1]) - 1) * 100:.2f}%)')
plt.show()
