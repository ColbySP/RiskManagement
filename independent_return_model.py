# imports
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt


class GenGauss:
    __slots__ = ('prices', 'returns', 'mu', 'std', 'beta', 'loc', 'scale')

    def __init__(self, prices):

        # store returns data
        self.prices = prices.to_numpy()
        self.returns = prices.pct_change(1).to_numpy()[1:]

        # calculate typical statistics
        self.mu = self.returns.mean()
        self.std = self.returns.std()

        # calculate generalized gaussian statistics
        beta, loc, scale = stats.gennorm.fit(self.returns, method='MM')
        self.beta = beta
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        return stats.gennorm.cdf(x, self.beta, self.loc, self.scale)

    def pdf(self, x):
        return stats.gennorm.pdf(x, self.beta, self.loc, self.scale)

    def inverse_pdf(self, x):
        return stats.gennorm.ppf(x, self.beta, self.loc, self.scale)

    def rand(self, n):
        return stats.gennorm.rvs(self.beta, self.loc, self.scale, size=n)

    def confidence(self, conf):
        return stats.gennorm.interval(conf, self.beta, self.loc, self.scale)

    def __str__(self):
        string = 'Distribution Details:\n'
        string += f'Mu: {self.mu:.3f}\n'
        string += f'STD: {self.std:.3f}\n'
        string += f'Loc: {self.loc:.3f}\n'
        string += f'Scale: {self.scale:.3f}\n'
        string += f'Beta: {self.beta:.3f}'
        return string


# load in stock price data
tickers = 'AAPL'
start = '2018-01-06'
end = '2023-01-06'

# fetch stock data
data = yf.download(tickers, start=start, end=end, group_by='column', progress=True, threads=True)['Adj Close']
data.fillna(method='ffill', inplace=True)
data.dropna(axis=0, inplace=True)

# make an object of the stock prices
a = GenGauss(data)
print(a)

# get an example of predicting stock returns
print(f"99% Confidence Interval For Returns: {np.round(a.confidence(0.99), 4) * 100}")

# plot the difference distributions to visualize their fits
x = np.linspace(min(a.returns), max(a.returns), 101)
plt.style.use('ggplot')
plt.title('Comparing Fitted CDFs')
plt.plot(ECDF(a.returns).x, ECDF(a.returns).y, label='Empirical CDF')
plt.plot(x, stats.norm.cdf(x, loc=a.mu, scale=a.std), label='Typical Gaussian CDF')
plt.plot(x, a.cdf(x), label='Generalized Gaussian CDF')
plt.ylabel('Cumulative Probability (%)')
plt.xlabel('Daily Return (%)')
plt.legend()
plt.figure()

plt.title('Comparing Fitted PDFs')
plt.hist(a.returns, density=True, bins=50, label='Empirical PDF')
plt.plot(x, stats.norm.pdf(x, loc=a.mu, scale=a.std), label='Typical Gaussian PDF')
plt.plot(x, a.pdf(x), label='Generalized Gaussian PDF')
# plt.ylabel('Probability Density')
plt.xlabel('Daily Return (%)')
plt.legend()
plt.figure()


def simulate(stock_dist, n_days, n_times, plot=None):

    final_arr = []
    for i in range(n_times):
        sim_prices = np.cumprod(stock_dist.rand(n_days) + 1) * stock_dist.prices[-1]
        sim_prices = np.insert(sim_prices, 0, stock_dist.prices[-1])

        if plot is not None:
            if i % plot == 0 or i == 0:
                plt.plot(sim_prices, c='dimgray')
            else:
                pass

        final_arr.append(sim_prices[-1])

    if plot is not None:
        plt.title("Hypothetical Stock Price Paths")
        plt.xlabel('Number Of Trading Days (#)')
        plt.ylabel('Stock Price ($)')

    return final_arr


projection_dist = simulate(stock_dist=a, n_days=21, n_times=100_000, plot=500)
lower = np.percentile(projection_dist, 0.5)
mid = np.percentile(projection_dist, 50)
upper = np.percentile(projection_dist, 99.5)
print(f'1st Percentile Price Projection: {lower:.2f} ({((lower / a.prices[-1]) - 1) * 100:.2f}%)')
print(f'50th Percentile Price Projection: {mid:.2f} ({((mid / a.prices[-1]) - 1) * 100:.2f}%)')
print(f'99th Percentile Price Projection: {upper:.2f} ({((upper / a.prices[-1]) - 1) * 100:.2f}%)')
plt.show()
