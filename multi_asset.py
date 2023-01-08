# imports
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import sqrt, gamma
from scipy import stats
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('ggplot')


class GenGauss:
    __slots__ = ('name', 'prices', 'returns', 'mu', 'beta', 'loc', 'scale', 'var', 'skew', 'kurt', 'ann_mu', 'ann_var')

    def __init__(self, name, prices):
        # store returns data
        self.name = name
        self.prices = prices.to_numpy()
        self.returns = prices.pct_change(1).to_numpy()[1:]

        # calculate generalized gaussian statistics
        self.mu = self.returns.mean()
        beta, loc, scale = stats.gennorm.fit(self.returns, method='MM')
        self.beta = beta
        self.loc = loc
        self.scale = scale

        # calculate higher moments
        self.var = (self.scale * gamma(3 / self.beta)) / gamma(1 / self.beta)
        self.skew = 0
        self.kurt = (gamma(5 / self.beta) * gamma(1 / self.beta)) / (gamma(3 / self.beta) ** 2)

        # calculate the annualised versions of returns and variance
        self.ann_mu = (1 + self.mu) ** 252 - 1
        self.ann_var = self.var * sqrt(252)

    def cdf(self, x):
        return stats.gennorm.cdf(x, self.beta, self.loc, self.scale)

    def pdf(self, x):
        return stats.gennorm.pdf(x, self.beta, self.loc, self.scale)

    def rand(self, n):
        return stats.gennorm.rvs(self.beta, self.loc, self.scale, size=n)

    def plot(self):
        # instantiate and populate plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.linspace(min(self.returns), max(self.returns), 100)
        ax.set_title('Comparing Fitted PDFs')
        ax.hist(self.returns, density=True, bins=50, label='Empirical PDF')
        ax.plot(x, self.pdf(x), label='Generalized Gaussian PDF')
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Daily Return (%)')
        ax.legend()

    def __str__(self):
        string = 'Distribution Details:\n'
        string += f'Mean (Mu): {self.mu:.3f}\n'
        string += f'Median (Loc): {self.loc:.3f}\n'
        string += f'Variance: {self.var:.3f}\n'
        string += f'Scale (Alpha): {self.scale:.3f}\n'
        string += f'Shape (Beta): {self.beta:.3f}\n'
        string += f'Skewness: {self.skew:.3f}\n'
        string += f'Kurtosis: {self.kurt:.3f}'
        return string


class Portfolio:
    __slots__ = ('tickers', 'data', 'returns', 'cov', 'dists', 'weights')

    def __init__(self, df):
        self.tickers = df.columns.values
        self.data = df.values
        self.returns = df.pct_change(1).to_numpy()[1:]
        self.cov = np.corrcoef(self.returns, rowvar=False)
        self.weights = np.full((len(self.tickers), 1), 1 / len(self.tickers))

        # make dictionary for each marginal distributions
        self.dists = {ticker: GenGauss(ticker, df[ticker]) for ticker in self.tickers}

    def rand(self, size=10_000, seed=None):
        mvnorm = stats.multivariate_normal(mean=[self.dists[ticker].loc for ticker in self.dists.keys()], cov=self.cov)
        samples = mvnorm.rvs(size=size, random_state=seed)

        # calculate some correlated samples using a Gaussian copula
        corr_samples = np.zeros_like(samples)
        for index, ticker in enumerate(self.dists.keys()):
            # select the current distribution
            distribution = self.dists[ticker]

            # make samples convert them to uniform and then to generalized gaussian
            x = samples[:, index]
            x_inter = stats.norm.cdf(x, loc=0, scale=1)
            x_prime = stats.gennorm.ppf(x_inter, distribution.beta, distribution.loc, distribution.scale)
            corr_samples[:, index] = x_prime

        return corr_samples

    def utility(self, w, target, method):
        """Scoring Function to return score for closeness to risk tolerance maximizing return"""
        r = self.returns
        port_r = np.dot(r, w)
        u = (port_r.mean() + 1) ** 252 - 1
        sigma = port_r.std(ddof=1) * sqrt(252)

        if method in ['risk', 'Risk']:
            return -u + (100 * abs(target - sigma))
        elif method in ['reward', 'Reward']:
            return sigma + (100 * abs(target - u))
        elif method in ['sharpe', 'Sharpe']:
            return -u / sigma
        else:
            return None

    def optimize(self, target, method):
        """Function to gradient decent to find best portfolio weights"""
        if method not in ['risk', 'reward', 'Risk', 'Reward', 'sharpe', 'Sharpe']:
            raise ValueError(f'{method} is not a valid optimization method')
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        bnds = tuple((0, 1) for _ in self.weights)  # change tuple to set lower and upper bounds for allocation
        res = minimize(self.utility, self.weights, args=(target, method), constraints=cons, bounds=bnds)
        return np.round(res.x, 3)

    def plot_frontier(self, w, sim=False):
        if sim:  # if we want to simulate points
            r = self.rand(10_000)
        else:
            r = self.returns

        # create a new figure for the visualization
        plt.figure()

        # plot individual assets
        assets_x, assets_y = np.std(r, axis=0) * sqrt(252), (np.mean(r, axis=0) + 1) ** 252 - 1
        plt.scatter(x=assets_x, y=assets_y, c='k', marker='x', linewidths=2, label='Individual Assets', zorder=2)
        coordinates = list(zip(self.tickers, assets_x, assets_y))
        for label in coordinates: plt.annotate(label[0], (label[1] + 0.005, label[2] + 0.005))

        # plot given portfolio
        given_r = np.dot(r, w)
        given_x = given_r.std(ddof=1, axis=0) * sqrt(252)
        given_y = (given_r.mean(axis=0) + 1) ** 252 - 1
        plt.scatter(given_x, given_y, c='g', marker='*', linewidths=2, label='Given Portfolio', zorder=5)

        # get random evenly spaced portfolio weights summing to 1
        rand_w = np.random.normal(loc=0, scale=1, size=(len(self.tickers), 10_000))
        mag = np.sqrt(np.square(rand_w).sum(axis=0))
        rand_w = np.square(np.divide(rand_w, mag))

        # create x and y coordinates for each simulated portfolio
        rand_r = np.dot(r, rand_w)
        x = rand_r.std(ddof=1, axis=0) * sqrt(252)
        y = (rand_r.mean(axis=0) + 1) ** 252 - 1

        # plot simulated portfolios
        plt.scatter(x, y, c=y / x, cmap='plasma', alpha=1, label='Hypothetical Portfolios', zorder=1)
        plt.colorbar(location='right', label='Sharpe Ratio')

        # format the plot nicely
        plt.xlabel('Standard Deviation of Returns (%)')
        plt.ylabel('Expected Return (%)')
        plt.title('Efficient Frontier (Simulated Returns)' if sim else 'Efficient Frontier')
        plt.legend()
        plt.tight_layout()

    def get_frontier(self, r_min, r_max):
        # make an array of target returns to scan through
        steps = int((r_max - r_min) * 100)
        targets = np.linspace(r_min, r_max, steps)
        weights = np.zeros(shape=(len(self.tickers), steps))

        # optimize the portfolio for each target rate
        for index, target in enumerate(targets):
            w = self.optimize(target=target, method='reward')
            weights[:, index] = w

        # instantiate the first plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # plot the allocation of the portfolio at each point
        for i in range(weights.shape[0]):
            ax.plot(targets, weights[i, :], label=self.tickers[i])  # run a regression for slope of weight value
        ax.set_title('Allocation Of Portfolio vs. Expected Return')
        ax.set_xlabel("Expected Return (%)")
        ax.set_ylabel("Percentage Of Portfolio (%)")
        ax.legend()
        fig.tight_layout()

        # instantiate the second plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # display the weights on the efficient frontier plot to check their accuracy
        ret = np.dot(self.returns, weights)
        x = ret.std(ddof=1, axis=0) * sqrt(252)
        y = (ret.mean(axis=0) + 1) ** 252 - 1
        ax.scatter(x, y, c='k', marker='x')
        ax.set_xlabel('Standard Deviation of Returns (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title('Portfolios On The Efficient Frontier')
        fig.tight_layout()
        return x, y

    def evaluate(self, w, sim=False, size=100_000, n=1, port_val=10_000):
        if sim:
            ret = np.zeros(shape=(size, n + 1))
            ret[:, 0] = 0
            for i in range(1, n+1):
                r = self.rand(size)
                ret[:, i] = np.dot(r, w)

            # generate paths of stocks
            rets = np.cumprod(ret + 1, axis=1)

            # calculate percentiles of overall returns
            lower = np.percentile(rets[:, -1], 0.5) - 1  # 99% Confidence Interval
            mid = np.percentile(rets[:, -1], 50) - 1
            upper = np.percentile(rets[:, -1], 99.5) - 1  # 99% Confidence Interval
            lower_str = f'Lower Bound: ${round(port_val + lower * port_val, 2):,} ({lower * 100:.2f}%)'
            mid_str = f'Expected Value: ${round(port_val + mid * port_val, 2):,} ({mid * 100:.2f}%)'
            upper_str = f'Upper Bound: ${round(port_val + upper * port_val, 2):,} ({upper * 100:.2f}%)'

            # instantiate the plot
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            # plot a random sample of 500 walks
            ind = np.random.choice(len(rets), 500, replace=False)
            ax.plot(rets[ind].T * port_val, c='dimgray')
            ax.scatter(0, port_val, label=upper_str, alpha=0)
            ax.scatter(0, port_val, label=mid_str, alpha=0)
            ax.scatter(0, port_val, label=lower_str, alpha=0)
            ax.set_title("Hypothetical Portfolio Value Paths")
            ax.set_xlabel('Number Of Trading Days (#)')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            fig.tight_layout()

            # return annualized return and risk for the simulated portfolio
            return round(rets[:, -1].mean() ** (252/n) - 1, 4), round(((rets[:, -1] - 1).std(ddof=1)) * sqrt(252/n), 4)

        else:
            # calculate daily portfolio returns given historical data
            r = self.returns
            ret = np.dot(r, w)

            # return annualized return and risk for the real portfolio
            return round((ret.mean() + 1) ** 252 - 1, 4), round(ret.std(ddof=1) * sqrt(252), 4)

    def drift_evaluate(self, w):
        # define initial portfolio value
        port_val = 10_000

        # before adjusting weights calculated equally weighted portfolio prices
        rebalanced_w = w
        equal_port_val = port_val * np.cumprod(np.dot(self.returns + 1, rebalanced_w))
        equal_port_val = np.insert(equal_port_val, 0, port_val)[:-1]

        w_arr = []
        port_val_arr = []
        for i in range(len(port.data)-1):
            # save values to arrays
            w_arr.append(w)
            port_val_arr.append(port_val)

            # calculate initial cash per stock
            cash_vals = port_val * w

            # get number of shares held per stock
            prices = self.data
            shares_held = cash_vals / prices[i, :]

            # step forward one day for position values
            updated_val = shares_held * prices[i+1, :]

            # get portfolio value and new weights
            port_val = sum(updated_val)
            w = updated_val / port_val

        # instantiate and populate the first plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(w_arr, label=self.tickers)
        ax.set_title('Drifting Asset Weights In A Non-Rebalanced Portfolio')
        ax.set_xlabel('Trading Days (#)')
        ax.set_ylabel('Portfolio Allocation (%)')
        ax.legend()
        fig.tight_layout()

        # instantiate and populate the second plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(port_val_arr, label='Non-Rebalanced Portfolio')
        ax.plot(equal_port_val, label='Portfolio Rebalanced Daily')
        ax.set_title('Comparison Of Rebalancing Frequencies')
        ax.set_xlabel('Trading Days (#)')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        fig.tight_layout()

        return w_arr, port_val_arr

    def plot(self, ticker1, ticker2):
        # empirical data
        print('\nEmpirical Correlation Matrix:')
        print(np.round(self.cov, 4))

        tick1 = list(self.tickers).index(ticker1)
        tick2 = list(self.tickers).index(ticker2)

        # show the empirical distribution
        plot = sns.jointplot(x=self.returns[:, tick1], y=self.returns[:, tick2], kind='hist')
        plot.set_axis_labels(f'{ticker1} Daily Returns', f'{ticker2} Daily Returns')
        plot.ax_marg_x.set_xlim(-0.1, 0.1)
        plot.ax_marg_y.set_ylim(-0.1, 0.1)
        plt.suptitle("Empirical Relation Of Daily Returns")
        plt.tight_layout()

        # simulated data
        corr_samples = self.rand()
        print('Simulated Correlation Matrix:')
        print(np.round(np.corrcoef(corr_samples, rowvar=False), 4))

        plot = sns.jointplot(x=corr_samples[:, tick1], y=corr_samples[:, tick2], kind='hist')
        plot.set_axis_labels(f'{ticker1} Daily Returns', f'{ticker2} Daily Returns')
        plot.ax_marg_x.set_xlim(-0.1, 0.1)
        plot.ax_marg_y.set_ylim(-0.1, 0.1)
        plt.suptitle("Correlated Sampling Of Daily Returns")
        plt.tight_layout()

    def __str__(self):
        return pd.DataFrame(columns=self.tickers, data=self.data).tail().to_string()


# load in stock price data
tickers = 'AAPL GOOG META AMD NKE'
start = '2017-01-06'
end = '2023-01-06'

# fetch stock data
df = yf.download(tickers, start=start, end=end, group_by='column', progress=True, threads=True)['Adj Close']
df.fillna(method='ffill', inplace=True)
df.dropna(axis=0, inplace=True)

# create the portfolio object
port = Portfolio(df)

# show an example of the synthetic returns data
port.plot('AAPL', 'GOOG')

# optimize the portfolio for best sharpe
w = port.optimize(target=0.30, method='sharpe')
print(f'Weights: {dict(zip(port.tickers, w))}')

# simulate the portfolio 10,000 times
mu, sigma = port.evaluate(w, sim=True, size=10_000, n=21, port_val=10_000)

# print stats of optimal portfolio
print(f'Expected Return: {mu * 100:.2f}%')
print(f'Expected Volatility: {sigma * 100:.2f}%')
print(f'Expected Sharpe: {mu / sigma:.2f}')

# plot the efficient frontier and its corresponding weights
port.plot_frontier(w=w)
front_x, front_y = port.get_frontier(r_min=0, r_max=0.7)

# show the drift of a portfolio given its weights
port_w, port_val = port.drift_evaluate(w=w)  # np.array([1/5, 1/5, 1/5, 1/5, 1/5])

# calculate portfolios efficiency as it drifts
weighted_ret = np.dot(port_w, port.returns.T)
weighted_u = (weighted_ret.mean(axis=1) + 1) ** 252 - 1
weighted_sigma = weighted_ret.std(axis=1, ddof=1) * sqrt(252)

# plot and format it nicely
plt.figure()
plt.plot(front_x, front_y, label='Efficient Frontier', c='k')
plt.scatter(weighted_sigma, weighted_u, c=np.linspace(0, len(weighted_ret), len(weighted_ret)), cmap='magma', label='Portfolio Drift')
plt.colorbar(location='right', label='Age Of Portfolio (Days)')
plt.xlabel('Standard Deviation of Returns (%)')
plt.ylabel('Expected Return (%)')
plt.title('Evolution Of A Drifting Portfolio')
plt.legend()
plt.tight_layout()
plt.show()
