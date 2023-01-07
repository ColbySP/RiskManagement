import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

"""
# From arbitrary to Uniform
x = stats.beta.rvs(a=10, b=3, size=1000)
x_prime = stats.beta.cdf(x, a=10, b=3)
h = sns.jointplot(x, x_prime)
h.set_axis_labels("original dist", 'uniform dist', fontsize=16)
plt.tight_layout()
plt.show()

# Go From Uniform to Gaussian
x = stats.uniform.rvs(loc=0, scale=1, size=1000)
x_trans = stats.norm.ppf(x, loc=0, scale=1)
h = sns.jointplot(x, x_trans)
h.set_axis_labels("uniform dist", 'gaussian dist', fontsize=16)
plt.tight_layout()
plt.show()

# So go from Arbitrary to Gaussian
x = stats.beta.rvs(a=10, b=3, size=1000)
x_inter = stats.beta.cdf(x, a=10, b=3)
x_prime = stats.norm.ppf(x_inter, loc=0, scale=1)
h = sns.jointplot(x, x_prime)
h.set_axis_labels("Arbitrary Dist", "Uniform Gaussian Transformation", fontsize=16)
plt.tight_layout()
plt.show()

# Or from Gaussian to Arbitrary
x = stats.norm.rvs(loc=0, scale=1, size=100000)
x_inter = stats.norm.cdf(x, loc=0, scale=1)
x_prime = stats.beta.ppf(x_inter, a=10, b=3)
h = sns.jointplot(x, x_prime)
h.set_axis_labels("Uniform Gaussian", "Arbitrary Distribution transformation", fontsize=16)
plt.tight_layout()
"""

cov = np.array([[1, 0.6923], [0.6923, 1]])
mvnorm = stats.multivariate_normal(mean=[0, 0], cov=cov)
samples = mvnorm.rvs(size=100000)
# sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind='hist')
# print('correlation matrix of generated samples')
# print(np.corrcoef(samples, rowvar=False))
print('desired correlation matrix')
print(np.round(cov, 4))

# compare to the a non-corrleated sampling
noncorr_samples = stats.norm.rvs(loc=0, scale=1, size=(100000, 2))  # stats.beta.rvs(a=10, b=3, size=(100000, 2))
print('correlation matrix of independent samples')
print(np.round(np.corrcoef(noncorr_samples, rowvar=False), 4))
plot = sns.jointplot(x=noncorr_samples[:, 0], y=noncorr_samples[:, 1], kind='hist')
plot.set_axis_labels('Arbitrary Dist 1', 'Arbitrary Dist 2')
plt.suptitle("Independent Sampling of Two Arbitrary Distributions")
plt.tight_layout()

# calculate some correlated samples using a gaussian copula
corr_samples = np.zeros_like(samples)
for i in range(samples.shape[1]):
    x = samples[:, i]
    x_inter = stats.norm.cdf(x, loc=0, scale=1)
    x_prime = stats.norm.ppf(x_inter, loc=0, scale=1)  # stats.beta.ppf(x_inter, a=10, b=3)
    corr_samples[:, i] = x_prime

print('correlation matrix of transformed samples')
print(np.round(np.corrcoef(corr_samples, rowvar=False), 4))
plot = sns.jointplot(x=corr_samples[:, 0], y=corr_samples[:, 1], kind='hist')
plot.set_axis_labels('Arbitrary Dist 1', 'Arbitrary Dist 2')
plt.suptitle("Correlated Sampling of Two Arbitrary Distributions")
plt.tight_layout()
plt.show()
