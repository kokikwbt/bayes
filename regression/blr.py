import time
import numpy as np
import pandas as pd
import timesynth as ts
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

class BLR(object):
    """
    Bayesian Linear Regression model
    """
    def __init__(self, x, y, d, var, setseed=False):
        if setseed: np.random.seed(seed=777)
        self.x = x
        self.y = y
        self.d = d
        self.a = 1 / var #1 / np.random.rand() * 10  # variance
        # model parameters
        self.w = np.random.rand(d)
        # hyper parameters
        self.m = np.random.normal(2, 1, d)
        self.A = np.linalg.pinv(np.random.rand(d, d))

    def infer_poly(self, x=None, y=None, max_iter=10):
        if x is None: x = self.x
        if y is None: y = self.y
        self.X = X = np.array([x ** i for i in range(self.d)]).T
        llh = [self.loglikelihood(model='poly')]

        # compute expectations
        Exx = sum(np.outer(xi, xi) for xi in X)  # Eq. (3.146)
        Eyx = sum(yi * xi for xi, yi in zip(X, y))

        # update hyper parameters -> Ez. (3.148)
        new_A = self.a * Exx + self.A
        new_m = np.linalg.inv(new_A) @ (self.a * Eyx + self.A @ self.m)
        self.A = new_A
        self.m = new_m

        llh.append(self.loglikelihood(model='poly'))
        self.log = llh

    def loglikelihood(self, model='poly'):
        llh = 0
        if model == 'poly':
            Ew = self.m
            Ainv = np.linalg.inv(self.A)
            for x, y in zip(self.X, self.y):
                Emu = np.dot(Ew, x)
                Ea = 1 / self.a + x @ Ainv @ x
                llh += norm.logpdf(y, loc=Emu, scale=Ea)
        return llh

    def predict(self, x, model='poly'):
        if model == 'poly':
            x = np.array([x ** i for i in range(self.d)])
            Emu = self.m @ x
            # Ea = 1 / self.a + x @ np.linalg.pinv(self.A) @ x
            return Emu


def main():
    sns.set()

    var = .3
    time_sampler = ts.TimeSampler(stop_time=20)
    time_samples = time_sampler.sample_regular_time(num_points=40)
    sinusoid = ts.signals.Sinusoidal(frequency=2)
    white_noise = ts.noise.GaussianNoise(std=np.sqrt(var))
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(time_samples)

    model = BLR(time_samples, samples, 4, var)
    model.infer_poly()
    pred = [model.predict(x) for x in time_samples]

    plt.plot(model.log)
    plt.figure()
    plt.fill_between(
        time_samples,
        pred-np.sqrt(1/model.a),
        pred+np.sqrt(1/model.a),
        facecolor='g', alpha=.5
    )
    plt.scatter(time_samples, samples, s=20, alpha=.5)
    plt.plot(time_samples, pred)
    plt.show()


if __name__ == '__main__':
    main()
