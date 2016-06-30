#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
"""
.. module:: ex10_template
    :synopsis example for discriminant analysis

.. moduleauthor Thomas Keck
"""
# ------------------------------------------------------------------------
# useful imports

import bisect
import numpy as np

import matplotlib
matplotlib.rcParams['backend']='TkAgg'

from matplotlib import pyplot as plt

### ------- load the Data set --------------------------------------------
# Load iris data
data = np.loadtxt('iris.data')

# Define dictionary with columns names: s=sepal, p=petal, l=lenght, w=width,
flowers = {'setosa', 'versicolor', 'virginica'}
columns = {0: 'L(Kelch)', 1: 'W(Kelch)',
        2: 'L(Blatt)', 3: 'W(Blatt)', 4: 'class'}

# Define boolean arrays corresponding to the three
#      classes setosa, versicolor and virginica
setosa = data[:, 4] == 0
versicolor = data[:, 4] == 1
virginica = data[:, 4] == 2

# Signal is versicolor (can be changed to setosa or virginica)
ndim = 2
signal = setosa
bckgrd = ~signal
# !! note: see indexing of arrays with boolenan array in pyhthon documentation
#
# exmaples how to access the data:
#    data[signal]     # All events classified as signal
#    data[background] # All events classified as background
#    data[setosa]     # All events classified as setosa
#    data[versicolor | virginica] # All events classified as versicolor or virginica
#    data[:, :2]      # The first two columns (sepal length and sepal width) of all events
#    data[signal, :2] # The first two columns of the signal events
#    data[background, 2:4] # The 3. and 4. column (petal length and petal width) of the background events
#    data[:, 4]       # The label column of all events
#             (see the numpy documentation for further examples)

### ------- helper functions ----------------------------------
class Plotter(object):
    """
        class to display and evaluate the performance of a test-statistic
    """
    def __init__(self, signal_data, bckgrd_data):
        self.signal_data = signal_data
        self.bckgrd_data = bckgrd_data
        self.data = np.vstack([signal_data, bckgrd_data])

    def plot_contour(self, classifier):
        # 1st variable as x-dimension in the plots
        xdim = 0
        # and 2nd variable as y-dimension
        ydim = 1
        # Draw the scatter-plots of signal and background
        plt.scatter(self.signal_data[:, xdim], self.signal_data[:, ydim],
                c='r', label='Signal')
        plt.scatter(self.bckgrd_data[:, xdim], self.bckgrd_data[:, ydim],
                c='b', label='Background')

        # Evaluate the response function on a two-dimensional grid ...
        #   ... using the mean-values of the data for the remaining dimensions.
        xs = np.arange(min(self.data[:, xdim])-1, max(self.data[:, xdim])+1, 0.1)
        ys = np.arange(min(self.data[:, ydim])-1, max(self.data[:, ydim])+1, 0.1)

        means = np.mean(self.data, axis = 0) # calculate mean of each column
        responses = np.zeros((len(ys), len(xs)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                values = np.copy(means)
                values[xdim] = x
                values[ydim] = y
                responses[j, i] = float(classifier.evaluate(values))

        # Draw a contour plot
        X, Y = np.meshgrid(xs, ys)
        c=plt.contourf(X, Y, responses, alpha=0.5, cmap=plt.cm.coolwarm)
        cbar=plt.colorbar(c, orientation='vertical')
        # add the direction of the fisher vector (if specified)
        if hasattr(classifier, 'fisher_vector'):
            vector = classifier.fisher_vector / np.linalg.norm(classifier.fisher_vector)
            plt.axes().set_aspect('equal', 'datalim')
            plt.plot([-vector[xdim]+means[xdim], vector[xdim]+means[xdim]],
                    [-vector[ydim]+means[ydim], vector[ydim]+means[ydim]],
                    'k-', lw=4, label="Fisher Projection")
            plt.title(
                    "scatter plot {} vs {} and classifier contour".format(
                        columns[xdim], columns[ydim] ) )
                    #cbar.draw_all()
        plt.show()

    def plot_test_statistic(self, classifier):
        # Draw Distribution of the test-statistic
        ns, binss, _ = plt.hist(list(map(classifier.evaluate, self.signal_data)),
                color='r', alpha=0.5, label='Signal' )
        nb, binsb, _ = plt.hist( list(map(classifier.evaluate, self.bckgrd_data)),
                color='b', alpha=0.5, label='Background' )
        plt.title("test statistic")
        plt.show()

    # calculate efficiencies and plot ROC-curves
    def plot_roc(self, classifier):
        ns, binss = np.histogram(list(map(classifier.evaluate, self.signal_data)))
        nb, binsb = np.histogram(list(map(classifier.evaluate, self.bckgrd_data)))
        # enforce common binning for response on bkg and sig
        minresp=min([ binss[0], binsb[0] ])
        maxresp=max([ binss[len(binss)-1], binsb[len(binsb)-1] ])
        nbins=100
        bins=np.linspace(minresp, maxresp, nbins)
        bwid=(maxresp-minresp)/nbins
        # calculate cumulative distributions (i.e. bkg and sig efficiencies)
        h, b = np.histogram( list(map(classifier.evaluate, self.signal_data)), bins, density=True)
        ns = np.cumsum(h)*bwid
        h, b = np.histogram( list(map(classifier.evaluate, self.bckgrd_data)), bins, density=True)
        nb = np.cumsum(h)*bwid
        # finally, draw bkg-eff vs. sig-eff
        f2, ax = plt.subplots(1, 1)
        ax.plot(1.-ns, nb, 'r-', 1.-ns, nb, 'bo', linewidth=2.0)
        ax.set_xlabel("signal efficiency")
        ax.set_ylabel("background rejection")
        ax.set_title("ROC curve")
        plt.show()

def find_bin(x, edges):
    """
        returns the bin number (in array of bin edges) corresponding to x
        @param x: value for which to find correspoding bin number
        @param edges: array of bin edges
    """
    return max(min(bisect.bisect(edges, x) - 1, len(edges) - 2), 0)


### ------- simple example: Classifier and usage with Plotter class ---------

# template of Classifier Class
class CutClassifier(object):
    """
        template implementation of a Classifier Class
    """
    def fit(self, signal_data, bckgrd_data):
        """
            set up classifier ("training")
        """
    # some examples of what might be useful:
      # 1. signal and background histograms with same binning
        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

      # 2. mean and covariance matrix
        self.signal_mean = np.mean(signal_data, axis=0)
        self.signal_cov = np.cov(signal_data.T)
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)
        self.bckgrd_cov = np.cov(bckgrd_data.T)

    def evaluate(self, x):
        # simple example of a cut-base classifier
        c=0
        for i in range(len(x)):
            c+=(x[i] < (self.signal_mean[i] + self.bckgrd_mean[i])/2.)
        return c

# initialise Classifier with training data
cut = CutClassifier()
cut.fit(data[signal, :ndim], data[bckgrd, :ndim])

# initialize Plotter Class
plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
# and amke plots
plotter.plot_contour(cut)
# plotter.plot_test_statistic(cut)
# plotter.plot_roc(cut)


# ------------------------------------------------------------------
# Exercise 1
nvar=4
f, axarr = plt.subplots(nvar, nvar)
plt.tight_layout()
nbins=20
# add your code here ...
colors = {'r', 'g', 'b'}
for i in range(nvar):
    for j in range(nvar):
        for flower, c in zip(flowers, colors):
            sig = eval(flower)
            if i == j:
                axarr[i, j].hist(data[sig, i], bins=20, color=c, label=flower)
                axarr[i, j].set_title(columns[i])
            else:
                axarr[i, j].scatter(data[sig, j], data[sig, i], color=c)
                axarr[i, j].set_xlabel(columns[j])
                axarr[i, j].set_ylabel(columns[i])

axarr[1, 1].legend(bbox_to_anchor=(0.5, -0.3), loc='center', ncol=3)
f.set_size_inches(20, 10, forward=True)
plt.show()

#-----------------------------------------------------------------
#Exercise 2

class LikelihoodClassifier(object):

    def fit(self, signal_data, bckgrd_data):

        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=2)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

    def evaluate(self, x):

        point = np.zeros_like(x, dtype=int)
        for i in range(len(x)):
            point[i] = find_bin(x[i], self.edges[i])
        return self.signal_hist[tuple(point)] / self.bckgrd_hist[tuple(point)]
        # return 1 if self.signal_hist[tuple(point)] / self.bckgrd_hist[tuple(point)] > 0 else -1

lh = LikelihoodClassifier()
lh.fit(data[signal, :ndim], data[bckgrd, :ndim])

plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
plotter.plot_contour(lh)
# plotter.plot_test_statistic(llh)
# plotter.plot_roc(llh)

#------------------------------------------------------------------
# Exercise 3

def gaussdd(x, mu, sig):
    t = np.zeros_like(x)
    for i in range(len(x)):
        t[i] = ((x[i] - mu[i])**2 / (2 * sig[i]**2))
    return np.exp(-(np.sum(t)))

class LogLikelihoodClassifier(object):

    def fit(self, signal_data, bckgrd_data):

        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

        self.signal_mean = np.mean(signal_data, axis=0)
        self.signal_cov = np.cov(signal_data.T)
        self.signal_stddev = np.sqrt(np.diag(self.signal_cov))
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)
        self.bckgrd_cov = np.cov(bckgrd_data.T)
        self.bckgrd_stddev = np.sqrt(np.diag(self.bckgrd_cov))

    def evaluate(self, x):

        p_signal = gaussdd(x, self.signal_mean, self.signal_stddev)
        p_bckgrd = gaussdd(x, self.bckgrd_mean, self.bckgrd_stddev)

        # prior_signal = 1/3
        # prior_bckgrd = 2/3
        # post_signal = prior_signal * np.prod(p_signal)
        # post_bckgrd = prior_bckgrd * np.prod(p_bckgrd)
        post_signal = np.prod(p_signal)
        post_bckgrd = np.prod(p_bckgrd)

        return np.log(post_signal / post_bckgrd)
        # return 1 if np.log(post_signal / post_bckgrd) > 0 else -1

llh = LogLikelihoodClassifier()
llh.fit(data[signal, :ndim], data[bckgrd, :ndim])

plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
plotter.plot_contour(llh)
# plotter.plot_test_statistic(llh)
# plotter.plot_roc(llh)

# ------------------------------------------------------------------
# Exercise 4

class FisherLinearClassifier(object):

    def fit(self, signal_data, bckgrd_data):

        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

        self.signal_data = signal_data
        self.bckgrd_data = bckgrd_data
        self.signal_mean = np.mean(signal_data, axis=0)
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)

        # calculate fisher weights which maximize linear separation
        self.signal_del = signal_data - self.signal_mean
        self.bckgrd_del = bckgrd_data - self.bckgrd_mean
        self.var = np.dot(self.signal_del.T, self.signal_del) + np.dot(self.bckgrd_del.T, self.bckgrd_del)
        self.fish = np.dot(np.linalg.inv(self.var), (self.bckgrd_mean - self.signal_mean))

    def evaluate(self, x):
        return np.dot(x, self.fish) / np.linalg.norm(self.fish)
        # return 1 if np.dot(x, self.fish) / np.linalg.norm(self.fish) > 0 else -1

    def plot_separation(self):
        sig = np.dot(self.signal_data, self.fish)
        bg = np.dot(self.bckgrd_data, self.fish)

        fg, ax = plt.subplots(1, 1)
        ax.plot(sig, np.zeros_like(sig), 'ro')
        ax.plot(bg, np.zeros_like(bg), 'go')
        plt.show()

fsh = FisherLinearClassifier()
fsh.fit(data[signal, :ndim], data[bckgrd, :ndim])
fsh.plot_separation()

plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
plotter.plot_contour(fsh)
# plotter.plot_test_statistic(llh)
# plotter.plot_roc(llh)
