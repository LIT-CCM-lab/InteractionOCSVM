import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import math

def plot_knee_normalized(ax, knee_finder):
    '''
    Plot the distribution of distances in order to identify the presence of a knee in the distribution
    :param ax: axis from matplotlib to make the plot
    :type ax: matplotlib.pyplot.ax
    :param knee_finder: Kneedle knee finder object containing information about the found knee
    :type knee_finder: object
    '''
    ax.plot(knee_finder.x_normalized,
            knee_finder.y_normalized,
            "b", label="normalized curve")
    ax.plot(knee_finder.x_difference,
            knee_finder.y_difference,
            "r", label="difference curve")
    ax.set_xticks(np.arange(knee_finder.x_normalized.min(),
                            knee_finder.x_normalized.max() + 0.1,
                            0.1))
    ax.set_yticks(np.arange(knee_finder.y_difference.min().round(1),
                  knee_finder.y_normalized.max() + 0.1,
                  0.1))
    ax.vlines(knee_finder.norm_knee,
              knee_finder.y_normalized.min(),
              knee_finder.y_normalized.max(),
              linestyles="--",
              label="knee/elbow",)
    ax.legend(loc="best")
    ax.set_xlabel('Normalized Rank')
    ax.set_ylabel('Normalized KNN-average similarity', labelpad = 1)


class Qms():
    '''
    Base class for the implementation of Quick Model Selection (QMS) 
    methods for parameters estimation for OCSVM models.
    This method can be used with any python module for graph kernel
    estimation as long as the kernel have a fit_transform method
    returning a square numpy array.
    Similairty is defined as the average similairty to the KNN points.
    :param data: Input graphs
    :type data: list of graphs
    :param kernel: The kernel used to compute the graph similairty
    :type kernel: graph kernel
    '''
    def __init__(self, dist, metric):
        self.dist = dist
        if metric in ['similarity', 'distance']:
            self.metric = metric
        else:
            raise ValueError(f'Unknown variable passed {metric}')

    def _find_vals(self, dist, n_end):
        if self.metric == 'similarity':
            return np.mean(dist[:,n_end-1:-1], axis = 1)
        else:
            return np.mean(dist[:,1:-n_end], axis = 1)

    def _find_knee(
            self,
            n = 7,
            curve = 'convex',
            direction = 'decreasing',
            S_knee = 1,
            plots = True,
            find_n = False,
            interp_method = 'interp1d',
            polynomial_degree = 7
        ):
        '''
        Find the knee for the pairwise similairty distribution.
        The method can be called multiple times changing the
        parameters, without the need to recompute the similarity
        matrix.
        Further information on the parameters used for knee
        identification can be found in the kneed module documentations:
        https://kneed.readthedocs.io/en/stable/
        :param n: Number of nearest neighbours for the analysis
        :type n: int
        :param curve: kneed parameter
        :type curve: string
        :param direction: kneed parameter
        :type direction: string
        :param S_knee: Sensibility of the kneed algorithm
        :type S_knee: int
        :param plots: show the plots of the obtained distribution
        :type plots: bool
        :param find_n: Defines K of KNN as the value corresponding to
        3% of the data
        :type find_n: bool
        :param interp_method: kneed parameter
        :type interp_method: string
        :param polynomial_degree: kneed parameter
        :tpye polynomial_degree: int
        '''

        if find_n:
            n = max(10, int(0.03*self.dist.shape[0]))
        if not hasattr(self, 'sorted_dist'):
            self.sorted_dist = np.sort(self.dist)
            self.vals = self._find_vals(self.sorted_dist, -n)
            self.vals[np.isnan(self.vals)] = 0
        self.knee_finder = KneeLocator(np.arange(len(self.vals)),
                                       np.sort(self.vals),
                                       curve = curve,
                                       direction = direction,
                                       S = S_knee,
                                       interp_method = interp_method,
                                       polynomial_degree=polynomial_degree)

    def find_gamma(self):
        if self.metric == 'similarity':
            self.gamma = 'auto'
            return
        
        qs = np.argsort(self.sorted_dist[:,1], axis = 0)
        for q in qs:
            if self.sorted_dist[q,1] != 0:
                break
        d_min = self.sorted_dist[q,1]
        d_max = np.mean(self.sorted_dist[q,1:])

        self.gamma = -math.log(d_min/d_max)/(d_max**2-d_min**2)

            


class Qms2(Qms):
    '''
    Implementation of the QMS2 method
    10.1109/TNNLS.2017.2785792
    '''
    def __init__(self, dist, metric):
        super().__init__(dist, metric)

    def fit(
            self,
            n = 7,
            curve = 'convex',
            direction = 'decreasing',
            S_knee = 1,
            plots = True,
            find_n = False,
            interp_method = 'interp1d',
            polynomial_degree = 7
        ):
        '''
        Select the graphs to be used for model training.
        :param n: Number of nearest neighbours for the analysis
        :type n: int
        :param curve: kneed parameter
        :type curve: string
        :param direction: kneed parameter
        :type direction: string
        :param S_knee: Sensibility of the kneed algorithm
        :type S_knee: int
        :param plots: show the plots of the obtained distribution
        :type plots: bool
        :param find_n: Defines K of KNN as the value corresponding to 3% of the data
        :type find_n: bool
        :param interp_method: kneed parameter
        :type interp_method: string
        :param polynomial_degree: kneed parameter
        :tpye polynomial_degree: int
        :returns: graphs to be used for model training
        :type: list of graphs
        '''

        self._find_knee(n = n,
                        curve = curve,
                        direction = direction,
                        S_knee = S_knee,
                        plots = plots,
                        find_n = find_n,
                        interp_method = interp_method,
                        polynomial_degree = polynomial_degree,
                        )
        if self.knee_finder.knee_y is None:
            return np.full(self.vals.shape[0], -1)
        labels = np.zeros(self.vals.shape[0])
        if self.metric == 'similarity':
            labels[self.vals < self.knee_finder.knee_y] = -1
        else:
            labels[self.vals > self.knee_finder.knee_y] = -1

        self.find_gamma()
        return labels


class Qms1(Qms):
    '''
    Implementation of the QMS1 method
    10.1007/978-3-319-31750-2_15
    '''
    def __init__(self, dist, metric):
        super().__init__(dist, metric)

    def fit(
            self,
            n = 7,
            curve = 'convex',
            direction = 'decreasing',
            S_knee = 1,
            eta = 0,
            plots = True,
            find_n = False,
            interp_method = 'interp1d',
            polynomial_degree = 7,
            ):
        '''
        Select the graphs to be used for model training.
        :param n: Number of nearest neighbours for the analysis
        :type n: int
        :param curve: kneed parameter
        :type curve: string
        :param direction: kneed parameter
        :type direction: string
        :param S_knee: Sensibility of the kneed algorithm
        :type S_knee: int
        :param eta: The eta value of QMS1, the values must be in [0,1)
        :type eta: float
        :param plots: show the plots of the obtained distribution
        :type plots: bool
        :param find_n: Defines K of KNN as the value corresponding to 3% of the data
        :type find_n: bool
        :param interp_method: kneed parameter
        :type interp_method: string
        :param polynomial_degree: kneed parameter
        :tpye polynomial_degree: int
        '''

        self._find_knee(n = n,
            curve = curve,
            direction = direction,
            S_knee = S_knee,
            plots = plots,
            find_n = find_n,
            interp_method = interp_method,
            polynomial_degree = polynomial_degree)

        if self.knee_finder.knee_y is None:
            return np.full(self.vals.shape[0], -1)
            self.nu = 0.001
        labels = np.zeros(self.vals.shape[0])
        if self.metric == 'similarity':
            labels[self.vals < self.knee_finder.knee_y] = -1
        else:
            labels[self.vals > self.knee_finder.knee_y] = -1
        self.nu = 1-((len(self.vals[labels == 0])*(1- eta))
                            /len(self.vals))

        self.find_gamma()

        return labels
        

class QmsDBSCAN(Qms):
    '''
    Implementation of a new method based on density based clustering (DBSCAN)
    Multiple training dataset are extracted from the original data by performing clustering.
    Each cluster is used to train a different model.
    Similar to what was done in 10.1137/1.9781611976236.13
    '''
    def __init__(self, dist, metric):
        super().__init__(dist, metric)

    def fit(
            self,
            n = 7,
            curve = 'convex',
            direction = 'increasing',
            S_knee = 1,
            plots = True,
            find_n = False,
            interp_method = 'interp1d',
            polynomial_degree = 7
        ):
        '''
        Select the graphs to be used for model training.
        :param n: Number of nearest neighbours for the analysis
        :type n: int
        :param curve: kneed parameter
        :type curve: string
        :param direction: kneed parameter
        :type direction: string
        :param S_knee: Sensibility of the kneed algorithm
        :type S_knee: int
        :param plots: show the plots of the obtained distribution
        :type plots: bool
        :param find_n: Defines K of KNN as the value corresponding to 3% of the data
        :type find_n: bool
        :param interp_method: kneed parameter
        :type interp_method: string
        :param polynomial_degree: kneed parameter
        :tpye polynomial_degree: int
        :returns: graphs to be used for model training
        :rtype: list of graphs
        '''

        self._find_knee(n = n,
                        curve = curve,
                        direction = direction,
                        S_knee = S_knee,
                        plots = plots,
                        find_n = find_n,
                        interp_method = interp_method,
                        polynomial_degree = polynomial_degree,
                        )
        if self.knee_finder.knee_y is None:
            return np.full(self.vals.shape[0], -1)
        
        if find_n:
            n = max(10, int(0.03*self.dist.shape[0]))
        eps = 1-self.knee_finder.knee_y if self.metric == 'similarity' else self.knee_finder.knee_y
        eps = eps if eps > 0 else 1e-15
        cluster = DBSCAN(eps=eps, min_samples=n, metric = 'precomputed')
        return cluster.fit_predict(1-self.dist)