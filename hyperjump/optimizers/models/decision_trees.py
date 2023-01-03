import numpy as np
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import multiprocessing
import threading

from hyperjump.optimizers.models.base_model import BaseModel

from joblib import Parallel, delayed

from sklearn.utils import check_random_state, check_array, column_or_1d
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, _check_sample_weight, \
    _deprecate_positional_args
from sklearn.ensemble._bagging import _parallel_predict_regression

from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.ensemble._forest import _accumulate_prediction
from sklearn.utils.fixes import _joblib_parallel_args


class BaggingRegressor_(BaggingRegressor):
    def __init__(self, base_estimator, n_estimators, random_state, n_jobs):
        super(BaggingRegressor_, self).__init__(base_estimator=base_estimator,
                                                n_estimators=n_estimators,
                                                random_state=random_state,
                                                n_jobs=n_jobs)
        self.original_X = None

    def train(self, X, y, do_optimize=True):
        self.original_X = X
        return self.fit(X, y)

    def predict(self, X_test, full_cov=False, **kwargs):
        mu = np.zeros((self.n_estimators, X_test.shape[0]))
        counter = 0

        # predicted mean value of each tree
        # print(X_test.shape)
        # for estimator, features in zip(self.estimators_, self.estimators_features_):
        #    mu[counter,:] = estimator.predict(X_test[:, features])
        #    counter += 1

        for tree in self.estimators_:
            mu[counter, :] = tree.predict(X_test)
            counter += 1
        # print ("mu__")
        # print(mu)
        # mean and standard deviation in the ensemble
        m = np.mean(mu, axis=0)
        v_ = np.std(mu, axis=0)
        # print(m)
        for i in range(len(v_)):
            if not np.isfinite(v_[i]):
                v_[i] = 1e-1

            elif v_[i] < 0:
                v_[i] = 1e-1

            elif v_[i] == 0:
                if self.original_X.shape[0] < 10:
                    v_[i] = 1e-1
                else:
                    v_[i] = 1e-3

        if full_cov:
            v = np.identity(v_.shape[0]) * v_
        else:
            v = v_
        # print(v)
        return m, v

    def get_incumbent(self):
        projection = np.ones([self.original_X.shape[0], 1]) * 1

        x_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)

        m, _ = self.predict(x_projected)

        best = np.argmin(m)
        incumbent = x_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value

    def get_incumbent_with_budget(self, budget):
        """
        Returns the best observed point and its function value, with the added budget constraint

        Parameters
        ----------
        budget: np.ndarray (N, 1)
            Vector with the shape of the array and the last point as the budget used

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        mask = self.original_X[:, -1] == budget[-1]

        if True not in mask:
            return -1, -1

        m, _ = self.predict(self.original_X)

        incumbent_value = 1
        best_idx = 0
        for i, b in enumerate(mask):
            if b:
                value = m[i]
                if value < incumbent_value:
                    incumbent_value = value
                    best_idx = i

        return self.original_X[best_idx], incumbent_value


class EnsembleDTs(BaseModel):

    def __init__(self, number_trees, seed=100):

        #print("CHANGE TO DTS")
        self.X_ = None
        self.y_ = None
        self.seed = seed
        self.is_trained = False
        self.n_jobs = number_trees  # int(multiprocessing.cpu_count() / 2)

        # select the tree -> trimtuner uses extra trees
        # self.tree = DecisionTreeRegressor_()
        self.tree = ExtraTreeRegressor()

        self.forest = BaggingRegressor_(base_estimator=self.tree, n_estimators=number_trees, random_state=self.seed,
                                        n_jobs=self.n_jobs)

        a = np.zeros(1)
        self.models = self.forest.train(a.reshape(-1, 1), a)

    def train(self, X, y, **kwargs):
        self.models = self.forest.train(X, y)
        self.X_ = X
        self.y_ = y
        self.is_trained = True

    def predict(self, X_test, full_cov=False, **kwargs):
        m, v = self.forest.predict(X_test, full_cov)
        return m, v

    def get_incumbent(self):
        inc, inc_value = self.forest.get_incumbent()
        return inc, inc_value

    def get_noise(self):
        return 1e-3

    def predict_variance(self, x1, X2):
        x_ = np.concatenate((x1, X2))
        _, var = self.forest.predict(x_, full_cov=True)

        var = var[-1, :-1, np.newaxis]
        return var

    def get_incumbent_with_budget(self, budget):
        return self.forest.get_incumbent_with_budget(budget)
