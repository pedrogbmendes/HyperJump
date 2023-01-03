import george
import emcee
import numpy as np
import copy

from copy import deepcopy

from hyperjump.optimizers.utils import normalization
from hyperjump.optimizers.models.gaussian_process_mcmc import GaussianProcessMCMC
from hyperjump.optimizers.models.gaussian_process import GaussianProcess
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FabolasGPMCMC(GaussianProcessMCMC):
    def __init__(self, kernel, basis_func,
                 prior=None, n_hypers=20,
                 chain_length=2000, burnin_steps=2000,
                 normalize_output=False,
                 rng=None,
                 lower=None,
                 upper=None,
                 noise=-8):

        self.basis_func = basis_func
        self.hypers = None
        super(FabolasGPMCMC, self).__init__(kernel, prior,
                                            n_hypers, chain_length,
                                            burnin_steps,
                                            normalize_output=normalize_output,
                                            normalize_input=False,
                                            rng=rng, lower=lower,
                                            upper=upper, noise=noise)

    def train(self, X, y, do_optimize=True, **kwargs):
        X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], self.lower, self.upper)
        s_ = self.basis_func(X[:, -1])[:, None]
        self.X = np.concatenate((X_norm, s_), axis=1)

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        # Use the mean of the data as mean for the GP
        mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            len(self.kernel) + 1,
                                            self.loglikelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = np.random.rand(self.n_hypers, len(self.kernel.pars) + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        else:
            if self.hypers is None:
                self.hypers = self.gp.kernel[:].tolist()
                self.hypers.append(self.noise)
                self.hypers = [self.hypers]

        self.models = []
        for sample in self.hypers:

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            kernel.set_parameter_vector(sample[:-1])
            noise = np.exp(sample[-1])
            model = FabolasGP(kernel,
                              basis_function=self.basis_func,
                              normalize_output=self.normalize_output,
                              noise=noise,
                              lower=self.lower,
                              upper=self.upper,
                              rng=self.rng)
            model.train(X, y, do_optimize=False)
            self.models.append(model)

        self.is_trained = True


class FabolasGP(GaussianProcess):
    def __init__(self, kernel, basis_function, prior=None,
                 noise=1e-3, use_gradients=False,
                 normalize_output=False,
                 lower=None, upper=None, rng=None):

        self.basis_function = basis_function
        super(FabolasGP, self).__init__(kernel=kernel,
                                        prior=prior,
                                        noise=noise,
                                        use_gradients=use_gradients,
                                        normalize_output=normalize_output,
                                        normalize_input=False,
                                        lower=lower[:-1],
                                        upper=upper[:-1],
                                        rng=rng)

        self.s_max = upper[-1]
        self.s_min = lower[-1]
        self.LogScale = True
        self.original_X = None

    def normalize(self, X):
        for x in X:
            x[-1] = self._transform(x[-1])

        X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], self.lower, self.upper)
        s_ = self.basis_function(X[:, -1])[:, None]
        X_norm = np.concatenate((X_norm, s_), axis=1)
        return X_norm

    def train(self, X, y, do_optimize=True):
        X_ = deepcopy(X)
        y_ = deepcopy(y)

        self.original_X = X_
        X_norm = self.normalize(X_)
        if self.LogScale:
            y_ = np.log(y_)

        return super(FabolasGP, self).train(X_norm, y_, do_optimize)

    def predict(self, X_test, full_cov=False, **kwargs):
        X_norm = self.normalize(deepcopy(X_test))
        mu, var = super(FabolasGP, self).predict(X_norm, full_cov)
        if self.LogScale:
            mu = np.exp(mu)
            var = mu * var

        return  mu, var

    def sample_functions(self, X_test, n_funcs=1):
        X_norm = self.normalize(deepcopy(X_test))
        return super(FabolasGP, self).sample_functions(X_norm, n_funcs)

    def _transform(self, s):
        s_transform = (np.log2(s) - np.log2(self.s_min)) / (np.log2(self.s_max) - np.log2(self.s_min))
        return s_transform

    def _retransform(self, s_transform):
        s = np.rint(2 ** (s_transform * (np.log2(self.s_max) - np.log2(self.s_min)) + np.log2(self.s_min)))
        return int(s)

    def get_incumbent(self):

        inc, inc_loss = super(FabolasGP, self).get_incumbent()
        inc_loss = np.exp(inc_loss)

        #print("Incumbent: ", inc, "\nAccuracy value: ", 1-inc_value)
        return inc, inc_loss


    def get_incumbent_with_budget(self, budget):

        budget_ = self._transform(budget[-1])

        mask = self.original_X[:, -1] == budget_

        if True not in mask: return -1, -1

        curMin = 1
        minIndex = 0
        for i, b in enumerate(mask):
            if b:
                value = self.y[i]
                if value < curMin:
                    curMin = value
                    minIndex = i

        inc = copy.copy(self.X[minIndex])
        inc[-1] = self._retransform(inc[-1])
        inc_loss = np.exp(self.y[minIndex])

        return inc, inc_loss



