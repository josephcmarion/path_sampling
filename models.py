import numpy as np
import scipy.stats as stats
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from smc.utils import load_stan_model


class LogisticVariableSelection:

    def __init__(self, load=True):
        """
        Variable selection example for linear model
        Uses a tempered geometric mixture.
        Markov kernels are implemented in stan
        """

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='logistic_variable_selection.stan',
            model_code_file='logistic_variable_selection.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )

    @staticmethod
    def generate_data(n_observations, n_dimensions, observation_sd=0.1, design_sd=0.1,
                      coefficient_mean=0, coefficient_sd=4.0, seed=1337):
        """ generates data from the model, using every other coefficient for the 'true' model

        attributes
        ----------
        n_observations: int
            number of observations to generate
        n_dimensions: even int
            number of covariate dimensions to use
        observation_sd: float > 0
            noise added to the observations prior to logistic transform
        design_sd: float in (-0.5, 0.5)
            roughly speaking, this is the correlation between adjacent columns in the design matrix
        coefficient_mean: float
            added to coefficients (used to make distance larger in space)
        coefficient_sd: float > 0
            standard deviation to use when sampling the coefficients
        seed: float
            used to control the RNG. If set to None does not use a seed

        returns
        -------
        np.array
            (n_observations, ) binary response vector
        np.array
            (n_observations, n_dimensions) covariate matrix for the first model
        np.array
            (n_observations, n_dimensions) covariate matrix for the second first model
        np.array
            true model coefficients for the first model
        """

        np.random.seed(seed)

        # create the design matrices
        X0 = stats.norm().rvs((n_observations, n_dimensions))
        X1 = (-X0 + stats.norm().rvs(X0.shape) * design_sd)

        X0 = StandardScaler().fit_transform(X0)
        X1 = StandardScaler().fit_transform(X1)

        # create the true parameters
        coefficients = stats.norm().rvs(n_dimensions) * coefficient_sd + coefficient_mean

        # Create the model
        mu = np.dot(X0, coefficients) + stats.norm().rvs(n_observations) * observation_sd
        pi = expit(mu)
        Y = (pi > stats.uniform().rvs(n_observations)).astype('float')

        return Y, X0, X1, coefficients

    @staticmethod
    def _stan_text_model():
        """ returns the text for a stan model, just in case you've lost it

        returns
        -------
        str
            a stan model file
        """

        model_code = """
        data {
            int<lower = 1> D0;
            int<lower = 1> D1;
            int<lower = 1> N;

            matrix[N, D0] X0;
            matrix[N, D1] X1;
            vector[N] Y;

            vector[D0+D1] prior_mean;
            matrix[D0+D1, D0+D1] prior_covariance;

            real<lower=0, upper=1> mixture;
            real<lower=0, upper=1> temperature;
        }

        parameters {
            vector[D0+D1] coefficients;
        }

        transformed parameters {
            vector[N] mu0;
            vector[N] mu1;

            vector[N] pi0;
            vector[N] pi1;

            mu0 = X0*coefficients[1:D0];
            pi0 = inv_logit(mu0);

            mu1 = X1*coefficients[D0+1:D0+D1];
            pi1 = inv_logit(mu1);

        }

        model {
            target += temperature*(1-mixture)*(Y .* log(pi0) + (1-Y) .* log(1-pi0));
            target += temperature*mixture*(Y .* log(pi1) + (1-Y) .* log(1-pi1));
            target += temperature*multi_normal_lpdf(coefficients | prior_mean, prior_covariance);
        }
        """

        return model_code

    def mcmc(self, Y, X0, X1, params, n_iters=10**2, n_warmup=10**3, n_chains=4, stan_kwargs={}):
        """
        uses MCMC via STAN to sample from the posterior for a given set of (mixture, temperature) parameters

        parameters
        ----------
        Y: np.array
            (n_observations, ) binary response vector
        X0: np.array
            (n_observations, n_dimensions) covariate matrix for the first model
        X1: np.array
            (n_observations, n_dimensions) covariate matrix for the second first model
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]
        n_iters: int
            number of samples to draw following burnin
        n_warmup: int
            number of burnin iterations to use
        n_chains: int
            number of MCMC chains to use
        stan_kwargs: dict
            additional arguments to stan

        returns
        -------
        pystan.StanFit
            contains details of the MCMC
        np.array
            samples from MCMC
        """

        mixture, temperature = params
        data = {
            'D0': X0.shape[1],
            'D1': X1.shape[1],
            'N': X0.shape[0],
            'X0': X0,
            'X1': X1,
            'Y': Y,
            'prior_mean': np.zeros(X0.shape[1] + X1.shape[1]),
            'prior_covariance': np.eye(X0.shape[1] + X1.shape[1]) * 10.0,
            'mixture': mixture,
            'temperature': temperature
        }

        fit = self.stan_model.sampling(data=data, pars='coefficients',
                                       iter=n_iters+n_warmup, warmup=n_warmup, chains=n_chains, n_jobs=-1,
                                       verbose=False, **stan_kwargs)

        return fit, fit.extract()['coefficients']

    @staticmethod
    def _potential(samples, Y, X0, X1, params):
        """ computes the potential of a set of samples for one set of mixture parameters

        parameters
        ----------
        samples: np.array
            (n_samples, n_dimension*2) samples from the posterior distribution @ params
        Y: np.array
            (n_observations, ) binary response vector
        X0: np.array
            (n_observations, n_dimensions) covariate matrix for the first model
        X1: np.array
            (n_observations, n_dimensions) covariate matrix for the second first model
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        returns
        -------
        np.array
            (n_samples, 2) array of potentials
        """

        mixture, temperature = params

        # important summary statistics
        mu0 = np.dot(samples[:, 0:X0.shape[1]], X0.T)
        mu1 = np.dot(samples[:, X0.shape[1]:X0.shape[1] + X1.shape[1]], X1.T)

        pi0 = expit(mu0)
        pi1 = expit(mu1)

        # fix some numerical issues
        pi0[pi0 < 10.0 ** -9] = 10 ** -9
        pi1[pi1 < 10.0 ** -9] = 10 ** -9

        pi0[pi0 > 1.0 - 10.0 ** -9] = 1.0 - 10.0 ** -9
        pi1[pi1 > 1.0 - 10.0 ** -9] = 1.0 - 10.0 ** -9

        # compute the pdfs
        log_pdf0 = (Y[None, :] * np.log(pi0) + (1.0 - Y[None, :]) * np.log(1.0 - pi0)).sum(1)
        log_pdf1 = (Y[None, :] * np.log(pi1) + (1.0 - Y[None, :]) * np.log(1.0 - pi1)).sum(1)

        # potentials
        u_mixture = temperature * (log_pdf1 - log_pdf0)
        u_temperature = (1.0 - mixture) * log_pdf0 + mixture * log_pdf1

        return np.vstack([u_mixture, u_temperature]).T


class RandomEffects:

    def __init__(self, a=2, b=2, load=True):
        """ Comparison of a simple random effects and fixed effects model

        The fixed effects model is:

            Y_ij ~ N(mu_j, sigma_2)
                p(mu_j) ~ 1
                p(mu, tau) ~ NIG(0, 1, a, b)
                p(sigma_2) ~ 1/sigma_2

        The random effects model is

            Y_ij ~ N(mu_j, sigma_2)
                p(mu_j) ~ N(mu, tau_2)
                p(mu, tau) ~ NIG(0, 1, a, b)
                p(sigma+2) ~ 1/sigma_2

        attributes
        ----------
        a: float > 0
            hyper parameter for tau_2 prior. Should be >2 to ensure finite variance
        b: float > 0
            hyper parameter for tau_2 prior

        """
        self.a = a
        self.b = b

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='random_effects.stan',
            model_code_file='random_effects.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )

    @staticmethod
    def generate_data(n_groups, n_observations, total_variance, interclass_correlation, seed=1337):
        """ generates summary statistics and true parameter values from the random effects model

        parameters
        ----------
        n_groups: int > 0
            number of groups to use
        n_observations: int > 1
            number of observations to use per group
        total_variance: float > 0
            combined variance of random effect + observation
        interclass_correlation: float in (0, 1]
            proportion of variance attributable to the random effect
            when close to 0, there is little between group variability (tau_2 ~ 0)
            when close to 1, there is no within group variability (sigma_2 ~ 0)
        seed: float > 0
                used to control the RNG.

        returns
        -------
        (int, np.array, np.array)
            number of observations per group, group specific means, group specific variances
        (np.array, 0, float, float)
            true values of random effects, mu,  observation variance (sigma_2), random effect variance (tau_2)
        """

        np.random.seed(seed)

        # true prior variance values
        tau_2 = interclass_correlation * total_variance
        sigma_2 = total_variance - tau_2

        # sample parameters from the model
        random_effects = stats.norm().rvs(n_groups) * tau_2 ** 0.5
        parameters = (random_effects, 0, sigma_2, tau_2)

        # data and summary statistics
        observations = stats.norm().rvs((n_groups, n_observations)) * sigma_2 ** 0.5 + random_effects[:, None]
        means = observations.mean(1)
        variances = observations.var(1, ddof=0)
        summary_statistics = (n_observations, means, variances)

        return summary_statistics, parameters

    def sample_fixed_effects(self, n_samples, summary_statistics):
        """samples from the posterior distribution of the fixed effects model

        parameters
        ----------
        n_samples: int
            number of samples to draw
        summary_statistics: (int, np.array, np.array)
            number of observations per group, group specific means, group specific variances

        returns
        -------
        np.array
            (n_samples, n_groups) mu_j/fixed effects
        np.array
            (n_samples, ) sigma_2/observation variances
        np.array
            (n_samples, ) mu
        np.array
            (n_samples, ) tau_2
        """
        n_observations, means, variances = summary_statistics
        n_groups = means.shape[0]

        # Sample from observation variance posterior
        a = n_observations * n_groups / 2.0
        b = n_observations * variances.sum() / 2.0
        sigma_2 = stats.invgamma(a=a, scale=b).rvs(n_samples)

        # Sample from the fixed effects
        fixed_effects = stats.norm().rvs((n_samples, n_groups)) * (sigma_2[:, None] / n_observations) ** 0.5 + means[None, :]

        # sample the NIG prior
        tau_2 = stats.invgamma(a=self.a, scale=self.b).rvs(n_samples)
        mu = stats.norm().rvs(n_samples)*tau_2**0.5

        return fixed_effects, mu, sigma_2, tau_2

    def mcmc(self, summary_statistics, params, n_iters=10 ** 2, n_warmup=10 ** 3, n_chains=4, stan_kwargs={}):
        """
        uses MCMC via STAN to sample from the posterior for a given set of (mixture, temperature) parameters

        parameters
        ----------
        summary_statistics: (int, np.array, np.array)
            number of observations per group, group specific means, group specific variances
        params: tuple
            first value is a mixing parameter in [0,1] second value is scale parameter in (0,1]
        n_iters: int
            number of samples to draw following burnin
        n_warmup: int
            number of burnin iterations to use
        n_chains: int
            number of MCMC chains to use
        stan_kwargs: dict
            additional arguments to stan

        returns
        -------
        pystan.StanFit
            contains details of the MCMC
        np.array
            sample potentials from MCMC
        """

        mixture, scale = params
        n_observations, means, variances = summary_statistics

        data = {
            'n_groups': means.shape[0],
            'n_observations': n_observations,
            'means': means,
            'variances': variances,
            'a': self.a,
            'b': self.b,
            'mixture': mixture,
            'scale': scale,
        }

        fit = self.stan_model.sampling(data=data,
                                       iter=n_iters + n_warmup, warmup=n_warmup, chains=n_chains, n_jobs=-1,
                                       verbose=False, **stan_kwargs)

        return fit, fit.extract()['U']

    @staticmethod
    def _stan_text_model():
        """ returns the text for a stan model, just in case you've lost it

        returns
        -------
        str
            a stan model file
        """

        model_code = """
        data {
            int<lower = 1> n_groups;
            int<lower = 1> n_observations;

            vector[n_groups] means;
            vector[n_groups] variances;

            real<lower = 0> a;
            real<lower = 0> b;
            real<lower = 0, upper =1> mixture;
            real<lower = 0, upper =1> scale;

        }

        parameters {
            vector[n_groups] random_effects;
            real mu;
            real<lower = 0> sigma_2;
            real<lower = 0> tau_2;
        }

        model {

            // shared priors
            target += log(1/sigma_2);
            target += normal_lpdf(mu | 0, tau_2^0.5);
            target += (a+1)*log(1/tau_2) - b/tau_2/scale;

            // likelihood (using summary statistics improves speed in N to O(1))
            target += 0.5*n_observations*n_groups*log(1/sigma_2);
            target += -0.5*n_observations/sigma_2*dot_self(random_effects-means);
            target += -0.5*n_observations/sigma_2*sum(variances);

            // random effects prior
            target += mixture * normal_lpdf(random_effects | mu, tau_2^0.5);
        }

        generated quantities{
          vector[2] U;

          U[1] = normal_lpdf(random_effects | mu, tau_2^0.5);
          U[2] = b/tau_2/scale^2;

        }

        """

        return model_code


class MixtureModel:

    def __init__(self, a=1, b=1, tau=10.0, load=True):
        """ Comparison of a mixture model to a model with one component
        and the prior

        attributes
        ----------
        a: float > 0
            hyper parameter for sigma_2 prior.
        b: float > 0
            hyper parameter for sigma_2 prior.
        tau: float > 0
            scale of the normal prior
        """

        self.a = a
        self.b = b
        self.tau = tau

        self.stan_power_model = load_stan_model(
            directory='stan',
            stan_file='mixture_power.stan',
            model_code_file='mixture_power.txt',
            model_code_text=self._stan_text_power_model(),
            load=load
        )

        self.stan_dirichlet_model = load_stan_model(
            directory='stan',
            stan_file='mixture_dirichlet.stan',
            model_code_file='mixture_dirichlet.txt',
            model_code_text=self._stan_text_dirichlet_model(),
            load=load
        )

    def mcmc_power(self, n_components, observations, params,
                   n_iters=10 ** 2, n_warmup=10 ** 3, n_chains=4, stan_kwargs={}):
        """
        uses MCMC via STAN to sample from the power posterior for a given mixture parameter

        parameters
        ----------
        n_components: int > 0
            number of mixture components to use
        observations: np.array
            (n_observations, ) array of centered and scaled data
        params: float in [0,1]
            mixing parameter
        n_iters: int
            number of samples to draw following burnin
        n_warmup: int
            number of burnin iterations to use
        n_chains: int
            number of MCMC chains to use
        stan_kwargs: dict
            additional arguments to stan

        returns
        -------
        pystan.StanFit
            contains details of the MCMC
        np.array
            sample potentials from MCMC
        """

        data = {
            'n_components': n_components,
            'n_observations': observations.shape[0],
            'Y': observations,
            'a': self.a,
            'b': self.b,
            'tau': self.tau,
            'alpha': np.ones(n_components) / n_components,
            'mixture': params,
        }

        fit = self.stan_power_model.sampling(data=data,
                                             iter=n_iters + n_warmup, warmup=n_warmup, chains=n_chains, n_jobs=-1,
                                             verbose=False, **stan_kwargs)

        return fit, fit.extract()['U']

    def mcmc_dirichlet(self, n_components, observations, params,
                       n_iters=10 ** 2, n_warmup=10 ** 3, n_chains=4, stan_kwargs={}):
        """
        uses MCMC via STAN to sample from the dirichlet posterior for a given scale vector

        parameters
        ----------
        n_components: int > 0
            number of mixture components to use
        observations: np.array
            (n_observations, ) array of centered and scaled data
        params: np.array
            (n_components, ) array of positive dirichlet scales. generally between 0 and 1.
        n_iters: int
            number of samples to draw following burnin
        n_warmup: int
            number of burnin iterations to use
        n_chains: int
            number of MCMC chains to use
        stan_kwargs: dict
            additional arguments to stan

        returns
        -------
        pystan.StanFit
            contains details of the MCMC
        np.array
            sample potentials from MCMC
        """

        data = {
            'n_components': n_components,
            'n_observations': observations.shape[0],
            'Y': observations,
            'a': self.a,
            'b': self.b,
            'tau': self.tau,
            'alpha': np.ones(n_components) / n_components,
            'scales': params,
        }

        fit = self.stan_dirichlet_model.sampling(data=data,
                                                 iter=n_iters + n_warmup, warmup=n_warmup, chains=n_chains, n_jobs=-1,
                                                 verbose=False, **stan_kwargs)

        return fit, fit.extract()['U']

    @staticmethod
    def generate_data(n_components, n_observations, total_variance=1.0, rho=0.75, sigma_cv=1.0, seed=None):
        """ generates synthetic data from the Gaussian mixture model

        attributes
        ----------
        n_components: int > 0
            number of mixture components to use
        n_observations: int > 1
            number of observations to use per group
        total_variance: float > 0
            variance of component means + expected variance observations.
            The scaling generally removes the need for this term
        rho: float in (0,1)
            controls the proportion of variance related to difference in means
            numbers close to 1 yield components with higher variance means
            numbers close to 0 yield high expected observation variance
        sigma_cv: float > 0
            coefficient of variation for the variances
            controls the dispersion of the variances
        seed: float
            used to control the RNG. If set to None does not use a seed

        returns
        -------
        np.array
            (n_observations, ) array of centered and scaled data
        """

        if seed is not None:
            np.random.seed(1337)

        # mixture probabilities
        pi = stats.dirichlet(alpha=np.ones(n_components)).rvs(1).flatten()
        pi.sort(-1)
        pi = pi[::-1]

        # gaussian means
        tau_2 = total_variance * rho
        mus = stats.norm().rvs(n_components) * tau_2 ** 0.5

        # gaussian variances
        sigma_2_mean = total_variance * (1 - rho)
        a = 1.0 / sigma_cv ** 2 + 2.0
        b = sigma_2_mean * (a - 1)
        sigma_2s = stats.invgamma(a=a, scale=b).rvs(n_components)

        # data (this approach is simple but a bit wasteful fo large n_components)
        data = np.array([stats.norm().rvs(n_observations) * sigma_2s[k] ** 0.5 + mus[k] for k in range(n_components)])
        data = data[np.random.choice(n_components, n_observations, p=pi), range(n_observations)]
        data = StandardScaler().fit_transform(data.reshape(-1, 1)).flatten()

        return data

    @staticmethod
    def _stan_text_power_model():
        """ returns the text for a stan model, just in case you've lost it

                returns
                -------
                str
                    a stan model file
        """

        model_code = """
        data {
            int<lower = 1> n_components;
            int<lower = 1> n_observations;
            real Y[n_observations];

            real<lower = 0> a;
            real<lower = 0> b;
            real<lower = 0> tau;
            vector<lower=0>[n_components] alpha;
            real<lower = 0, upper=1> mixture;
        }

        parameters {

            vector[n_components] mu;
            simplex[n_components] pi;
            vector<lower=0>[n_components] sigma_2;

        }

        transformed parameters {

            vector[n_components] lps =  rep_vector(0, n_components);
            vector[n_components] log_pi = log(pi);
            vector[n_components] sigma = sqrt(sigma_2);
            real log_likelihood = 0;
            real log_prior = 0;

            // this is basically the likelihood computation
            for (n in 1:n_observations) {
                for (k in 1:n_components) {
                    lps[k] = log_pi[k] + normal_lpdf(Y[n] | mu[k], sigma[k]);
                }
                log_likelihood = log_likelihood + log_sum_exp(lps);
            }
        }

        model {

            vector[n_components] ones =  rep_vector(0, n_components);

            // priors
            target += normal_lpdf(mu | 0, tau);
            target += log_prior + inv_gamma_lpdf(sigma_2 | a, b);
            target += sum((alpha-ones) .* log_pi);

            //likelihood
            target += mixture * log_likelihood;

        }

        generated quantities {
            real U;
            U = log_likelihood;

        }
        """
        return model_code

    @staticmethod
    def _stan_text_dirichlet_model():
        """ returns the text for a stan model, just in case you've lost it

                returns
                -------
                str
                    a stan model file
        """

        model_code = """
        data {
            int<lower = 1> n_components;
            int<lower = 1> n_observations;
            real Y[n_observations];

            real<lower = 0> a;
            real<lower = 0> b;
            real<lower = 0> tau;
            vector<lower=0>[n_components] alpha;
            vector<lower = 0, upper=1>[n_components] scales;
        }

        parameters {

            vector[n_components] mu;
            simplex[n_components] pi;
            vector<lower=0>[n_components] sigma_2;

        }

        transformed parameters {

            vector[n_components] lps =  rep_vector(0, n_components);
            vector[n_components] log_pi = log(pi);
            vector[n_components] sigma = sqrt(sigma_2);
            real log_likelihood = 0;
            real log_prior = 0;

            // this is basically the likelihood computation
            for (n in 1:n_observations) {
                for (k in 1:n_components) {
                    lps[k] = log_pi[k] + normal_lpdf(Y[n] | mu[k], sigma[k]);
                }
                log_likelihood = log_likelihood + log_sum_exp(lps);
            }
        }

        model {

            vector[n_components] ones =  rep_vector(0, n_components);

            // priors
            target += normal_lpdf(mu | 0, tau);
            target += log_prior + inv_gamma_lpdf(sigma_2 | a, b);
            target += sum(scales .* (alpha-ones) .* log_pi);

            //likelihood
            target += log_likelihood;

        }

        generated quantities {

            vector[n_components] U;

            for (k in 1:n_components){
                U[k] = (alpha[k]-1) * log_pi[k];
            }
        }
        """

        return model_code


def generate_covariance_matrix(D, sigma, correlation):
    """ creates a covariance matrix by specifying a precision matrix with an off-diagonal elements"""
    precision = np.eye(D)
    precision[range(1, D), range(0, D-1)] = correlation
    precision[range(0, D-1), range(1, D)] = correlation
    covariance = np.linalg.inv(precision)*sigma**2
    return covariance