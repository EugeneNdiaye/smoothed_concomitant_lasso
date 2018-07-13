import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from smoothed_concomitant import SC_path


class SCRegressor(BaseEstimator, RegressorMixin):

	""" Estimator for learning a Smoothed Concomitant Lasso with coordinate descent.
	The objective functions is defined as
    P(beta, sigma) = 0.5 * norm(y - X beta, 2)^2 / sigma + sigma / 2 + lambda * norm(beta, 1).   
    We solve argmin_{beta, sigma >= sigma_0} P(beta, sigma).
	"""

	def __init__(self, eps=1e-4, max_iter=5000, f=10.):

		self.eps = eps
		self.max_iter = max_iter
		self.f = f
		self.lambdas = None
		self.sigma_0 = None
		self.beta_init = None


    def fit(self, X, y):

    	""" Fit smooth_conco according to X and y.

	    Parameters
	    ----------
	    X : {array-like}, shape (n_samples, n_features)
	        Training data.
	    y : ndarray, shape = (n_samples,)
	        Target values

        Returns
        -------
        self : regressor
            Returns self.
		"""

    	n_samples, n_features = X.shape

    	if self.sigma_0 is None:
    		self.sigma_0 = (np.linalg.norm(y) / np.sqrt(n_samples)) * 1e-2

    	if self.lambdas is None:  # default values
    		n_lambdas = 30
    		sigstar = max(self.sigma_0, np.linalg.norm(y) / np.sqrt(n_samples))
    		lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf) / (n_samples * sigstar)
    		self.lambdas = np.logspace(np.log10(lambda_max / 10.),
									   np.log10(lambda_max), n_lambdas)[::-1]

    	model = SC_path(X, y, self.lambdas, self.beta_init, self.sigma_0,
    				    self.eps, self.max_iter, self.f)

    	self.betas = model[0]
    	self.sigmas = model[1]
    	self.gaps = model[2]
    	self.n_iters = model[3]

    	return self


    def predict(self, X):

     	""" Compute a prediction vector based on X.
    	
	    Parameters
	    ----------
	    X : {array-like}, shape (n_samples, n_features)
	        Testing data.

        Returns
        -------
        pred : ndarray, shape = (n_samples, n_lambdas)
        	prediction of target values for different parameter lambda
		"""

		pred = np.dot(X, self.betas.T)
        return pred


    def score(self, X, y):

      	""" Compute a prediction error wrt y.
    	
	    Parameters
	    ----------
	    X : {array-like}, shape (n_samples, n_features)
	        Testing data.
	    y : ndarray, shape = (n_samples,)
	        Testing target values

        Returns
        -------
        pred_error : ndarray, shape = (n_lambdas,)
        	Prediction error wrt target values for different parameter lambda.
		"""

    	n_lambdas = self.betas.shape[0]
    	pred_error = np.array([np.linalg.norm(np.dot(X, self.betas.T)[:, l] - y) 
    		    		 	   for l in range(n_lambdas)])
    	return pred_error


if __name__ == "__main__":

	from data_generation import generate_data

	# Generate dataset

	n_samples = 100
	n_features = 300
	sigma = 1.
	sparsity = 0.9
	snr = 1
	correlation = 0.5
	random_state = 42

	X, y, true_beta, true_sigma = generate_data(n_samples, n_features, sigma,
												snr, sparsity, correlation,
	                                            random_state=random_state)

	clf = SCRegressor()
	clf.fit(X, y)
	clf.predict(X)
	clf.score(X, y)
