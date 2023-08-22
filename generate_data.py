import numpy as np
import scipy.stats as stats
import random
import pandas as pd
import argparse
import os 


def simulate_raw_counts(num_samples, num_otus, target_dist='log_normal', return_dists=False, seed=25):
	'''
	Input
	num_samples : (int)
		number of samples 
	num_otus : (int)
		number of OTUs
	target_dist : (str)
		distribution used to generate counts
		Options:
			- log_normal --> mean = (244.692, 897.847), std = (320.75, 1176.927)
			- exponential --> mean = (250, 500), std = (250, 500)
			- gamma --> mean = (250, 750), std = (250, 474.3416)
			- negative_binomial --> mean = (245, 1188), std = (110.68, 344.674)
			- beta_negative_binomial --> mean = (245, 1188), std = (110.68, 344.674)
	return_dists : (bool)
		whether or not to return specific distributions
	----------------------
	Output
	count_table : numpy array (int) of size (num_otus x num_samples)
		raw count table with counts generated from the specified distribution (rows as otus and columns as samples)
	dists (optional) : list of length num_otus
		lists of distributions
	'''
	assert target_dist in {'log_normal', 'exponential', 'gamma', 'negative_binomial', 'beta_negative_binomial'}, "Target distribution must"\
		"be one of : log_normal, exponential, gamma, negative_binomial, or beta_negative_binomial"
	distributions = {
		'log_normal' : 'stats.lognorm(s=1, scale=np.exp(rng.uniform(5,6.3)))',
		'exponential' :'stats.expon(scale=rng.integers(250,501))',
		'gamma' : 'stats.gamma(a=rng.uniform(1,2.5), scale=rng.uniform(250,300))',
		'negative_binomial' : 'stats.nbinom(n=rng.integers(5,13), p=rng.uniform(0.01,0.02))'
	}
	rng = np.random.default_rng(seed)
	cor_target = rng.uniform(-0.01,0.01, size=(num_otus,num_otus))
	cor_target = np.zeros((num_otus, num_otus)) + np.triu(cor_target) + np.triu(cor_target).T
	np.fill_diagonal(cor_target,1)
	rand_U = stats.norm.cdf(stats.multivariate_normal.rvs(mean=np.zeros(num_otus),cov=cor_target,size=num_samples))
	
	if target_dist == 'beta_negative_binomial':
		beta_dists = [stats.beta(rng.uniform(1.3,5), rng.uniform(1.3,5)) for i in range(num_otus)]
		dists = [stats.nbinom(n=rng.integers(5,13), p=0.01*beta_dists[i].rvs()+0.01) for i in range(num_otus)]
		if return_dists and return_cor_target:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round(), dists, beta_dists, cor_target
		elif return_dists:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round(), dists, beta_dists
		elif return_cor_target:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round(), cor_target
		else:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round()
	else:
		globs = globals()
		locs = locals()
		dists = [eval(distributions[target_dist],globs,locs) for i in range(num_otus)]
		if return_dists:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round(), dists
		else:
			return np.stack([dists[i].ppf(rand_U[:,i]) for i in range(num_otus)]).round()


def adjust_raw_counts(counts, strength=3, num_pairs=1, model='mutual', idx=None, seed=100):
	'''
	Input
	counts : (int) of shape (num_otus, num_samples)
		raw count matrix with otus as rows and samples as columns
	strength : (float)
		strength of relationships (max 5)
	num_pairs : (int)
		number of relationships to model (max is num_otus//2)
	model : (str)
		type of ecological relationship to model 
		Options:
			- mutual
			- competitive
			- exploitative
			- amensal
			- commensal
	idx : (int) of shape (num_pairs,2)
	----------------------
	Output
	count_table : (int) of shape (num_otus, num_samples)
		adjusted count table with specified ecologocial relationship
	idx : (int) of shape (num_pairs x 2)
		indices of count table that were changed
	'''
	rng = np.random.default_rng(seed)

	assert model in {'mutual', 'competitive', 'exploitative', 'amensal', 'commensal'}, "Expected model from : mutual, competitive, exploitative, amensal, commensal"
	assert strength >= 0, "Input strength must be >= 0"
	assert num_pairs <= counts.shape[0]//2, "Specified number of pairs to change must be less than (number of OTUs)//2"

	if strength == 0:
		return counts
	else:
		strength = strength/5

	if idx is None:
		idx = rng.choice(counts.shape[0], size=(num_pairs,2), replace=False)
	else:
		assert idx.shape[0] == num_pairs, "Number of pairs to change must match number of indices provided"

	x_max = np.expand_dims(counts[idx[:,0],:].max(axis=1), axis=-1).copy()
	y_max = np.expand_dims(counts[idx[:,1],:].max(axis=1), axis=-1).copy()
	x_xy = np.divide(counts[idx[:,0],:], counts[idx[:,0],:] + counts[idx[:,1],:],
		out=np.zeros_like(counts[idx[:,0],:]), 
		where=(counts[idx[:,0],:] + counts[idx[:,1],:]) != 0)

	temp_x = counts[idx[:,0],:].copy()
	if model == 'mutual':

		counts[idx[:,0],:] += strength * counts[idx[:,1],:] * np.log1p(np.e-np.exp(x_xy))
		counts[idx[:,0],:] *= x_max / counts[idx[:,0],:].max(axis=1).reshape(-1,1)
		counts[idx[:,0],:] += rng.normal(0, x_max*0.01/strength, counts[idx[:,0],:].shape)

		counts[idx[:,1],:] += strength * counts[idx[:,0],:] * np.log1p(np.e-np.exp(1-x_xy))
		counts[idx[:,1],:] *= y_max / counts[idx[:,1],:].max(axis=1).reshape(-1,1)
		counts[idx[:,1],:] += rng.normal(0, y_max*0.01/strength, counts[idx[:,1],:].shape)
		
	elif model == 'competitive':

		counts[idx[:,0],:] -= strength * counts[idx[:,1],:] * np.log1p(np.e-np.exp(x_xy))
		counts[idx[:,0],:] *= x_max / counts[idx[:,0],:].max(axis=1).reshape(-1,1)
		counts[idx[:,0],:] += rng.normal(0, x_max*0.01/strength, counts[idx[:,0],:].shape)

		counts[idx[:,1],:] -= strength * counts[idx[:,0],:] * np.log1p(np.e-np.exp(1-x_xy))
		counts[idx[:,1],:] *= y_max / counts[idx[:,1],:].max(axis=1).reshape(-1,1)
		counts[idx[:,1],:] += rng.normal(0, y_max*0.01/strength, counts[idx[:,1],:].shape)

	elif model == 'exploitative':
		counts[idx[:,0],:] += strength * counts[idx[:,1],:] * np.log1p(np.e-np.exp(x_xy))
		counts[idx[:,0],:] *= x_max / counts[idx[:,0],:].max(axis=1).reshape(-1,1)
		counts[idx[:,0],:] += rng.normal(0, x_max*0.01/strength, counts[idx[:,0],:].shape)

		counts[idx[:,1],:] -= strength * counts[idx[:,0],:] * np.log1p(np.e-np.exp(1-x_xy))
		counts[idx[:,1],:] *= y_max / counts[idx[:,1],:].max(axis=1).reshape(-1,1)
		counts[idx[:,1],:] += rng.normal(0, y_max*0.01/strength, counts[idx[:,1],:].shape)

	elif model == 'amensal':
		counts[idx[:,1],:] -= strength * counts[idx[:,0],:] * np.log1p(np.e-np.exp(1-x_xy))
		counts[idx[:,1],:] *= y_max / counts[idx[:,1],:].max(axis=1).reshape(-1,1)
		counts[idx[:,1],:] += rng.normal(0, y_max*0.01/strength, counts[idx[:,1],:].shape)

	elif model == 'commensal':
		counts[idx[:,1],:] += strength * counts[idx[:,0],:] * np.log1p(np.e-np.exp(1-x_xy))
		counts[idx[:,1],:] *= y_max / counts[idx[:,1],:].max(axis=1).reshape(-1,1)
		counts[idx[:,1],:] += rng.normal(0, y_max*0.01/strength, counts[idx[:,1],:].shape)

	del temp_x
	del x_xy
	counts[counts<0] = 0
	return counts.round(), idx

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MI-Benchmark Data Generation')
	parser.add_argument('--num_samples', type=int, default=50, choices=[50, 200], help='Number of samples')
	parser.add_argument('--strength', default='weak', choices=['weak', 'strong'])
	parser.add_argument('--distribution', default='log_normal', 
		choices=['log_normal', 'exponential', 'gamma', 'negative_binomial', 'beta_negative_binomial'])
	parser.add_argument('--zi', action='store_true', help='Zero inflated data?')
	args = parser.parse_args()

	NUM_SAMPLES = args.num_samples
	STRENGTH = 1.5 if args.strength == 'weak' else 3
	DISTRIBUTION = args.distribution
	ZI = args.zi
	NUM_OTUS=400

	if not os.path.exists('./Simulated_Data/'):
		os.mkdir('./Simulated_Data/')

	rng = np.random.default_rng(5123)

	eco_conditions = ['amensal', 'commensal', 'mutual', 'competitive', 'exploitative']
	if not ZI:
		if not os.path.exists('./Simulated_Data/{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES)):
			os.mkdir('./Simulated_Data/{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES))
	else:
		if not os.path.exists('./Simulated_Data/ZI_{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES)):
			os.mkdir('./Simulated_Data/ZI_{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES))

	for _ in eco_conditions:
		if DISTRIBUTION == 'beta_negative_binomial':
			counts, generated_dists, generated_beta_dists = simulate_raw_counts(num_samples=NUM_SAMPLES, num_otus=NUM_OTUS, target_dist=DISTRIBUTION, return_dists=True, seed=rng.integers(10000))
		else:
			counts, generated_dists = simulate_raw_counts(num_samples=NUM_SAMPLES, num_otus=NUM_OTUS, target_dist=DISTRIBUTION, return_dists=True, seed=rng.integers(10000))
		counts *= counts[:,:]<np.quantile(counts[:,:],0.95)
		adjusted_counts, idx = adjust_raw_counts(counts, strength=STRENGTH, num_pairs=100, model=_)

		df = pd.DataFrame(adjusted_counts, columns = [i for i in range(NUM_SAMPLES)], index = ['OTU{}'.format(i) for i in range(NUM_OTUS)])
		if ZI:
			# Save count table
			df -= np.mean(df.values).round()
			df[df<0] = 0.0
			df.to_csv('./Simulated_Data/ZI_{}_{}_samples/{}_{}.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t')
			# Save changed indices
			df = pd.DataFrame(columns = ['Var1', 'Var2'])
			df['Var1'] = np.asarray(['OTU{}'.format(idx[i,0]) for i in range(len(idx[:,0]))])
			df['Var2'] = np.asarray(['OTU{}'.format(idx[i,1]) for i in range(len(idx[:,1]))])
			df.to_csv('./Simulated_Data/ZI_{}_{}_samples/{}_{}_pairs.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t', index=False)

		else:
			# Save count table
			df.to_csv('./Simulated_Data/{}_{}_samples/{}_{}.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t')
			# Save distributions
			df = pd.DataFrame([generated_dists[d].kwds for d in range(len(generated_dists))], index = ['OTU{}'.format(i) for i in range(NUM_OTUS)])
			df.to_csv('./Simulated_Data/{}_{}_samples/{}_{}_dists.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t')
			if DISTRIBUTION == 'beta_negative_binomial':
				df = pd.DataFrame([generated_beta_dists[d].args for d in range(len(generated_beta_dists))], index = ['OTU{}'.format(i) for i in range(NUM_OTUS)], columns=['alpha', 'beta'])
				df.to_csv('./Simulated_Data/{}_{}_samples/{}_{}_beta_params.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t')
			
			# Save changed indices
			df = pd.DataFrame(columns = ['Var1', 'Var2'])
			df['Var1'] = np.asarray(['OTU{}'.format(idx[i,0]) for i in range(len(idx[:,0]))])
			df['Var2'] = np.asarray(['OTU{}'.format(idx[i,1]) for i in range(len(idx[:,1]))])
			df.to_csv('./Simulated_Data/{}_{}_samples/{}_{}_pairs.csv'.format(DISTRIBUTION, NUM_SAMPLES, args.strength, _), sep='\t', index=False)

