import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
import dataframe_image as dfi
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import argparse


#####################################################
################ Helper fxns ########################
#####################################################

def load_results(num_samples, d, c, s, m, zi=False):
	if zi:
		return pd.read_csv('./Simulated_Data_results/ZI_{}_{}_samples/{}_{}/{}.csv'.format(d, num_samples, s, c, m), sep='\t', header=0).fillna(0)
	else:
		return pd.read_csv('./Simulated_Data_results/{}_{}_samples/{}_{}/{}.csv'.format(d, num_samples, s, c, m), sep='\t', header=0).fillna(0)

def get_true_positives(num_samples, d, c, s, zi=False):
	if zi:
		df = pd.read_csv('./Simulated_Data/ZI_{}_{}_samples/{}_{}_pairs.csv'.format(d, num_samples, s, c), sep='\t', header=0)
	else:
		df = pd.read_csv('./Simulated_Data/{}_{}_samples/{}_{}_pairs.csv'.format(d, num_samples, s, c), sep='\t', header=0)
	pairs = [(int(df.iloc[i][0][3:]), int(df.iloc[i][1][3:])) for i in range(len(df))]
	return pairs

def get_scores(d, c, s, m, norm, num_samples, group='all', zi=False):

	scores = load_results(num_samples, d, c, s, m, zi=zi)[norm].to_numpy()
	if group == 'all':
		return scores
		
	if group == 'null':
		return scores[np.where(get_true_labels(num_samples, d, c, s, m, zi=zi)==0)]
	
	elif group == 'tp':
		return scores[np.where(get_true_labels(num_samples, d, c, s, m, zi=zi)==1)]
	
def get_null_scores(d, c, s, m, norm, num_samples, zi=False):

	return get_scores(d, c, s, m, norm, num_samples, group='null', zi=zi)

def get_tp_scores(d, c, s, m, norm, num_samples, zi=False):

	return get_scores(d, c, s, m, norm, num_samples, group='tp', zi=zi)

def get_true_labels(num_samples, d, c, s, m, zi=False):

	tp_inds = get_true_positives(num_samples, d, c, s, zi=zi)
	df = load_results(num_samples, d, c, s, m, zi=zi)
	true_labels = np.array([1 if ((int(df['Var1'][i][3:]), int(df['Var2'][i][3:])) in tp_inds or 
							   (int(df['Var2'][i][3:]), int(df['Var1'][i][3:])) in tp_inds) else 0 
						 for i in range(len(df))])
	return true_labels

def get_all_nulls(d, s, m, norm, num_samples, zi=False):
	conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	null_dist = np.array(())
	for i in range(len(conditions)):
		null_dist = np.append(null_dist, get_null_scores(d, conditions[i], s, m, norm, num_samples, zi=zi))
	return null_dist

def get_all_tp(d, s, m, norm, num_samples, zi=False):
	conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	tp = np.array(())
	for i in range(len(conditions)):
		tp = np.append(tp, get_tp_scores(d, conditions[i], s, m, norm, num_samples, zi=zi))
	return tp

def get_empirical_p(test, nulls, m, num_samples=50):
	'''
	calculates empirical p values based on a set of null scores
	'''
	if m in ('pearson','spearman'):
		nulls_abs = np.abs(nulls)
		nulls_abs.sort()
		test_abs = np.abs(test)
		num_nulls = len(nulls)
		p_vals = [ min( ((num_nulls-np.searchsorted(nulls_abs, x)+1)/ (num_nulls + 1)),  num_nulls/(num_nulls + 1))  for x in test_abs]
	else:
		sorted_nulls = nulls.copy()
		sorted_nulls.sort()
		num_nulls = len(nulls)
		p_vals = [ min( ((num_nulls-np.searchsorted(sorted_nulls, x)+1)/ (num_nulls + 1)),  num_nulls/(num_nulls + 1))  for x in test]

	return np.asarray(p_vals)

def get_adjusted_p(p_vals, correction, alpha=0.05):
	'''
	adjusts p values for multiple testing
	correction = ('fdr_bh', 'bonferroni')
	'''
	with np.errstate(divide='ignore'):
		return multipletests(p_vals, method=correction, alpha=alpha)[1]

def get_empirical_q(p_vals, lam=0.5):
	'''
	calculates an empirical q value based on p values
	'''
	sorted_pvals = p_vals.copy()
	sorted_pvals.sort()
	num_pvals = len(p_vals)
	pi_hat = sum(p_vals>lam) / ((1-lam)*num_pvals)
	return np.asarray([ ((len(p_vals)+1)*pi_hat*x) / (np.searchsorted(sorted_pvals, x, side='right')+1) for x in p_vals])

def get_parametric_q(p_vals, local=False, x0 = [0.5, 0.5]):
	'''
	calculates a parametric q value based on empirical p values
	p values are modeled with a beta-uniform mixture distribution
	'''
	def neg_log_BUM_logit(params, x):
		'''
		params correspond to the logits of the lambda and a parameters for beta uniform mixture
		returns the negative log likelihood of the observations
		'''
		psi, phi = params
		l = np.exp(phi)/(1+np.exp(phi))
		a = np.exp(psi)/(1+np.exp(psi))
		return -(np.log(l + (1-l)*a*(x**(a-1))).sum())

	def calc_q_BUM(params, x):
		'''
		params correspond to the logits of the lambda and a parameters for beta uniform mixture
		returns FDR for each observation
		'''
		psi, phi = params
		l = np.exp(phi)/(1+np.exp(phi))
		a = np.exp(psi)/(1+np.exp(psi))
		pi_ub = l + (1-l)*a    
		F_x = l*x + (1-l)*(x**a)
		p_a = F_x - pi_ub*x
		p_c = pi_ub*x
		return p_c / (p_a+p_c)

	def calc_local_q_BUM(params, x):
		'''
		params correspond to the logits of the lambda and a parameters for beta uniform mixture
		returns FDR for each observation
		'''
		psi, phi = params
		l = np.exp(phi)/(1+np.exp(phi))
		a = np.exp(psi)/(1+np.exp(psi))
		pi_ub = l + (1-l)*a    
		
		return pi_ub / (l + (1-l)*a*(x**(a-1)))

	res = minimize(neg_log_BUM_logit, x0, args=p_vals)
	if not local:
		return calc_q_BUM(res.x, p_vals)
	else:
		return calc_local_q_BUM(res.x, p_vals)

def get_pred_labels(m, num_samples, tp_scores, null_scores, alpha=0.05, correction_method='fdr_bh', seed=111):
	'''
	Returns predicted labels after applying indicated p value correction method
	correction_method = ('fdr_bh', 'bonferroni', 'empirical_q', 'parametric_q', 'local_q')
	'''
	rng = np.random.default_rng(seed)
	null_test_ind = rng.permutation(len(null_scores))
	null_test = null_scores[null_test_ind[:len(tp_scores)]]
	null_dist = null_scores[null_test_ind[len(tp_scores):]]
	
	scores = np.append(tp_scores, null_test)
	labels = np.append(np.ones_like(tp_scores), np.zeros_like(null_test))
	
	p_vals = get_empirical_p(scores, null_dist, m)

	if correction_method in ('fdr_bh', 'bonferroni'):
		ret = get_adjusted_p(p_vals, correction_method, alpha=alpha)

	elif correction_method == 'empirical_q':
		ret = get_empirical_q(p_vals)

	elif correction_method == 'parametric_q':
		ret = get_parametric_q(p_vals)

	elif correction_method == 'local_q':
		ret = get_parametric_q(p_vals, local=True)

	return np.asarray([1 if x<=alpha else 0 for x in ret])


def calc_AUC(m, tp_scores, null_scores, conf=False):
	'''
	Raw score based AUC
	'''

	rng = np.random.default_rng(500)
	if conf:
		scores = np.append(tp_scores, null_scores)
		if m in ('pearson','spearman'):
			scores = np.abs(scores)
		labels = np.append(np.ones_like(tp_scores), np.zeros_like(null_scores))
		return AUC_ci(scores, labels)
	else:
		scores = np.append(tp_scores, rng.choice(null_scores, len(tp_scores)))
		if m in ('pearson','spearman'):
			scores = np.abs(scores)
		labels = np.append(np.ones_like(tp_scores), np.zeros_like(tp_scores))
		return roc_auc_score(labels, scores)

def AUC_ci(scores, labels, num_bootstraps=1000):
	rng = np.random.default_rng(95400)
	vals = []
	pos_inds = np.asarray(labels==1).nonzero()[0]
	null_inds = np.asarray(labels==0).nonzero()[0]
	for i in range(num_bootstraps):
		inds = np.concatenate((rng.choice(pos_inds, len(pos_inds)), rng.choice(null_inds, len(pos_inds))), axis=None)
		vals.append(roc_auc_score(labels[inds], scores[inds]))
	vals = np.asarray(vals)
	vals.sort()
	return vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals))]

def calc_metrics(m, num_samples, tp_scores, null_scores, alpha=0.05, correction_method='fdr_bh', seed=123, conf=False):

	# returns (TPR, FDR)
	rng = np.random.default_rng(seed)
	if not conf:
		res = get_pred_labels(m, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction_method, seed=seed)
		labels = np.append(np.ones_like(tp_scores), np.zeros_like(tp_scores))
		tn, fp, fn, tp = confusion_matrix(labels, res).ravel()
		tpr = 0.0 if tp == 0 else tp/(tp+fn)
		fdr = 0.0 if fp == 0 else fp/(tp+fp)
		return (tpr,fdr)
	else:
		tpr = []
		fdr = []
		for i in range(1000):
			cur_tpr, cur_fdr = calc_metrics(m, num_samples, rng.choice(tp_scores, 250), null_scores, alpha=alpha, correction_method=correction_method, seed=i)
			tpr.append(cur_tpr)
			fdr.append(cur_fdr)
		tpr = np.asarray(tpr)
		tpr.sort()
		fdr = np.asarray(fdr)
		fdr.sort()
		
		return ((np.mean(tpr), tpr[int(0.05 * len(tpr))], tpr[int(0.95 * len(tpr))]), (np.mean(fdr), fdr[int(0.05 * len(fdr))], fdr[int(0.95 * len(fdr))]))


def calc_condition_specific_tpr_bootstrap(m, num_samples, tp_scores, null_scores, num_edges=100, alpha=0.05, correction_method='fdr_bh', seed=111, boot=False):
	
	conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	if not boot:
		rng = np.random.default_rng(seed)
		res = {}
		temp = get_pred_labels(m, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction_method, seed=seed)
		for i in range(5):
			res[conditions[i]] = sum(temp[i*num_edges:(i+1)*num_edges])/num_edges
		return res
	else:
		res = {cond : [] for cond in conditions}
		for i in range(5000):
			rng = np.random.default_rng(i)
			temp_tp_scores = np.array([])
			for j in range(len(conditions)):
				temp_tp_scores = np.append(temp_tp_scores, rng.choice(tp_scores[j*num_edges:(j+1)*num_edges], num_edges//2))
			temp = calc_condition_specific_tpr_bootstrap(m, num_samples, temp_tp_scores, null_scores, num_edges=num_edges//2, alpha=alpha, correction_method=correction_method, seed=i+1)
			for cond in conditions:
				res[cond].append(temp[cond])
		for cond in conditions:
			res[cond] = np.asarray(res[cond])
		return res


#######################################################
################ Plotting fxns ########################
#######################################################

def bold_max(s):
	is_max = s == s.max()
	return ['font-weight : bold' if cell else '' for cell in is_max]

def bold_min(s):
	is_min = s == s.min()
	return ['font-weight : bold' if cell else '' for cell in is_min]

def method_x_distribution_table(num_samples=50, methods=None, distributions=None, strength='weak',
								norms=None, conditions=None, zi=False, alpha=0.05, correction_method=None,
								measure='AUC'):
	'''
	generates tables for overall method results by distribution
	'''
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']
	if norms is None:
		norms = ['TMM','RLE','TSS']
	if correction_method is None:
		correction_method = ['fdr_bh', 'bonferroni', 'empirical_q', 'parametric_q', 'local_q']

	for norm in norms:
		for correction in correction_method:
			if measure == 'AUC':
				# do stuff
				######### get best performing KSG, LNC and Partition number to prevent crowding
				best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
				cur_methods = []
				for method in methods:
					if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
						auc = []
						for dist in distributions:
							tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
							auc.append(calc_AUC(method, tp_scores, null_scores))
						auc = sum(auc)/len(auc)
						if auc > best[method.split('_')[0]][1]:
							best[method.split('_')[0]][0] = method
							best[method.split('_')[0]][1] = auc
					else:
						cur_methods.append(method)
				cur_methods.extend([best[key][0] for key in list(best.keys())])
				########## Build table
				row_names = [x.upper() for x in cur_methods]
				col_names = [dist.capitalize().replace("_"," ") for dist in distributions]
				res = pd.DataFrame(columns=col_names, index=row_names)
				for i in range(len(cur_methods)):
					temp = []
					for dist in distributions:
						tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						ci = calc_AUC(cur_methods[i], tp_scores, null_scores, conf=True)
						temp.append('{:.3f} +/- {:.3f}'.format(np.mean(ci), ci[1]-np.mean(ci) ))
					res.iloc[i] = temp
				res.fillna('', inplace=True)
				res = res.style.apply(bold_max, axis=1)

			else:
				######### get best performing KSG, LNC and Partition number to prevent crowding
				best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
				cur_methods = []
				for method in methods:
					if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
						tpr = []
						for dist in distributions:
							tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
							tpr.append(calc_metrics(method, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction)[0])
						tpr = sum(tpr)/len(tpr)
						if tpr > best[method.split('_')[0]][1]:
							best[method.split('_')[0]][0] = method
							best[method.split('_')[0]][1] = tpr
					else:
						cur_methods.append(method)
				cur_methods.extend([best[key][0] for key in list(best.keys())])
				########## Build table
				row_names = [x.upper() for x in cur_methods]
				col_names = [dist.capitalize().replace("_"," ") for dist in distributions]
				
				tpr_res = pd.DataFrame(columns=col_names, index=row_names)
				fdr_res = pd.DataFrame(columns=col_names, index=row_names)
				for i in range(len(cur_methods)):
					tpr_temp = []
					fdr_temp = []
					for dist in distributions:
						tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						ci = calc_metrics(cur_methods[i], num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction, conf=True)
						tpr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[0][0], ci[0][1], ci[0][2]))
						fdr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[1][0], ci[1][1], ci[1][2]))
					tpr_res.iloc[i] = tpr_temp
					fdr_res.iloc[i] = fdr_temp
				tpr_res.fillna('', inplace=True)
				fdr_res.fillna('', inplace=True)
				tpr_res = tpr_res.style.apply(bold_max, axis=1)
				fdr_res = fdr_res.style.apply(bold_min, axis=1)
			# SAVE 
			if zi:
				if measure == 'AUC':
					if not os.path.exists('./Output_tables/ZI_method_x_distribution/AUC_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_distribution/AUC_{}_samples/'.format(num_samples))

					res.set_caption('{} by Distribution ({} data, {} samples, Zero inflated)'.format(measure, norm, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/ZI_method_x_distribution/AUC_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)
					break
				else:
					if not os.path.exists('./Output_tables/ZI_method_x_distribution/TPR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_distribution/TPR_{}_samples/'.format(num_samples))
					if not os.path.exists('./Output_tables/ZI_method_x_distribution/FDR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_distribution/FDR_{}_samples/'.format(num_samples))

					tpr_res.set_caption('TPR by Distribution ({} data, {}, {} samples, Zero inflated)'.format(norm, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					fdr_res.set_caption('FDR by Distribution ({} data, {}, {} samples, Zero inflated)'.format(norm, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(tpr_res, './Output_tables/ZI_method_x_distribution/TPR_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)
					dfi.export(fdr_res, './Output_tables/ZI_method_x_distribution/FDR_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)
			else:
				if measure == 'AUC':
					if not os.path.exists('./Output_tables/method_x_distribution/AUC_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/method_x_distribution/AUC_{}_samples/'.format(num_samples))

					res.set_caption('{} by Distribution ({} data, {} samples)'.format(measure, norm, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/method_x_distribution/AUC_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)
					break
				else:
					if not os.path.exists('./Output_tables/method_x_distribution/TPR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/method_x_distribution/TPR_{}_samples/'.format(num_samples))
					if not os.path.exists('./Output_tables/method_x_distribution/FDR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/method_x_distribution/FDR_{}_samples/'.format(num_samples))
					tpr_res.set_caption('TPR by Distribution ({} data, {}, {} samples)'.format(norm, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					fdr_res.set_caption('FDR by Distribution ({} data, {}, {} samples)'.format(norm, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(tpr_res, './Output_tables/method_x_distribution/TPR_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)
					dfi.export(fdr_res, './Output_tables/method_x_distribution/FDR_{}_samples/{}_{}.png'.format(num_samples, strength, norm), fontsize=3, dpi=800)



def method_x_condition_table(num_samples=50, methods=None, distributions=None, strengths=None,
							norms=None, conditions=None, zi=False, alpha=0.05, correction_method=None, measure='TPR'):
	'''
	generates tables for method results by condition
	'''
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']
	if conditions is None:
		conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	if strengths is None:
		strengths = ['weak', 'strong']
	if norms is None:
		norms = ['TMM','RLE','TSS']
	if correction_method is None:
		correction_method = ['fdr_bh', 'bonferroni', 'empirical_q', 'parametric_q', 'local_q']

	for norm in norms:
		for correction in correction_method:
			for dist in distributions:
				######### get best performing KSG, LNC and Partition number to prevent crowding
				best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
				cur_methods = []
				for method in methods:
					if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
						tpr = []
						for strength in strengths:
							tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
							tpr.append(calc_metrics(method, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction)[0])
						tpr = sum(tpr)/len(tpr)
						if tpr > best[method.split('_')[0]][1]:
							best[method.split('_')[0]][0] = method
							best[method.split('_')[0]][1] = tpr
					else:
						cur_methods.append(method)
				cur_methods.extend([best[key][0] for key in list(best.keys())])
				########## Build table
				row_names = [x.upper() for x in cur_methods]
				if len(strengths)==1:
					col_names = [cond.capitalize() for cond in conditions]
				else:
					col_names = [cond.capitalize()+'_'+strength for cond in conditions for strength in strengths]
				res = pd.DataFrame(columns=col_names, index=row_names)
				for i in range(len(cur_methods)):
					temp = {cond : [] for cond in conditions}
					for strength in strengths:
						tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						ci = calc_condition_specific_tpr(cur_methods[i], num_samples, tp_scores, null_scores, correction_method=correction, conf=True)
						for cond in conditions:
							temp[cond].append('{:.3f} ({:.3f}, {:.3f})'.format(ci[cond][0], ci[cond][1], ci[cond][2]))
					final = []
					for k in range(len(conditions)):
						final.extend(temp[conditions[k]])
					res.iloc[i] = final
				res.fillna('', inplace=True)
				res = res.style.apply(bold_max)
				# save stuff
				if zi:
					if not os.path.exists('./Output_tables/ZI_method_x_condition/{}_{}_samples/'.format(measure, num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_condition/{}_{}_samples/'.format(measure, num_samples))
						os.mkdir('./Output_tables/ZI_method_x_condition/{}_{}_samples/{}_{}_{}/'.format(measure, num_samples, strengths[0],
									dist, norm))

					res.set_caption('{} by Ecological Condition ({} {} data, {}, {} samples, Zero inflated)'.format(measure, norm, dist.capitalize().replace("_"," "), correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/ZI_method_x_condition/{}_{}_samples/{}_{}_{}/{}.png'.format(measure, num_samples, strengths[0],
									dist, norm, correction), fontsize=3, dpi=800)
				else:
					if not os.path.exists('./Output_tables/method_x_condition/{}_{}_samples/'.format(measure, num_samples)):
						os.mkdir('./Output_tables/method_x_condition/{}_{}_samples/'.format(measure, num_samples))
						os.mkdir('./Output_tables/method_x_condition/{}_{}_samples/{}_{}_{}/'.format(measure, num_samples, strengths[0],
									dist, norm))
					res.set_caption('{} by Ecological Condition ({} {} data, {}, {} samples)'.format(measure, norm, dist.capitalize().replace("_"," "), correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/method_x_condition/{}_{}_samples/{}_{}_{}/{}.png'.format(measure, num_samples, strengths[0],
									dist, norm, correction), fontsize=3, dpi=800)


def method_x_correction_table(num_samples=50, methods=None, distributions=None, strengths=None,
							norms=None, conditions=None, zi=False, alpha=0.05, correction_method=None):
	'''
	generates tables for method results by testing correction approach
	'''
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']
	if conditions is None:
		conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	if strengths is None:
		strengths = ['weak', 'strong']
	if norms is None:
		norms = ['TMM','RLE','TSS']
	if correction_method is None:
		correction_method = ['fdr_bh', 'bonferroni', 'empirical_q', 'parametric_q', 'local_q']

	for norm in norms:
		for dist in distributions:
			######### get best performing KSG, LNC and Partition number to prevent crowding
			best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
			cur_methods = []
			for method in methods:
				if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
					tpr = []
					for correction in correction_method:
						for strength in strengths:
							tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
							tpr.append(calc_metrics(method, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction)[0])
					tpr = sum(tpr)/len(tpr)
					if tpr > best[method.split('_')[0]][1]:
						best[method.split('_')[0]][0] = method
						best[method.split('_')[0]][1] = tpr
				else:
					cur_methods.append(method)
			cur_methods.extend([best[key][0] for key in list(best.keys())])
			########## Build table
			row_names = [x.upper() for x in cur_methods]
			if len(strengths)==1:
				col_names = [correction for correction in correction_method]
			else:
				col_names = [correction+'_'+strength for correction in correction_method for strength in strengths]
			tpr_res = pd.DataFrame(columns=col_names, index=row_names)
			fdr_res = pd.DataFrame(columns=col_names, index=row_names)
			for i in range(len(cur_methods)):
				tpr_temp = []
				fdr_temp = []
				for correction in correction_method:
					for strength in strengths:
						tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
						ci = calc_metrics(cur_methods[i], num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction, conf=True)
						# temp.append('{:.3f} +/- {:.3f}'.format(ci[0], np.mean(ci[1:]), ci[2]-np.mean(ci[1:])))
						tpr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[0][0], ci[0][1], ci[0][2]))
						fdr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[1][0], ci[1][1], ci[1][2]))
				tpr_res.iloc[i] = tpr_temp
				fdr_res.iloc[i] = fdr_temp
			tpr_res.fillna('', inplace=True)
			fdr_res.fillna('', inplace=True)
			tpr_res = tpr_res.style.apply(bold_max, axis=1)
			fdr_res = fdr_res.style.apply(bold_min, axis=1)
			# SAVE 
			if zi:
				if not os.path.exists('./Output_tables/ZI_method_x_correction/TPR_{}_samples/'.format(num_samples)):
					os.mkdir('./Output_tables/ZI_method_x_correction/TPR_{}_samples/'.format(num_samples))

				if not os.path.exists('./Output_tables/ZI_method_x_correction/FDR_{}_samples/'.format(num_samples)):
					os.mkdir('./Output_tables/ZI_method_x_correction/FDR_{}_samples/'.format(num_samples))

				tpr_res.set_caption('TPR by Testing Correction Method ({} {} data, {} samples, Zero inflated)'.format(norm, dist.capitalize().replace("_"," "), num_samples)).set_table_styles([{
					'selector': 'caption',
					'props': [('font-weight', 'bold')]}])
				fdr_res.set_caption('FDR by Testing Correction Method ({} {} data, {} samples, Zero inflated)'.format(norm, dist.capitalize().replace("_"," "), num_samples)).set_table_styles([{
					'selector': 'caption',
					'props': [('font-weight', 'bold')]}])
				dfi.export(tpr_res, './Output_tables/ZI_method_x_correction/TPR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, norm), fontsize=3, dpi=800)
				dfi.export(fdr_res, './Output_tables/ZI_method_x_correction/FDR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, norm), fontsize=3, dpi=800)
			else:
				if not os.path.exists('./Output_tables/method_x_correction/TPR_{}_samples/'.format(num_samples)):
					os.mkdir('./Output_tables/method_x_correction/TPR_{}_samples/'.format(num_samples))

				if not os.path.exists('./Output_tables/method_x_correction/FDR_{}_samples/'.format(num_samples)):
					os.mkdir('./Output_tables/method_x_correction/FDR_{}_samples/'.format(num_samples))

				tpr_res.set_caption('TPR by Testing Correction Method ({} {} data, {} samples)'.format(norm, dist.capitalize().replace("_"," "), num_samples)).set_table_styles([{
					'selector': 'caption',
					'props': [('font-weight', 'bold')]}])
				fdr_res.set_caption('FDR by Testing Correction Method ({} {} data, {} samples)'.format(norm, dist.capitalize().replace("_"," "), num_samples)).set_table_styles([{
					'selector': 'caption',
					'props': [('font-weight', 'bold')]}])
				dfi.export(tpr_res, './Output_tables/method_x_correction/TPR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, norm), fontsize=3, dpi=800)
				dfi.export(fdr_res, './Output_tables/method_x_correction/FDR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, norm), fontsize=3, dpi=800)


def method_x_norm_table(num_samples=50, methods=None, distributions=None, strengths=None,
							norms=None, conditions=None, zi=False, alpha=0.05, correction_method=None, measure='AUC'):
	'''
	generates tables for method results by norm
	25 tables total (num_corrections x num_distributions)
	'''
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']
	if conditions is None:
		conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']
	if strengths is None:
		strengths = ['weak', 'strong']
	if norms is None:
		norms = ['TMM','RLE','TSS']
	if correction_method is None:
		correction_method = ['fdr_bh', 'bonferroni', 'empirical_q', 'parametric_q', 'local_q']

	for dist in distributions:	
		for correction in correction_method:
			if measure == 'AUC':
				######### get best performing KSG, LNC and Partition number to prevent crowding
				best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
				cur_methods = []
				for method in methods:
					if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
						auc = []
						for norm in norms:
							for strength in strengths:
								tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
								null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
								auc.append(calc_AUC(method, tp_scores, null_scores))
						auc = sum(auc)/len(auc)
						if auc > best[method.split('_')[0]][1]:
							best[method.split('_')[0]][0] = method
							best[method.split('_')[0]][1] = auc
					else:
						cur_methods.append(method)
				cur_methods.extend([best[key][0] for key in list(best.keys())])
				########## Build table
				row_names = [x.upper() for x in cur_methods]
				if len(strengths)==1:
					col_names = [norm for norm in norms]
				else:
					col_names = [norm+'_'+strength for norm in norms for strength in strengths]
				res = pd.DataFrame(columns=col_names, index=row_names)
				for i in range(len(cur_methods)):
					temp = []
					for norm in norms:
						for strength in strengths:
							tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
							ci = calc_AUC(cur_methods[i], tp_scores, null_scores, conf=True)
							temp.append('{:.3f} +/- {:.3f}'.format(np.mean(ci), ci[1]-np.mean(ci) ))
					res.iloc[i] = temp
				res.fillna('', inplace=True)
				res = res.style.apply(bold_max, axis=1)

			else:
				# do stuff
				######### get best performing KSG, LNC and Partition number to prevent crowding
				best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
				cur_methods = []
				for method in methods:
					if method.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
						tpr = []
						for norm in norms:
							for strength in strengths:
								tp_scores = get_all_tp(dist, strength, method, norm, num_samples, zi=zi)
								null_scores = get_all_nulls(dist, strength, method, norm, num_samples, zi=zi)
								tpr.append(calc_metrics(method, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction)[0])
						tpr = sum(tpr)/len(tpr)
						if tpr > best[method.split('_')[0]][1]:
							best[method.split('_')[0]][0] = method
							best[method.split('_')[0]][1] = tpr
					else:
						cur_methods.append(method)
				cur_methods.extend([best[key][0] for key in list(best.keys())])
				########## Build table
				row_names = [x.upper() for x in cur_methods]
				if len(strengths)==1:
					col_names = [norm for norm in norms]
				else:
					col_names = [norm+'_'+strength for norm in norms for strength in strengths]
				tpr_res = pd.DataFrame(columns=col_names, index=row_names)
				fdr_res = pd.DataFrame(columns=col_names, index=row_names)
				for i in range(len(cur_methods)):
					tpr_temp = []
					fdr_temp = []
					for norm in norms:
						for strength in strengths:
							tp_scores = get_all_tp(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
							null_scores = get_all_nulls(dist, strength, cur_methods[i], norm, num_samples, zi=zi)
							ci = calc_metrics(cur_methods[i], num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction, conf=True)
							# temp.append('{:.3f} +/- {:.3f}'.format(ci[0], np.mean(ci[1:]), ci[2]-np.mean(ci[1:])))
							tpr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[0][0], ci[0][1], ci[0][2]))
							fdr_temp.append('{:.3f} ({:.3f}, {:.3f})'.format(ci[1][0], ci[1][1], ci[1][2]))
					tpr_res.iloc[i] = tpr_temp
					fdr_res.iloc[i] = fdr_temp
				tpr_res.fillna('', inplace=True)
				fdr_res.fillna('', inplace=True)
				tpr_res = tpr_res.style.apply(bold_max, axis=1)
				fdr_res = fdr_res.style.apply(bold_min, axis=1)
			# SAVE 
			if zi:
				if measure == 'AUC':
					if not os.path.exists('./Output_tables/ZI_method_x_norm/{}_{}_samples/'.format(measure, num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_norm/{}_{}_samples/'.format(measure, num_samples))
					res.set_caption('{} by Normalization ({} data, {} samples, Zero inflated)'.format(measure, dist, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/ZI_method_x_norm/{}_{}_samples/{}_{}_{}.png'.format(measure, num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)
					break

				else:
					if not os.path.exists('./Output_tables/ZI_method_x_norm/TPR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_norm/TPR_{}_samples/'.format(num_samples))
					if not os.path.exists('./Output_tables/ZI_method_x_norm/FDR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/ZI_method_x_norm/FDR_{}_samples/'.format(num_samples))

					tpr_res.set_caption('TPR by Normalization ({} data, {}, {} samples, Zero inflated)'.format(dist, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					fdr_res.set_caption('FDR by Normalization ({} data, {}, {} samples, Zero inflated)'.format(dist, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(tpr_res, './Output_tables/ZI_method_x_norm/TPR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)
					dfi.export(fdr_res, './Output_tables/ZI_method_x_norm/FDR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)

			else:
				if measure == 'AUC':
					if not os.path.exists('./Output_tables/method_x_norm/{}_{}_samples/'.format(measure, num_samples)):
						os.mkdir('./Output_tables/method_x_norm/{}_{}_samples/'.format(measure, num_samples))
					res.set_caption('{} by Normalization ({} data, {} samples)'.format(measure, dist, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(res, './Output_tables/method_x_norm/{}_{}_samples/{}_{}_{}.png'.format(measure, num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)
					break

				else:
					if not os.path.exists('./Output_tables/method_x_norm/TPR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/method_x_norm/TPR_{}_samples/'.format(num_samples))
					if not os.path.exists('./Output_tables/method_x_norm/FDR_{}_samples/'.format(num_samples)):
						os.mkdir('./Output_tables/method_x_norm/FDR_{}_samples/'.format(num_samples))

					tpr_res.set_caption('TPR by Normalization ({} data, {}, {} samples)'.format(dist, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					fdr_res.set_caption('FDR by Normalization ({} data, {}, {} samples)'.format(dist, correction, num_samples)).set_table_styles([{
						'selector': 'caption',
						'props': [('font-weight', 'bold')]}])
					dfi.export(tpr_res, './Output_tables/method_x_norm/TPR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)
					dfi.export(fdr_res, './Output_tables/method_x_norm/FDR_{}_samples/{}_{}_{}.png'.format(num_samples, strengths[0], dist, correction), fontsize=3, dpi=800)

def sensitivity_x_condition_box_plot(num_samples=50, methods=None, distributions=None, strength='weak',
							norm='TMM', zi=False, alpha=0.05, correction_method='fdr_bh', save=False):
	"""
	x axis = distributions
	y axis = sensitivity
	"""

	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']

	conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']

	df = pd.DataFrame(columns=["Method", "Condition", "Distribution", "TPR"])
	temp_df = pd.DataFrame(columns=["Method", "Condition", "Distribution", "TPR"])
	for m in methods:
		for d in distributions:
			tp_scores = get_all_tp(d, strength, m, norm, num_samples, zi=zi)
			null_scores = get_all_nulls(d, strength, m, norm, num_samples, zi=zi)
			temp = calc_condition_specific_tpr_bootstrap(m, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction_method, boot=True)
			for cond in conditions:
				if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
					temp_df = pd.concat([temp_df, pd.concat([pd.DataFrame([[m, cond, d, temp[cond][i]]], columns=["Method", "Condition", "Distribution", "TPR"]) for i in range(len(temp[cond]))], ignore_index=True)], ignore_index=True)
				else:
					df = pd.concat([df, pd.concat([pd.DataFrame([[m, cond, d, temp[cond][i]]], columns=["Method", "Condition", "Distribution", "TPR"]) for i in range(len(temp[cond]))], ignore_index=True)], ignore_index=True)
	# Get best KSG, LNC, Partitioning results for each distribution
	for d in distributions:
		best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
		for m in methods:
			if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
				cur = temp_df.loc[(temp_df["Method"]==m) & (temp_df["Distribution"]==d)]['TPR'].mean()
				if cur > best[m.split('_')[0]][1]:
					best[m.split('_')[0]][0] = m
					best[m.split('_')[0]][1] = cur
		for m in best:
			temp = temp_df.loc[(temp_df["Method"]==best[m][0]) & (temp_df["Distribution"]==d)].copy()
			temp["Method"] = m
			df = pd.concat([df, temp], ignore_index=True)
		if not zi:
			if not os.path.exists('./Output_boxplots/{}_samples/{}_{}/alpha_{}/'.format(num_samples, strength, norm, alpha)):
				os.mkdir('./Output_boxplots/{}_samples/{}_{}/alpha_{}/'.format(num_samples, strength, norm, alpha))

			with open('./Output_boxplots/{}_samples/{}_{}/alpha_{}/{}_besthyper.txt'.format(num_samples, strength, norm, alpha, correction_method),'a') as f:
				f.write('{}\n'.format(d))
				f.write(str(best))
				f.write('\n')
		else:
			if not os.path.exists('./Output_boxplots/ZI_{}_samples/{}_{}/alpha_{}/'.format(num_samples, strength, norm, alpha)):
				os.mkdir('./Output_boxplots/ZI_{}_samples/{}_{}/alpha_{}/'.format(num_samples, strength, norm, alpha))
			with open('./Output_boxplots/ZI_{}_samples/{}_{}/alpha_{}/{}_besthyper.txt'.format(num_samples, strength, norm, alpha, correction_method),'a') as f:
				f.write('{}\n'.format(d))
				f.write(str(best))
				f.write('\n')

	methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 'ksg', 'lnc', 'partitioning']
	plt_titles = ['MINE', 'NWJ', 'DoE', 'MIC', 'Pearson', 'Spearman', 'KSG', 'LNC', 'Partitioning']
	df['Condition'] = df['Condition'].map({'mutual': 'Mutual', 'competitive': 'Competitive', 'amensal': 'Amensal', 
											'commensal': 'Commensal', 'exploitative': 'Exploitative'})
	# Plot
	sns.set_theme()
	fig, axs = plt.subplots(len(distributions),9,figsize =(20,13), sharey='row', dpi=80)
	# Set titles
	for ax, col in zip(axs[0], plt_titles):
		ax.set_title(col, fontsize=15)

	for i in range(len(distributions)):
		for j in range(9):
			sns.boxplot(ax=axs[i][j], data=df.loc[(df["Method"]==methods[j]) & (df["Distribution"]==distributions[i])], x="Distribution", y='TPR', hue="Condition", 
					width=0.8, flierprops={"marker": "x"}, fliersize=3)
			axs[i][j].set(xlabel=None)
			axs[i][j].set_xticklabels([])
			axs[i][j].get_legend().remove()
			if j == 0:
				axs[i][j].set_ylabel("{}\n\nSensitivity".format(distributions[i].capitalize().replace("_"," ")), fontsize = 15)
			else:
				axs[i][j].set(ylabel=None)

	fig.tight_layout()
	fig.subplots_adjust(wspace=0.01, hspace=0.075)
	axs[-1,4].legend(loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, ncol=5)
	
	if save:
		if not zi:
			fig.savefig('./Output_boxplots/{}_samples/{}_{}/alpha_{}/{}.pdf'.format(num_samples, strength, norm, alpha, correction_method), 
				bbox_inches="tight", format='pdf')
		else:
			fig.savefig('./Output_boxplots/ZI_{}_samples/{}_{}/alpha_{}/{}.pdf'.format(num_samples, strength, norm, alpha, correction_method), 
			bbox_inches="tight",format='pdf')

def sensitivity_heatmap(num_samples=50, methods=None, distributions=None, strength='weak', 
						norms=None, zi=False, alpha=0.05, correction_method='fdr_bh', save=False):
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman',
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	if distributions is None:
		distributions = ['log_normal', 'exponential','negative_binomial', 'gamma', 'beta_negative_binomial']

	if norms is None:
		norms = ['TMM','RLE','TSS']
		
	df = pd.DataFrame(columns=["Method", "Distribution", "Norm", "TPR", "FDR"])
	temp_df = pd.DataFrame(columns=["Method", "Distribution", "Norm", "TPR", "FDR"])
	for m in methods:
		for d in distributions:
			for norm in norms:
				tp_scores = get_all_tp(d, strength, m, norm, num_samples, zi=zi)
				null_scores = get_all_nulls(d, strength, m, norm, num_samples, zi=zi)
				tpr, fdr = calc_metrics(m, num_samples, tp_scores, null_scores, alpha=alpha, correction_method=correction_method, seed=123, conf=True)
				if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
					temp_df = pd.concat([temp_df, pd.DataFrame([[m, d, norm, tpr[0], fdr[0]]], columns=["Method", "Distribution", "Norm", "TPR", "FDR"])], ignore_index=True)
				else:
					df = pd.concat([df, pd.DataFrame([[m, d, norm, tpr[0], fdr[0]]], columns=["Method", "Distribution", "Norm", "TPR", "FDR"])], ignore_index=True)
	
	# save best performing KSG, LNC, Partition
	for d in distributions:
		for norm in norms:
			best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
			for m in methods:
				if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
					cur = temp_df.loc[(temp_df["Method"]==m) & (temp_df["Distribution"]==d) & (temp_df["Norm"]==norm)]['TPR'].values
					if cur > best[m.split('_')[0]][1]:
						best[m.split('_')[0]][1] = cur
						best[m.split('_')[0]][0] = m
			for m in best:
				temp = temp_df.loc[(temp_df["Method"]==best[m][0]) & (temp_df["Distribution"]==d) & (temp_df["Norm"]==norm)].copy()
				temp["Method"] = m
				df = pd.concat([df, temp], ignore_index=True)
			if not zi:
				if not os.path.exists('./Output_heatmaps/{}_samples/{}_alpha_{}/'.format(num_samples, strength, alpha)):
					os.mkdir('./Output_heatmaps/{}_samples/{}_alpha_{}/'.format(num_samples, strength, alpha))
				with open('./Output_heatmaps/{}_samples/{}_alpha_{}/{}_besthyper.txt'.format(num_samples, strength, alpha, correction_method),'a') as f:
					f.write('{}, {}\n'.format(d,norm))
					f.write(str(best))
					f.write('\n')
			else:
				if not os.path.exists('./Output_heatmaps/ZI_{}_samples/{}_alpha_{}/'.format(num_samples, strength, alpha)):
					os.mkdir('./Output_heatmaps/ZI_{}_samples/{}_alpha_{}/'.format(num_samples, strength, alpha))
				with open('./Output_heatmaps/ZI_{}_samples/{}_alpha_{}/{}_besthyper.txt'.format(num_samples, strength, alpha, correction_method),'a') as f:
					f.write('{}, {}\n'.format(d,norm))
					f.write(str(best))
					f.write('\n')


	methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman', 'ksg', 'lnc', 'partitioning']
	plt_titles = ['MINE', 'NWJ', 'DoE', 'MIC', 'Pearson', 'Spearman', 'KSG', 'LNC', 'Partitioning']
	df['Distribution'] = df['Distribution'].map({'log_normal': 'Log-normal', 'exponential': 'Exponential', 'negative_binomial': 'Negative Binomial', 
											'gamma': 'Gamma', 'beta_negative_binomial': 'Beta Negative Binomial'})

	for val in ['TPR','FDR']:
		fig, axs = plt.subplots(3,3, figsize=(10, 8), sharex=True, sharey=True)
		vmax = 1.0 if val=='TPR' else 0.2
		for i in range(9):
			summary = pd.pivot_table(data=df[df["Method"] == methods[i]], index='Norm', columns='Distribution', values=val)
			sns.heatmap(ax=axs.flat[i], data=summary, cmap='rocket', vmin=0, vmax=vmax, annot=True, linewidths=0.5, cbar=False, fmt='.3f')
			axs.flat[i].tick_params(axis='x', labelrotation=75)
			axs.flat[i].tick_params(axis='y', labelrotation=0)
			axs.flat[i].set(ylabel=None, xlabel=None)
			axs.flat[i].set_title('{}'.format(plt_titles[i]), fontsize = 12)
		fig.subplots_adjust(wspace=0.05, hspace=0.18)
		fig.subplots_adjust(right=0.85)
		cbar_ax = fig.add_axes([0.875, 0.2, 0.025, 0.6])
		fig.colorbar(axs.flat[0].collections[0], cax=cbar_ax)

		if save:
			if not zi:
				fig.savefig('./Output_heatmaps/{}_samples/{}_alpha_{}/{}_{}.pdf'.format(num_samples, strength, alpha, correction_method, val), 
				bbox_inches="tight",format='pdf')
			else:
				fig.savefig('./Output_heatmaps/ZI_{}_samples/{}_alpha_{}/{}_{}.pdf'.format(num_samples, strength, alpha, correction_method, val), 
				bbox_inches="tight",format='pdf')


def alpha_line_graphs(num_samples=50, methods=None, distribution='log_normal', strength='weak', 
						norm='TMM', zi=False, correction_method='fdr_bh', save=False):
	if methods is None:
		methods = ['mine', 'nwj', 'doe', 'mic', 'pearson', 'spearman',
				'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
				'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
				'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']
	alphas = [0.01, 0.05, 0.10, 0.15, 0.2]
	df = pd.DataFrame(columns=["Method", "Alpha", "TPR", "FDR"])
	temp_df = pd.DataFrame(columns=["Method", "TPR",])
	sns.set(style="darkgrid")
	fig, axs = plt.subplots(2,1,figsize =(10,8), sharex=True)
	
	for m in methods:
		tp_scores = get_all_tp(distribution, strength, m, norm, num_samples, zi=zi)
		null_scores = get_all_nulls(distribution, strength, m, norm, num_samples, zi=zi)
		if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
			t, f = calc_metrics(m, num_samples, tp_scores, null_scores, alpha=0.05, correction_method=correction_method, conf=True)
			temp_df = pd.concat([temp_df, pd.DataFrame([[m, t[0]]], columns=["Method", "TPR"])] , ignore_index=True)
			continue

		for a in alphas:
			t, f = calc_metrics(m, num_samples, tp_scores, null_scores, alpha=a, correction_method=correction_method, conf=True)
			df = pd.concat([df, pd.DataFrame([[m, a, t[0], f[0]]], columns=["Method", "Alpha", "TPR", "FDR"])] , ignore_index=True)

	# save best performing KSG, LNC, Partition
	best = {'ksg': [None, float('-inf')], 'lnc': [None, float('-inf')], 'partitioning': [None, float('-inf')]}
	for m in methods:
		if m.split('_')[0] in ('ksg', 'lnc', 'partitioning'):
			cur = temp_df.loc[temp_df["Method"]==m]['TPR'].values
			if cur > best[m.split('_')[0]][1]:
				best[m.split('_')[0]][1] = cur
				best[m.split('_')[0]][0] = m
	for m in best:
		tp_scores = get_all_tp(distribution, strength, best[m][0], norm, num_samples, zi=zi)
		null_scores = get_all_nulls(distribution, strength, best[m][0], norm, num_samples, zi=zi)
		for a in alphas:
			t, f = calc_metrics(best[m][0], num_samples, tp_scores, null_scores, alpha=a, correction_method=correction_method, conf=True)
			df = pd.concat([df, pd.DataFrame([[m, a, t[0], f[0]]], columns=["Method", "Alpha", "TPR", "FDR"])] , ignore_index=True)

	if not zi:
		if not os.path.exists('./Output_line_graphs/{}_samples/{}_{}_{}/'.format(num_samples, strength, distribution, norm)):
			os.mkdir('./Output_line_graphs/{}_samples/{}_{}_{}/'.format(num_samples, strength, distribution, norm))
		with open('./Output_line_graphs/{}_samples/{}_{}_{}/{}_besthyper.txt'.format(num_samples, strength, distribution, norm, correction_method),'a') as f:
			f.write(str(best))
			f.write('\n')
	else:
		if not os.path.exists('./Output_line_graphs/ZI_{}_samples/{}_{}_{}/'.format(num_samples, strength, distribution, norm)):
			os.mkdir('./Output_line_graphs/ZI_{}_samples/{}_{}_{}/'.format(num_samples, strength, distribution, norm))
		with open('./Output_line_graphs/ZI_{}_samples/{}_{}_{}/{}_besthyper.txt'.format(num_samples, strength, distribution, norm, correction_method),'a') as f:
			f.write(str(best))
			f.write('\n')

	plt_titles = ['MINE', 'NWJ', 'DoE', 'MIC', 'Pearson', 'Spearman', 'KSG', 'LNC', 'Partitioning']
	df['Method'] = df['Method'].map({
		'mine': 'MINE', 
		'nwj': 'NWJ', 
		'doe': 'DoE', 
		'mic': 'MIC', 
		'pearson': 'Pearson',
		'spearman': 'Spearman',
		'ksg': 'KSG',
		'lnc': 'LNC',
		'partitioning': 'Parititioning'
		})

	sns.lineplot(ax=axs[0], data=df, x="Alpha", y="TPR", hue="Method", style="Method", markers=True, markersize=7, linewidth=2.5)
	sns.lineplot(ax=axs[1], data=df, x="Alpha", y="FDR", hue="Method", style="Method", markers=True, markersize=8, linewidth=2.5)

	axs[0].set_ylabel("TPR", fontsize = 15)
	axs[1].axhspan(0, 0.05, alpha=0.2)

	axs[1].set_xlabel("Significance Threshold (alpha)", fontsize = 15)
	axs[1].set_ylabel("FDR", fontsize = 15)
	axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
				mode="expand", borderaxespad=0, ncol=9)
	plt.xticks(alphas)
	axs[0].get_legend().remove()
	fig.tight_layout()

	if save:
		if not zi:
			fig.savefig('./Output_line_graphs/{}_samples/{}_{}_{}/{}.txt'.format(num_samples, strength, distribution, norm, correction_method), 
				bbox_inches="tight", format='pdf')
		else:
			fig.savefig('./Output_line_graphs/ZI_{}_samples/{}_{}_{}/{}.txt'.format(num_samples, strength, distribution, norm, correction_method), 
				bbox_inches="tight", format='pdf')



if __name__ == '__main__':

		#Fig1
		sensitivity_x_condition_box_plot(num_samples=50, strength='weak', norm='TMM', correction_method='fdr_bh')

		#Fig2
		sensitivity_heatmap(num_samples=50, strength='weak', correction_method='fdr_bh')

		#Fig3
		sensitivity_heatmap(num_samples=50, strength='strong', correction_method='fdr_bh')

		#Fig4
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='strong', norm='TMM', correction_method='fdr_bh')
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='strong', norm='TMM', correction_method='empirical_q')
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='strong', norm='TMM', correction_method='local_q')


	
		# Supplementary Table S1
		method_x_distribution_table(num_samples=50, strength='weak', norms=['RLE'], zi=False, measure='AUC')

		# Supplementary Table S2
		method_x_distribution_table(num_samples=50, strength='strong', norms=['RLE'], zi=False, measure='AUC')

		# Supplementary Table S3
		method_x_distribution_table(num_samples=50, strength='weak', norms=['TSS'], zi=False, measure='AUC')

		# Supplementary Table S4
		method_x_distribution_table(num_samples=50, strength='strong', norms=['TSS'], zi=False, measure='AUC')

		# Supplementary Table S5
		method_x_distribution_table(num_samples=50, strength='strong', norms=['TMM'], zi=True, measure='AUC')

		# Supplementary Figure S6
		sensitivity_x_condition_box_plot(num_samples=50, strength='strong', norm='TMM', correction_method='fdr_bh')

		# Supplementary Figure S7
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='weak', norm='TMM', correction_method='bonferroni')

		# Supplementary Figure S8
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='weak', norm='TMM', correction_method='fdr_bh')
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='weak', norm='TMM', correction_method='empirical_q')
		alpha_line_graphs(num_samples=50, distribution='log_normal', strength='weak', norm='TMM', correction_method='local_q')
