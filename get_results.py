import os
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy import stats
from minepy import pstats
from lnc import MI
from sklearn import metrics
import util
import torch
from torch.utils.data import DataLoader,TensorDataset
from multiprocessing import Pool
import argparse


def calc_results(model, eco_condition, distribution, strength, zi=False, num_workers=1, NUM_SAMPLES=50, NUM_OTUS=400, BATCH_SIZE=16):
	###############################
	NUM_PAIRS = 100
	###############################
	rng = default_rng(1209)

	## Results will be calculated for the following:
	# 1) NUM_PAIRS=100 true positive pairs
	# 2) 100 mixed (one null OTU with one altered OTU)
	# 3) 100 nulls (OTUs not used in true positives)

	# true positive pairs
	if zi:
		df = pd.read_csv('./Simulated_Data/ZI_{}_{}_samples/{}_{}_pairs.csv'.format(distribution, NUM_SAMPLES, strength, eco_condition), sep='\t', header=0)
	else:
		df = pd.read_csv('./Simulated_Data/{}_{}_samples/{}_{}_pairs.csv'.format(distribution, NUM_SAMPLES, strength, eco_condition), sep='\t', header=0)

	pool_input = [(int(df.iloc[i][0][3:]), int(df.iloc[i][1][3:])) for i in range(NUM_PAIRS)]
	del df

	tp_inds = [x for pair in pool_input for x in pair]
	null_inds = [x for x in range(NUM_OTUS) if x not in tp_inds]

	for i in range(NUM_PAIRS):
		# mixed pairs
		pool_input.append((int(rng.choice(tp_inds,1)), int(rng.choice(null_inds,1))))
		# null pairs
		pool_input.append(tuple(rng.choice(null_inds,2)))

	pool_input = [{'model': model, 'eco_condition': eco_condition, 'distribution' : distribution, 'strength' : strength,
					'zi' : zi, 'batch_size': BATCH_SIZE, 'num_samples': NUM_SAMPLES,
					'inds': pool_input[i::(num_workers)]} for i in range(num_workers)]
	res = pd.DataFrame(columns = ['Var1','Var2','TMM','RLE','TSS'])
	if num_workers>1:
		with Pool(num_workers) as pool:
			for result in pool.imap(calculate_MI, pool_input):
				res = pd.concat([res, result], ignore_index=True)
	else:
		res = pd.concat([res, calculate_MI(pool_input[0])], ignore_index=True)

	if zi:
		res.to_csv('./Simulated_Data_results/ZI_{}_{}_samples/{}_{}/{}.csv'.format(distribution, NUM_SAMPLES, strength, eco_condition, model),
			sep='\t',index=False)
	else:
		res.to_csv('./Simulated_Data_results/{}_{}_samples/{}_{}/{}.csv'.format(distribution, NUM_SAMPLES, strength, eco_condition, model),
			sep='\t',index=False)

def calculate_MI(input_dict):
	hypers = {
		'dim_hidden': 64, 
		'num_layers': 3, 
		'batch_size': input_dict['batch_size'], 
		'dropout_rate': 0.3,
		'activation': 'leaky_relu', 
		'alpha': 0.95, 
		'lr': 0.01,
		'seed': 1337, 
		###############################
		'num_epochs': 5000,
		'window_size': 500,
		###############################
		'epsilon': 0.05
	}

	ml_models = {
	'mine': "util.MINE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], alpha=hypers['alpha'])",
	'nwj': "util.NWJ(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'])",
	'doe_gauss': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='gauss')",
	'doe_log_normal': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='log_normal')",
	'doe_exponential': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='exponential')",
	'doe_negative_binomial': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='negative_binomial')",
	'doe_beta_negative_binomial': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='negative_binomial')",
	'doe_gamma': "util.DoE(dim_hidden=hypers['dim_hidden'], num_layers=hypers['num_layers'], activation=hypers['activation'], \
			dropout_rate=hypers['dropout_rate'], pdf='gamma')"
	}

	if input_dict['zi']:
		data = pd.read_csv('./Simulated_Data/ZI_{}_{}_samples/{}_{}.csv'.format(input_dict['distribution'], input_dict['num_samples'], 
			input_dict['strength'], input_dict['eco_condition']), sep='\t', header=0, index_col=0).values + 1
	else:
		data = pd.read_csv('./Simulated_Data/{}_{}_samples/{}_{}.csv'.format(input_dict['distribution'], input_dict['num_samples'], 
			input_dict['strength'], input_dict['eco_condition']), sep='\t', header=0, index_col=0).values + 1
	
	results = pd.DataFrame(columns = ['Var1','Var2', 'TMM','RLE','TSS'])

	for pair in input_dict['inds']:
		norm_res = []

		# MINE, NWJ, DoE
		if input_dict['model'] in ['mine', 'nwj', 'doe']:
			for norm in ['TMM','RLE','TSS']:
				if input_dict['model'] == 'doe':
					if norm == 'TSS':
						model = eval(ml_models['doe_gauss'])
					else:
						model = eval(ml_models['doe_'+input_dict['distribution']])
				else:
					model = eval(ml_models[input_dict['model']])

				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:]
				optim = torch.optim.Adam(model.parameters(), lr=hypers['lr'])
				dataloader = DataLoader(TensorDataset(temp_data[0],temp_data[1]), batch_size=hypers['batch_size'], shuffle=True, drop_last=True)
				trainer = util.Trainer(model, optim, window_size=hypers['window_size'], epsilon=hypers['epsilon'])
				trainer.train(temp_data, dataloader, hypers['num_epochs'])
				norm_res.append(max(0,trainer.get_ma_losses()[1][-1]))
				del temp_data
				del dataloader
				del trainer
				del optim
				del model
		# KSG
		elif input_dict['model'] in ['ksg_3','ksg_5','ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15']:
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:].numpy()
				norm_res.append(MI.mi_Kraskov(temp_data, k=int(input_dict['model'].split('_')[-1])))
				del temp_data 

		# LNC 
		elif input_dict['model'] in ['lnc_3','lnc_5','lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15']:
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:].numpy()
				norm_res.append(MI.mi_LNC(temp_data, k=int(input_dict['model'].split('_')[-1])))
				del temp_data

		# Partitioning
		elif input_dict['model'] in ['partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15']:
			BINS = int(input_dict['model'].split('_')[-1])
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:].numpy()
				norm_res.append(metrics.mutual_info_score(None,None,contingency=np.histogram2d(temp_data[0,:], temp_data[1,:], bins=BINS)[0]))
				del temp_data

		# MIC
		elif input_dict['model'] in ['mic']:
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:].numpy()
				norm_res.append(pstats(temp_data)[0][0])
				del temp_data
		
		# Pearson Correlation
		elif input_dict['model'] in ['pearson']:
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:]
				norm_res.append(stats.pearsonr(temp_data[0,:], temp_data[1,:])[0])
				del temp_data
		
		# Spearman's Rank Coefficient
		elif input_dict['model'] in ['spearman']:
			for norm in ['TMM','RLE','TSS']:
				temp_data = util.normalize_counts(data, norm)[(pair[0],pair[1]),:]
				norm_res.append(stats.spearmanr(temp_data[0,:], temp_data[1,:])[0])
				del temp_data
		results = pd.concat([results, pd.DataFrame.from_dict(
			{
			'Var1': ['OTU{}'.format(pair[0])],
			'Var2': ['OTU{}'.format(pair[1])],
			'TMM': [norm_res[0]],
			'RLE': [norm_res[1]],
			'TSS': [norm_res[2]]})], ignore_index=True)
		del norm_res

	return results

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Script for meta-entropy analysis')
	parser.add_argument('--model', nargs='+', type=str, default=['all'], help='Model(s) to run for score calculation')
	parser.add_argument('--num_samples', type=int, default=50, choices=[50, 200], help='Number of samples')
	parser.add_argument('--strength', default='weak', choices=['weak', 'strong'])
	parser.add_argument('--distribution', default='log_normal', 
		choices=['log_normal', 'exponential', 'gamma', 'negative_binomial', 'beta_negative_binomial'])
	parser.add_argument('--zi', action='store_true', help='Zero inflated data?')
	parser.add_argument('--parallel', action='store_true', help='Use cpu cores to parallelize computations?')
	args = parser.parse_args()

	NUM_WORKERS = os.cpu_count()-2 if args.parallel else 1
	print('Num of workers:',NUM_WORKERS)

	for model in args.model:
		assert model in {'all', 'mine', 'nwj', 'doe', 'pearson', 'spearman', 'mic',
		'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
		'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15', 
		'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15'}
	models = ['mine', 'nwj', 'doe', 'pearson', 'spearman', 'mic',
	'ksg_3', 'ksg_5', 'ksg_7', 'ksg_9', 'ksg_11', 'ksg_13', 'ksg_15', 
	'lnc_3', 'lnc_5', 'lnc_7', 'lnc_9', 'lnc_11', 'lnc_13', 'lnc_15',
	'partitioning_3', 'partitioning_5', 'partitioning_7', 'partitioning_10', 'partitioning_15'] if args.model == ['all'] else args.model

	NUM_SAMPLES = args.num_samples
	BATCH_SIZE = 16 if NUM_SAMPLES==50 else 64
	STRENGTH = args.strength
	DISTRIBUTION = args.distribution
	ZI = args.zi
	NUM_OTUS=400
	
	eco_conditions = ['mutual', 'competitive', 'amensal', 'commensal', 'exploitative']

	if not os.path.exists('./Simulated_Data_results/'):
		os.mkdir('./Simulated_Data_results/')

	if not ZI:
		if not os.path.exists('./Simulated_Data_results/{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES)):
			os.mkdir('./Simulated_Data_results/{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES))
	else:
		if not os.path.exists('./Simulated_Data_results/ZI_{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES)):
			os.mkdir('./Simulated_Data_results/ZI_{}_{}_samples/'.format(DISTRIBUTION, NUM_SAMPLES))

	for c in eco_conditions:
		if not ZI:
			if not os.path.exists('./Simulated_Data_results/{}_{}_samples/{}_{}'.format(DISTRIBUTION, NUM_SAMPLES, STRENGTH, c)):
				os.mkdir('./Simulated_Data_results/{}_{}_samples/{}_{}'.format(DISTRIBUTION, NUM_SAMPLES, STRENGTH, c))
		else:
			if not os.path.exists('./Simulated_Data_results/ZI_{}_{}_samples/{}_{}'.format(DISTRIBUTION, NUM_SAMPLES, STRENGTH, c)):
				os.mkdir('./Simulated_Data_results/ZI_{}_{}_samples/{}_{}'.format(DISTRIBUTION, NUM_SAMPLES, STRENGTH, c))
		for m in models:
			calc_results(model=m, eco_condition=c, distribution=DISTRIBUTION, strength=STRENGTH, zi=ZI, num_workers=NUM_WORKERS, 
				NUM_SAMPLES=NUM_SAMPLES, BATCH_SIZE=BATCH_SIZE)

