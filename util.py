import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from conorm import tmm
from pydeseq2.preprocessing import deseq2_norm
import time

class FF(nn.Module):
	def __init__(self, dim_input=2, dim_hidden=50, dim_output=1, num_layers=3,
				activation='tanh', dropout_rate=0.3):
		'''
		General feed forward module
		
		dim_input : (int)
			length of each input
		dim_hidden : (int)
			number of hidden nodes in each layer
		dim_output : (int)
			length of each output
		num_layers : (int)
			number of hidden layers
		activation : (str)
			activation function used between hidden layers
		dropout_rate : (float)
			droupout rate
		'''
		super(FF, self).__init__()
		blocks = [lin_block(dim_input, dim_hidden, activation, dropout_rate)] + \
					[lin_block(dim_hidden, dim_hidden, activation, dropout_rate) for i in range(num_layers-2)]
		self.model = nn.Sequential(
			*blocks,
			nn.Linear(dim_hidden, dim_output),
			nn.Softplus()
		)

	def forward(self, X):
		return self.model(X)

def lin_block(dim_in, dim_out, activation, dropout_rate):
	activations = {'tanh': nn.Tanh(), 'leaky_relu': nn.LeakyReLU(0.2), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU()}
	return nn.Sequential(
		nn.Linear(dim_in, dim_out),
		nn.LayerNorm(dim_out),
		activations[activation],
		nn.Dropout(dropout_rate)
		)

class DifferentiableClamp(torch.autograd.Function):
	"""
	In the forward pass this operation behaves like torch.clamp.
	But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
	"""
	@staticmethod
	def forward(ctx, x, min_val, max_val=float("Inf")):
		return x.clamp(min=min_val, max=max_val)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.clone(), None, None

def custom_clamp(x, min_val, max_val=float("Inf")):
	return DifferentiableClamp.apply(x, min_val, max_val)

class MINE(nn.Module):
	def __init__(self, dim_hidden=50, num_layers=3, activation='tanh',
				dropout_rate=0.3, alpha=0.95):
		'''
		Implementation of MINE (Mutual Information Neural Estimation)
		
		dim_hidden : (int)
			number of hidden nodes in each layer
		num_layers : (int)
			number of hidden layers
		activation : (str)
			activation function used between hidden layers
		dropout_rate : (float)
			droupout rate
		alpha : (float)
			constant used in EMA (exponential moving average) calculation 
		'''

		super(MINE, self).__init__()
		self.T_xy = FF(dim_hidden=dim_hidden, num_layers=num_layers, activation=activation, dropout_rate=dropout_rate)
		self.alpha = alpha
		self.ema = torch.tensor(float('inf'))

	def forward(self, X, Y):
		'''
		Input
		X : (float) of size (batch_size x 1)
			tensor of X values
		Y : (float) of size (batch_size x 1)
			tensor of Y values
		----------------------
		Output
		-I : (float) of size (batch_size x 1)
			Negative estimate of mutual information
		'''
		T_joint = self.T_xy(torch.cat((X,Y), -1)).mean()
		exp_T_marginal = self.T_xy(torch.cat((X[torch.randperm(X.shape[0])],Y[torch.randperm(Y.shape[0])]), -1)).exp().mean()
		self.ema = exp_T_marginal.detach() if self.ema == torch.tensor(float('inf')) else \
				   (self.alpha * self.ema + (1 - self.alpha) * exp_T_marginal).detach()
		I_estimate = (T_joint - torch.log(exp_T_marginal)).detach()
		gradient_estimate = (T_joint - (exp_T_marginal/self.ema))
		return -I_estimate + gradient_estimate.detach() - gradient_estimate



class NWJ(nn.Module):
	def __init__(self, dim_hidden=50, num_layers=3, activation='tanh', dropout_rate=0.3):
		'''
		Implementation of the NWJ mutual information method
		
		dim_hidden : (int)
			number of hidden nodes in each layer
		num_layers : (int)
			number of hidden layers
		activation : (str)
			activation function used between hidden layers
		dropout_rate : (float)
			droupout rate
		'''
		super(NWJ, self).__init__()
		self.T_xy = FF(dim_hidden=dim_hidden, num_layers=num_layers, activation=activation, dropout_rate=dropout_rate)

	def forward(self, X, Y):
		'''
		Input
		X : (float) of size (batch_size x 1)
			tensor of X values
		Y : (float) of size (batch_size x 1)
			tensor of Y values
		----------------------
		Output
		-I : (float) of size (batch_size x 1)
			Negative estimate of mutual information
		'''
		T_joint = self.T_xy(torch.cat((X,Y), -1)).mean()
		exp_T_marginal = (self.T_xy(torch.cat((X[torch.randperm(X.shape[0])],Y[torch.randperm(Y.shape[0])]), -1))-1).exp().mean()
		return -T_joint + exp_T_marginal

class DoE(nn.Module):
	def __init__(self, dim_hidden=50, num_layers=3, activation='tanh', dropout_rate=0.3, pdf='gauss'):
		'''
		Implementation of DoE (Difference of Entropies)
		
		dim_hidden : (int)
			number of hidden nodes in each layer
		num_layers : (int)
			number of hidden layers
		activation : (str)
			activation function used between hidden layers
		dropout_rate : (float)
			droupout rate
		pdf : (str)
			prior distribution
		'''
		super(DoE, self).__init__()

		self.qX = PDF(pdf)
		self.qX_Y = ConditionalPDF(dim_hidden=dim_hidden, num_layers=num_layers, activation=activation, 
									dropout_rate=dropout_rate, pdf=pdf)

	def forward(self, X, Y):
		'''
		Input
		X : (float) of size (batch_size x 1)
			tensor of X values
		Y : (float) of size (batch_size x 1)
			tensor of Y values
		----------------------
		Output
		-I : (float) of size (batch_size x 1)
			Negative estimate of mutual information
		'''
		hX = self.qX(X)
		hX_Y = self.qX_Y(X,Y)
		negative_I = hX_Y.detach() - hX.detach()
		return negative_I - hX.detach() - hX_Y.detach() + hX + hX_Y

class PDF(nn.Module):
	def __init__(self, pdf):
		'''
		Helper class for DoE - calculates cross entropy estimate of data
	
		pdf : (str)
			prior distribution to be used in calculations
		'''
		super(PDF, self).__init__()
		assert pdf in {'gauss', 'log_normal', 'exponential', 'negative_binomial', 'gamma', 'beta_negative_binomial'}
		self.pdf = pdf
		if pdf in {'exponential'}:
			self.param1 = nn.Parameter(torch.tensor(0.5).reshape(-1,1))
		elif pdf in {'gauss', 'log_normal', 'gamma'}:
			self.param1 = nn.Parameter(torch.tensor(0.5).reshape(-1,1))
			self.param2 = nn.Parameter(torch.tensor(0.5).reshape(-1,1))
		elif pdf in {'negative_binomial'}:
			self.param1 = nn.Parameter(torch.tensor(5.0).reshape(-1,1))
			self.param2 = nn.Parameter(torch.tensor(0.5).reshape(-1,1))
			
	def forward(self, X):
		'''
		Input
		X : (float) of size (batch_size x 1)
			tensor of X values
		----------------------
		Output
		cross_entropy : (float) of size (batch_size x 1)
			Negative log likelihood of X
		'''

		if self.pdf in {'exponential'}: #param1 = lambda
			cross_entropy = compute_negative_ln_prob(X, self.pdf, custom_clamp(self.param1, min_val=0.0001)).mean()

		elif self.pdf in {'gauss','log_normal'}: #param1 = mu, param2 = sigma
			cross_entropy = compute_negative_ln_prob(X, self.pdf, self.param1, custom_clamp(self.param2, min_val=0.0001)).mean()

		elif self.pdf in {'gamma'}: #param1 = alpha, param2 = beta
			cross_entropy = compute_negative_ln_prob(X, self.pdf, custom_clamp(self.param1, min_val=0.0001), 
				custom_clamp(self.param2, min_val=0.0001)).mean()

		elif self.pdf in {'negative_binomial'}: #param1 = r, param2 = p
			cross_entropy = compute_negative_ln_prob(X, self.pdf, custom_clamp(self.param1, min_val=1), 
				custom_clamp(self.param2, min_val=0.0001, max_val=0.99999)).mean()

		return cross_entropy

class ConditionalPDF(nn.Module):
	def __init__(self, dim_hidden=50, num_layers=3, activation='tanh', dropout_rate=0.3, pdf = 'gauss'):
		'''
		Helper class for DoE - calculates cross entropy estimate of conditional data
		
		dim_hidden : (int)
			number of hidden nodes in each layer
		num_layers : (int)
			number of hidden layers
		activation : (str)
			activation function used between hidden layers
		dropout_rate : (float)
			droupout rate
		pdf : (str)
			prior distribution
		'''
		super(ConditionalPDF, self).__init__()
		assert pdf in {'gauss', 'log_normal', 'exponential', 'gamma', 'negative_binomial','beta_negative_binomial'}
		self.pdf = pdf
		if pdf in {'exponential'}:
			self.Y2X = FF(dim_input=1, dim_hidden=dim_hidden, dim_output=1, num_layers=num_layers, 
					activation=activation, dropout_rate=dropout_rate)
			self.param1 = None

		elif pdf in {'gauss', 'log_normal', 'gamma', 'negative_binomial'}:
			self.Y2X = FF(dim_input=1, dim_hidden=dim_hidden, dim_output=2, num_layers=num_layers, 
					activation=activation, dropout_rate=dropout_rate)
			self.param1 = None
			self.param2 = None

	def forward(self, X, Y):
		'''
		Input
		X : (float) of size (batch_size x 1)
			tensor of X values
		Y : (float) of size (batch_size x 1)
			tensor of Y values
		----------------------
		Output
		cross_entropy : (float) of size (batch_size x 1)
			Negative log likelihood of X|Y
		'''
		if self.pdf in {'exponential'}: #param1 = lambda
			param1 = self.Y2X(Y)
			self.param1 = param1.detach()
			return compute_negative_ln_prob(X, self.pdf, custom_clamp(param1, min_val=0.0001)).mean()

		elif self.pdf in {'gauss', 'log_normal', 'gamma', 'negative_binomial'}:
			param1,param2 = torch.split(self.Y2X(Y), 1, dim=-1)
			self.param1 = param1.detach()
			self.param2 = param2.detach()

		if self.pdf in {'gauss','log_normal'}: #param1 = mu, param2 = sigma
			return compute_negative_ln_prob(X, self.pdf, param1, custom_clamp(param2, min_val=0.0001)).mean()

		elif self.pdf in {'gamma'}: #param1 = alpha, param2 = beta
			return compute_negative_ln_prob(X, self.pdf, custom_clamp(param1, min_val=0.0001), 
				custom_clamp(param2, min_val=0.0001)).mean()

		elif self.pdf in {'negative_binomial'}: #param1 = r, param2 = p
			return compute_negative_ln_prob(X, self.pdf, custom_clamp(param1, min_val=1), 
				custom_clamp(param2, min_val=0.0001, max_val=0.99999)).mean()

def compute_negative_ln_prob(X, pdf, param1, param2=None, param3=None):
	'''
	Helper class for DoE - calculates cross entropy estimate of data

	Input
	X : (float) of size (batch_size x 1)
		tensor of X values
	param1 : (float)
	param2 : (float)
	pdf : (str)
		prior distribution to be used in calculations

	----------------------
	Output
	cross_entropy : (float) of size (batch_size x 1)
		Negative log likelihood of X
	'''

	if pdf in {'exponential'}: #param1 = lambda
		return param1*X - torch.log(param1)

	elif pdf in {'gauss'}: #param1 = mu, param2 = sigma
		return torch.log(param2) + torch.log(torch.sqrt(2*torch.tensor(math.pi))) + \
				0.5 * (((X - param1)/param2) ** 2)

	elif pdf in {'log_normal'}: #param1 = mu, param2 = sigma
		return torch.log1p(X) + torch.log(param2) + torch.log(torch.sqrt(2*torch.tensor(math.pi))) + \
				0.5 * (((torch.log1p(X) - param1)/param2) ** 2)

	elif pdf in {'gamma'}: #param1 = alpha, param2 = beta
		return -param1*torch.log(param2) + torch.lgamma(param1) - (param1-1)*torch.log1p(X) + \
				param2*X

	elif pdf in {'negative_binomial'}: #param1 = n, param2 = p
		return -torch.lgamma(X+param1) + torch.lgamma(X+1) + torch.lgamma(param1) - \
				param1*torch.log(param2) - X*torch.log(1-param2)
	else:
		raise ValueError('Unknown PDF: %s' % (pdf))

def ma(a, window_size=50, return_all=False):
	if return_all:
		return [np.mean(a[i-window_size:i]) for i in range(window_size,len(a)+1)]
	else:
		return np.mean(a[-window_size:])


class Trainer():
	def __init__(self, model, optim, epsilon=0.1, window_size=50):
		'''
		General class for training the models en masse

		model : (torch nn.module object)
			model to be trained
		optim : (torch optim object)
			optimizer to be used in model training
		epsilon : (float)
			early stopping condition (stop when MA of last window_size losses are within epsilon% of current overall loss)
		window_size : (int)
			size of window for early stopping condition
		'''
		self.model=model
		self.optim=optim
		self.training_scores = []
		self.full_scores = []
		self.ma_training_scores = []
		self.ma_full_scores = []
		self.epsilon = epsilon
		self.window_size = window_size
		self.running_training_loss = 0.0

	def _train_epoch(self, data):
		self.optim.zero_grad()
		loss = self.model(data[0].unsqueeze(-1), data[1].unsqueeze(-1))
		loss.backward()
		self.optim.step()
		self.running_training_loss += -(loss.detach())

	def _eval_model(self, data):
		self.model.eval()
		ret = -self.model(data[0].unsqueeze(-1), data[1].unsqueeze(-1)).detach()
		self.model.train()
		return ret

	def train(self, data, data_loader, num_epochs):
		# Complete at least 2*self.window_size number of epochs
		for epoch in range(2*self.window_size):
			self.running_training_loss = 0.0
			for i, d in enumerate(data_loader):
				self._train_epoch(d)
			self.training_scores.append(self.running_training_loss/(i+1))
			self.full_scores.append(self._eval_model(data))
			
		self.ma_training_scores = ma(self.training_scores, self.window_size, return_all=True)
		self.ma_full_scores = ma(self.full_scores, self.window_size, return_all=True)
		for epoch in range(num_epochs - 2*self.window_size):
			self.running_training_loss = 0.0
			for i, d in enumerate(data_loader):
				self._train_epoch(d)
			self.training_scores.append(self.running_training_loss/(i+1))
			self.full_scores.append(self._eval_model(data))
			self.ma_training_scores.append(ma(self.training_scores, self.window_size))
			self.ma_full_scores.append(ma(self.full_scores, self.window_size))
			if abs(max(self.ma_training_scores[-self.window_size:-1]) - self.ma_training_scores[-1]) <= max(self.epsilon*self.ma_training_scores[-1], 0.0015) and \
				abs(min(self.ma_training_scores[-self.window_size:-1]) - self.ma_training_scores[-1]) <= max(self.epsilon*self.ma_training_scores[-1], 0.0015):
				break
	

	def get_losses(self):
		'''
		Returns: lists of training and validation scores
		'''
		return self.training_scores, self.full_scores

	def get_ma_losses(self):
		'''
		Returns: lists of moving averages of training and validation scores
		'''
		return self.ma_training_scores, self.ma_full_scores

def normalize_counts(data, norm_type):
	'''
	Applies a method of normalization to a raw count table
	Input
	data : np array of counts with rows as OTUs/genes and columns as samples 
	----------------------
	Output
	normalized_data : normalized counts table according to input (represented as a tensor)
	'''
	assert norm_type in {'TMM', 'RLE', 'TSS'}
	if norm_type == 'TMM': # rows are genes, columns are samples
		return torch.tensor(tmm(data).round()).float()

	elif norm_type == 'RLE': # rows are samples, columns are genes
		return torch.tensor(deseq2_norm(data.T)[0].T.round()).float()

	elif norm_type == 'TSS': # rows are genes, columns are samples
		return torch.tensor((data/data.sum(axis=0))).float()