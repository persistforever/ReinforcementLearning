# -*- encoding = utf8 -*-
import numpy
import theano
import theano.tensor as tensor


class QNetwork:

	def __init__(self, rng, n_state, n_action):
		# input
		self.states = tensor.matrix(name='states', dtype=theano.config.floatX)
		
		# hyper params
		self.input_dim = n_state
		self.first_hidden_dim = 10
		self.second_hidden_dim = 10
		self.output_dim = n_action
		
		# network params and structure
		## first hidden layer
		### params
		self.first_W = theano.shared(numpy.array(rng.uniform(\
				low=-numpy.sqrt(6.0 / (self.input_dim + self.first_hidden_dim)), \
				high=numpy.sqrt(6.0 / (self.input_dim + self.first_hidden_dim)), \
				size=(self.input_dim, self.first_hidden_dim)), dtype=theano.config.floatX), \
			name='first_W', borrow=True)
		self.first_b = theano.shared(value=numpy.zeros(\
				shape=(self.first_hidden_dim,), dtype=theano.config.floatX), \
			name='first_b', borrow=True)
		### data flow
		self.first_hidden_vector = \
			tensor.tanh(tensor.dot(self.states, self.first_W) + self.first_b)
		
		## second hidden layer
		### params
		self.second_W = theano.shared(numpy.array(rng.uniform(\
				low=-numpy.sqrt(6.0 / (self.first_hidden_dim + self.second_hidden_dim)), \
				high=numpy.sqrt(6.0 / (self.first_hidden_dim + self.second_hidden_dim)), \
				size=(self.first_hidden_dim, self.second_hidden_dim)), dtype=theano.config.floatX), \
			name='second_W', borrow=True)
		self.second_b = theano.shared(value=numpy.zeros(\
				shape=(self.second_hidden_dim,), dtype=theano.config.floatX), \
			name='second_b', borrow=True)
		### data flow
		self.second_hidden_vector = \
			tensor.tanh(tensor.dot(self.first_hidden_vector, self.second_W) + self.second_b)
			
		## output layer
		### params
		self.output_W = theano.shared(numpy.array(rng.uniform(\
				low=-numpy.sqrt(6.0 / (self.second_hidden_dim + self.output_dim)), \
				high=numpy.sqrt(6.0 / (self.second_hidden_dim + self.output_dim)), \
				size=(self.second_hidden_dim, self.output_dim)), dtype=theano.config.floatX), \
			name='output_W', borrow=True)
		self.output_b = theano.shared(value=numpy.zeros(\
				shape=(self.output_dim,), dtype=theano.config.floatX), \
			name='output_b', borrow=True)
		### data flow
		self.output_vector = \
			tensor.tanh(tensor.dot(self.second_hidden_vector, self.output_W) + self.output_b)
			
		# network params and structure
		self.params = {'first_W': self.first_W, 'first_b': self.first_b, \
					'second_W': self.second_W, 'second_b': self.second_b, \
					'output_W': self.output_W, 'output_b': self.output_b}
			
	def get_q_func(self):
		q_func = theano.function(inputs=[self.states], \
								outputs=[self.output_vector], \
								name='q_func')
		
		def get_output_vector(states):
			output_vector = q_func(states)[0]
			return output_vector
		
		return get_output_vector
	
	def train_one_batch(self):
		self.actions = tensor.vector(name='actions', dtype='int64')
		self.y = tensor.vector(name='y', dtype=theano.config.floatX)
		cost = self.output_vector[self.actions].sum() / self.actions.shape[0]
		coef = (self.y - self.output_vector[self.actions]).sum() / self.actions.shape[0]
		grads = tensor.grad(cost, wrt=self.params.values())
		grads = [coef*t for t in grads]
		
		lr = tensor.scalar(name='lr')
		f_update = self._adadelta(lr, self.params, grads)
		
		def update_function(states, actions, y, yita):
			f_update(numpy.array(yita, dtype=theano.config.floatX))
			return
		
		return update_function
	
	def _adadelta(self, lr, tparams, grads, givens=None):
		zipped_grads = [theano.shared(p.get_value() * \
									numpy.array(0.0, dtype=theano.config.floatX), \
									name='%s_grad' % k) for k, p in tparams.iteritems()]
		running_up2 = [theano.shared(p.get_value() * \
									numpy.array(0.0, dtype=theano.config.floatX), \
									name='%s_rup2' % k) for k, p in tparams.iteritems()]
		running_grads2 = [theano.shared(p.get_value() * \
									numpy.array(0.0, dtype=theano.config.floatX), \
									name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
		
		updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg \
				for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
		ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
		param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
		
		f_update = theano.function([lr], [], updates=ru2up + param_up, \
								on_unused_input='ignore', name='adadelta_f_update', givens=givens)
		
		return f_update