import tensorflow as tf
from copy import deepcopy

class model:
	def __init__(self, params):
		self.params = params
		self.inp = tf.placeholder(tf.int32, shape=[None, None])
		self.weights = tf.placeholder(tf.float32, shape=[None, None])
		self.trgt = tf.placeholder(tf.int32, shape=[None, None])
		self.embeddings = tf.Variable(tf.random_uniform([self.params['vocab_size'], self.params['embedding_size']], -1., 1.), trainable=True)

	def forward(self):
		inputs = tf.nn.embedding_lookup(self.embeddings, self.inp)
		batch_size = tf.shape(inputs)[0]
		cell = tf.nn.rnn_cell.LSTMCell(self.params['embedding_size'])
		colony = tf.nn.rnn_cell.MultiRNNCell(cells=[deepcopy(cell) for _ in range(self.params['num_layers'])])
		outputs, _ = tf.nn.dynamic_rnn(colony, inputs, dtype=tf.float32)
		outputs_reshaped = tf.reshape(outputs, [-1, self.params['embedding_size']])
		W = tf.Variable(tf.random_uniform([self.params['embedding_size'], self.params['vocab_size']], -1., 1.), trainable=True)
		b = tf.Variable(tf.ones(self.params['vocab_size']))
		logits = tf.add(tf.matmul(outputs_reshaped, W), b)
		self.logits = tf.reshape(logits, [batch_size, -1, self.params['vocab_size']])
		self.predictions = tf.nn.softmax(self.logits)

	def backward(self):
		self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.trgt, self.weights)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss)