import tensorflow as tf
from model import model
import ujson as json
import utils
from reader import reader

with open('./params.json', 'rb') as f:
	params = json.load(f)

vocab, vocab_map, reverse_map = utils.vocab_maker(params['raw_data_dir'])
params['vocab_size'] = max(vocab_map.values())+1

reader = reader()
model = model(params)
model.forward()
model.backward()

fp = open('./data/dataset.pkl', 'rb')
init_op = [tf.local_variables_initializer(), tf.global_variables_initializer()]
sess = tf.Session()
sess.run(init_op)
for epoch in range(params['n_epochs']):
	batch = reader.read_batches(fp)
	inps, trgts, weights = utils.process_batches(batch, vocab_map)
	feed_dict = {
		model.inp : inps,
		model.trgt : trgts,
		model.weights : weights
	}
	logits, loss, _ = sess.run([model.logits, model.loss, model.train_op], feed_dict)
	print(loss)