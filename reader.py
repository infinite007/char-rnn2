import _pickle as pkl
import utils
from tqdm import tqdm

class reader:
	def __init__(self):
		pass

	def make_dataset(self, data_dir, save_dir):
		data = utils.read_text(data_dir)
		f = open(save_dir, 'wb')
		print('generating sentences')
		sents = utils.generate_sentences(data)
		print('converting sentences to code points')
		sents_cp = utils.convert_sent_to_id(sents)
		for i in tqdm(sents_cp):
			x = i[:-1]
			y = i[1:]
			pkl.dump((x, y), f)
		f.close()
		print('done')

	def read_data(self, fp):
		yield pkl.load(fp)

	def read_batches(self, fp, n_epoch=1, b_size=32):
		self.curr_epoch = 0
		while self.curr_epoch<n_epoch:
			try:
				return [self.read_data(fp) for _ in range(b_size)]
			except:
				self.curr_epoch += 1