from nltk import sent_tokenize

def to_codepoint(text):
	return [ord(i) for i in list(text)]

def to_char(codepoints):
	return ''.join([chr(i) for i in codepoints])

def read_text(data_dir):
	with open(data_dir, 'r') as f:
		return f.read()

def vocab_maker(data_dir):
	text = read_text(data_dir)
	vocab = list(text)
	vocab = list(set(vocab))
	vocab_map = {i: ord(i) for i in vocab}
	vocab.append('<EOS>')
	vocab_map['<EOS>'] = max(vocab_map.values())+1
	rev_vocab_map = {v:k for k, v in vocab_map.items()}
	return  vocab, vocab_map, rev_vocab_map

def generate_sentences(text):
	return sent_tokenize(text)

def convert_sent_to_id(sentences, vocab_map):
	return [vocab_map[j] for i in sentences for j in list(i)]

def convert_id_to_sent(ids, rev_vocab_map):
	return [''.join([rev_vocab_map[j] if j!='<EOS>' else '\n' for j in i]) for i in ids]

def process_batches(batch, vocab_map):
	b_list = [next(i) for i in batch]
	print(b_list)
	ins, outs = [i for i, _ in b_list],[i for _, i in b_list]
	max_seq_len = max(len(i) for i in ins)
	weights = [[1]*len(i)+[0]*(max_seq_len-len(i)+1) for i in outs]
	for i in ins:
		i.extend([vocab_map['<EOS>']]*(max_seq_len-len(i)+1))
	for i in outs:
		i.extend([vocab_map['<EOS>']]*(max_seq_len-len(i)+1))
	return ins, outs, weights