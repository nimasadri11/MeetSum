import csv
import os
import subprocess
import stanza
from tensorflow.core.example import example_pb2
import tensorflow as tf
import collections
import struct
stanza.download('en')
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def process(summary, text, nlp):
    s = nlp(summary)
    s_um = []
    for i, sentence in enumerate(s.sentences):
        s_um.append('<s>')
        for token in sentence.tokens:
            s_um.append(token.text)
        s_um.append('</s>')
    t = nlp(text)
    t_ex = []
    for i, sentence in enumerate(t.sentences):
        for token in sentence.tokens:
            t_ex.append(token.text)
    article = ' '.join(t_ex)
    abstract = ' '.join(s_um)
    return abstract, article




def write(file, makevocab=False):
    nlp = stanza.Pipeline(lang='en', processors='tokenize', use_gpu = True)
    if makevocab:
        vocab_counter = collections.Counter()
    with open(file, 'r') as f, open(os.path.join(finish_dir, 'train.bin') , 'wb') as writer:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            # get the text
            summary = (row[0]).lower()
            if file == 'wikihowAll.csv':
                text = (row[2]).lower()
            else:
                text = (row[1]).lower()
            abstract, article = process(summary, text, nlp)
            count += 1
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)
    if makevocab:
        with open(os.path.join(vocab_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
                

def chunk_file():
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % ('train', chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

vocab_dir = "./vocab"
if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)
                
file = 'cl_news_summary_more.csv'
finish_dir = './outdir_news/'
chunks_dir = './outdirChunks_news/'
in_file = './outdir_news/train.bin'

if not os.path.exists(finish_dir):
    os.makedirs(finish_dir)
if not os.path.exists(chunks_dir):
    os.makedirs(chunks_dir)
write(file, True)
chunk_file()


file = 'wikihowAll.csv'
finish_dir = './outdir_wiki/'
chunks_dir = './outdirChunks_wiki/'
if not os.path.exists(finish_dir):
    os.makedirs(finish_dir)
if not os.path.exists(chunks_dir):
    os.makedirs(chunks_dir)
write(file, True)
chunk_file()

file = 'AMItrain.csv'
finish_dir = './outdir_AMItrain/'
if not os.path.exists(finish_dir):
    os.makedirs(finish_dir)
write(file, True)

file = 'AMItest.csv'
finish_dir = './outdir_AMItest/'
if not os.path.exists(finish_dir):
    os.makedirs(finish_dir)
write(file, True)

file = 'AMIval.csv'
finish_dir = './outdir_AMIval/'
if not os.path.exists(finish_dir):
    os.makedirs(finish_dir)
write(file, True)
