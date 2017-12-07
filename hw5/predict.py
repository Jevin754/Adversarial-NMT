import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable

from model import seq2seq
import codecs

import sys
import heapq

model_name = sys.argv[1]
gpu_id = int(sys.argv[2])
cuda.set_device(gpu_id)
use_gpu = True

reload(sys)  
sys.setdefaultencoding('utf8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

src_train, src_dev, src_test, src_vocab = torch.load(open("data/hw5.de", 'rb'))
trg_train, trg_dev, trg_test, trg_vocab = torch.load(open("data/hw5.en", 'rb'))

#nmt = seq2seq()
#print "Loading model..."
nmt = torch.load(open(model_name, 'rb'), map_location=lambda storage, loc: storage, pickle_module=dill)
#print "model loaded..."
if use_gpu:
    nmt.cuda()

batch_size = 1
batched_test, batched_test_mask = utils.tensor.advanced_batchize_no_sort(src_test, batch_size, src_vocab.stoi['<blank>'])
#print "batch made..."

for i, batch_i in enumerate(src_test):
    #print "handling batch: ", i
    batch_i = Variable(batch_i.view(-1,1), volatile=True)
    if use_gpu:
        batch_i = batch_i.cuda()

    decoded = torch.max(nmt(src_vocab, trg_vocab, batch_i), 2)[1]

    for d in decoded.permute(1, 0):
        out = ' '.join([trg_vocab.itos[v] for v in d.data])
        id = out.find('</s>')
        if id != -1:
            print out[:id].replace('<s> ','')
        else:
            print out.replace('<s>','')