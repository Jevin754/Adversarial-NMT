from __future__ import print_function
import utils.tensor
import utils.rand

import argparse
import dill
import logging

import sys

import torch
from torch import cuda
from torch.autograd import Variable
from model import NMT
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", default="g2p/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="words",
                    help="Source Language. (default = words)")
parser.add_argument("--trg_lang", default="phoneme",
                    help="Target Language. (default = phoneme)")
parser.add_argument("--model_file", required=True,
                    help="Location to load the models.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  _, _, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, _, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  trg_vocab_size = len(trg_vocab)
  src_vocab_size = len(src_vocab)

  nmt = torch.load(options.model_file, map_location={'cuda:0': 'cpu'})
  nmt.use_cuda = False
  nmt.eval()
  if use_cuda > 0:
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()

  with open('output.txt', 'w') as f_write:
    for batch_i in range(len(src_test)):
      test_src_batch = Variable(src_test[batch_i], volatile = True)
      test_trg_batch = Variable(trg_test[batch_i], volatile = True)
      test_src_batch = test_src_batch.view(-1, 1)
      test_trg_batch = test_trg_batch.view(-1, 1)

      sys_out_batch = nmt(test_src_batch, test_trg_batch)
      s =""
      for i in sys_out_batch:
        ix = np.argmax(i.data.cpu().numpy())
        if ix == 2:
          continue
        if ix == 3:
          break
        s += trg_vocab.itos[ix] + " "

      s += '\n'
      f_write.write(s.encode('utf-8'))


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
