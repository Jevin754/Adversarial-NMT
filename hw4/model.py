import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    # word embedding lookup table
    self.lookup = torch.randn(vocab_size, 32)
    self.weight_x = nn.Parameter(torch.Tensor(32,16), requires_grad = True)
    self.weight_h = nn.Parameter(torch.Tensor(16,16), requires_grad = True)
    self.weight_o = nn.Parameter(torch.Tensor(16,vocab_size), requires_grad = True)
  def forward(self, input_batch):
    X = self.lookup[input_batch.data,:]
    sequence_length = input_batch.size()[0]
    batch_length = input_batch.size()[1]
    output = Variable(torch.zeros(sequence_length, batch_length, self.lookup.size()[0]), requires_grad = False)
    H = nn.Parameter(torch.randn(batch_length, 16), requires_grad = True)

    for i in range(sequence_length):

      input_x = nn.Parameter(X[i,:,:], requires_grad = False)
      H_tmp1 = input_x.mm(self.weight_x)
      H_tmp2 = H.mm(self.weight_h)
      H_sum = H_tmp1 + H_tmp2
      H_cur = torch.tanh(H_sum)
      H = H_cur
      outlayer_in = H_cur.mm(self.weight_o)
      m = nn.Softmax()
      softwax = m(outlayer_in)
      output[i,:,:] = torch.log(softwax)

    return output



    





    


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
