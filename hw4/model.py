import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    # word embedding lookup table
    self.lookup   = nn.Parameter(torch.Tensor(vocab_size, 32).uniform_())
    self.weight_x = nn.Parameter(torch.Tensor(32, 16).uniform_(-1.0/math.sqrt(32), 1.0/math.sqrt(32)))
    self.weight_h = nn.Parameter(torch.Tensor(16, 16).uniform_(-1.0/math.sqrt(16), 1.0/math.sqrt(16)))
    self.weight_o = nn.Parameter(torch.Tensor(16, vocab_size).uniform_(-1.0/math.sqrt(16), 1.0/math.sqrt(16)))

    self.bias_x = nn.Parameter(torch.zeros(1))
    self.bias_h = nn.Parameter(torch.zeros(1))
    self.bias_o = nn.Parameter(torch.zeros(1))

    self.H = nn.Parameter(torch.Tensor(1,16).uniform_())


  def forward(self, input_batch):
    #print "lookup table:"
    #print self.lookup[0]
    #print "weight_x:"
    #print self.weight_x[0]
    #print "weight_h:"
    #print self.weight_h[0]
    #print "weight_o:"
    #print self.weight_o[0]
    print "bias_x:"
    print self.bias_x
    print "bias_h:"
    print self.bias_h
    print "bias_o:"
    print self.bias_o
    #print "H:"
    #print self.H

    sequence_length = input_batch.size()[0]
    batch_length    = input_batch.size()[1]

    X      = Variable(self.lookup[input_batch.data,:].data)
    output = Variable(torch.zeros(sequence_length, batch_length, self.lookup.size()[0]))

    H_pre = self.H

    for i in range(sequence_length):

      input_x = X[i,:,:]
      H_tmp1  = input_x.mm(self.weight_x) + self.bias_x
      H_tmp2  = H_pre.mm(self.weight_h) + self.bias_h
      H_sum   = H_tmp1 + H_tmp2
      H_cur   = torch.tanh(H_sum)
      H_pre  = H_cur[0].view(1,16)

      outlayer_in = H_cur.mm(self.weight_o) + self.bias_o
      
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
