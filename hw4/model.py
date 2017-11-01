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
    self.vocab_size = vocab_size
    self.embed_size = 32
    self.hidden_size = 16
    self.lookup   = nn.Parameter(torch.Tensor(vocab_size, self.embed_size).uniform_())
    self.weight_x = nn.Parameter(torch.Tensor(self.embed_size, self.hidden_size).uniform_(-1.0/math.sqrt(self.embed_size), 1.0/math.sqrt(self.embed_size)))
    self.weight_h = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-1.0/math.sqrt( self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.weight_o = nn.Parameter(torch.Tensor(self.hidden_size, vocab_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt( self.hidden_size)))

    self.bias_x = nn.Parameter(torch.rand(self.hidden_size))
    self.bias_h = nn.Parameter(torch.rand(self.hidden_size))
    self.bias_o = nn.Parameter(torch.rand(vocab_size))

    self.H = nn.Parameter(torch.Tensor(self.hidden_size).uniform_())


  def forward(self, input_batch):
    sequence_length = input_batch.size()[0]
    batch_length    = input_batch.size()[1]

    X = torch.index_select(self.lookup, 0, input_batch.view(-1)).view(sequence_length, batch_length,  self.embed_size)
    H_pre = self.H.expand(batch_length, self.hidden_size)

    outs = []

    for i in range(sequence_length):

      input_x = X[i,:,:]
      H_tmp1  = input_x.mm(self.weight_x) + self.bias_x
      H_tmp2  = H_pre.mm(self.weight_h) + self.bias_h
      H_sum   = H_tmp1 + H_tmp2
      H_cur   = torch.tanh(H_sum)

      H_pre = H_cur

      outlayer_in = H_cur.mm(self.weight_o) + self.bias_o
      
      m = nn.Softmax()

      softwax = m(outlayer_in)

      outs.append(torch.log(softwax))

    outs = torch.cat(outs)
    return outs.view(sequence_length, batch_length, self.vocab_size)



 
# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    # word embedding lookup table
    self.vocab_size = vocab_size
    self.hidden_size = 8
    self.embed_size = 32
    self.lookup   = nn.Parameter(torch.Tensor(vocab_size, self.embed_size).uniform_(-1.0/math.sqrt(self.embed_size), 1.0/math.sqrt(self.embed_size)))

    self.weight_xf = nn.Parameter(torch.Tensor(self.embed_size, self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.weight_hf = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))

    self.weight_xb = nn.Parameter(torch.Tensor(self.embed_size, self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.weight_hb = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.weight_o = nn.Parameter(torch.Tensor(self.hidden_size*2, vocab_size).uniform_(-1.0/math.sqrt(vocab_size), 1.0/math.sqrt(vocab_size)))

    self.Hf = nn.Parameter(torch.Tensor(self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.Hb = nn.Parameter(torch.Tensor(self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))

    #self.Hf = nn.Parameter(torch.ones(8))
    #self.Hb = nn.Parameter(torch.ones(8))


    self.bias_x = nn.Parameter(torch.Tensor(self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.bias_hf = nn.Parameter(torch.Tensor(self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.bias_hb = nn.Parameter(torch.Tensor(self.hidden_size).uniform_(-1.0/math.sqrt(self.hidden_size), 1.0/math.sqrt(self.hidden_size)))
    self.bias_o = nn.Parameter(torch.Tensor(vocab_size).uniform_(-1.0/math.sqrt(vocab_size), 1.0/math.sqrt(vocab_size)))

  def forward(self, input_batch):
    sequence_length = input_batch.size()[0]
    batch_length    = input_batch.size()[1]

    X = torch.index_select(self.lookup, 0, input_batch.view(-1)).view(sequence_length, batch_length, -1)
    Hf_pre = self.Hf.expand(batch_length, self.hidden_size)
    Hb_pre = self.Hb.expand(batch_length, self.hidden_size)

    Hf_table = [] 
    Hb_table = [] 

    # forward
    # save the inital H state
    Hf_table.append(Hf_pre)
    for i in range(sequence_length):

      input_x = X[i,:,:]

      H_tmp1  = input_x.mm(self.weight_xf) + self.bias_x
      H_tmp2  = Hf_pre.mm(self.weight_hf) + self.bias_hf
      H_sum   = H_tmp1 + H_tmp2
      H_cur   = torch.tanh(H_sum)
      Hf_pre = H_cur
      Hf_table.append(H_cur)

    Hf_table = Hf_table[:-1]
    Hf_table = torch.cat(Hf_table)
    Hf_table = Hf_table.view(sequence_length, batch_length, -1)

    # backward
    # save the initial H state
    Hb_table.append(Hb_pre)
    for i in range(sequence_length-1,-1, -1):

      input_x = X[i,:,:]
      
      H_tmp1  = input_x.mm(self.weight_xb) + self.bias_x
      H_tmp2  = Hb_pre.mm(self.weight_hb) + self.bias_hb
      H_sum   = H_tmp1 + H_tmp2
      H_cur   = torch.tanh(H_sum)
      Hb_pre = H_cur
      Hb_table.append(H_cur)

    # reverse backward hidden layer result to fit forward
    Hb_table = Hb_table[:-1]
    Hb_table.reverse()
    Hb_table = torch.cat(Hb_table)
    Hb_table = Hb_table.view(sequence_length, batch_length, -1)

    outs = []

    for i in range(sequence_length):
      H_i = torch.cat((Hf_table[i],Hb_table[i]),1)
      outlayer_in = H_i.mm(self.weight_o) + self.bias_o
      
      m = nn.Softmax()

      softwax = m(outlayer_in)

      outs.append(torch.log(softwax))
    outs = torch.cat(outs)
    return outs.view(sequence_length, batch_length, self.vocab_size)


