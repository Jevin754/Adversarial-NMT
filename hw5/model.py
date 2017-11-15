
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class NMT(nn.Module):
  def __init__(self, vocab_size):
    super(NMT, self).__init__()

    self.src_word_emb_size = 300
    self.encoder_hidden_size = 512
    self.decoder_hidden_size = 1024
    self.context_vector_size = 1024
    self.vocab_size = vocab_size

    model_param = torch.load(open("data/model.param", 'rb'))

    self.embeddings_en = nn.Embedding(36616, 300)
    self.embeddings_en.weight.data = (model_param["encoder.embeddings.emb_luts.0.weight"])
    self.embeddings_de = nn.Embedding(23262, 300)
    self.embeddings_de.weight.data = (model_param["decoder.embeddings.emb_luts.0.weight"])

    # encoder
    self.lstm_en = nn.LSTM(self.src_word_emb_size, self.encoder_hidden_size, bidirectional = True)
    # lstm_en forward
    self.lstm_en.weight_ih_l0.data = (model_param["encoder.rnn.weight_ih_l0"])
    self.lstm_en.weight_hh_l0.data = (model_param["encoder.rnn.weight_hh_l0"])
    self.lstm_en.bias_ih_l0.data = (model_param["encoder.rnn.bias_ih_l0"])
    self.lstm_en.bias_hh_l0.data = (model_param["encoder.rnn.bias_hh_l0"])

    # lstm_en backward
    self.lstm_en.weight_ih_l0_reverse.data = (model_param["encoder.rnn.weight_ih_l0_reverse"])
    self.lstm_en.weight_hh_l0_reverse.data = (model_param["encoder.rnn.weight_hh_l0_reverse"])
    self.lstm_en.bias_ih_l0_reverse.data = (model_param["encoder.rnn.bias_ih_l0_reverse"])
    self.lstm_en.bias_hh_l0_reverse.data = (model_param["encoder.rnn.bias_hh_l0_reverse"])

    # decoder
    self.lstm_de = nn.LSTMCell(self.src_word_emb_size + self.context_vector_size, self.decoder_hidden_size)
    # lstm_de forward
    self.lstm_de.weight_ih.data = (model_param["decoder.rnn.layers.0.weight_ih"])
    self.lstm_de.weight_hh.data = (model_param["decoder.rnn.layers.0.weight_hh"])
    self.lstm_de.bias_ih.data = (model_param["decoder.rnn.layers.0.bias_ih"])
    self.lstm_de.bias_hh.data = (model_param["decoder.rnn.layers.0.bias_hh"])

    # generator
    self.generator = nn.Linear(23262, self.decoder_hidden_size)
    self.generator.weight.data = (model_param["0.weight"])
    self.generator.bias.data = (model_param["0.bias"])

    #attention
    self.weight_i = nn.Linear(1024,1024,bias = False)
    self.weight_i.weight.data = (model_param["decoder.attn.linear_in.weight"])

    self.weight_o = nn.Linear(2048,1024,bias = False)
    self.weight_o.weight.data = (model_param["decoder.attn.linear_out.weight"])

    #self.weight_o = nn.Parameter(model_param["decoder.attn.linear_out.weight"])

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()
    self.logsoftmax = nn.LogSoftmax()


  def forward(self, train_src_batch, train_trg_batch = None):

    sequence_length = train_src_batch.size()[0] 
    batch_length    = train_src_batch.size()[1]

    if train_trg_batch is not None:
        trg_sequence_lentgh = train_trg_batch.size()[0]
    else:
        trg_sequence_lentgh = sequence_length


    # encoder
    word_embed_en = self.embeddings_en(train_src_batch)
    output_hs, (h,c) = self.lstm_en(word_embed_en) # sequence_length x batch_length x 1024

    h = h.contiguous().view(batch_length, 1024)
    c = c.contiguous().view(batch_length, 1024)
   
    vocab_distrubition = self.logsoftmax(self.generator(h))


    #output = Variable(torch.Tensor(trg_sequence_lentgh, batch_length, 23262))
    #output[0] = Variable(torch.Tensor(batch_length,23262).fill_(0))

    output = Variable(torch.Tensor(trg_sequence_lentgh, batch_length, 23262)).cuda()
    output[0] = Variable(torch.Tensor(batch_length,23262).fill_(0)).cuda()

    # attention, output is context vector ct 

    for i in range(1,trg_sequence_lentgh):

        # a = Variable(torch.Tensor(sequence_length, batch_length))
        # s_t = Variable(torch.Tensor(batch_length, 1024))

        a = Variable(torch.Tensor(sequence_length, batch_length)).cuda()
        s_t = Variable(torch.Tensor(batch_length, 1024)).cuda()
        #print i

        for j in range(sequence_length):        
            score_pre =self.weight_i(h)
            #print score_pre

            score = output_hs[j].mm(torch.t(score_pre))   # 48 x 48
            #print score 
            score = torch.diag(score)   # 48 x 1
            a[j] = score

        a = self.softmax(a) # seq x batch
        #print a.size()
        a = a.contiguous().view(sequence_length,batch_length,1)
        #print a

        s_t = torch.sum(a * output_hs, dim=0)   # 48 x 1024
        #print s_t

        ct_input = torch.cat((s_t,h),1)  # 48 x 2048
        #print ct_input  

        c_t = self.tanh(self.weight_o((ct_input))) # 1024 x 48
        #print c_t
           
        # decoder
        if train_trg_batch is None:
            (_, argmax) = torch.max(vocab_distrubition,1)
            word_embed_de = self.embeddings_de(argmax)
        else:
            word_embed_de = self.embeddings_de(train_trg_batch[i-1])

       
        decoder_input = torch.cat((c_t, word_embed_de),1) # 48 x1324
        #print decoder_input.size()
        
        h, c = self.lstm_de(decoder_input, (h,c))

        #print h.size()

        # generator
        vocab_distrubition = self.logsoftmax(self.generator(h))
        output[i] = vocab_distrubition

    return output
    









