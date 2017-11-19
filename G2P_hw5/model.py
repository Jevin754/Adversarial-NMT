import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class NMT(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size, use_cuda = False):
    super(NMT, self).__init__()

    self.src_word_emb_size = 32
    self.encoder_hidden_size = 64
    self.decoder_hidden_size = 128
    self.trg_vocab_size = trg_vocab_size
    self.src_vocab_size = src_vocab_size
    self.use_cuda = use_cuda

    self.embeddings_en = nn.Embedding(src_vocab_size, self.src_word_emb_size)
    self.embeddings_de = nn.Embedding(trg_vocab_size, self.src_word_emb_size)

    # encoder
    self.lstm_en = nn.LSTM(self.src_word_emb_size, self.encoder_hidden_size, bidirectional = True)

    # decoder
    self.lstm_de = nn.LSTMCell(self.src_word_emb_size + self.decoder_hidden_size, self.decoder_hidden_size)

    # generator
    self.generator = nn.Linear(self.decoder_hidden_size, trg_vocab_size)

    #attention
    self.weight_i = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias = False)

    self.weight_o = nn.Linear(2 * self.decoder_hidden_size, self.decoder_hidden_size, bias = False)

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()
    self.logsoftmax = nn.LogSoftmax()

    if use_cuda:
        self.embeddings_en = self.embeddings_en.cuda()
        self.embeddings_de = self.embeddings_de.cuda()
        self.lstm_en = self.lstm_en.cuda()
        self.lstm_de = self.lstm_de.cuda()
        self.generator = self.generator.cuda()
        self.weight_i = self.weight_i.cuda()
        self.weight_o = self.weight_o.cuda()


  def forward(self, train_src_batch, train_trg_batch):

    # encoder
    word_embed_en = self.embeddings_en(train_src_batch)
    output_hs, (h,c) = self.lstm_en(word_embed_en)

    sequence_length = output_hs.size()[0]
    batch_length = output_hs.size()[1]
    trg_sequence_lentgh = train_trg_batch.size()[0]

    h = h.permute(1,2,0).contiguous().view(batch_length, 2 * self.encoder_hidden_size)
    c = c.permute(1,2,0).contiguous().view(batch_length, 2 * self.encoder_hidden_size)

    vocab_distrubition = Variable(torch.LongTensor(batch_length).fill_(1))

    output = Variable(torch.FloatTensor(trg_sequence_lentgh, batch_length, self.trg_vocab_size))
    
    if self.use_cuda:
        output = output.cuda()
        vocab_distrubition = vocab_distrubition.cuda()

    output[0] = Variable(torch.Tensor(batch_length,self.trg_vocab_size).fill_(0))

    for i in range(1,trg_sequence_lentgh):
        
        # attention layer
        # compute score
        ht_wi = self.weight_i(h).view(1, batch_length, 2*self.encoder_hidden_size).expand_as(output_hs)
        score = torch.sum(output_hs * ht_wi, dim=2)

        # compute a
        a = self.softmax(torch.t(score))
        a = torch.t(a).contiguous().view(sequence_length, batch_length, 1)

        # compute st and ct
        s_t = torch.sum(a * output_hs, dim=0)
        c_t = self.tanh(self.weight_o(torch.cat([s_t, h], dim=1)))

        # decoder and generator layer
        if self.training:
            de_input = torch.cat([c_t, self.embeddings_de(train_trg_batch[i-1])], dim=1)
            h, c = self.lstm_de(de_input,(h,c))
            vocab_distrubition = self.logsoftmax(self.generator(h))
            output[i] = vocab_distrubition
        else:
            de_input = torch.cat([c_t, self.embeddings_de(vocab_distrubition)], dim=1)
            h, c = self.lstm_de(de_input,(h,c))
            output[i] = self.logsoftmax(self.generator(h))
            _, vocab_distrubition = torch.max(output[i], dim=1)

    return output


    