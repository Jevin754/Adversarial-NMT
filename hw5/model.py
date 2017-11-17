
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class NMT(nn.Module):
    def __init__(self, trg_vocab_size):
        super(NMT, self).__init__()

        self.train = True

        self.src_word_emb_size = 300
        self.encoder_hidden_size = 512
        self.decoder_hidden_size = 1024
        self.context_vector_size = 1024
        self.trg_vocab_size = trg_vocab_size

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
        self.generator = nn.Linear(self.decoder_hidden_size, 23262)
        self.generator.weight.data = (model_param["0.weight"])
        self.generator.bias.data = (model_param["0.bias"])

        #attention
        self.weight_i = nn.Linear(1024,1024,bias = False)
        self.weight_i.weight.data = (model_param["decoder.attn.linear_in.weight"])

        self.weight_o = nn.Linear(2048,1024,bias = False)
        self.weight_o.weight.data = (model_param["decoder.attn.linear_out.weight"])

        #self.weight_o = nn.Parameter(model_param["decoder.attn.linear_out.weight"])

        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax()
        # self.logsoftmax = nn.functional.log_softmax()


    def forward(self, train_src_batch, train_trg_batch = None, is_train = True, use_cuda = True):
        trg_seq_length = train_trg_batch.size()[0]
        batch_length   = train_src_batch.size()[1]

        # Embedding
        src_embed = self.embeddings_en(train_src_batch)

        # Encoding
        # encoding_o, (encoding_h,encoding_c) = self.lstm_en(src_embed) # (seq_len, batch, hidden_size * num_directions)
        encoding_o, (_, _) = self.lstm_en(src_embed) # (seq_len, batch, hidden_size * num_directions)

        # if use_cuda:
        #     output = Variable(torch.Tensor(trg_seq_length, batch_length, 23262)).cuda()
        # else:
        #     output = Variable(torch.Tensor(trg_seq_length, batch_length, 23262))

        output = []

        # Initialize start vocab_distribution
        stdv = 1.0 / math.sqrt(23262)
        vocab_distrubition = torch.Tensor(batch_length, 23262).uniform_(-stdv, stdv)

        # Pre-compute attention score matrix left part
        mat_left_mul = encoding_o.matmul(self.weight_i.weight)  # (seq_len, batch_size, 1024)

        # Decoding for training
        if train_trg_batch is not None:
            d_embed = self.embeddings_de(train_trg_batch)

        for i in range(trg_seq_length):
            # Initialize the pesudo decoding hidden state, cell state
            if i == 0:
                stdv = 1.0 / math.sqrt(self.decoder_hidden_size)
                if use_cuda:
                    d_h = Variable(torch.Tensor(batch_length, self.decoder_hidden_size).uniform_(-stdv, stdv)).cuda()
                    d_c = Variable(torch.Tensor(batch_length, self.decoder_hidden_size).uniform_(-stdv, stdv)).cuda()
                else:
                    d_h = Variable(torch.Tensor(batch_length, self.decoder_hidden_size).uniform_(-stdv, stdv))
                    d_c = Variable(torch.Tensor(batch_length, self.decoder_hidden_size).uniform_(-stdv, stdv))

            # if use_cuda:
            #     d_h = Variable(d_h.data).cuda()
            #     d_c = Variable(d_c.data).cuda()
            # else:
            #     d_h = Variable(d_h.data)
            #     d_c = Variable(d_c.data)

            d_h = d_h.detach()
            d_c = d_c.detach()

            # Compute attention
            # scores = []

            # Compute score for each word
            
            # for idx, hidden_state in enumerate(encoding_o):
            #     tmp = torch.sum(mat_left_mul[idx] * decoding_h, 1).view(-1, 1)
            #     scores.append(tmp)

            # scores = torch.cat([torch.sum(mat_left_mul[idx] * d_h, 1).view(-1, 1) for idx, _ in enumerate(encoding_o)], 1) # (batch_size, sequence_len)
            scores = torch.cat([torch.sum(mat_left_mul[idx] * d_h, 1).view(-1, 1) for idx in range(encoding_o.size(0))],1)  # (batch_size, sequence_len)

            association = torch.exp(scores)
            association = (association / torch.sum(association, 1).unsqueeze(1) \
                            .expand(scores.size(0), scores.size(1))).t() # (sequence_len, batch)

            # Compute context vector
            s_t = association.unsqueeze(2) \
                .expand(association.size(0), association.size(1), encoding_o.size(2)) * encoding_o # (sequence_len, batch, hidden_size)--sum->(batch, hidden_size)
            s_t = torch.sum(s_t, 0) #

            c_t = self.tanh(self.weight_o(torch.cat([s_t, d_h], 1)))

            # Decoding
            if is_train:
                decoder_input = torch.cat(((Variable(c_t.data)), d_embed[i,:,:]), 1)
            else:
                (_, argmax) = torch.max(vocab_distrubition,1)
                # word_embed_de = self.embeddings_de(argmax)
                decoder_input = torch.cat((c_t, self.embeddings_de(argmax)),1) # 48 x1324

            d_h, d_c = self.lstm_de(decoder_input, (d_h, d_c))

            # Generator
            # vocab_distrubition = self.logsoftmax((self.generator(d_h)).clamp(min=1e-8))
            # vocab_distrubition = self.logsoftmax((self.generator(d_h)))
            # vocab_distrubition = nn.functional.log_softmax(self.generator(d_h))
            vocab_distrubition = nn.functional.log_softmax(self.generator(d_h)).unsqueeze(0)

            output.append(vocab_distrubition)


        return torch.cat(output, 0)