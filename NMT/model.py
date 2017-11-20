import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class NMT(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, word_emb_size, 
            hidden_size, src_vocab, trg_vocab, attn_model = 'general', use_cuda = False):
        super(NMT, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        self.attn_model = attn_model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.use_cuda = use_cuda

        # Initialize models
        self.encoder = EncoderRNN(src_vocab_size, word_emb_size, hidden_size) 
        self.decoder = LuongAttnDecoderRNN(attn_model, trg_vocab_size, word_emb_size,hidden_size, trg_vocab_size)

        if use_cuda > 0:
            self.encoder.cuda()
            self.decoder.cuda()
        else:
            self.encoder.cpu()
            self.decoder.cpu()

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
          weight.data.uniform_(-0.1, 0.1)

    def forward(self, src_batch, trg_batch, is_train):
        # Encoding
        encoder_outputs, (e_h,e_c) = self.encoder(src_batch)

        # Preparing for decoding
        trg_seq_len = trg_batch.size(0)
        batch_size = trg_batch.size(1)
        sys_out_batch = Variable(torch.zeros(trg_seq_len, batch_size, self.trg_vocab_size)) # (trg_seq_len, batch_size, trg_vocab_size)
        decoder_input = Variable(torch.LongTensor([self.trg_vocab.stoi['<s>']] * batch_size))
        d_h = e_h[:self.decoder.n_layers]
        d_c = e_c[:self.decoder.n_layers]

        if self.use_cuda > 0:
            decoder_input = decoder_input.cuda()
            sys_out_batch = sys_out_batch.cuda()

        # Decoding
        for d_idx in range(trg_seq_len):
            decoder_output, (d_h, d_c) = self.decoder(decoder_input, (d_h, d_c), encoder_outputs)
            sys_out_batch[d_idx] = decoder_output
            if is_train:
                decoder_input = trg_batch[d_idx]
            else:
                top_val, top_inx = decoder_output.topk(1)
                decoder_input = top_inx

        return sys_out_batch


# Encoder Module
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_vocab_size = input_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs_batch):
        
        embedded = self.embedding(input_seqs_batch)

        e_outputs, (e_h, e_c) = self.lstm(embedded)

        e_outputs = e_outputs[:, :, :self.hidden_size] + e_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        return e_outputs, (e_h, e_c)


# Attention Module
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        else:
            raise NotImplementedError


    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies)
    
    def score(self, hidden, encoder_output):

        if self.method == 'general':
            energy = self.attn(encoder_output)
            hidden = hidden.expand_as(energy)
            energy = torch.sum(energy * hidden, dim = 2)
            return energy

        else:
            raise NotImplementedError

# Luong Attention Decoder Module
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_vocab_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.input_vocab_size = input_vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # instantiate attention class
        if attn_model != None:
        	self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embed_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, (d_h, d_c) = self.lstm(embedded, last_hidden) # rnn_output: 1*batch*hidden

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) # batch * hidden
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim = 0) # batch * hidden

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # 1*batch*hidden -> batch*hidden
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
        output = nn.functional.log_softmax(self.out(concat_output))

        # Return final output, hidden state
        return output, (d_h, d_c)

