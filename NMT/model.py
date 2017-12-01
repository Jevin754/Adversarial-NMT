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
        self.encoder = EncoderRNN(src_vocab_size, word_emb_size, hidden_size, use_cuda)
        self.decoder = LuongAttnDecoderRNN(attn_model, trg_vocab_size, word_emb_size,hidden_size, trg_vocab_size, use_cuda)

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
        d_h = e_h
        d_c = e_c

        # Decoding
        decoder_output = self.decoder(trg_batch, (d_h, d_c), encoder_outputs, is_train)

        return decoder_output

# Encoder Module
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, use_cuda, dropout=0.3, num_directions=2, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_vocab_size = input_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size // num_directions
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, n_layers, dropout=dropout, bidirectional=True)
        
    def forward(self, input_seqs_batch):

        batch_size = input_seqs_batch.size(1)
        
        embedded = self.embedding(input_seqs_batch)

        e_outputs, (e_h, e_c) = self.lstm(embedded)

        # e_outputs = e_outputs[:, :, :self.hidden_size] + e_outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        e_h_ = e_h.transpose(0,1).contiguous().view(batch_size, -1)
        e_c_ = e_c.transpose(0,1).contiguous().view(batch_size, -1)
        return e_outputs, (e_h_, e_c_)


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
        align = self.score(hidden, encoder_outputs)
        batch, targetL, sourceL = align.size()
        # Normalize energies to weights in range 0 to 1
        align_vectors = F.softmax(align.view(batch * targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
        return align_vectors
    
    def score(self, hidden, encoder_output):

        src_len, src_batch, src_dim = encoder_output.size()
        tgt_len, tgt_batch, tgt_dim = hidden.size()

        if self.method == 'general':
            h_t_ = hidden.view(tgt_batch*tgt_len, tgt_dim)
            h_t_ = self.attn(h_t_)
            h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = encoder_output.permute(1,2,0)
            #h_s_ = h_s_.permute(0,2,1)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            try:
                return torch.bmm(h_t, h_s_)
            except:
                print "error"

        else:
            raise NotImplementedError

# Luong Attention Decoder Module
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, trg_vocab_size, embed_size, hidden_size, output_size, use_cuda, dropout=0.3, n_layers=1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.use_cuda = use_cuda
        self.attn_model = attn_model
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        # Define layers
        self.embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # instantiate attention class
        if attn_model != None:
        	self.attn = Attn(attn_model, hidden_size)

    def forward(self, trg_batch, (d_h, d_c), encoder_outputs, is_train):

        trg_seq_len, batch_size = trg_batch.size()

        sys_out_batch = Variable(torch.Tensor(trg_seq_len, batch_size, self.trg_vocab_size)).fill_(0) # (trg_seq_len, batch_size, trg_vocab_size)
        # the token <s> is 2 in trg_vocab
        initial_input = Variable(torch.LongTensor(batch_size).fill_(2))
        if self.use_cuda > 0:
            initial_input = initial_input.cuda()
            sys_out_batch = sys_out_batch.cuda()

        initial_emb = self.embedding(initial_input)

        # remove the end of sentence token when doing embedding
        trg_batch = trg_batch[:-1]
        embedded = self.embedding(trg_batch)

        # Initial context
        context = None
        if self.use_cuda:
            init_context = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).cuda()
        else:
            init_context = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)

        for i in range(trg_seq_len):

            if is_train :
                if i == 0:
                    emb_t = initial_emb
                else:
                    emb_t = embedded[i-1,:,:].squeeze(0)
            else:
                if i == 0:
                    emb_t = initial_emb
                else:
                    top_val, top_inx = output.view(batch_size, -1).topk(1)
                    next_token = top_inx.squeeze(1)
                    next_emb = self.embedding(next_token)
                    emb_t = next_emb

            decoder_input = torch.cat([emb_t, context.squeeze(1) if context is not None else init_context], 1)

            # Detach hidden variable
            d_h = d_h.detach()
            d_c = d_c.detach()

            d_h, d_c = self.lstm(decoder_input, (d_h, d_c)) # rnn_output: 1*batch*hidden

            # Calculate attention from current RNN state and all encoder outputs;
            # apply to encoder outputs to get weighted average
            decoder_hidden = d_h.view(1, batch_size, -1)
            attn_weights = self.attn(decoder_hidden, encoder_outputs) # batch * hidden
            encoder_out = encoder_outputs.transpose(0, 1)
            context = torch.bmm(attn_weights, encoder_out)

            # Attentional vector using the RNN hidden state and context vector
            # concatenated together (Luong eq. 5)
            decoder_hidden = decoder_hidden.transpose(0,1)
            concat_c = torch.cat((decoder_hidden, context), 2)
            concat_output = F.tanh(self.concat(concat_c))
            decoder_out = self.dropout(concat_output)

            # Finally predict next token (Luong eq. 6)
            output = nn.functional.log_softmax(self.out(decoder_out))

            sys_out_batch[i] = output.squeeze(1)



        return sys_out_batch

