import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

# Encoder Module
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs_batch):
        
        embedded = self.embedding(input_seqs_batch)

        e_outputs, (e_h, e_c) = self.LSTM(embedded)

        e_outputs = e_outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        return e_outputs, (e_h, e_c)


# Attention Module
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(batch_size, 1, hidden_size))


    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies)
    
    def score(self, hidden, encoder_output):

        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.transpose(2, 1)
            energy = hidden.bmm(energy)
            return energy

        elif self.method == 'concat':
            hidden = hidden * Variable(encoder_output.data.new(encoder_output.size()).fill_(1)) # broadcast hidden to encoder_outputs size
            energy = self.attn(torch.cat((hidden, encoder_output), -1))
            energy = F.tanh(energy.transpose(2, 1))
            energy = self.v.bmm(energy)
            return energy
        else:
            #self.method == 'dot':
            encoder_output = encoder_output.transpose(2, 1)
            energy = hidden.bmm(encoder_output)
            return energy

# Luong Attention Decoder Module
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # instantiate attention class
        if attn_model != None:
        	self.attn = Attn(attn_model, hidden_size)

    def forward(self, attn_model, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, (d_h, d_c) = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
        output = nn.functional.log_softmax(self.out(concat_output))

        # Return final output, hidden state
        return output, (d_h, d_c)

