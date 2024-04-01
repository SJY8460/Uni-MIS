import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    Thanks Qin for sharing code in his github
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import numpy as np
from utils.matrix_utils import flat2matrix, matrix2flat
from utils.graph import normalize_adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]
        a_input = torch.cat(
            [h.repeat(1, 1, N).view(B, N * N, -1),
             h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [
            GraphAttentionLayer(nfeat,
                                nhid,
                                dropout=dropout,
                                alpha=alpha,
                                concat=True) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module(
                        'attention_{}_{}'.format(i + 1, j),
                        GraphAttentionLayer(nhid * nheads,
                                            nhid,
                                            dropout=dropout,
                                            alpha=alpha,
                                            concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(
                        self.__getattr__('attention_{}_{}'.format(i + 1,
                                                                  j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


class GlobalLocalDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 dropout_rate,
                 n_heads=8,
                 decoder_gat_hidden_dim=16,
                 n_layers_decoder_global=1,
                 alpha=0.2):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        """

        super(GlobalLocalDecoder, self).__init__()
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.alpha = alpha
        self.gat_dropout_rate = dropout_rate
        self.decoder_gat_hidden_dim = decoder_gat_hidden_dim
        self.n_heads = n_heads
        self.n_layers_decoder_global = n_layers_decoder_global
        # Network parameter definition.

        self.__slot_graph = GAT(self.__hidden_dim, self.decoder_gat_hidden_dim,
                                self.__hidden_dim, self.gat_dropout_rate,
                                self.alpha, self.n_heads,
                                self.n_layers_decoder_global)

        self.__global_graph = GAT(self.__hidden_dim,
                                  self.decoder_gat_hidden_dim,
                                  self.__hidden_dim, self.gat_dropout_rate,
                                  self.alpha, self.n_heads,
                                  self.n_layers_decoder_global)

        # self.__linear_layer = nn.Sequential(
        #     nn.Linear(self.__hidden_dim, self.__hidden_dim),
        #     nn.LeakyReLU(alpha),
        #     nn.Linear(self.__hidden_dim, self.__output_dim),
        # )

    def forward(self, inputs):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :return: is distribution of prediction labels.
        """
        encoded_hiddens = inputs['hidden']
        seq_lens = inputs['seq_lens']
        global_adj = inputs['global_adj']
        slot_adj = inputs['slot_adj']
        intent_embedding = inputs['intent_embedding']
        output_tensor_list, sent_start_pos = [], 0

        batch = len(seq_lens)
        slot_graph_out = self.__slot_graph(encoded_hiddens, slot_adj)
        intent_in = intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        global_graph_in = torch.cat([intent_in, slot_graph_out], dim=1)
        global_graph_out = self.__global_graph(global_graph_in, global_adj)
        num_intent = intent_embedding.size(0)
        output_tensor_list = global_graph_out[:, num_intent:, :]
        return {"hidden": output_tensor_list}


class AGIFDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 intent_num,
                 decoder_gat_hidden_dim,
                 dropout_rate ,
                 alpha=0.2,
                 n_heads=4,
                 gpu = True,
                 row_normalized = True,
                 n_layers_decoder_global=1,
                 embedding_dim=None,
                 extra_dim=None):
        """ Construction function for Decoder.
        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(AGIFDecoder, self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim
        self.__num_intent = intent_num 
        self.decoder_gat_hidden_dim = decoder_gat_hidden_dim
        self.gat_dropout_rate = dropout_rate
        self.alpha = alpha
        self.n_heads = n_heads
        self.n_layers_decoder_global = n_layers_decoder_global
        self.row_normalized = row_normalized
        self.gpu = gpu

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(torch.randn(
                1, self.__embedding_dim),
                                              requires_grad=True)

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim,
                                    hidden_size=self.__hidden_dim,
                                    batch_first=True,
                                    bidirectional=False,
                                    dropout=self.__dropout_rate,
                                    num_layers=1)

        self.__graph = GAT(self.__hidden_dim, self.decoder_gat_hidden_dim,
                           self.__hidden_dim, self.gat_dropout_rate,
                           self.alpha, self.n_heads, self.n_layers_decoder_global)

        self.__linear_layer = nn.Linear(self.__hidden_dim, self.__output_dim)

    def forward(self, inputs):
        """ Forward process for decoder.
        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """
        encoded_hiddens = inputs.get('hidden', None)
        seq_lens = inputs.get('seq_lens', None)
        forced_input = inputs.get('force_input', None)
        intent_index = inputs.get('intent_index', None)
        seq_lens, sorted_idx = torch.sort(seq_lens, descending=True)
        _, origin_idx = torch.sort(sorted_idx, dim=0)

        encoded_hiddens = encoded_hiddens[sorted_idx]
        if forced_input is not None:
            forced_input = forced_input[sorted_idx]

        batch_size, _, _ = encoded_hiddens.shape
        adj = self.generate_adj_gat(intent_index, batch_size)[sorted_idx]
        intent_embedding = inputs.get('intent_embedding', None)
        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is not None and forced_input is not None:

            forced_tensor = self.__embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat((self.__init_tensor.unsqueeze(0).repeat(
                len(forced_tensor), 1, 1), forced_tensor),
                                    dim=1)
            combined_input = torch.cat([input_tensor, prev_tensor], dim=2)
            dropout_input = self.__dropout_layer(combined_input)
            packed_input = pack_padded_sequence(dropout_input,
                                                seq_lens,
                                                batch_first=True)
            lstm_out, _ = self.__lstm_layer(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            for sent_i in range(0, len(seq_lens)):
                if adj is not None:
                    lstm_out_i = torch.cat(
                        (lstm_out[sent_i][:seq_lens[sent_i]].unsqueeze(1),
                         intent_embedding.unsqueeze(0).repeat(
                             seq_lens[sent_i], 1, 1)),
                        dim=1)
                    lstm_out_i = self.__graph(
                        lstm_out_i, adj[sent_i].unsqueeze(0).repeat(
                            seq_lens[sent_i], 1, 1))[:, 0]
                else:
                    lstm_out_i = lstm_out[sent_i][:seq_lens[sent_i]]
                linear_out = self.__linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.__init_tensor.unsqueeze(0).repeat(
                len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat(
                    (input_tensor[:, word_i].unsqueeze(1), prev_tensor), dim=2)
                dropout_input = self.__dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h,
                               last_c) = self.__lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(
                        dropout_input, (last_h, last_c))

                if adj is not None:
                    lstm_out = torch.cat(
                        (lstm_out, intent_embedding.unsqueeze(0).repeat(
                            len(lstm_out), 1, 1)),
                        dim=1)
                    lstm_out = self.__graph(lstm_out, adj)[:, 0]

                lstm_out = self.__linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.__embedding_layer(
                    index.squeeze(1)).unsqueeze(1)
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [
                output_tensor[:length, i] for i, length in enumerate(seq_lens)
            ]

        flat_result = torch.cat(output_tensor_list, dim=0)
        matrix_result = flat2matrix(flat_result, seq_lens)
        return {"hidden" : matrix_result[origin_idx]}

    def generate_adj_gat(self, index, batch):
        intent_idx_ = [[torch.tensor(0)] for i in range(batch)]
        for item in index:
            intent_idx_[item[0]].append(item[1] + 1)
        intent_idx = intent_idx_
        adj = torch.cat([
            torch.eye(self.__num_intent + 1).unsqueeze(0) for i in range(batch)
        ])
        for i in range(batch):
            for j in intent_idx[i]:
                adj[i, j, intent_idx[i]] = 1.
        if self.row_normalized:
            adj = normalize_adj(adj)
        if self.gpu:
            adj = adj.cuda()
        return adj
