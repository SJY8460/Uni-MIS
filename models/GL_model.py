import torch
from torch import nn
from utils.graph import normalize_adj
from modules.graph import GlobalLocalDecoder
from utils.matrix_utils import flat2matrix


class GL_Model(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 num_intent,
                 gpu=None,
                 dropout_rate=0.4,
                 n_head=4,
                 decoder_gat_hidden_dim=16,
                 slot_graph_window = 2,
                 use_normalized=False) -> None:
        super().__init__()
        self.use_normalized = use_normalized
        self.gpu = gpu or torch.cuda.is_available()
        self.__num_intent = num_intent
        self.__slot_graph_window = slot_graph_window
        self.__slot_decoder = GlobalLocalDecoder(
            hidden_dim,
            output_dim,
            dropout_rate,
            n_heads=n_head,
            decoder_gat_hidden_dim=decoder_gat_hidden_dim)


    def generate_global_adj_gat(self, seq_len, index, batch, window):
        global_intent_idx = [[] for i in range(batch)]
        global_slot_idx = [[] for i in range(batch)]
        for item in index:
            global_intent_idx[item[0]].append(item[1])

        for i, len in enumerate(seq_len):
            global_slot_idx[i].extend(
                list(range(self.__num_intent, self.__num_intent + len)))

        adj = torch.cat([
            torch.eye(self.__num_intent + max(seq_len)).unsqueeze(0)
            for i in range(batch)
        ])

        for i in range(batch):
            for j in global_intent_idx[i]:
                adj[i, j, global_slot_idx[i]] = 1.
                adj[i, j, global_intent_idx[i]] = 1.
            for j in global_slot_idx[i]:
                adj[i, j, global_intent_idx[i]] = 1.

        for i in range(batch):
            for j in range(self.__num_intent, self.__num_intent + seq_len[i]):
                adj[i, j,
                    max(self.__num_intent, j -
                        window):min(seq_len[i] + self.__num_intent, j +
                                    window + 1)] = 1.

        if self.use_normalized:
            adj = normalize_adj(adj)
        if self.gpu:
            adj = adj.cuda()
        return adj

    def generate_slot_adj_gat(self, seq_len, batch, window):
        slot_idx_ = [[] for i in range(batch)]
        adj = torch.cat(
            [torch.eye(max(seq_len)).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                adj[i, j,
                    max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
        if self.use_normalized:
            adj = normalize_adj(adj)
        if self.gpu:
            adj = adj.cuda()
        return adj

    def forward(self, inputs):
        slot_lstm_out = inputs['hidden']
        seq_lens = inputs['seq_lens']
        intent_index = inputs['intent_index']
        intent_embedding = inputs['intent_embedding']
        global_adj = self.generate_global_adj_gat(seq_lens, intent_index,
                                                  len(seq_lens),
                                                  self.__slot_graph_window)
        slot_adj = self.generate_slot_adj_gat(seq_lens, len(seq_lens),
                                              self.__slot_graph_window)
        inputs = {
            "hidden": slot_lstm_out,
            "global_adj": global_adj,
            "slot_adj": slot_adj,
            "intent_embedding": intent_embedding,
            "seq_lens": seq_lens
        }
        pred_slot = self.__slot_decoder(inputs)['hidden']
        return {"hidden": pred_slot}
