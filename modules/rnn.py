from modules.mlp import MLPAdapter
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from modules.attention import SelfAttention
import torch
from utils.matrix_utils import flat2matrix, matrix2flat
"""
    Thanks Qin for sharing code in his github
"""


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=self.__embedding_dim,
                                    hidden_size=self.__hidden_dim,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=self.__dropout_rate,
                                    num_layers=1)

    def forward(self, embedded_text, seq_lens: torch.Tensor):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        sorted_length, sorted_idx = torch.sort(seq_lens, descending=True)
        embedded_text = embedded_text[sorted_idx]
        _, origin_idx = torch.sort(sorted_idx, dim=0)
        dropout_text = self.__dropout_layer(embedded_text)
        # Pack and Pad process for input of variable length.
        sorted_length = sorted_length.cpu()
        packed_text = pack_padded_sequence(dropout_text,
                                           sorted_length.cpu(),
                                           batch_first=True)

        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        padded_hiddens = padded_hiddens[origin_idx]
        return padded_hiddens


class Encoder(nn.Module):
    def __init__(self,
                 word_embedding_dim,
                 encoder_hidden_dim,
                 attention_hidden_dim,
                 attention_output_dim,
                 dropout_rate=0.4):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.dropout_rate = dropout_rate
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_output_dim = attention_output_dim
        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(self.word_embedding_dim,
                                     self.encoder_hidden_dim,
                                     self.dropout_rate)

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(self.word_embedding_dim,
                                         self.attention_hidden_dim,
                                         self.attention_output_dim,
                                         self.dropout_rate)

    def forward(self, word_tensor, seq_lens):
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens


class LSTMDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 dropout_rate=0.4,):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim,
                                   hidden_dim,
                                   dropout_rate=dropout_rate)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        input, seq_lens = inputs['hidden'], inputs['seq_lens']
        extra_input = inputs.get('extra_input')
        if extra_input is None:
            input = torch.cat((input, extra_input), dim = -1)
        hidden = self.encoder(input, seq_lens)
        return {"hidden": hidden}


class Seq2SeqDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout_rate,
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

        super(Seq2SeqDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

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
        extra_input = inputs.get('extra_input', None)
        # Flat matrix
        encoded_hiddens = matrix2flat(encoded_hiddens, seq_lens)
        if extra_input is not None:
            extra_input = matrix2flat(extra_input, seq_lens)
        if forced_input is not None:
            batch_size, seq_len = forced_input.shape
            forced_input = forced_input.reshape(batch_size * seq_len)
        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos:sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[
                            sent_start_pos:sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(
                            seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat(
                            [self.__init_tensor, seg_forced_tensor[:-1, :]],
                            dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor],
                                               dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(
                    dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(
                    lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(
                        1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h,
                                   last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(
                            dropout_input, (last_h, last_c))

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos
        flat_result = torch.cat(output_tensor_list, dim=0)
        matrix_result = flat2matrix(flat_result, seq_lens)
        return {"hidden": matrix_result}


class Seq2SeqModel(nn.Module):
    def __init__(self,
                 embed,
                 encoder_hidden_dim,
                 decoder_type,
                 decoder_hidden_dim,
                 output_dim,
                 mlp_type='qin',
                 dropout_rate=0.4):
        super().__init__()
        self.embed = embed
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.encoder = LSTMEncoder(self.encoder_hidden_dim,
                                   self.decoder_hidden_dim,
                                   dropout_rate=self.dropout_rate)

        if decoder_type == 'mlp':
            self.decoder = MLPAdapter(mlp_type, self.decoder_hidden_dim,
                                      output_dim)
        elif decoder_type == 'seq2seq':
            self.decoder = Seq2SeqDecoder(self.decoder_hidden_dim,
                                          self.decoder_hidden_dim,
                                          output_dim,
                                          dropout_rate=dropout_rate,
                                          embedding_dim=32,
                                          extra_dim=len(
                                              self.dataset.intent_vocab))
        elif decoder_type == 'lstm':
            self.decoder = LSTMDecoder(self.decoder_hidden_dim,
                                       self.decoder_hidden_dim,
                                       self.output_dim,
                                       mlp_type=mlp_type,
                                       dropout_rate=self.dropout_rate)

    def forward(self, inputs):
        if self.embed is not None:
            word_idx = inputs['word_idx']
            hidden = self.embed(word_idx)
        else:
            hidden = inputs['hidden']
        seq_lens = inputs['seq_lens']
        # print(hidden.shape)
        hidden = self.encoder(hidden, seq_lens)
        decoder_inputs = {
            "hidden": hidden,
            "seq_lens": seq_lens,
        }
        hidden = self.decoder(decoder_inputs)['hidden']
        return {'hidden': hidden}