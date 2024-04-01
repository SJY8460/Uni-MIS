from unicodedata import bidirectional
from torch import nn
from fastNLP.embeddings import LSTMCharEmbedding, StackEmbedding, StaticEmbedding, BertEmbedding, ElmoEmbedding, BertWordPieceEncoder,RobertaEmbedding
from fastNLP.modules import LSTM, MLP
import torch
from modules.rnn import Encoder, Seq2SeqDecoder, LSTMEncoder, LSTMDecoder, Seq2SeqModel
from modules.mlp import MLPAdapter
# from utils.matrix_utils import flat2matrix, matrix2flat
import torch.nn.functional as F
from models.GL_model import GL_Model
from modules.graph import AGIFDecoder
import copy


class MainModel(nn.Module):
    def __init__(self,dataset,args):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.vocab = dataset.vocab['word']
        if self.args.embed_type == 'bert':
            self.embedding_dim = 768 #用bert写死768
            self.embedding = BertEmbedding(self.vocab,model_dir_or_name='en')
        elif self.args.embed_type == 'w2v':
            self.embedding = StaticEmbedding(self.vocab,'en-word2vec-300d')
            self.embedding_dim = 300
        elif self.args.embed_type == 'glove':
            self.embedding = StaticEmbedding(self.vocab,'en-glove-6b-300d')
            self.embedding_dim = 300
        elif self.args.embed_type == 'roberta':
            self.embedding = RobertaEmbedding(self.vocab,'en',include_cls_sep = True)
            self.embedding_dim = 768
        elif self.args.embed_type == 'roberta-large':
            self.embedding = RobertaEmbedding(self.vocab,'en-large')
            self.embedding_dim = 1024
        else:
            self.embedding = StaticEmbedding(self.vocab,model_dir_or_name=None,embedding_dim=self.args.embedding_dim)
            self.embedding_dim = self.args.embedding_dim
        char_embed = LSTMCharEmbedding(self.vocab,
                                        embed_size=64,
                                        char_emb_size=50)
        if self.args.embed_type == 'roberta' or self.args.embed_type == 'bert' or self.args.embed_type == 'roberta-large':
            self.embedding_dim = self.embedding.embedding_dim
        else:
            self.embedding = StackEmbedding([self.embedding, char_embed])
            self.embedding_dim = self.embedding.embedding_dim
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dim_feedforward=2048, dropout=0.1, activation='relu')
        self.encoder_hidden_dim = args.encoder_hidden_dim
        self.attention_hidden_dim =args.attention_hidden_dim
        self.attention_output_dim = args.attention_output_dim
        self.dropout = nn.Dropout(args.drop_out)
        self.dropout_rate = args.drop_out
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.args.tf_layer_num)

        self.bertliner = nn.Sequential (nn.Linear(self.embedding_dim,self.encoder_hidden_dim),
                                        nn.Dropout(0.1))
        self.encoder = Encoder(self.embedding_dim,
                               self.encoder_hidden_dim,
                               self.attention_hidden_dim,
                               self.attention_output_dim,
                               dropout_rate=self.dropout_rate)
        
        self.num_intent = len(dataset.vocab['intent'])
        self.num_slot = len(dataset.vocab['slot'])
        self.window_size = args.window_size
        self.is_hard = args.use_hard_vote
        self.decoder_gat_hidden_dim = args.decoder_gat_hidden_dim
        self.slot_graph_window=args.slot_graph_window
        self.n_head = args.n_head
        self.slot_decoder_hidden_dim = args.slot_decoder_hidden_dim
        self.cls_liner = nn.Sequential(nn.Linear(self.encoder_hidden_dim,self.slot_decoder_hidden_dim),
                                       nn.Dropout(0.1))
                                       


        self.__intent_decoder = Window_Attention_Intent_Decoder(self.args,self.window_size,self.slot_decoder_hidden_dim,self.num_intent)
        self.align_liner = nn.Linear(2*(self.encoder_hidden_dim+self.attention_output_dim),self.encoder_hidden_dim)
        self.align_intent_decoder = Window_Attention_Intent_Decoder(self.args,self.window_size,self.encoder_hidden_dim,self.num_intent)
        self.decoder = self.args.decoder
            
    
        self.__slot_lstm = LSTMEncoder(
        self.encoder_hidden_dim,
        self.slot_decoder_hidden_dim,
        self.dropout_rate
    )

        self.window_intent_slot_lstm = LSTMEncoder(
            2* self.slot_decoder_hidden_dim,
            self.slot_decoder_hidden_dim,
            self.dropout_rate
        )
        self.token_intent_slot_lstm = LSTMEncoder(
            2* self.slot_decoder_hidden_dim,
            self.slot_decoder_hidden_dim,
            self.dropout_rate
    )       
        self.global_intent_slot_lstm = LSTMEncoder(
            2* self.slot_decoder_hidden_dim,
            self.slot_decoder_hidden_dim,
            self.dropout_rate
        )      

        self.decoder_hidden_dim = 2* self.slot_decoder_hidden_dim

    

        if self.decoder == 'mlp':
            self.__slot_decoder = MLPAdapter('qin', self.decoder_hidden_dim,
                                             self.decoder_hidden_dim)
        elif self.decoder == 'seq2seq':
            self.__slot_decoder = Seq2SeqDecoder(self.decoder_hidden_dim,
                                                 self.decoder_hidden_dim,
                                                 self.num_slot,
                                                 dropout_rate=self.dropout_rate,
                                                 embedding_dim=32,
                                                 extra_dim=self.num_intent)
        elif self.decoder == 'lstm':
            self.__slot_decoder = LSTMDecoder(self.decoder_hidden_dim,
                                              self.decoder_hidden_dim,
                                              dropout_rate=self.dropout_rate)
        elif self.decoder == 'gat':
            self.__slot_decoder = GL_Model(
                self.slot_decoder_hidden_dim,
                self.num_slot,
                self.num_intent,
                n_head=self.n_head,
                decoder_gat_hidden_dim=self.decoder_gat_hidden_dim,
                slot_graph_window=self.slot_graph_window,
                use_normalized=True)

        elif self.decoder == 'agif':
            self.__slot_decoder = AGIFDecoder(
                # self.decoder_hidden_dim,
                self.slot_decoder_hidden_dim,
                self.slot_decoder_hidden_dim,
                self.num_slot,
                self.num_intent,
                self.decoder_gat_hidden_dim,
                self.dropout_rate,
                n_heads=self.n_head,
                row_normalized=True,
                embedding_dim=128
            )
        
            self.__intent_decoder_hidden_dim = self.slot_decoder_hidden_dim
            self.intent_encoder = LSTMEncoder(
                self.encoder_hidden_dim ,
                self.__intent_decoder_hidden_dim,
                self.dropout_rate
            )



        self.__slot_predict = MLPAdapter('qin',
                                         self.slot_decoder_hidden_dim,
                                         self.num_slot,
                                         drop_out=self.dropout_rate)

        self.intent_embedding = nn.Parameter(
            torch.FloatTensor(self.num_intent,
                                self.slot_decoder_hidden_dim))  # 191, 32
        nn.init.normal_(self.intent_embedding.data)
    
    def hard_vote(self,pred_intent,window_nums):
        # token_intent的版本
        window_intent = torch.argmax(F.softmax(pred_intent,dim=2),dim=-1)
        window_intent_list = [window_intent[i,:window_nums[i]].cpu().data.numpy().tolist() for i in range(len(window_nums)) ]
        intent_index = []
        start_idx,end_idx = [],[]
        for sen_idx,sen in enumerate(window_intent_list):
            sep_idx = [i for i,x in enumerate(sen) if x == self.dataset.vocab['intent'].word2idx['SEP'] ]
            start_idx = [i + 1 for i in sep_idx]
            start_idx.insert(0,0)
            end_idx = sep_idx[:]
            end_idx.append(len(sen))
            sen_intent = []
            for start,end in zip(start_idx,end_idx):
                partition = sen[start:end]
                if len(partition) == 0:
                    continue
                partition_intent = max(partition,key=partition.count)
                sen_intent.append([sen_idx,partition_intent])
            intent_index.extend(sen_intent)
        return {'intent_index':intent_index,
                'window_intent_list':window_intent_list}

    def soft_vote(self,pred_intent,window_num_tensor,window_num,threshold):
        intent_index_sum = torch.cat([ #[batch_size,intent_num]
            # [intent_num]
            torch.sum( 
                # [seq_lens,intent_num]
                torch.sigmoid(pred_intent[i,0:window_num[i], :]) > threshold,
                dim=0).unsqueeze(0) for i in range(len(window_num))
        ],dim = 0)
        intent_index = (intent_index_sum >
                        (window_num_tensor // 2).unsqueeze(1)).nonzero() #保存为true的位置  
        return intent_index.cpu().data.numpy().tolist()

    def forward(self,inputs,n_predict=None,threshold=0.5):
        
        words = inputs['word_idx']
        seq_lens = inputs['seq_lens']

        # if you do not to add cls token you need to delete embedded_w_cls as it's redundent
        embedded_w_cls = self.embedding(words)
        
        embedded = embedded_w_cls[:,1:-1,:]

        
        if self.args.embed_type == 'roberta' or self.args.embed_type == 'bert' or self.args.embed_type == 'roberta-large':
            hidden = self.bertliner(embedded_w_cls)
            cls_token = hidden[:,0,:]
            cls_token = cls_token.unsqueeze(1).repeat(1,embedded.shape[1],1)
            cls_token = self.cls_liner(cls_token)

        else:
            hidden = self.encoder(embedded,seq_lens)
        

        intent_hidden= self.intent_encoder(hidden,seq_lens)

        #window-level的intent版本
        output = self.__intent_decoder({
            # "hidden": intent_hidden,
            "hidden" : intent_hidden,
            "seq_lens": seq_lens
        })


        pred_intent = output['hidden']#[batch_size,window_num,intent_num+1]
        window_nums = output['window_num']
        align_intent = output['align_intent']

        # align_token_window_intent = self.align_liner(torch.cat((intent_hidden,align_intent),dim=-1)) 
        #window_token align的intent版本
        # align_out = self.align_intent_decoder({
        #     # "hidden": intent_hidden,
        #     "hidden" : align_token_window_intent,
        #     "seq_lens": seq_lens
        # })       
        # pred_intent = align_out['hidden']#[batch_size,window_num,intent_num+1]


        window_num_tensor = window_nums #[batch_size]

        if self.is_hard and not self.args.ablation:
            intent_index,window_intent_list  = self.hard_vote(pred_intent,window_nums)['intent_index'],\
                           self.hard_vote(pred_intent,window_nums)['window_intent_list']
        else:
            intent_index = self.soft_vote(pred_intent,window_num_tensor,window_nums,threshold)
            window_intent_list = None

        slot_lstm_out = self.__slot_lstm(hidden, seq_lens)
        

        window_intent_slot = self.window_intent_slot_lstm(torch.cat((slot_lstm_out, align_intent),dim=-1),seq_lens)
        token_intent_slot = self.token_intent_slot_lstm(torch.cat((slot_lstm_out, intent_hidden),dim=-1),seq_lens)
        if self.args.embed_type == 'roberta' or self.args.embed_type == 'bert' or self.args.embed_type == 'roberta-large':
            global_intent_slot = self.global_intent_slot_lstm(torch.cat((slot_lstm_out, cls_token),dim=-1),seq_lens)
            
            slot_lstm_out =  token_intent_slot  + global_intent_slot + window_intent_slot  + 3 * slot_lstm_out
             
             

        else:
            slot_lstm_out = window_intent_slot + token_intent_slot + slot_lstm_out
        

        force_slot_idx = inputs['slot_idx'] if 'slot_idx' in inputs else None
        slot_inputs = {
            "hidden": slot_lstm_out,
            # "hidden" : hidden,
            "seq_lens": seq_lens,
            "extra_input": pred_intent,
            "intent_index" : intent_index,
            "intent_embedding" : self.intent_embedding,
            "force_input":force_slot_idx
        }
        pred_slot = self.__slot_decoder(slot_inputs)['hidden']
        if not (self.decoder == 'seq2seq' or self.decoder == 'agif'):
            pred_slot = self.__slot_predict({"hidden": pred_slot})['hidden']

        if n_predict is None:
            return {
                "pred_intent": pred_intent, #[batch_size,seq_len,intent_num]/[batch_size,window_num,intent_num]
                "pred_slot": pred_slot,
                'window_num':window_nums,
            }
        else:
            pred_slot = torch.argmax(pred_slot,
                                     dim=-1).cpu().data.numpy().tolist()

            return {
                "pred_intent": intent_index,
                # "pred_intent": intent_index.cpu().data.numpy().tolist(),
                "pred_slot": pred_slot,
                "window_intent_list":window_intent_list
            }

class Window_Attention_Intent_Decoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """
    def __init__(self,
                 args,
                 window_size,
                 input_dim,
                 num_intent,
                 layer_num = 1,
                 drop_out=0.4,
                 alpha=0.2):
        super(Window_Attention_Intent_Decoder, self).__init__()
        self.args = args
        self.window_type = self.args.window_type
        self.window_size = window_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=2048, dropout=drop_out, activation='relu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layer_num)
        # self.embedding = BertEmbedding(self.vocab,model_dir_or_name='en')


        self.__intent_decoder = MLPAdapter(
            'qin', input_dim,
            num_intent)
        padding = 1 if window_size == 3 else 0
        self.conv = nn.Conv2d(in_channels=1,out_channels=input_dim,kernel_size=(window_size,input_dim))
        

    def forward(self, inputs):
        hidden = inputs['hidden']
        seq_lens = inputs['seq_lens']
        if self.window_type == 'tf':
            window_num = seq_lens - self.window_size + 1
            window_num = torch.where(window_num<=0,1,window_num)
            max_window_num = torch.max(window_num)
            
            
            hidden_stack = None
            align_hidden_stack = None
            if self.window_size > 1:
                for start_idx in range(max_window_num):
                    end_idx = start_idx + self.window_size # [start_idx,end_idx]构成一个窗口 左闭右开
                    hidden_window = hidden[:,start_idx:end_idx,:] # chunk
                    if self.args.embed_type == 'roberta' or self.args.embed_type == 'bert' or self.args.embed_type == 'roberta-large':
                        hidden_window = hidden_window
                    else:
                        hidden_window = self.encoder(hidden_window) 

                    # hidden_window_sum = torch.sum(hidden_window,dim=1).unsqueeze(1)
                    hidden_window_sum = hidden_window.max(1)[0].unsqueeze(1)

                    if hidden_stack is None:
                        hidden_stack = hidden_window_sum
                        align_hidden_stack = hidden_window_sum
                    else:
                        hidden_stack = torch.cat([hidden_stack,hidden_window_sum],dim=1)
                        align_hidden_stack = torch.cat([align_hidden_stack,hidden_window_sum],dim=1)
                    #  对齐slot的长度
                    if (start_idx) == 0 or (start_idx == max_window_num - 1):
                            align_hidden_stack = torch.cat([align_hidden_stack,hidden_window_sum],dim=1)

                    

            # [batch, num_of_window_scalar * max_window_num, hidden_size]
            else:
                hidden_stack = hidden

            pred_intent = self.__intent_decoder({'hidden':hidden_stack})['hidden']
        else:
            window_num = seq_lens - self.window_size + 1
            window_num = torch.where(window_num<=0,1,window_num)
            hidden = hidden.unsqueeze(1)
            conved = F.relu(self.conv(hidden)).squeeze(3).permute(0,2,1)


            pred_intent = self.__intent_decoder({'hidden':conved})['hidden']
        
        output = {
            'hidden':pred_intent,
            'window_num':window_num,
            'seq_lens':seq_lens,
            'align_intent': align_hidden_stack
        }

        return output#[batch_size,window_num,intent_num]

