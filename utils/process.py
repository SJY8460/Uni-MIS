import logging
import pickle
from fitlog.fastlog import logger
from torch import nn
import torch.optim as optim
import torch
import os
import time
from fastNLP import BucketSampler, SequentialSampler
from fastNLP import DataSetIter
from fastNLP import AccuracyMetric
from tqdm import tqdm
import fitlog
from utils.loss import get_loss, get_slot_label_loss,get_token_intent_loss
from utils.cuda_utils import load_data_to_device
from utils.metris import get_multi_acc, computeF1Score, semantic_acc
from utils.dataset import batch_idx2batch_label, clean_multi_label
# from utils.label_utils import get_label_description, get_label_description_matrix, get_multi_intent_index
import random
import json
# from dataset import word2slot
loger = logging.getLogger()

class Processor:
    def __init__(self, args, dataset, model) -> None:
        self.window_size = args.window_size
        self.dataset = dataset
        self.model = model
        self.args = args
        if args.task_type == 'multi':
            if args.ablation:
                self.criterion_intent = nn.BCEWithLogitsLoss()
            else:
                self.criterion_intent = nn.CrossEntropyLoss()
            # self.criterion_intent = nn.BCEWithLogitsLoss()
        self.criterion_slot = nn.CrossEntropyLoss()
        if self.args.embed_type == 'bert' or self.args.embed_type =='roberta' or self.args.embed_type =='roberta-large':
            embedding_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if 'embedding' in name.lower():
                    embedding_params.append(param)
                else:
                    other_params.append(param)
                # print(name)
            # print(embedding_params)
            # exit()
            params = [
                {"params": embedding_params, "lr": args.bert_lr},
                {"params": other_params, "lr": args.learning_rate},
                ]

        if self.args.optimizer == 'SGD':
            if self.args.embed_type == 'bert' or self.args.embed_type =='roberta':
                self.optimizer = optim.SGD(params)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        else:
             if self.args.embed_type == 'bert' or self.args.embed_type =='roberta' or self.args.embed_type =='roberta-large':
                self.optimizer = optim.Adam(params, weight_decay=self.args.l2_penalty)
             else: 
                self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.learning_rate,weight_decay=self.args.l2_penalty)

        if self.args.gpu:
            self.model = self.model.cuda()

    def train(self):
        best_dev_sent = 0.0
        best_dev_slot = 0.0
        best_dev_acc = 0.0
        batch_size = self.args.batch_size
        train_dataset = self.dataset.dataset['train']
        model = self.model
        if self.args.gpu:
            model = model.cuda()

        sampler = BucketSampler(batch_size=batch_size,
                                seq_len_field_name='seq_lens')
        dataloader = DataSetIter(batch_size=batch_size,
                                 dataset=train_dataset,
                                 sampler=sampler)

        best_test_result = {}
        for epoch in range(0, self.args.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0
            total_loss = 0.0
            time_start = time.time()
            model.train()
            
            for batch_x,batch_y in tqdm(dataloader,ncols=70):
                self.optimizer.zero_grad()
                intent_idx = clean_multi_label(batch_y['intent_idx'],
                                               batch_y['intent_num'])
                seq_lens = batch_x['seq_lens']
                token_intent_idx = batch_y['token_intent_idx']
                if self.args.gpu:
                    load_data_to_device(batch_x, [
                        'word_idx', 'seq_lens','slot_idx'
                    ], 'cuda')

                slot_force = random.random()
                if slot_force > 0.9:
                    batch_x['slot_idx'] = None
                
                output = model(batch_x)
                window_num = output['window_num']
                if self.args.ablation:
                    # intent_loss = get_token_intent_loss(self.criterion_intent,
                    #                     output['pred_intent'],
                    #                     token_intent_idx,
                    #                     len(self.dataset.vocab['token_intent']),
                    #                     #seq_lens,
                    #                     self.window_size,
                    #                     window_num,
                    #                     )
                    intent_loss = get_loss(self.criterion_intent,
                                       output['pred_intent'],
                                       intent_idx,
                                       len(self.dataset.vocab['intent']),
                                       window_num,
                                       self.window_size,
                                       to_token_level=True,
                                       multi=True)
                else:
                    intent_loss = get_loss(self.criterion_intent,
                                        output['pred_intent'],
                                        batch_y['token_intent_idx'],
                                        len(self.dataset.vocab['intent']),
                                        window_num,
                                        self.args.window_size,
                                        is_token_level=True,
                                        multi=False)
                slot_loss = get_loss(self.criterion_slot,
                                     output['pred_slot'],
                                     batch_y['slot_idx'],
                                     len(self.dataset.vocab['slot']),
                                     seq_lens,
                                     1,
                                     is_token_level=True,
                                     multi=False,
                                     print_info=True)
                loss = self.args.intent_weight * intent_loss + self.args.slot_weight * slot_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            time_end = time.time()
            loger.info("Epoch {} finish".format(epoch))
            loger.info("Trainning time :{} s".format(time_end - time_start))
            loger.info("loss : {}".format(total_loss))
            fitlog.add_loss(total_loss, name='loss', step=epoch)

            output = self.validate(self.args, model, self.dataset,
                                   self.dataset.dataset['valid'])
            acc, f1, sem= output['metrics'][
                'intent acc'], output['metrics']['slot f1'], output['metrics'][
                    'sem acc']
            loger.info("dev: {}".format(
                    json.dumps(output['metrics'], indent=4,
                               ensure_ascii=False)))
            acc, f1, sem = output['metrics'][
                'intent acc'], output['metrics']['slot f1'], output['metrics'][
                    'sem acc']
            fitlog.add_metric(
                {
                    'dev': {
                        "intent acc": acc,
                        "slot f1": f1,
                        "sem acc": sem,
                    }
                },
                step=epoch)
            if acc > best_dev_acc or f1 > best_dev_slot or sem > best_dev_sent:
            # if sem > best_dev_sent:
                output = self.validate(self.args, model, self.dataset,
                                       self.dataset.dataset['test'])
                best_test_sem = 0.0
                test_acc, test_f1, test_sem = output[
                    'metrics']['intent acc'], output['metrics'][
                        'slot f1'], output['metrics']['sem acc']
                best_dev_acc = acc
                best_dev_slot = f1
                best_dev_sent = sem
                best_epoch = epoch
                best_test_sem = test_sem
                fitlog.add_metric(
                    {
                        'test': {
                            "intent acc": test_acc,
                            "slot f1": test_f1,
                            "sem acc": test_sem,
                        }
                    },
                    step=epoch)
                fitlog.add_best_metric({
                    'dev': {
                        "intent acc": acc,
                        "slot f1": f1,
                        "sem acc": sem,
                    }
                })
                fitlog.add_best_metric({
                    'test': {
                        "intent acc": test_acc,
                        "slot f1": test_f1,
                        "sem acc": test_sem,
                    }
                })

                if best_test_sem < test_sem:
                    best_test_sem = test_sem
                    fitlog.add_best_metric({
                    'best_dev && test': {
                        "intent acc": test_acc,
                        "slot f1": test_f1,
                        "sem acc": test_sem,
                         "epoch": epoch
                    }
                })
                # wandb.log({ 'best_test_sem': best_test_sem,'intent acc': test_acc,'slot f1': test_f1,'sem acc': test_sem,'epoch': epoch})
                
                    

                loger.info("test: {}".format(
                    json.dumps(output['metrics'], indent=4,
                               ensure_ascii=False)))
                best_test_result = output['metrics']
                model_save_dir = os.path.join(self.args.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    try:
                        os.mkdir(model_save_dir)
                    except:
                        pass
                # torch.save(
                #     self.model,
                #     os.path.join(model_save_dir, "model_{}.pkl".format(epoch)))
                torch.save(
                    self.model,
                    os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.dataset,
                           os.path.join(model_save_dir, 'dataset.pkl'))
                save_case(self,output)
            
    

    def validate(self,args, model, data_manager, dataset):
        batch_size = args.batch_size
        if args.gpu:
            model = model.cuda()
        model = model.eval()
        dataset.set_input('word')
        sampler = SequentialSampler()
        dataloader = DataSetIter(batch_size=batch_size,
                                 dataset=dataset,
                                 sampler=sampler)
        pred_intent = []
        target_intent = []              
        raw_pred_slot = []
        raw_target_slot = []
        raw_pred_intent = []
        raw_target_intent = []
        word_list = []
        raw_window_intent = []

        for batch_x, batch_y in tqdm(dataloader, ncols=70):
            seq_lens = batch_x['seq_lens']
            golds_intent = batch_y['intent_idx'].data.numpy().tolist()
            gold_num = batch_y['intent_num'].data.numpy().tolist()
            golds_intent = clean_multi_label(golds_intent, gold_num)
            slot_idx = batch_y['slot_idx'].data.numpy().tolist()


            if args.gpu:
                field_name = ['word_idx','seq_lens']
                load_data_to_device(batch_x,field_name = field_name,device='cuda')
            
            output = model(batch_x,n_predict = 1)
            batch_word = batch_x['word']
            word_list.extend(batch_word)
            # pred_slot_label =  torch.argmax(output,
            #                          dim=-1).cpu().data.numpy().tolist()
            intent_output = output['pred_intent']
            intent_idx_ = [[] for _ in range(len(golds_intent))]
            for p in intent_output:
                if p[1] == self.dataset.vocab['intent'].word2idx['SEP']:
                    continue
                intent_idx_[p[0]].append(p[1])
            pred_intent.extend(intent_idx_)
            target_intent.extend(golds_intent)
            pred_slot = output['pred_slot']

            window_intent = output['window_intent_list']
            if window_intent is not None:
                for idx, sen_window_intent in enumerate(window_intent):#[batch,seq_lens]
                    raw_window_intent.append([
                        data_manager.vocab['intent'].idx2word[t] for t in sen_window_intent
                    ])
            else:
                raw_window_intent = None


            for idx, slot in enumerate(pred_slot):
                raw_pred_slot.append([
                    data_manager.vocab['slot'].idx2word[s] 
                     for s in slot 
                ][:seq_lens[idx]])
            for idx, slot in enumerate(slot_idx):
                raw_target_slot.append([
                    data_manager.vocab['slot'].idx2word[s] 
                        for s in slot
                ][:seq_lens[idx]])

            intent_idx_ = [[] for _ in range(len(golds_intent))]
            for p in intent_output:
                if p[1] == self.dataset.vocab['intent'].word2idx['SEP']:
                    continue
                intent_idx_[p[0]].append(
                    data_manager.vocab['intent'].idx2word[p[1]])

            for line in intent_idx_:
                raw_pred_intent.append(sorted(line))

            for line in batch_idx2batch_label(data_manager.vocab['intent'],
                                              golds_intent):
                assert len(line) == len(set(line))
                raw_target_intent.append(sorted(line))

        acc = get_multi_acc(raw_pred_intent, raw_target_intent)
        f1, precision, recall = computeF1Score(raw_target_slot, raw_pred_slot)
        sem_acc = semantic_acc(raw_pred_slot, raw_target_slot, raw_pred_intent,
                               raw_target_intent)
        return {
            "metrics": {
                "intent acc": acc,
                "slot f1": f1,
                "sem acc": sem_acc,
            },
            "word": word_list,
            "pred": {
                "intent": raw_pred_intent,
                "slot": raw_pred_slot,
                "window_intent":raw_window_intent
            },
            "target": {
                "intent": raw_target_intent,
                "slot": raw_target_slot
            }
        }

    def prediction(self,args, load_model_dir):
        batch_size = args.batch_size
        if args.gpu:
            print("MODEL {} LOADED".format(str(load_model_dir)))
            model = torch.load(os.path.join(load_model_dir, 'model/model.pkl'))
        else:
            print("MODEL {} LOADED".format(str(load_model_dir)))
            model = torch.load(os.path.join(load_model_dir, 'model/model.pkl'),
                               map_location=torch.device('cpu'))
        datamanager = torch.load(
            os.path.join(load_model_dir, 'model/dataset.pkl'))
        print(model)
        dataset = datamanager.dataset['test']
        dataset.set_input('word')
        output = self.validate(args, model, datamanager, dataset)
        print("final score : {}".format(
            json.dumps(output['metrics'], indent=4, ensure_ascii=False)))
        save_case(self,output)
        return output

def save_case(self,output):
        with open(os.path.join(self.args.save_dir,'all_case.txt'),'w',encoding='utf-8') as fw:
            words = output['word']
            pred_slot = output['pred']['slot']
            pred_intent = output['pred']['intent']
            window_intent = output['pred']['window_intent']
            real_slot = output['target']['slot']
            real_intent =output['target']['intent']
            if window_intent is not None and (self.window_size==1 or self.window_size ==3):
                for p_slot_list,r_slot_list,p_intent_list,r_intent_list,word_list,window_intent_list in zip(pred_slot,real_slot,pred_intent,real_intent,words,window_intent): 
                    if len(window_intent_list) != len(p_slot_list):
                        window_intent_list.insert(0,'HEAD')
                        window_intent_list.append('END')
                    for token,p_slot,r_slot,token_intent in zip(word_list,p_slot_list,r_slot_list,window_intent_list):
                        if p_slot != r_slot:
                            fw.write(token+'\t'+r_slot+'\t'+p_slot+'\t'+token_intent+'\t    <==\n')
                        else:
                            fw.write(token+'\t'+r_slot+'\t'+p_slot+'\t'+token_intent+'\n')
                    if set(r_intent_list) != set(p_intent_list):
                        fw.write('#'.join(r_intent_list)+'  <=>  '+'#'.join(p_intent_list)+'       <===\n') 
                    else:
                        fw.write('#'.join(r_intent_list)+'  <=>  '+'#'.join(p_intent_list)+'\n')
                    fw.write('\n')

        with open(os.path.join(self.args.save_dir,'all_case_pure.txt'),'w',encoding='utf-8') as fw:
            words = output['word']
            pred_slot = output['pred']['slot']
            pred_intent = output['pred']['intent']
            window_intent = output['pred']['window_intent']
            real_slot = output['target']['slot']
            real_intent =output['target']['intent']
            if window_intent is not None and (self.window_size==1 or self.window_size ==3):
                for p_slot_list,r_slot_list,p_intent_list,r_intent_list,word_list,window_intent_list in zip(pred_slot,real_slot,pred_intent,real_intent,words,window_intent):                
                    if len(window_intent_list) != len(p_slot_list):
                        window_intent_list.insert(0,'HEAD')
                        window_intent_list.append('END')
                    for token,p_slot,r_slot,token_intent in zip(word_list,p_slot_list,r_slot_list,window_intent_list):
                        if p_slot != r_slot:
                            fw.write(token+'\t'+r_slot+'\t'+p_slot+'   <==\n')
                        else:
                            fw.write(token+'\t'+r_slot+'\t'+p_slot+'\n')
                    if set(r_intent_list) != set(p_intent_list):
                        fw.write('#'.join(r_intent_list)+'  <=>  '+'#'.join(p_intent_list)+'       <===\n') 
                    else:
                        fw.write('#'.join(r_intent_list)+'  <=>  '+'#'.join(p_intent_list)+'\n') 
                    fw.write('\n')
        return 

