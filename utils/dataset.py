from fastNLP import Vocabulary, Instance
from fastNLP import DataSet
import os
from typing import List, Tuple
import logging

from fastNLP.core import dataset, instance

logger = logging.getLogger()
from utils.get_chunk import get_chunks
# import re
# word2slot = {}

def bio2bmes(slots):
    chunks = get_chunks(['O'] + slots + ['O'])
    bmes_slots = ['O'] * len(slots)
    for chunk in chunks:
        s, e, label = chunk
        s = s - 1
        e = e - 1
        if s == e:
            bmes_slots[s] = 'S-' + label
        else:
            bmes_slots[s] = 'B-' + label
            for idx in range(s, e + 1):
                bmes_slots[idx] = 'M-' + label
            bmes_slots[s] = 'B-' + label
            bmes_slots[e] = "E-" + chunk[2]
    return bmes_slots

class DataManager:
    def __init__(self, args, fields: List[Tuple]):

        self.fields = fields
        self.vocab = self.create_vocab(fields)
        self.dataset = {}
        self.__args = args
        self.vis_map = {
            "train":{
                "input":["word_idx","seq_lens",'slot_idx'],
                "target":["intent_idx",'slot_idx',"intent_num","token_intent_idx"]
            },
            "valid":{
                "input":["word_idx","seq_lens"],
                "target":["intent","intent_idx","slot_idx","intent_num"]
            },
            "test":{
                "input":["word_idx","seq_lens"],
                "target":["intent","intent_idx","slot_idx","intent_num"]
            }
        }


    def create_vocab(self, fields: List[Tuple]):
        field_map = {}
        for item in fields:
            name, is_label = item
            print(name)
            if is_label:
                field_map[name] = Vocabulary(padding=None, unknown=None)
            else:
                field_map[name] = Vocabulary()
        return field_map

    def show_summary(self):
        print("Training parameters are listed as follows:\n")
        print("the number of vocab : {}".format(len(self.word_vocab)))

    def quick_build(self):
        train_path = os.path.join(self.__args.data_dir, 'train.txt')
        dev_path = os.path.join(self.__args.data_dir, 'dev.txt')
        test_path = os.path.join(self.__args.data_dir, 'test.txt')

        train_dataset = self.create_dataset(train_path, is_train_file=True)
        valid_dataset = self.create_dataset(dev_path)
        test_dataset = self.create_dataset(test_path)
        self.vocab_init(train_dataset, valid_dataset, test_dataset)
        self.dataset = {
            'train': train_dataset,
            'valid': valid_dataset,
            "test": test_dataset
        }
        self.to_digit([train_dataset, valid_dataset, test_dataset])
        self.set_input_and_target()
        self.save_datamanager()

    def vocab_init(self, train_dataset, dev_dataset, test_dataset):
        for field_item in self.fields:
            field_name, is_label = field_item
            vocab = self.vocab[field_name]
            if field_name == 'token_intent':
                vocab = self.vocab['intent']
                vocab.add('SEP')
                self.vocab['token_intent'] = vocab
            else:
                if is_label:
                    vocab.from_dataset(train_dataset,
                                    dev_dataset,
                                    test_dataset,
                                    field_name=field_name)
                else:
                    vocab.from_dataset(
                        train_dataset,
                        field_name=field_name,
                        no_create_entry_dataset=[dev_dataset, test_dataset])

    @staticmethod
    def __read_file(file_path,is_train = False):
        """ Read data file of given path.

        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents, token_intents = [], [], [], []
        text, slot, token_intent = [], [], []

        with open(file_path, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    if is_train:
                        token_intents.append(token_intent)
                    if "/" not in items[0]:
                        intents.append(items)
                    else:
                        new = items[0].split("/")
                        intents.append([new[1]])

                    # clear buffer lists.
                    text, slot,token_intent = [], [], []

                elif len(items) >=2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

# ---------------------------------------------------------rule
                
                    # temp_slot = []
                    # for i in slot:
                    #     if '.' in i:
                    #         prefix, subfix= i.split('.')
                    #         temp = i[:2]+prefix[2:]
                    #         temp_slot.append(temp)
                    #     else:
                    #         temp_slot.append(i)
                    # slot = temp_slot
# ------------------------------------------------------

                    if is_train:
                        token_intent.append(items[2].strip())
        if is_train:
            return texts, slots, intents, token_intents
        else:
            return texts, slots, intents

    def create_dataset(self, file_path, is_train_file=False):
        if is_train_file:
            text, slot, intent, token_intent = self.__read_file(file_path,True)
        else:
            text, slot, intent = self.__read_file(file_path)
        dataset = DataSet()
        if is_train_file:
            for w, i, s, t in zip(text, intent, slot, token_intent):
                    intance = Instance(
                            word=w,
                            intent=i[0].split('#')
                            if self.__args.task_type == 'multi' else i[0],
                            slot=s,
                            token_intent = t,
                            intent_num=len(i[0].split('#')),
                            seq_lens=len(w),)
                    dataset.append(intance)

                        
        else:
            for w, i, s in zip(text, intent, slot):
                
# ---------------------------------rule format----------------------------------
#                 a = []
#                 for i in s:
#                     if '#' in i and (i.startswith('B') or i.startswith('I') or i.startswith('E')):
#                         print(s)
#                         exit()
#                     else:
#                         a.append(i)
#                 s = a
# # ----------------------------------------
                # if len(i[0].split('#')) == 3:
                    intance = Instance(
                        word=w,
                        intent=i[0].split('#')
                        if self.__args.task_type == 'multi' else i[0],
                        slot=s,
                        intent_num=len(i[0].split('#')),
                        seq_lens=len(w),)
                    dataset.append(intance)
        print(dataset)
        return dataset

    def to_word_piece(self,
                      bert_encoder,
                      datasets,
                      field_name,
                      new_field_name='word_pieces',
                      add_cls_sep=True):
        bert_encoder.index_datasets(*datasets,
                                    field_name=field_name,
                                    add_cls_sep=add_cls_sep)
        for dataset in datasets:
            dataset.rename_field(field_name, new_field_name)

    def to_digit(self, datasets):
        for dataset in datasets:
            for field_item in self.fields:
                field_name, _ = field_item
                if not dataset.has_field(field_name):
                    continue
                vocab = self.vocab[field_name]
                vocab.index_dataset(dataset,
                                    field_name=field_name,
                                    new_field_name=field_name + '_idx')

    def _use_vocab_to_digit_dataset(self, vocab, field_name, dataset):
        vocab.index_dataset(dataset,
                            field_name=field_name,
                            new_field_name=field_name + '_idx')

    def set_input_and_target(self):
        for dataset_type in self.vis_map:
            dataset = self.dataset[dataset_type]
            for field_name in self.vis_map[dataset_type]['input']:
                dataset.set_input(field_name)
            for field_name in self.vis_map[dataset_type]['target']:
                dataset.set_target(field_name)

    def save_datamanager(self):
        if not os.path.exists(os.path.join(self.__args.save_dir, 'alphabet')):
            os.mkdir(os.path.join(self.__args.save_dir, 'alphabet'))
        if not os.path.exists(os.path.join(self.__args.save_dir, 'dataset')):
            os.mkdir(os.path.join(self.__args.save_dir, 'dataset'))

        for vocab_name in self.vocab:
            vocab = self.vocab[vocab_name]
            vocab.save(
                os.path.join(self.__args.save_dir, 'alphabet',
                             vocab_name + '.txt'))
        for dataset_name in self.dataset:
            dataset = self.dataset[dataset_name]
            dataset.save(
                os.path.join(self.__args.save_dir, 'dataset', dataset_name))

    def show_summary(self):
        for dataset_name in self.dataset:
            dataset = self.dataset[dataset_name]
            logger.info(dataset.print_field_meta())
            logger.info(dataset)



def batch_idx2batch_label(vocab, batch):
    result = []
    for list_item in batch:
        if type(list_item) == type([]):
            r = [vocab.idx2word[idx] for idx in list_item]
        else:
            r = vocab.idx2word[list_item]
        result.append(r)
    return result

def clean_multi_label(labels, labels_num):
    result = []
    for label, num in zip(labels, labels_num):
        result.append(label[:num])
    return result