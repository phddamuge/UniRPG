from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
from tag_op.data.operation_util import GRAMMER_CLASS, AUX_NUM, GRAMMER_ID, SCALE_CLASS
from tqdm import tqdm

import re
import string
np.set_printoptions(threshold=np.inf)
import pandas as pd
from typing import List, Dict, Tuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tag_op.data.file_utils import is_scatter_available
from tatqa_utils import  *
from tag_op.data.data_util import *
from tag_op.data.tatqa_dataset import table_tokenize, paragraph_tokenize, question_tokenizer, _concat

from tag_op.data.logical_former import LogicalFormer, LogicalFormerWS
import pickle
import spacy
from multiprocessing import Pool
import math

class BartTatQATestPipe(Pipe):
    def __init__(self, tokenizer='plm/bart-base'):
        super(BartTatQATestPipe, self).__init__()
        import pdb
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.question_length_limit = 46
        self.passage_length_limit = 463
        self.max_pieces = 512
        
        all_grammars = [key for key in GRAMMER_CLASS.keys()] + [key for key in AUX_NUM.keys() ]
        all_grammers_vocab = ["<<"+grammar+">>" for grammar in all_grammars]
        self.mapping = dict(zip(all_grammars, all_grammers_vocab))
        # so that the label word can be initialized in a better embedding.
        
        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)
        
    def process(self, path, mode):
         # 是由于第一位是sos，紧接着是eos, 然后是
        skip_count = 0
        instances = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        print("instance have been loaded from pickle file in desk")
        print("start to build the data_bundle")
        import pdb
#         pdb.set_trace()
        
        for ins in tqdm(data):
            instance = self.prepare_target(ins, mode)
            instances.append(Instance(**instance))
            assert 0<len(instance["src_tokens"])<=512
       
        print('test number', len(instances)) 
        data_bundle = DataSet(instances)
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.set_input( 'src_tokens', 'src_seq_len', 'question_id', 'table_mask', 'paragraph_mask', 'table_cell_number_value', 'paragraph_number_value', 'table_cell_index', 'paragraph_index', 'row_include_cells', 'col_include_cells', 'question_mask', 'attention_mask',  'all_cell_index', 'all_cell_index_mask', 'all_cell_token_index', 'att_mask_vm_lower', 'att_mask_vm_upper')
        
        data_bundle.table_cell_number_value.dtype=np.float     
        data_bundle.paragraph_number_value.dtype=np.float
        return data_bundle

    
    def prepare_target(self, ins, mode):
        input_ids_list = ins["input_ids_list"]
        logical_forms = ins["logical_forms"]
        question_id = ins["question_id"]
        
        table_mask = ins['table_mask'][0].tolist()
        paragraph_mask = ins['paragraph_mask'][0].tolist()
        table_cell_number_value = ins['table_cell_number_value'].tolist()
        paragraph_number_value = ins['paragraph_number_value'].tolist()
        table_cell_index = ins['table_cell_index'][0].tolist()
        paragraph_index = ins['paragraph_index'][0].tolist()
        
        row_include_cells = [value for value in ins['row_include_cells'].values()]
        col_include_cells = [value for value in ins['col_include_cells'].values()]
        
        question_mask = ins['question_mask'][0].tolist()
        attention_mask = ins['attention_mask'][0].tolist()
        all_cell_index = [item[0] for item in ins['all_cell_index']]

        all_cell_token_index = [list(item) for item in ins['all_cell_token_index']]
        att_mask_vm_lower, att_mask_vm_upper = self.build_vm(input_ids_list, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells)
        
        example = {
                'src_tokens':input_ids_list,
                'question_id':question_id,
                'table_mask':table_mask,
                'paragraph_mask':paragraph_mask,
                'table_cell_number_value':table_cell_number_value,
                'paragraph_number_value':paragraph_number_value,
                'table_cell_index':table_cell_index,
                'paragraph_index':paragraph_index,
                'row_include_cells':row_include_cells,
                'col_include_cells':col_include_cells,
                'question_mask':question_mask,
                'attention_mask':attention_mask,
                'all_cell_index':all_cell_index,
                'all_cell_index_mask':[1]*len(all_cell_index),
            
                'att_mask_vm_lower':att_mask_vm_lower,
                'att_mask_vm_upper':att_mask_vm_upper,
                'all_cell_token_index':all_cell_token_index
        }
        return example
    
    
    
    def build_vm(self, input_ids_list, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells):
        import pdb
        
        seq_len = len(input_ids_list)
        att_mask_vm_lower = np.zeros([512, 512])
        att_mask_vm_upper = np.zeros([512, 512])
        table_start = int(sum(question_mask)) + 2
        paragraph_start = table_start + int(sum(table_mask))
        att_mask_vm_lower[0:table_start, 0:seq_len]=1
        att_mask_vm_upper[0:table_start, 0:seq_len]=1
        att_mask_vm_lower[paragraph_start:seq_len, 0:seq_len]=1
        att_mask_vm_upper[paragraph_start:seq_len, 0:seq_len]=1
        att_mask_vm_lower[table_start:paragraph_start, 0:table_start]=1
        att_mask_vm_lower[table_start:paragraph_start, paragraph_start:seq_len]=1
        att_mask_vm_upper[table_start:paragraph_start, 0:table_start]=1
        att_mask_vm_upper[table_start:paragraph_start, paragraph_start:seq_len]=1

        ## lower mask
        
        for row_cells in row_include_cells:
            tokens_idx_offset = [all_cell_token_index[cell-1] for cell in row_cells]
            tokens_idx = []
            for item in tokens_idx_offset:
                tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

            for i in tokens_idx:
                for j in tokens_idx:
                    att_mask_vm_lower[i][j] = 1
                    att_mask_vm_upper[i][j] = 1
        
        
       
        ## upper mask     
        for col_cells in col_include_cells:
            tokens_idx_offset = [all_cell_token_index[cell-1] for cell in col_cells]
            tokens_idx = []
            for item in tokens_idx_offset:
                tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

            for i in tokens_idx:
                for j in tokens_idx:
                    att_mask_vm_upper[i][j] = 1
                    
        att_mask_vm_lower= att_mask_vm_lower[:seq_len, :seq_len]
        att_mask_vm_upper = att_mask_vm_upper[:seq_len, :seq_len]
        return att_mask_vm_lower, att_mask_vm_upper
    
    def _to_instance(self, ins, mode):
        assert mode=='test'
        import pdb
        
        question = ins["question"]
        question_id = ins["question_id"]
        table = ins["table"]
        paragraphs = ins["paragraphs"]
        answer = ins["answer"]
        answer_from = ins["answer_from"]
        answer_type = ins["answer_type"]
        scale = ins["scale"]

        question_text = question.strip()
        question_split_tokens, question_ids = question_tokenizer(question_text, self.tokenizer)
        scale_class = SCALE.index(scale)
        
        table_cell_tokens, table_split_cell_tokens, table_ids, table_cell_number_value,table_cell_index,\
    table_row_header_index, table_col_header_index, table_data_cell_index, \
    same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel,  table_cell_number_value_with_time, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index\
    =table_tokenize(table, self.tokenizer)
        
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)
        
        paragraph_tokens, paragraph_split_tokens, paragraph_split_tokens_offsets, paragraph_text, paragraph_ids, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index= \
            paragraph_tokenize(question, paragraphs, self.tokenizer)
        
        example = {
                    "question_ids":question_ids, 
                    "question_split_tokens":question_split_tokens,
            
                    "table_ids":table_ids, 
                    "table_split_cell_tokens":table_split_cell_tokens,
                    "table_cell_index":table_cell_index, 
                    "table_cell_number_value":table_cell_number_value,
                    "table_row_header_index":table_row_header_index, 
                    "table_col_header_index":table_col_header_index,
                    "table_data_cell_index":table_data_cell_index,
                    "same_row_data_cell_rel":same_row_data_cell_rel,
                    "same_col_data_cell_rel":same_col_data_cell_rel, 
                    "data_cell_row_rel":data_cell_row_rel, 
                    "data_cell_col_rel":data_cell_col_rel,
                    "table_cell_number_value_with_time":table_cell_number_value_with_time,
                    "row_include_cells":row_include_cells, 
                    "col_include_cells":col_include_cells, 
                    "all_cell_index":all_cell_index, 
                    "all_cell_token_index":all_cell_token_index,

                    "paragraph_ids":paragraph_ids, 
                    "paragraph_split_tokens":paragraph_split_tokens,
                    "paragraph_index":paragraph_index, 
                    "paragraph_number_value":paragraph_number_value, 
                    "paragraph_split_tokens_offsets":paragraph_split_tokens_offsets,
                    "paragraph_text":paragraph_text,
            
                    "sep_start":self.bos_token_id, 
                    "sep_end":self.eos_token_id, 
                    "question_length_limitation":self.question_length_limit, 
                    "passage_length_limitation":self.passage_length_limit, 
                    "max_pieces":self.max_pieces
        }
        

        input_ids, input_ids_list, input_tokens, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_split_tokens_offsets, paragraph_text, \
        table_mask, table_number_value, table_index, token_type_ids,table_row_header_index, \
        table_col_header_index,table_data_cell_index,same_row_data_cell_rel,\
                    same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index = \
            _concat(**example)
#         pdb.set_trace()
        
        # get_logical_form
        table_start = int(torch.sum(question_mask[0])) + 2
        instance = {
            "question":question,
            "tokenizer":self.tokenizer,
            "table":table,
            "input_tokens":input_tokens,
            "table_start":table_start,
            "table_mask":table_mask,
            "paragraph_mask":paragraph_mask,
            "table_number_value":table_number_value,
            "all_cell_index":all_cell_index, 
            "all_cell_token_index":all_cell_token_index,
            "answer_from":answer_from,
            "answer_type":answer_type, 
            "answer":answer,
            "scale":scale,
        }
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from}
        
        return {
            "input_ids": np.array(input_ids),
            "input_ids_list":input_ids_list,
            "question_mask":np.array(question_mask),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_index),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            ## add table structure
            "table_row_header_index":table_row_header_index,  
            "table_col_header_index":table_col_header_index, 
            "table_data_cell_index":table_data_cell_index, 
            "same_row_data_cell_rel":same_row_data_cell_rel, 
            "same_col_data_cell_rel":same_col_data_cell_rel, 
            "data_cell_row_rel":data_cell_row_rel, 
            "data_cell_col_rel":data_cell_col_rel,
            "table_cell_number_value_with_time":np.array(table_cell_number_value_with_time),
            "row_include_cells":row_include_cells,
            "col_include_cells":col_include_cells,
            "all_cell_index":all_cell_index,
            "logical_forms":None,
            "question":question_text,
            "all_cell_token_index":all_cell_token_index
        } 
    

    def process_from_file(self, paths, output_path, mode, demo=False, processor_num=1) -> DataBundle:
        # 读取数据
        data = TaTQALoader(demo=demo)._load(paths, mode)
        instances = []
        import pdb
#         pdb.set_trace()
        for ins in tqdm(data):
            instance = self._to_instance(ins, mode)
            instances.append(instance)
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)
            print("data processed success and saved in disk")
        f.close()




class BartTatQATrainPipe(Pipe):
    def __init__(self, tokenizer='plm/bart-base'):
        super(BartTatQATrainPipe, self).__init__()
        import pdb
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.question_length_limit = 46
        self.passage_length_limit = 463
        self.max_pieces = 512
        
#         all_grammars = [key for key in GRAMMER_CLASS.keys()] + [key for key in AUX_NUM.keys() ] + [key for key in SCALE_CLASS.keys()]
        
        all_grammars = [key for key in GRAMMER_CLASS.keys()] + [key for key in AUX_NUM.keys() ]
        
        all_grammers_vocab = ["<<"+grammar+">>" for grammar in all_grammars]
        self.mapping = dict(zip(all_grammars, all_grammers_vocab))
        # so that the label word can be initialized in a better embedding.
        
        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens
        import pdb
        
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)
#         pdb.set_trace()
        self.logical_former = LogicalFormerWS(GRAMMER_CLASS, GRAMMER_ID)
        
    def process(self, path, mode):
         # 是由于第一位是sos，紧接着是eos, 然后是
        skip_count = 0
        all_instances = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        print("instance have been loaded from pickle file in desk")
        print("start to build the data_bundle")
        import pdb
        #pdb.set_trace()
        total_num = 0
        if mode == 'train':
            for ins in tqdm(data):
                instances, skip_flag = self.prepare_target(ins, 'train')
                if mode=="train"  and (instances==None or skip_flag==1):
                    skip_count+=1
                    continue

                for instance in instances:
                    all_instances.append(Instance(**instance))
                    total_num+=1
                    if  mode=="train":
                        assert 0<len(instance["tgt_tokens"])<=512 
                        assert 0<len(instance["src_tokens"])<=512 
                    else:
                        assert 0<len(instance["src_tokens"])<=512

                    if not len(instance["table_cell_index"])==512:
                        import pdb
                        pdb.set_trace()
        if mode =='dev':
            for ins in tqdm(data):
                instances, skip_flag = self.prepare_target(ins, 'dev')
                for instance in instances[:1]:
                    all_instances.append(Instance(**instance))
                    total_num+=1
                    if  mode=="train":
                        assert 0<len(instance["tgt_tokens"])<=512 
                        assert 0<len(instance["src_tokens"])<=512 
                    else:
                        assert 0<len(instance["src_tokens"])<=512

                    if not len(instance["table_cell_index"])==512:
                        import pdb
                        pdb.set_trace()
            
            
        print('skip_num', skip_count) 
        data_bundle = DataSet(all_instances)
        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
#         data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'question_id', 'table_mask', 'paragraph_mask', 'table_cell_number_value', 'paragraph_number_value', 'table_cell_index', 'paragraph_index', 'row_include_cells', 'col_include_cells', 'question_mask', 'attention_mask',  'all_cell_index', 'all_cell_index_mask', 'all_cell_token_index', 'att_mask_vm_lower', 'att_mask_vm_upper' )
#         data_bundle.set_target('tgt_tokens', 'tgt_seq_len', "target_span", 'scale_label')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', "target_span", 'scale_label', 'question_id', 'table_cell_number_value', 'paragraph_number_value', 'table_cell_index', 'paragraph_index', 'table_cell_tokens', 'paragraph_tokens', 'answer', 'answer_from', 'answer_type', 'weight')
        
        
        data_bundle.weight.dtype=float
        data_bundle.table_cell_number_value.dtype=float     
        data_bundle.paragraph_number_value.dtype=float
        
        for i in range(len(data_bundle)):
            assert len(data_bundle[i]['table_cell_index'])==512
#         pdb.set_trace()
        print('total instance:', total_num)
        return data_bundle

    def build_vm(self, input_ids_list, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells):
        import pdb
        
        seq_len = len(input_ids_list)
        att_mask_vm_lower = np.zeros([512, 512])
        att_mask_vm_upper = np.zeros([512, 512])
        table_start = int(sum(question_mask)) + 2
        paragraph_start = table_start + int(sum(table_mask))
        att_mask_vm_lower[0:table_start, 0:seq_len]=1
        att_mask_vm_upper[0:table_start, 0:seq_len]=1
        att_mask_vm_lower[paragraph_start:seq_len, 0:seq_len]=1
        att_mask_vm_upper[paragraph_start:seq_len, 0:seq_len]=1
        att_mask_vm_lower[table_start:paragraph_start, 0:table_start]=1
        att_mask_vm_lower[table_start:paragraph_start, paragraph_start:seq_len]=1
        att_mask_vm_upper[table_start:paragraph_start, 0:table_start]=1
        att_mask_vm_upper[table_start:paragraph_start, paragraph_start:seq_len]=1

        ## lower mask
        
        for row_cells in row_include_cells:
            tokens_idx_offset = [all_cell_token_index[cell-1] for cell in row_cells]
            tokens_idx = []
            for item in tokens_idx_offset:
                tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

            for i in tokens_idx:
                for j in tokens_idx:
                    att_mask_vm_lower[i][j] = 1
                    att_mask_vm_upper[i][j] = 1
                    
                ## upper mask     
        for col_cells in col_include_cells:
            tokens_idx_offset = [all_cell_token_index[cell-1] for cell in col_cells]
            tokens_idx = []
            for item in tokens_idx_offset:
                tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

            for i in tokens_idx:
                for j in tokens_idx:
                    att_mask_vm_upper[i][j] = 1
                    
        att_mask_vm_lower= att_mask_vm_lower[:seq_len, :seq_len]
        att_mask_vm_upper = att_mask_vm_upper[:seq_len, :seq_len]
        return att_mask_vm_lower, att_mask_vm_upper
                    
    def prepare_target(self, ins, mode):
        input_ids_list = ins["input_ids_list"]
        logical_forms = ins["logical_forms"]
        question_id = ins["question_id"]
        
        table_mask = ins['table_mask'][0].tolist()
        paragraph_mask = ins['paragraph_mask'][0].tolist()
        table_cell_number_value = ins['table_cell_number_value'].tolist()
        paragraph_number_value = ins['paragraph_number_value'].tolist()
        table_cell_index = ins['table_cell_index'][0].tolist()
        paragraph_index = ins['paragraph_index'][0].tolist()
        table_cell_tokens = ins['table_cell_tokens']
        paragraph_tokens = ins['paragraph_tokens']
        
        row_include_cells = [value for value in ins['row_include_cells'].values()]
        col_include_cells = [value for value in ins['col_include_cells'].values()]
        
        all_cell_token_index = [list(item) for item in ins['all_cell_token_index']]
        
        question_mask = ins['question_mask'][0].tolist()
        attention_mask = ins['attention_mask'][0].tolist()
        all_cell_index = [item[0] for item in ins['all_cell_index']]
        
        scale = ins['answer_dict']['scale']
        answer_type = ins['answer_dict']['answer_type']
        answer_from = ins['answer_dict']['answer_from']
        answer = ins['answer_dict']['answer']
            
        scale_label = [SCALE.index(scale)]
        
        att_mask_vm_lower, att_mask_vm_upper = self.build_vm(input_ids_list, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells)
       
        examples = []
        example = {
                'src_tokens':input_ids_list,
                'question_id':question_id,
                'table_mask':table_mask,
                'paragraph_mask':paragraph_mask,
                'table_cell_number_value':table_cell_number_value,
                'paragraph_number_value':paragraph_number_value,
                'table_cell_index':table_cell_index,
                'paragraph_index':paragraph_index,
                'table_cell_tokens':table_cell_tokens,
                'paragraph_tokens':paragraph_tokens,
            
                'row_include_cells':row_include_cells,
                'col_include_cells':col_include_cells,
                
                'question_mask':question_mask,
                'attention_mask':attention_mask,
                'all_cell_index':all_cell_index,
                'all_cell_index_mask':[1]*len(all_cell_index),
            
                'scale_label':scale_label,
                'answer_type':answer_type,
                'answer_from':answer_from,
                'answer':answer,
            
                'att_mask_vm_lower':att_mask_vm_lower,
                'att_mask_vm_upper':att_mask_vm_upper,
                "all_cell_token_index":all_cell_token_index,
        }
        
        target_shift = len(self.mapping) + 2
        if mode=='dev':
            if logical_forms==None or len(logical_forms) == 0:
                example_tmp = example.copy()
                example_tmp.update({
                    'tgt_tokens':[], 
                    'target_span':[],
                    'weight':1
                })
                examples.append(example_tmp)
                return examples, 1
            else:
                try:
                    import pdb

                    for logical_form in logical_forms[:1]:
                        target = [0]
                        example_tmp = example.copy()
                        for item in logical_form:
                            if item in self.mapping.keys():
                                target.append(self.mapping2targetid[item]+2)
                            else:
                                target.append(int(item)+target_shift)
                        target.append(1)
                        example_tmp.update({
                                        'tgt_tokens':target, 
                                        'target_span':[tuple(target[1:-1])],
                                        'weight':1
                                    })

    #                     pdb.set_trace()
                        check_target = [item - target_shift for item in target]
                        try:
                            assert all([item<len(input_ids_list) for item in check_target])
                        except:
                            import pdb
                            #pdb.set_trace()
                            continue

                        examples.append(example_tmp)
                    return examples, 0
                except:
                    import pdb
                    pdb.set_trace()
                    
        if mode == 'train':
            if logical_forms==None or len(logical_forms) == 0:
                example_tmp = example.copy()
                example_tmp.update({
                    'tgt_tokens':[], 
                    'target_span':[],
                    'weight':1
                })
                examples.append(example_tmp)
                return examples, 1
            else:
                try:
#                     import pdb
#                     pdb.set_trace()
                    for logical_form in logical_forms[:5]:
                        target = [0]
                        example_tmp = example.copy()
                        for item in logical_form:
                            if item in self.mapping.keys():
                                target.append(self.mapping2targetid[item]+2)
                            else:
                                target.append(int(item)+target_shift)
                        target.append(1)
                        example_tmp.update({
                                        'tgt_tokens':target, 
                                        'target_span':[tuple(target[1:-1])],
                                       #'weight': round(1/len(logical_forms[:5]), 4)
                                         'weight':1
                                    })

    #                     pdb.set_trace()
                        check_target = [item - target_shift for item in target]
                        try:
                            assert all([item<len(input_ids_list) for item in check_target])
                        except:
                            import pdb
                            #pdb.set_trace()
                            continue

                        examples.append(example_tmp)
                    return examples, 0
                except:
                    import pdb
                    pdb.set_trace()
          
    def _to_instance(self, ins, mode):
        import pdb
        #pdb.set_trace()
        question = ins["question"]
        question_id = ins["question_id"]
        table = ins["table"]
        paragraphs = ins["paragraphs"]
        answer = ins["answer"]
        scale = ins["scale"]
        answer_type = ins['answer_type']
        answer_from = ins['answer_from']
        aux_info = ins["aux_info"]

        question_text = question.strip()
        question_split_tokens, question_ids = question_tokenizer(question_text, self.tokenizer)
        scale_class = SCALE.index(scale)
        
        ## NER table text
        
#         table_text, cell_offset = self.build_table_text(table)
        
        table_cell_tokens, table_split_cell_tokens, table_ids, table_cell_number_value,table_cell_index,\
    table_row_header_index, table_col_header_index, table_data_cell_index, \
    same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel,  table_cell_number_value_with_time, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index\
    =table_tokenize(table, self.tokenizer)
        
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)
        
#         pdb.set_trace()
        
        paragraph_tokens, paragraph_split_tokens, paragraph_split_tokens_offsets, paragraph_text, paragraph_ids,paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index= \
            paragraph_tokenize(question, paragraphs, self.tokenizer)
        
#         pdb.set_trace()
        example = {
                    "question_ids":question_ids, 
                    "question_split_tokens":question_split_tokens,
            
                    "table_ids":table_ids, 
                    "table_split_cell_tokens":table_split_cell_tokens,
                    "table_cell_index":table_cell_index, 
                    "table_cell_number_value":table_cell_number_value,
                    "table_row_header_index":table_row_header_index, 
                    "table_col_header_index":table_col_header_index,
                    "table_data_cell_index":table_data_cell_index,
                    "same_row_data_cell_rel":same_row_data_cell_rel,
                    "same_col_data_cell_rel":same_col_data_cell_rel, 
                    "data_cell_row_rel":data_cell_row_rel, 
                    "data_cell_col_rel":data_cell_col_rel,
                    "table_cell_number_value_with_time":table_cell_number_value_with_time,
                    "row_include_cells":row_include_cells, 
                    "col_include_cells":col_include_cells, 
                    "all_cell_index":all_cell_index, 
                    "all_cell_token_index":all_cell_token_index,

                    "paragraph_ids":paragraph_ids, 
                    "paragraph_split_tokens":paragraph_split_tokens,
                    "paragraph_index":paragraph_index, 
                    "paragraph_number_value":paragraph_number_value, 
                    "paragraph_split_tokens_offsets":paragraph_split_tokens_offsets,
                    "paragraph_text":paragraph_text,
            
                    "sep_start":self.bos_token_id, 
                    "sep_end":self.eos_token_id, 
                    "question_length_limitation":self.question_length_limit, 
                    "passage_length_limitation":self.passage_length_limit, 
                    "max_pieces":self.max_pieces
        }
        
        input_ids, input_ids_list, input_tokens, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_split_tokens_offsets, paragraph_text, \
        table_mask, table_number_value, table_index, token_type_ids,table_row_header_index, \
        table_col_header_index,table_data_cell_index,same_row_data_cell_rel,\
                    same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index = \
            _concat(**example)
#         pdb.set_trace()
        
        # get_logical_form
        table_start = int(torch.sum(question_mask[0])) + 2
        instance = {
            "question_id":question_id,
            "question":question,
            "table":table,
            "paragraphs":paragraphs,
            "tokenizer":self.tokenizer,
            "input_tokens":input_tokens,
            "table_cell_tokens":table_cell_tokens,
            "paragraph_tokens":paragraph_tokens,
            
            "table_start":table_start,
            "table_mask":table_mask,
            "paragraph_mask":paragraph_mask,
            "table_number_value":table_number_value,
            "paragraph_number_value":paragraph_number_value,
            "all_cell_index":all_cell_index, 
            "all_cell_token_index":all_cell_token_index,
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_index),
            "paragraph_split_tokens_offsets":paragraph_split_tokens_offsets,
            "paragraph_text":paragraph_text,
            "row_include_cells":row_include_cells,
            "col_include_cells":col_include_cells,
            "answer":answer,
            "scale":scale,
            'aux_info':aux_info
            
        }
        answer_dict = {"answer": answer, "scale": scale, 'answer_from':answer_from, 'answer_type':answer_type}
        logical_forms = self.logical_former.get_logical_forms(**instance)  
        return {
            "input_ids": np.array(input_ids),
            "input_ids_list":input_ids_list,
            "question_mask":np.array(question_mask),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_index),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            ## add table structure
            "table_row_header_index":table_row_header_index,  
            "table_col_header_index":table_col_header_index, 
            "table_data_cell_index":table_data_cell_index, 
            "same_row_data_cell_rel":same_row_data_cell_rel, 
            "same_col_data_cell_rel":same_col_data_cell_rel, 
            "data_cell_row_rel":data_cell_row_rel, 
            "data_cell_col_rel":data_cell_col_rel,
            "table_cell_number_value_with_time":np.array(table_cell_number_value_with_time),
            "row_include_cells":row_include_cells,
            "col_include_cells":col_include_cells,
            "all_cell_index":all_cell_index,
            "logical_forms":logical_forms,
            "question":question_text,
            
             "all_cell_token_index":all_cell_token_index
        } 
    
    def read_tatqa_data(self, data, mode):
        instances = []
        logical_forms = []
        for ins in tqdm(data):
            instance = self._to_instance(ins, mode)
            logical_form = {
                "question":instance["question"],
                "logical_forms": str(instance["logical_forms"])
            }
            logical_forms.append(logical_form)
            instances.append(instance)
    
        return instances, logical_forms
    
    def process_from_file(self, paths, output_path, mode, demo=False, processor_num=1) -> DataBundle:
        # 读取数据
        data = TaTQALoader(demo=demo)._load(paths, mode)
        import pdb
#         pdb.set_trace()
        instances, logical_forms = [], []
    
        total_num = 0
        valid_logical_forms = 0
        unvalid_instances = []
        
        total_logical_forms_num = 0
    
        if processor_num ==1:
            instances, logical_forms = self.read_tatqa_data(data, mode)
        elif processor_num >1:
            import pdb
#             pdb.set_trace()
            dataset_size = len(data)
            chunk_size = math.ceil(dataset_size / processor_num)
            res = []
            p = Pool(processor_num)
            for i in range(processor_num):
                if i == processor_num - 1:
                    sub_dataset = data[i * chunk_size:dataset_size]
                else:
                    sub_dataset = data[i * chunk_size:(i + 1) * chunk_size]
                res.append(p.apply_async(self.read_tatqa_data, args=(sub_dataset,mode, )))
                #                 p.apply_async(func=DropReader.read_drop_data, args=(i,))
                print(str(i) + ' processor started !')
                
#             pdb.set_trace()
            for i in res:
                sub_instances, sub_logical_forms = i.get()
#                 import pdb
#                 pdb.set_trace()
                assert isinstance(sub_logical_forms, list)
                instances.extend(sub_instances)
                logical_forms.extend(sub_logical_forms)
                
                
                total_num +=len(sub_instances)

                for instance in sub_instances:
                    if len(instance['logical_forms'])>0:
                        valid_logical_forms+=1
                        total_logical_forms_num +=len(instance['logical_forms'])
                    else:
                        unvalid_instances.append({'question_id':instance['question_id']})
        else:
            print("preocessor num must larger than 0, error!!!")
            return
         
        print('total_num', total_num)
        print('valid_logical_forms', valid_logical_forms)
        
        if 'train' in paths:
            with open('./logical_forms_train.json','w') as f:
                json.dump(logical_forms, f, indent=4)
            f.close()
        elif 'dev' in paths:
            with open('./logical_forms_dev.json','w') as f:
                json.dump(logical_forms, f, indent=4)
            f.close()
            
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)
            print("data processed success and saved in disk")
        f.close()
        
        with open('./unvalid_logical_forms_dev.json', 'w') as f:
            json.dump(unvalid_instances, f, indent=4)
        f.close() 
        
        print('avg number of logical forms', total_logical_forms_num/valid_logical_forms)
    
    
class TaTQALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, file_path, mode):
        print("Reading file at %s", file_path)
        with open(file_path) as f:
            dataset = json.load(f)
        ds = []
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            import pdb
#             pdb.set_trace()
            for question_answer in questions:
                question = question_answer["question"].strip()
                question_id = question_answer["uid"]
                
                if mode=='test':
                    answer = ""
                    scale = ""
                    answer_type = ""
                    answer_from = ""
#                     answer_mapping = None
#                     facts = ""
                else:
                    answer = question_answer["answer"]
                    scale = question_answer["scale"]
                    answer_from = question_answer["answer_from"]
                    answer_type = question_answer["answer_type"]
#                     facts = question_answer["facts"]
#                     answer_mapping = None
                
                aux_info = ""
                if 'aux_info' in question_answer.keys():
                    aux_info = question_answer['aux_info']
                
                instance = {
                    "question":question,
                    "question_id":question_id,
                    "table":table,
                    "paragraphs":paragraphs,
                    "answer":answer,
                    "answer_from":answer_from,
                    "answer_type":answer_type,
                    'aux_info':aux_info,
                    "scale":scale,
#                      "facts":facts,
#                      "mapping":answer_mapping
                    
                }
                ds.append(instance)
        return ds
    
if __name__ == '__main__':
    path = "./dataset_tagop/tatqa_dataset_dev.json"
    output_path = "tag_op/cache/tagop_roberta_cached_dev.pkl"
    pipe = BartTatQATrainPipe()
    pipe.process_from_file(path, output_path)
    pipe.process(output_path)
