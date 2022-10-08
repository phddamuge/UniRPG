from tatqa_utils import to_number, is_number
# from data_util import *
from itertools import product
# from tag_op.data.data_util import facts_to_nums, _is_average, _is_change_ratio, _is_sum, _is_times, _is_diff,_is_division
import numpy as np
import re
from itertools import product
import itertools
from tag_op.data.operation_util import GRAMMER_CLASS, GRAMMER_ID, AUX_NUM, SCALECLASS, SCALE
from tag_op.executor import TATExecutor
import  functools



USTRIPPED_CHARACTERS = ''.join([u"Ġ"])
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def string_tokenizer(string: str, tokenizer):
    if not string:
        return [], []
    tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return split_tokens, ids


class LogicalFormer(object):
    def __init__(self, GRAMMER_CLASS, GRAMMER_ID):
        self.GRAMMER_CLASS = GRAMMER_CLASS
        self.GRAMMER_ID = GRAMMER_ID
        
    def get_logical_forms(self, question, tokenizer, table, input_tokens, tags, table_start, table_mask, paragraph_mask, table_number_value, all_cell_index, all_cell_token_index, answer_from, answer_type, answer, derivation, facts, answer_mapping, scale):
#         pass
        logical_forms = []
        try:
            if answer_type == "span":
                if derivation=="":
                     logical_forms.extend(self.get_single_span_logical_forms(tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens))
#                     pass
                else:
                    logical_form_tmp = self.get_span_comparison_logical_forms(table_start, table_number_value, derivation, all_cell_index, all_cell_token_index, answer_mapping, answer, input_tokens)
                    if len(logical_form_tmp)>0:
                        logical_forms.extend(logical_form_tmp)
                    else:
                        ## Bottoming
                        logical_forms.extend(self.get_single_span_logical_forms(tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens))
#                     pass
                
            elif answer_type == "multi-span":
                logical_forms.extend(self.get_multi_span_logical_forms(tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, table_mask, paragraph_mask,derivation, answer_mapping, answer, input_tokens))
#                 #pass
            elif answer_type == "count":
                logical_forms.extend(self.get_count_logical_forms(table_start, all_cell_index, all_cell_token_index, tags,table_mask, paragraph_mask, answer_mapping, answer, derivation, input_tokens))
                #pass
            elif answer_type == "arithmetic":
                logical_forms_tmp = self.get_arithemtic_logical_forms(question, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, derivation, answer_type, facts, answer, answer_mapping, scale, input_tokens)
                if len(logical_forms_tmp)>0:
                    logical_forms.extend(logical_forms_tmp)
#                 #pass
            
        except KeyError:
            logical_forms = []
            
#         for id, logical_form in enumerate(logical_forms):
#             logical_form = [SCALE2CLASS[scale]] + logical_form + [")"]
#             logical_forms[id] = logical_form
        
        return logical_forms
    
            
            
    def find_cell(self, all_cell_index, pos):
        ## return the index  given the row/col
        for idx, item in enumerate(all_cell_index):
            if item[1] == pos[0]  and item[2] == pos[1]:
                return idx
        return -1
    
    
    def get_single_span_logical_forms(self, tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens):
        '''
        Deal for single span 
        '''
        logical_forms = []
        try:
            if "table" in answer_mapping.keys() and len(answer_mapping['table'])>0:
                answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                answer_cell_index = self.find_cell(all_cell_index, answer_pos)
                if answer_cell_index == -1:
                    import pdb
                    pdb.set_trace()
                answer_token_start, answer_token_end = all_cell_token_index[answer_cell_index][0]+table_start, all_cell_token_index[answer_cell_index][1] + table_start
#                 answer_token_start, answer_token_end = self.check_boundary_for_table(tokenizer, input_tokens, answer[0], answer_token_start, answer_token_end)
                
                logical_form = ["CELL("]
                logical_form.extend([answer_token_start, answer_token_end-1, ")"])
                logical_forms.append(logical_form)

            if "paragraph" in answer_mapping.keys() and len(answer_mapping['paragraph'])>0:
                
                tags_tmp = tags * paragraph_mask
                paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp)
                if len(paragraph_token_index_answer)>0:
                    logical_form = ["SPAN("]
                    answer_token_start , answer_token_end = paragraph_token_index_answer[0][0], paragraph_token_index_answer[0][1]-1
#                     answer_token_start, answer_token_end = self.check_boundary(tokenizer, input_tokens, answer[0], answer_token_start, answer_token_end)
                    
                    logical_form.extend([answer_token_start , answer_token_end, ")"])
                    logical_forms.append(logical_form)
        except:
            import pdb
            pdb.set_trace()
        return logical_forms
    
    
    
    def check_boundary_for_table(self, tokenizer, input_tokens, answer_str, answer_token_start, answer_token_end):
        import pdb
        checked_answer_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in input_tokens[answer_token_start:answer_token_end]]
        answer_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in answer_str.split()]
        
        if ''.join(checked_answer_tokens) == ''.join(answer_tokens):
            return answer_token_start, answer_token_end
        else:
            start = 0 
            
            for id, token in enumerate(checked_answer_tokens):
                if token in answer_str.lower():
                    start = id
                    break
            
            end = start
            
            while end+1 < len(checked_answer_tokens):
                if checked_answer_tokens[end+1] in answer_str.lower():
                    end+=1
                else:
                    break
            if ''.join(checked_answer_tokens[start:end+1])==''.join(answer_tokens):
                print("re correct")
                import pdb
#                 pdb.set_trace()
                return answer_token_start+start, answer_token_start+end+1
                
            else:
                return answer_token_start, answer_token_end
                
    def get_span_comparison_logical_forms(self, table_start, table_number_value, derivation, all_cell_index, all_cell_token_index, answer_mapping, answer, input_tokens):
        logical_forms = []
        try:
            if ">" in derivation:
                items = derivation.split(">")
                items = [to_number(item.replace("%", "")) for item in items]
                answer_str = answer[0]
                try:
                    items_index = [table_number_value.index(item) for item in items]
                    items_pos = [(all_cell_index[i][1], all_cell_index[i][2]) for i in items_index]
                    answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                except:
                    return logical_forms
                if answer_pos[0] in [item[0] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((item[0], answer_pos[1]))
                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMAX(", "["]
                    logical_form = ["ARGMAX("]


                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["COL_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    
                    logical_forms.append(logical_form)

                if answer_pos[1] in [item[1] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((answer_pos[0], item[1]))

                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMAX(", "["]
                    logical_form = ["ARGMAX("]
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["ROW_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    
                    logical_forms.append(logical_form)

            ## deal for argmax  
            elif "<" in derivation:
                import pdb
    #             #pdb.set_trace()
                items = derivation.split("<")
                items = [to_number(item.replace("%", "")) for item in items]
                answer_str = answer[0]
                try:
                    items_index = [table_number_value.index(item) for item in items]
                    items_pos = [(all_cell_index[i][1], all_cell_index[i][2]) for i in items_index]
                    answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                except:
                    return logical_forms
                
                if answer_pos[0] in [item[0] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((item[0], answer_pos[1]))
                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMIN(", "["]
                    logical_form = ["ARGMIN("]
                    
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["COL_KEY_VALUE(",  "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")",")"])
                        logical_form.extend(["KEY_VALUE(",  "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")",")"])
                        
                        
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    logical_forms.append(logical_form)

                if answer_pos[1] in [item[1] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((answer_pos[0], item[1]))

                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMIN(", "["]
                    logical_form = ["ARGMIN("]
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["ROW_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    logical_forms.append(logical_form)
            else:
                return logical_forms
        except:
            import pdb
            pdb.set_trace()
        return logical_forms
    
    def get_all_answer_pos_in_table(self, answer, answer_mapping, table):
        '''
        mapping is not repeated, but answer may include same items
        also, there may be some Multiple combinations in the answer mapping
        '''
        answer_cells_pos = []
        for item in answer:
            answer_cells_pos.append([])
            for answer_cell in answer_mapping["table"]:
                cell_str = table.iloc[answer_cell[0],answer_cell[1]]
                if item in cell_str:
                    answer_cells_pos[-1].append((answer_cell[0], answer_cell[1]))
        
        result = list(product(*answer_cells_pos))
        import pdb
#         pdb.set_trace()
        return result
    
    def get_all_answer_pos_in_text(self,tokenizer, answer, paragraph_token_evidence_index, input_tokens):
        
        answer_token_pos = []
        for answer_item in answer:
            answer_token_pos.append([])
            answer_item_split_token, _ = string_tokenizer(answer_item, tokenizer)
            normlized_answer_item_split_token = [token.strip(USTRIPPED_CHARACTERS) for token in answer_item_split_token]
            for paragraph_token_evidence_index_item in paragraph_token_evidence_index:
                item_tokens = input_tokens[paragraph_token_evidence_index_item[0]:paragraph_token_evidence_index_item[1]]
                normlized_item_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in item_tokens]
                if normlized_answer_item_split_token == normlized_item_tokens:
                    answer_token_pos[-1].append((paragraph_token_evidence_index_item[0], paragraph_token_evidence_index_item[1]))
        result = list(product(*answer_token_pos))
        import pdb
#         pdb.set_trace()
        return result
        
    def get_all_answer_pos_in_text_patch(self, tokenizer, answer, paragraph_token_index_answer, input_tokens):
        
        
        def find_index(index, len_sum):
            id = -1
            for id in range(len(len_sum)):
                if index == len_sum[id]:
                    return id
            return id
        
        pos = []
        
        checked_answer = [input_tokens[offset[0]:offset[1]] for offset in paragraph_token_index_answer]
        normalized_checked_answer = [ [token.strip(USTRIPPED_CHARACTERS).lower() for token in item]   for item in checked_answer]
        
        start = 0
        import pdb
        
        for answer_item in answer:
            tokenized_answer_item, _ = string_tokenizer(answer_item, tokenizer)
            normalized_tokenized_answer_item = [token.strip(USTRIPPED_CHARACTERS).lower() for token in tokenized_answer_item]
            index=-1
            for id, normalized_checked_answer_item in enumerate(normalized_checked_answer):
                index  = ''.join(normalized_checked_answer_item).find(''.join(normalized_tokenized_answer_item))
                if index != -1:
                    start = paragraph_token_index_answer[id][0]
                    break
                
            
            if index!=-1:
               
                start_index = index
                end_index = index + len(''.join(normalized_tokenized_answer_item))
                len_sum = [ len(''.join(normalized_checked_answer_item[:i])) for i in range(0, len(normalized_checked_answer_item)+1)]
                
                start_index = find_index(start_index, len_sum)

                end_index = find_index(end_index, len_sum)
                
                if start_index == -1 or end_index==-1:
                    return []
                else:
                    pos.append((start+start_index, start+end_index))
#         pdb.set_trace()
        
        if len(pos) >0:
            return [pos]
        else:
            return []
            
        
        
    def get_multi_span_logical_forms(self, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, table_mask, paragraph_mask, derivation, answer_mapping, answer, input_tokens):
        logical_forms = []
        if "table" in answer_mapping.keys() and len(answer_mapping["table"])>0 and ("paragraph" not in answer_mapping.keys() or len(answer_mapping["paragraph"])==0):
            answer_cells_pos = answer_mapping["table"]
            all_possible_answer_cells_pos = []
            all_possible_answer_cells_pos.append(answer_cells_pos)
            if len(answer_cells_pos)!=len(answer):
                all_possible_answer_cells_pos= self.get_all_answer_pos_in_table(answer, answer_mapping, table) ## makeup 
#             if len(all_possible_answer_cells_pos) ==0:
#                 import pdb
#                 pdb.set_trace()
            for answer_cells_pos in all_possible_answer_cells_pos:
                answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
                answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                
                for item in answer_cells_token_index:
                    logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                
                logical_forms.append(logical_form)
            
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" not in answer_mapping.keys() or len(answer_mapping["table"])==0):
            
            import pdb
#             pdb.set_trace()
            
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            all_possible_paragraph_token_index_answer = []
#             all_possible_paragraph_token_index_answer.append(paragraph_token_index_answer)
            
            if len(paragraph_token_index_answer)>=len(answer):
                all_possible_paragraph_token_index_answer = self.get_all_answer_pos_in_text(tokenizer, answer,paragraph_token_index_answer, input_tokens)
            
            if len(paragraph_token_index_answer) < len(answer):
                all_possible_paragraph_token_index_answer = self.get_all_answer_pos_in_text_patch(tokenizer, answer,paragraph_token_index_answer, input_tokens)
            
            if len(all_possible_paragraph_token_index_answer) ==0:
                all_possible_paragraph_token_index_answer = [paragraph_token_index_answer]
#                 import pdb
#                 pdb.set_trace()
            for paragraph_token_index_answer in all_possible_paragraph_token_index_answer:
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)

        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" in answer_mapping.keys() and len(answer_mapping["table"])!=0):
            answer_cells_pos = answer_mapping["table"]
            answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
            answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
            
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            
            if len(answer_cells_pos) + len(paragraph_token_index_answer) == len(answer):
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                for item in answer_cells_token_index:
                    logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)
                
            else:
#                 import pdb
#                 pdb.set_trace()
                return logical_forms
        import pdb
        #pdb.set_trace()
        return logical_forms
                
    
    def get_count_logical_forms(self, table_start, all_cell_index, all_cell_token_index, tags,table_mask, paragraph_mask, answer_mapping, answer, derivation, input_tokens):
        logical_forms = []
        if "table" in answer_mapping.keys() and len(answer_mapping["table"])>0 and ("paragraph" not in answer_mapping.keys() or len(answer_mapping["paragraph"])==0):
            answer_cells_pos = answer_mapping["table"]
            answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
            answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
            if len(answer_cells_token_index) != int(answer):
                import pdb
                pdb.set_trace()
#             logical_form = ["COUNT(", "["]
            logical_form = ["COUNT("]
            
            for item in answer_cells_token_index:
                logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
#             logical_form.extend(["]", ")"])
            logical_form.extend([")"])
            logical_forms.append(logical_form)
        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" not in answer_mapping.keys() or len(answer_mapping["table"])==0):
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            if len(paragraph_token_index_answer)>0:
#                 if len(paragraph_token_index_answer) != int(answer):
#                     import pdb
#                     pdb.set_trace()
#                 logical_form = ["COUNT(", "["]
                logical_form = ["COUNT("]
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
                
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)
        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" in answer_mapping.keys() and len(answer_mapping["table"])!=0):
            return logical_forms
        
        return logical_forms

    
    def get_arithemtic_logical_forms(self, question, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, derivation, answer_type, facts, answer, answer_mapping, scale, input_tokens):
        grammer_space = ["[", "]", "(", ")", "+", "-", "*", "/"]
        ## mapping the derivation into the grammer_number_sequence
        logical_forms = []
        
#         if question =="What is the percentage increase / (decrease) in the loss from operations from 2018 to 2019?":
#             import pdb
#             pdb.set_trace()
        try:
#             if '[' in derivation:
#                 import pdb
#                 pdb.set_trace()
            grammer_number_sequence = get_operators_from_derivation(derivation, grammer_space)
            
            normalized_grammer_number_sequence = normalize_grammer_number_sequence(grammer_number_sequence, grammer_space)
            post_grammer_number_sequence = convert_inorder_to_postorder(normalized_grammer_number_sequence, grammer_space)
            lf = get_lf_from_post_grammer_number_sequence(post_grammer_number_sequence, grammer_space)
            
            '''
            transfer arguments to index
            '''
            lf = self.mapping_arithmatic_arguments_to_index(question, tokenizer, table, lf, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, GRAMMER_CLASS, input_tokens)
            logical_forms.extend(lf)
        except:
            import pdb
#             pdb.set_trace()
            return []
        
        return logical_forms
    
    
    def mapping_arithmatic_arguments_to_index(self, question, tokenizer, table, lf, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, GRAMMER_CLASS, input_tokens):
        '''
        mapping arguments into index
        '''
        arguments = []
        for idx, op in enumerate(lf):
            if op not in GRAMMER_CLASS and op!="0" and op!="1":
                arguments.append((idx, op))

        import pdb
        
        answer_table_cell_token_index = []
        paragraph_token_index_answer = []
        
        if "table" in answer_mapping.keys():
            answer_table_cell = answer_mapping["table"]
            answer_table_cell_index = [self.find_cell(all_cell_index, cell) for cell in answer_table_cell]
            answer_table_cell_token_index = [(table_start+all_cell_token_index[cell_index][0], table_start+all_cell_token_index[cell_index][1])  for cell_index in answer_table_cell_index]
        if "paragraph" in answer_mapping.keys():
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
        
        all_op_index = []
#         pdb.set_trace()
        
#         if question == "What is the increase / (decrease) in the Fabless design companies from 2018 to 2019?":
#             import pdb
#             pdb.set_trace()
            
        try:
            for idx, op in arguments:
                op_index = []
                if '%' in op:
                    op = op.replace("%","")
                    op = op.strip()
                if len(answer_table_cell_token_index) > 0:
                    for j, item in enumerate(answer_table_cell):
#                         tokenized_item, _ =  string_tokenizer(table.iloc[item[0], item[1]], tokenizer)
#                         normlized_item = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_item]   
#                         tokenized_op, _ = string_tokenizer(op, tokenizer)
#                         normlized_tokenized_op = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_op]
#                         if set(normlized_tokenized_op) in set(normlized_item):
#                       
                        candidates = []
                        candidate = to_number(table.iloc[item[0], item[1]])
                        candidates.extend([candidate, round(100*candidate,2), -1*candidate, round(-100*candidate,2)])
                        if to_number(op) in candidates:
                            op_index.append((0,j))

                if len(paragraph_token_index_answer) > 0:
                    for j, item in enumerate(paragraph_token_index_answer):
                        normlized_item = [token.strip(USTRIPPED_CHARACTERS) for token in input_tokens[item[0]: item[1]]]
                        tokenized_op, _ = string_tokenizer(op, tokenizer)
                        normlized_tokenized_op = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_op]
                        if "".join(normlized_tokenized_op) in "".join(normlized_item):
                             op_index.append((1,j))
                
                all_op_index.append(op_index)
        except:
            import pdb
#             pdb.set_trace()
                        
        all_op_comps = list(product(*all_op_index))
        all_lfs = []
        
        for op_comp in all_op_comps:
            lf_tmp = lf.copy()
            for idx in range(len(op_comp)):
                item = op_comp[idx]
                if item[0] == 1:
                    s = paragraph_token_index_answer[item[1]][0]
                    e =  paragraph_token_index_answer[item[1]][1]
                    unit = ['VALUE(', s, e-1, ')']
                    lf_tmp = lf_tmp[:arguments[idx][0]+ 3* idx] + unit + lf_tmp[arguments[idx][0]+ 3* idx +1:]
                if item[0] == 0:
                    s = answer_table_cell_token_index[item[1]][0]
                    e =  answer_table_cell_token_index[item[1]][1]
                    unit = ['CELL_VALUE(', s, e-1, ')']
                    lf_tmp = lf_tmp[:arguments[idx][0]+ 3* idx] + unit + lf_tmp[arguments[idx][0]+ 3* idx +1:]
            all_lfs.append(lf_tmp)

        return all_lfs       
            
    
    def find_evidence_from_tags(self, tags):
        '''
        extract all the evidence
        '''
        evidence_pos = []
        len = tags.size(1)
        start = 0
        flag=False
        for i in range(len):
            if flag==False and tags[0][i]==1:
                flag=True
                start = i
                continue
            elif flag==False and tags[0][i]==0:
                continue
            elif (i==len-1 or tags[0][i]==0) and flag:
                flag=False
                evidence_pos.append((start, i))
                start = i+1
                continue
            else:
                flag=True
                continue
        return evidence_pos
                
def convert_inorder_to_postorder(grammer_number_sequence, grammer_space):
    '''
    in-order 
    '''
    stack = []
    post_grammer_number_sequence = []
    import pdb
#     pdb.set_trace()
    for i in range(len(grammer_number_sequence)):
        if grammer_number_sequence[i] not in grammer_space:
            post_grammer_number_sequence.append(grammer_number_sequence[i])
        elif grammer_number_sequence[i]== "(" or grammer_number_sequence[i]=="[":
            stack.append(grammer_number_sequence[i])
        elif grammer_number_sequence[i] in ["+", "-", "*", "/"]:
            while(len(stack)>0):
                if compare_priority(stack[-1], grammer_number_sequence[i]):
                    post_grammer_number_sequence.append(stack.pop())

                else:
                    break
            stack.append(grammer_number_sequence[i])
        else:
            ## "]" or ")"
            while(stack[-1]!="(" and stack[-1]!="["):
                post_grammer_number_sequence.append(stack.pop())
            stack.pop()
    while len(stack)>0:
        post_grammer_number_sequence.append(stack.pop())
        
    return post_grammer_number_sequence
                
def compare_priority(c1, c2):
    priority={ "*":1, "/":1, "+":0, "-":0}
    if c1 not in priority or c2 not  in priority:
        return False
    priority_value_1 = priority[c1]
    priority_value_2 = priority[c2]

    return priority_value_1>=priority_value_2

def get_operators_from_derivation(derivation, grammer_space):
    grammer_number_sequence = []
    num=""
    for idx, char in enumerate(derivation):
        if is_whitespace(char):
            if len(num)>0:
                grammer_number_sequence.append(num)
                num=""
            continue
        if char in grammer_space:
            if len(num)>0:
                grammer_number_sequence.append(num)
                num=""
            grammer_number_sequence.append(char)
        else:
            num+=char
            if idx == len(derivation)-1:
                grammer_number_sequence.append(num)
    return grammer_number_sequence


def get_lf_from_post_grammer_number_sequence(grammer_number_sequence, grammer_space):
    '''
    get the logical forms from the post-order grammer and number sequence
    '''
    lf_stack = []
    
    for grammer_number  in grammer_number_sequence:
        if grammer_number not in grammer_space:
            lf_stack.append([grammer_number])
        else:
            a = lf_stack.pop()
            b = lf_stack.pop()
            
            if grammer_number == "+":
                lf_stack.append(["SUM("] + b + a + [")"])
            elif grammer_number == "-":
                lf_stack.append(["DIFF("] + b + a + [")"])
            elif grammer_number == "*":
                lf_stack.append(["TIMES("] + b + a + [")"])
            else:
                lf_stack.append(["DIV("] + b + a + [")"])
    lf = lf_stack.pop()
    '''
    change for change_ration and avg
    '''
    lf = mapping_into_change_ratio(lf)
    lf = mapping_into_avg(lf)
    
    return lf


def normalize_grammer_number_sequence(grammer_number_sequence, grammer_space):
    '''
    deal with the negative number
    '''
    normalized_grammer_number_sequence = grammer_number_sequence.copy()
    add_grammer_count = 0
    for idx, grammer_number in enumerate(grammer_number_sequence):
        if grammer_number == "-" and idx==0 :
            normalized_grammer_number_sequence = ['0'] + normalized_grammer_number_sequence
            add_grammer_count +=1
            
        elif grammer_number == "-" and (grammer_number_sequence[idx-1]=="(" or grammer_number_sequence[idx-1]=="["):
            normalized_grammer_number_sequence = normalized_grammer_number_sequence[:idx+add_grammer_count] + ['0'] + normalized_grammer_number_sequence[idx+add_grammer_count:]
            add_grammer_count +=1
        elif grammer_number == "-" and (grammer_number_sequence[idx-1]=="*" or grammer_number_sequence[idx-1]=="/"):
#             import pdb
#             pdb.set_trace()
            normalized_grammer_number_sequence = normalized_grammer_number_sequence[:idx+add_grammer_count] + ["("] + ['0'] + normalized_grammer_number_sequence[idx+add_grammer_count:idx+add_grammer_count+2] + [")"] + normalized_grammer_number_sequence[idx+add_grammer_count+3:]
            add_grammer_count +=3
        else:
            continue
        
    return  normalized_grammer_number_sequence   
            
                
def mapping_into_change_ratio(logical_form):
    rval = []
    i = 0
    while i < len(logical_form):
        if logical_form[i] == "DIV("  and i + 6 < len(logical_form) and logical_form[i+1] == "DIFF(" and logical_form[i+4] == ')' and logical_form[i+3] == logical_form[i+5] and logical_form[i+6] == ")":
            rval.extend(['CHANGE_R(', logical_form[i+2], logical_form[i+3], ')'])
            i += 7
        else:
            rval.append(logical_form[i])
            i += 1
    return rval
    
def mapping_into_avg(logical_form):
    '''
    AVG: 
    '''
    def find_end_idx(logical_form):
        stack = []
        for i in range(len(logical_form)):
            if logical_form[i][-1] == '(':
                stack.append('(')
            elif logical_form[i][0] == ')':
                stack.pop()
                if len(stack) == 0:
                    return i

    rval = []
    i = 0
    import pdb
    
#     pdb.set_trace()
    while i < len(logical_form):
        if logical_form[i] == "DIV(":
            end_idx = i + find_end_idx(logical_form[i:])
            try:
                num = int(logical_form[end_idx-1])
            except:
                i += 1
                continue
            j = 0
            while j < num-1:
                if logical_form[i+1+j] != "SUM(":
                    break
                j += 1
            if j == num-1 and j!=0:
#                 pdb.set_trace()
                tmp = ["AVG("]
                import pdb
#                 pdb.set_trace()
                last_end_idx = i+num-1
                for k in range(0, num-1):
                    cur_end_idx = find_end_idx(logical_form[i+num-1-k:]) + (i+num-1-k)
                    tmp.extend(logical_form[last_end_idx+1: cur_end_idx])
                    last_end_idx = cur_end_idx
#                 tmp.append("]")
                tmp.append(")")
                logical_form = logical_form[:i] + tmp + logical_form[end_idx+1:]

        i += 1

    return logical_form


class LogicalFormerWS(object):
    '''
    obtain the groundtruth of logical forms by heurisitc search
    i.e. we train the model with weak supervision
    '''
    def __init__(self, GRAMMER_CLASS, GRAMMER_ID):
        self.GRAMMER_CLASS = GRAMMER_CLASS
        self.GRAMMER_ID = GRAMMER_ID
        self.span_num = 0
        self.none_span_num = 0
        self.mul_valid_span_num = 0
        self.arithmetic_expression_library, self.arithmetic_expression_function = self.arithmetic_expression_constrcutor(self.GRAMMER_CLASS, max_avg_arg=3, max_arithmetic_arg=3)
        self.patt_function_mapping = dict(zip([str(item) for item in self.arithmetic_expression_library], self.arithmetic_expression_function))
        self.executor = TATExecutor(GRAMMER_CLASS, GRAMMER_ID, AUX_NUM)
        
    def get_logical_forms(self, question_id, question, table, paragraphs, tokenizer, input_tokens, table_cell_tokens, paragraph_tokens, table_start, table_mask, paragraph_mask, table_number_value, paragraph_number_value, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells, answer, scale, aux_info):
        '''
        question:str
        table: dataframe
        paragraphs: [dict(uid, order, text)]
        input_tokens
        table_mask: tensor(bsz, 512)
        paragraph_mask:tensor(bsz, 512)
        table_number_value: [nan, n_1, n_2, ..., n_k]
        all_cell_index: [(cell_id, row_id, col_id)]
        all_cell_token_index:[(cell_start, cell_end)]
        answer: [str]
        scale: ['', 'million', 'thousand', 'billion', 'percent']
        '''
        import pdb
        logical_forms = []
        if isinstance(answer, list):
            ans_item_num = len(answer)
            is_num = is_number(answer[0])
            if ans_item_num == 1:
                if not is_num or str(answer[0]) in [str(num) for num in table_number_value] + [str(num) for num in paragraph_number_value]: ## a span and not a number
                    answer_text = answer[0]
                    span_logical_forms = self.get_valid_span_answer(table, paragraphs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start)
                    if len(span_logical_forms)>0:
                        logical_forms.extend(span_logical_forms)
                    ## a span answer could be derived by operations argmax/argmin
                    compare_logical_forms = self.get_compare_span_answer(table, paragraphs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells, table_number_value, table_start)
                    if len(compare_logical_forms)>0:
                        logical_forms.extend(compare_logical_forms)
# #                     if is_num:
# #                         arithmetic_logical_forms = self.get_valid_expression_answer(question_id, question, table, paragraphs, answer, input_tokens, table_cell_tokens, paragraph_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, table_number_value, paragraph_number_value, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells,table_start, scale)
# #                         if len(arithmetic_logical_forms)>0:
# #                             logical_forms.extend(arithmetic_logical_forms)
#                 pass
              
            else:
#                 assert ans_item_num>1
                multi_span_logical_forms = self.get_multi_span_answer(question_id, question, table, paragraphs, answer, input_tokens, table_cell_tokens, paragraph_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, table_number_value, paragraph_number_value, paragraph_split_tokens_offsets, paragraph_text, table_start)
                if len(multi_span_logical_forms)>0:
                    logical_forms.extend(multi_span_logical_forms)
#                 pass
            return logical_forms
        else:
#             ## if a count number
            if not is_number(answer):
                return logical_forms
            ## a value in table/ text
            if isinstance(aux_info, list):
                count_logical_forms = self.get_count_answer(table, paragraphs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start, aux_info)
                if len(count_logical_forms)>0:
                    logical_forms.extend(count_logical_forms)
            else:
                ## expression
                arithmetic_logical_forms = self.get_valid_expression_answer(question_id, question, table, paragraphs, answer, input_tokens, table_cell_tokens, paragraph_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, table_number_value, paragraph_number_value, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells,table_start, scale)
                if len(arithmetic_logical_forms)>0:
                    logical_forms.extend(arithmetic_logical_forms)
#             pass
            return logical_forms
    
    def find_cell(self, all_cell_index, pos):
        ## return the index  given the row/col
        for idx, item in enumerate(all_cell_index):
            if item[1] == pos[0]  and item[2] == pos[1]:
                return idx
        return -1
    
    def get_valid_expression_answer(self, question_id, question,  table, paragraphs, answer, input_tokens, table_cell_tokens, paragraph_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, table_number_value, paragraph_number_value, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells, table_start, scale):
        ## arithematic expression logical forms contruction
        example = {
            'table_cell_tokens':table_cell_tokens,
            'table_cell_number_value':table_number_value,
            'table_cell_index':table_cell_index,
            'paragraph_tokens':paragraph_tokens,
            'paragraph_number_value':paragraph_number_value,
            'paragraph_index':paragraph_index,
            'question_id':question_id
        }

        arithmetic_expression_patt_candidates = self.arithmetic_expression_patt_filter(question, self.arithmetic_expression_library)

        logical_forms = []
        number_candidates = [] ## all the numbers that not nan
        number_candidates_index = []
        
        cell_nums = int(max(table_cell_index[0]))
        table_number_value = table_number_value[:cell_nums]
        
        table_number_value = [number if number>=0 else -1*number for number in table_number_value]
        
        paragraph_word_nums = int(max(paragraph_index[0]))
        paragraph_number_value = paragraph_number_value[:paragraph_word_nums]
        paragraph_number_value = [number if number>=0 else -1*number for number in paragraph_number_value]
        
        
        table_cell_num = len(table_number_value)
        
        row_include_num  = {}
        col_include_num = {}
        paragraph_number_index = []
        for row_id, row in row_include_cells.items():
            number_in_row = []
            for cell_id in row:
                if not np.isnan(table_number_value[cell_id-1]):
                    number_in_row.append(cell_id-1)
            if len(number_in_row)>0:
                row_include_num.update({row_id:number_in_row})
        
        for col_id, col in col_include_cells.items():
            number_in_col = []
            for cell_id in col:
                if not np.isnan(table_number_value[cell_id-1]):
                    number_in_col.append(cell_id-1)
            if len(number_in_col)>0:
                col_include_num.update({col_id:number_in_col})
        
        for number_id, number in enumerate(paragraph_number_value):
            if not np.isnan(number):
                paragraph_number_index.append(number_id+table_cell_num)
        
        try:
            for arithmetic_expression_patt in arithmetic_expression_patt_candidates:
                args_pos = self.get_arguments_pos(arithmetic_expression_patt) ## arguments
                args_num = len(args_pos)
                all_args_combinations, all_args_number_combinations = self.get_all_args_combinations(question_id, arithmetic_expression_patt, table_number_value, paragraph_number_value, row_include_num, col_include_num, paragraph_number_index, args_num)
                for id, args_candidates in enumerate(all_args_combinations):
                    '''
                    first compute the result then save the correct argument candidates 
                    '''
                    if id>=1e6:
                        break
                        
                    result = None
                    args_number_candidates = all_args_number_combinations[id]
                    result = self.patt_function_mapping[str(arithmetic_expression_patt)](arithmetic_expression_patt, args_number_candidates)

                    if scale=='percent' and ( "DIV(" in arithmetic_expression_patt or "CHANGE_R(" in arithmetic_expression_patt):
                        result = np.round(result*100, 4)

                    if float(answer) == result or float(answer)==np.round(result, 2):
                        args_logical_form = []
                        for arg in args_candidates:
                            if arg < table_cell_num:
                                start = table_start + all_cell_token_index[arg][0]
                                end = table_start + all_cell_token_index[arg][1]-1
                                args_logical_form.append(['CELL_VALUE(', start, end, ')'])

                            if arg >= table_cell_num:
                                arg -=table_cell_num
                                start = paragraph_index[0].tolist().index(arg+1)
                                reversed_paragraph_index = list(reversed(paragraph_index[0].tolist()))
                                end = 512 - reversed_paragraph_index.index(arg+1) - 1
                                args_logical_form.append(['VALUE(', start, end, ')'])

                        logical_form = []
                        for i,  pos in enumerate(args_pos):
                            if i==0:
                                logical_form += arithmetic_expression_patt[:pos]
                                logical_form += args_logical_form[i]
                            elif i== len(args_pos)-1:
                                logical_form += arithmetic_expression_patt[args_pos[i-1]+1:pos]
                                logical_form += args_logical_form[i]
                                logical_form += arithmetic_expression_patt[pos+1:]
                            else:
                                logical_form += arithmetic_expression_patt[args_pos[i-1]+1:pos]
                                logical_form += args_logical_form[i]

                        execution_result = self.executor.execute(logical_form, example)[0]
                        if scale=='percent' and ( "DIV(" in arithmetic_expression_patt or "CHANGE_R(" in arithmetic_expression_patt):
                            execution_result = np.round(execution_result*100, 4)

                        assert float(answer) == execution_result or float(answer)==np.round(execution_result, 2)
                        logical_forms.append(logical_form)
              
        except:
            #print('question_id', question_id)
            #print('question', question)
            #print('arithmetic_expression_patt_candidates', arithmetic_expression_patt_candidates)
#             print('args_number_candidates', args_number_candidates)
#             print('result', result)
#             print('execution_result', execution_result)
            return logical_forms
            
        return logical_forms
    
    def get_arguments_pos(self, arithmetic_expression_patt):
        '''
        get the number of arguments of expression pattern
        '''
        length = len(arithmetic_expression_patt)
        i = 0
        args_num = 0
        args_pos = []
        while(i<length):
            if arithmetic_expression_patt[i].startswith('arg_'):
                args_pos.append(i)  
                i+=1
            else:
                i+=1
        assert len(args_pos) > 0
        return args_pos

        
        
        
    
    
    
    
    def get_all_args_combinations(self, question_id, arithmetic_expression_patt,  table_number_value, paragraph_number_value, row_include_num, col_include_num, paragraph_number_index, args_num):
        '''
        get all the possible permutations of selected number
        '''
#         assert args_num <= len(number_candidates)

        ops, op_num = get_op_num(arithmetic_expression_patt)
        number_index_combinations = []
        number_combinations = []
        import pdb
        
        if op_num == 1:
            if arithmetic_expression_patt[0] in ['DIFF(', 'DIV(', 'CHANGE_R(']:
#                 pdb.set_trace()
                for row_id, row_nums in row_include_num.items():
                    if len(row_nums)<args_num:
                        continue
                    else:
                        for item in itertools.permutations(row_nums, args_num):
                            number_index_combinations.append(item)
                            number_combinations.append([table_number_value[id] for id in item])
                          
                for col_id, col_nums in col_include_num.items():
                    if len(col_nums)<args_num:
                        continue
                    else:
                        for item in itertools.permutations(col_nums, args_num):
                            number_index_combinations.append(item) 
                            number_combinations.append([table_number_value[id] for id in item])
                          
                if len(paragraph_number_index)>=args_num:
                    for item in itertools.permutations(paragraph_number_index, args_num):
                        number_index_combinations.append(item)
                        number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
            else:
                for row_id, row_nums in row_include_num.items():
                    if len(row_nums)<args_num:
                        continue
                    else:
                        for item in itertools.combinations(row_nums, args_num):
                            number_index_combinations.append(item)
                            number_combinations.append([table_number_value[id] for id in item])
                            
                for col_id, col_nums in col_include_num.items():
                    if len(col_nums)<args_num:
                        continue
                    else:
                        for item in itertools.combinations(col_nums, args_num):
                            number_index_combinations.append(item) 
                            number_combinations.append([table_number_value[id] for id in item])
                            
                if len(paragraph_number_index)>=args_num:
                    for item in itertools.combinations(paragraph_number_index, args_num):
                        number_index_combinations.append(item)
                        number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
#                 pdb.set_trace()
        elif op_num == 2:
        
#             assert args_num == 3
            import pdb
            if ops[-1] in ['DIFF(', 'DIV(', 'CHANGE_R(']:
                for row_id, row_nums in row_include_num.items():
                    if len(row_nums)<args_num:
                        continue
                    else:
                        for item in itertools.permutations(row_nums, args_num):
                            number_index_combinations.append(item)
                            number_combinations.append([table_number_value[id] for id in item])
                          
                for col_id, col_nums in col_include_num.items():
                    if len(col_nums)<args_num:
                        continue
                    else:
                        for item in itertools.permutations(col_nums, args_num):
                            number_index_combinations.append(item) 
                            number_combinations.append([table_number_value[id] for id in item])
                          
                if len(paragraph_number_index)>=args_num:
                    for item in itertools.permutations(paragraph_number_index, args_num):
                        number_index_combinations.append(item)
                        number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
                
                            
#                 pdb.set_trace()
                
            else:
                if arithmetic_expression_patt in [['SUM(', 'SUM(','arg_1', 'arg_2', ')', 'arg_3', ')'], 
                                                      ['TIMES(', 'TIMES(', 'arg_1', 'arg_2', ')', 'arg_3', ')']]:
                    
#                     pdb.set_trace()
                    for row_id, row_nums in row_include_num.items():
                        if len(row_nums)<args_num:
                            continue
                        else:
                            for item in itertools.combinations(row_nums, args_num):
                                number_index_combinations.append(item)
                                number_combinations.append([table_number_value[id] for id in item])

                    for col_id, col_nums in col_include_num.items():
                        if len(col_nums)<args_num:
                            continue
                        else:
                            for item in itertools.combinations(col_nums, args_num):
                                number_index_combinations.append(item) 
                                number_combinations.append([table_number_value[id] for id in item])

                    if len(paragraph_number_index)>=args_num:
                        for item in itertools.combinations(paragraph_number_index, args_num):
                            number_index_combinations.append(item)
                            number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
                
#                     pdb.set_trace()
                else:
#                     pdb.set_trace()
                    for row_id, row_nums in row_include_num.items():
                        if len(row_nums)<args_num:
                            continue
                        else:
                            for item in itertools.combinations(row_nums, args_num):
                                number_index_combinations.append(item)
                                number_combinations.append([table_number_value[id] for id in item])
                                
                                if arithmetic_expression_patt[1] in GRAMMER_CLASS.keys():
                                    aux_items = [(item[0], item[1], item[0]), (item[0], item[2], item[1])]
                                    for aux_item in aux_items:
                                        number_index_combinations.append(aux_item)
                                        number_combinations.append([table_number_value[id] for id in aux_item])
                                else:
                                    assert arithmetic_expression_patt[1] not in GRAMMER_CLASS.keys()
                                    aux_items = [(item[1], item[0], item[2]), (item[2], item[0], item[1])]
                                    for aux_item in aux_items:
                                        number_index_combinations.append(aux_item)
                                        number_combinations.append([table_number_value[id] for id in aux_item])
                                
                                

                    for col_id, col_nums in col_include_num.items():
                        if len(col_nums)<args_num:
                            continue
                        else:
                            for item in itertools.combinations(col_nums, args_num):
                                number_index_combinations.append(item) 
                                number_combinations.append([table_number_value[id] for id in item])
                                if arithmetic_expression_patt[1] in GRAMMER_CLASS.keys():
                                    aux_items = [(item[0], item[1], item[0]), (item[0], item[2], item[1])]
                                    for aux_item in aux_items:
                                        number_index_combinations.append(aux_item)
                                        number_combinations.append([table_number_value[id] for id in aux_item])
                                else:
                                    assert arithmetic_expression_patt[1] not in GRAMMER_CLASS.keys()
                                    aux_items = [(item[1], item[0], item[2]), (item[2], item[0], item[1])]
                                    for aux_item in aux_items:
                                        number_index_combinations.append(aux_item)
                                        number_combinations.append([table_number_value[id] for id in aux_item])
                                
                                

                    if len(paragraph_number_index)>=args_num:
                        for item in itertools.combinations(paragraph_number_index, args_num):
                            number_index_combinations.append(item)
                            number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
                            
                            if arithmetic_expression_patt[1] in GRAMMER_CLASS.keys():
                                aux_items = [(item[0], item[1], item[0]), (item[0], item[2], item[1])]
                                for aux_item in aux_items:
                                    number_index_combinations.append(aux_item)
                                    number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in aux_item])
                            else:
                                assert arithmetic_expression_patt[1] not in GRAMMER_CLASS.keys()
                                aux_items = [(item[1], item[0], item[2]), (item[2], item[0], item[1])]
                                for aux_item in aux_items:
                                    number_index_combinations.append(aux_item)
                                    number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in aux_item])
                                
#                     pdb.set_trace()
        else:
            speical_patterns = [ ['DIFF(', 'AVG(', 'arg_1', 'arg_2', ')', 'AVG(', 'arg_3', 'arg_4', ')', ')'], ['DIFF(', 'AVG(', 'arg_1', 'arg_2', 'arg_3', ')', 'AVG(', 'arg_4', 'arg_5', 'arg_6', ')', ')'], ['DIFF(', 'CHANGE_R(', 'arg_1', 'arg_2', ')', 'CHANGE_R(', 'arg_3', 'arg_4', ')', ')']]
            assert arithmetic_expression_patt in speical_patterns
            
            tmp_number_combinations = []
            tmp_number_index_combinations = []
            tmp_args_num = int(args_num/2)
            for row_id, row_nums in row_include_num.items():
                if len(row_nums)<tmp_args_num:
                    continue
                else:
                    for item in itertools.combinations(row_nums, tmp_args_num):
                        tmp_number_index_combinations.append(item)
                        tmp_number_combinations.append([table_number_value[id] for id in item])
                        if arithmetic_expression_patt[1] == 'CHANGE_R(':
                            aux_item = (item[1], item[0])
                            tmp_number_index_combinations.append(aux_item)
                            tmp_number_combinations.append([table_number_value[id] for id in aux_item])
                            
                        
            for col_id, col_nums in col_include_num.items():
                if len(col_nums)<tmp_args_num:
                    continue
                else:
                    for item in itertools.combinations(col_nums, tmp_args_num):
                        tmp_number_index_combinations.append(item) 
                        tmp_number_combinations.append([table_number_value[id] for id in item])
                        if arithmetic_expression_patt[1] == 'CHANGE_R(':   
                            aux_item = (item[1], item[0])
                            tmp_number_index_combinations.append(aux_item)
                            tmp_number_combinations.append([table_number_value[id] for id in aux_item])

            if len(paragraph_number_index)>=tmp_args_num:
                for item in itertools.combinations(paragraph_number_index, tmp_args_num):
                    tmp_number_index_combinations.append(item)
                    tmp_number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in item])
                    if arithmetic_expression_patt[1] == 'CHANGE_R(':
                        aux_item = (item[1], item[0])
                        tmp_number_index_combinations.append(aux_item)
                        tmp_number_combinations.append([paragraph_number_value[id-len(table_number_value)] for id in aux_item])
            
            
            for item in itertools.permutations(tmp_number_index_combinations, 2):
                new_item = []
                for it in item:
                    new_item +=list(it)
                number_index_combinations.append(new_item)
                
            for item in itertools.permutations(tmp_number_combinations, 2):
                new_item = []
                for it in item:
                    new_item +=list(it)
                number_combinations.append(new_item)
            
            
            
            
        return number_index_combinations, number_combinations



    def arithmetic_expression_patt_filter(self, question, arithmetic_expression_library):
        '''
        filter some unreasonable expression depand on question
        '''
        AVG_TRIGGER_LEXICO = ['average']
        DIFF_TRIGGER_LEXICO = ['difference', 'change', 'decrease']
        CHANGE_R_TRIGGER_LEXICO = ['percentage change']
        DIFF_TRUGGER_LEXICO=['What is the change', 'What was the change', 'What was the difference', 'What is the difference', 'What is the increase / (decrease)', 'What was the increase / (decrease)', 'What is the increase/ (decrease)']
        SUM_TRGGER_LEXICO = ['What is the total', 'What was the total', 'What is the sum', 'What was the sum']
        DIV_TRGGER_LEXICO = ['What is the percentage', 'What was the percentage', 'How many percent of', 'What is the ratio', 'What percentage of', 'What is the proportion', 'What proportion', 'What is the value of']
        
        
        filted_arithmetic_expression_library = []
        
        if any([trigger.lower() in question.lower() for trigger in CHANGE_R_TRIGGER_LEXICO]) and all([trigger.lower() not in question.lower() for trigger in ['difference', 'decrease']]):
            arithmetic_expression_patts=[['CHANGE_R(', 'arg_1', 'arg_2', ')'], ['DIFF(', 'arg_1', 'arg_2', ')']]
            for arithmetic_expression_patt in arithmetic_expression_patts:
                if arithmetic_expression_patt in arithmetic_expression_library:
                    filted_arithmetic_expression_library.append(arithmetic_expression_patt)
            return filted_arithmetic_expression_library
        
        if any([trigger.lower() in question.lower() for trigger in CHANGE_R_TRIGGER_LEXICO]) and any([trigger.lower() in question.lower() for trigger in DIFF_TRIGGER_LEXICO]):
            diff_change_r_patt = ['DIFF(', 'CHANGE_R(', 'arg_1', 'arg_2', ')', 'CHANGE_R(', 'arg_3', 'arg_4', ')', ')']
            if diff_change_r_patt in arithmetic_expression_library:
                filted_arithmetic_expression_library.append(diff_change_r_patt)
            return filted_arithmetic_expression_library
        
        if any([trigger.lower() in question.lower() for trigger in AVG_TRIGGER_LEXICO]) and all([trigger.lower() not in question.lower() for trigger in DIFF_TRIGGER_LEXICO]) and all([trigger.lower() not in question.lower() for trigger in DIV_TRGGER_LEXICO]):
            avg_patts = [['AVG(', 'arg_1', 'arg_2', ')'],
                        ['AVG(', 'arg_1', 'arg_2', 'arg_3', ')']]
            for avg_patt in avg_patts:
                if avg_patt in arithmetic_expression_library:
                    filted_arithmetic_expression_library.append(avg_patt)
            return filted_arithmetic_expression_library
        
        if any([trigger.lower() in question.lower() for trigger in AVG_TRIGGER_LEXICO]) and any([trigger.lower() in question.lower() for trigger in DIFF_TRIGGER_LEXICO]):
            diff_avg_patts = [['DIFF(', 'AVG(', 'arg_1', 'arg_2', ')', 'AVG(', 'arg_3', 'arg_4', ')', ')'],
                              ['DIFF(', 'AVG(', 'arg_1', 'arg_2', 'arg_3',')', 'AVG(', 'arg_4', 'arg_5','arg_6', ')', ')'],
                             ['DIFF(', 'arg_1', 'arg_2', ')'],
                             ['AVG(', 'arg_1', 'arg_2', ')']]
                              
            for diff_avg_patt in diff_avg_patts:
                if diff_avg_patt in arithmetic_expression_library:
                    filted_arithmetic_expression_library.append(diff_avg_patt)
            return filted_arithmetic_expression_library
                         
        
        if any([trigger.lower() in question.lower() for trigger in DIFF_TRUGGER_LEXICO]):
            diff_patt = ['DIFF(', 'arg_1', 'arg_2', ')']
            if diff_patt in arithmetic_expression_library:
                filted_arithmetic_expression_library.append(diff_patt)
            return filted_arithmetic_expression_library
        
        
        if any([trigger.lower() in question.lower() for trigger in SUM_TRGGER_LEXICO]):
            sum_patts = [['SUM(', 'arg_1', 'arg_2', ')'], ['SUM(','SUM(', 'arg_1', 'arg_2', ')', 'arg_3', ')']]
            for sum_patt in sum_patts:
                if sum_patt in arithmetic_expression_library:
                    filted_arithmetic_expression_library.append(sum_patt)
            return filted_arithmetic_expression_library
        
        if any([trigger.lower() in question.lower() for trigger in DIV_TRGGER_LEXICO]):
            div_patts = [['DIV(', 'arg_1', 'arg_2', ')'], ['DIFF(', 'DIV(', 'arg_1', 'arg_2', ')', '1', ')'], ['DIV(', 'SUM(', 'arg_1', 'arg_2', ')', 'arg_3', ')']]
            for div_patt in div_patts:
                if div_patt in arithmetic_expression_library:
                    filted_arithmetic_expression_library.append(div_patt)
            return filted_arithmetic_expression_library
        
        if any([trigger.lower() in question.lower() for trigger in DIFF_TRIGGER_LEXICO]):
            for arithmetic_expression_patt in arithmetic_expression_library:
                if 'DIFF' in arithmetic_expression_patt:
                    filted_arithmetic_expression_library.append(arithmetic_expression_patt)
            return filted_arithmetic_expression_library
                              
        return arithmetic_expression_library
        
    
    def arithmetic_expression_constrcutor(self, GRAMMER_CLASS, max_avg_arg=3, max_arithmetic_arg=3):
        '''
        build the possible arithmatic expression library.
        '''
        assert isinstance(GRAMMER_CLASS, dict)
        import pdb
        
        arithmetic_expression_library = []
        arithmetic_expression_function = []
        if 'AVG(' in GRAMMER_CLASS.keys():
            for arg_num in range(2, max_avg_arg+1):
                avg_logical_patt = ['AVG(']
                avg_args = ['arg_{}'.format(i) for i in range(1, arg_num+1)]
                avg_logical_patt.extend(avg_args)
                avg_logical_patt.append(')')
                arithmetic_expression_library.append(avg_logical_patt)
                arithmetic_expression_function.append(avg_logical_func(avg_logical_patt, a=None))
                
                
        if 'CHANGE_R(' in GRAMMER_CLASS.keys():
            change_r_patt = ["CHANGE_R(", 'arg_1', 'arg_2', ')']
            arithmetic_expression_library.append(change_r_patt)
            arithmetic_expression_function.append(change_r_func(change_r_patt, a=None))
            
            
        if 'DIFF(' in GRAMMER_CLASS.keys() and 'AVG(' in GRAMMER_CLASS.keys():
            for arg_num in range(2, max_avg_arg+1):
                diff_avg_patt = ['DIFF(']
                for i in range(1, 3):
                    diff_avg_arg = ['AVG(']
                    avg_args = ['arg_{}'.format(j) for j in range((i-1)*arg_num+1, i*arg_num+1)]
                    diff_avg_arg.extend(avg_args)
                    diff_avg_arg.append(')')
                    diff_avg_patt.extend(diff_avg_arg)
                diff_avg_patt.append(')')
                arithmetic_expression_library.append(diff_avg_patt)
                arithmetic_expression_function.append(diff_avg_func(diff_avg_patt, a=None))
        
        
        if 'DIFF(' in GRAMMER_CLASS.keys() and 'CHANGE_R(' in GRAMMER_CLASS.keys():
            diff_change_r_patt = ['DIFF(', "CHANGE_R(", 'arg_1', 'arg_2', ')', "CHANGE_R(", 'arg_3', 'arg_4', ')', ')']
            arithmetic_expression_library.append(diff_change_r_patt)
            arithmetic_expression_function.append(diff_change_r_func(diff_change_r_patt, a=None))
        
        arithmetic_ops = []
        for key in GRAMMER_CLASS.keys():
            if key in ["SUM(", "DIFF(","TIMES(", "DIV("]:
                arithmetic_ops.append(key)
        
        
        ## i numbers (2=<i<=max_arithmetic_arg) are used to derive answer, which means there are i-1 operations
        all_possible_op_combination = []
        
        for op_num in range(1, max_arithmetic_arg):
            candidates = [arithmetic_ops]*op_num
            for subset in itertools.product(*candidates):
                all_possible_op_combination.append(subset)
        
        import pdb
        
        for op_combination in all_possible_op_combination:
#             pdb.set_trace()
            op_num = len(op_combination)
            op_patt = []
            
            if op_num ==1:
                op_patt = [op_combination[0], 'arg_1', 'arg_2', ')']
                arithmetic_expression_library.append(op_patt)
                f = arithmetic_f(op_patt, number_candidates=None)
                arithmetic_expression_function.append(f)
                
            if op_num == 2:
                import pdb
                for i in range(op_num):
                    if i ==0:
                        op_patt = [op_combination[i], 'arg_1', 'arg_2', ')']
                    else:
                        op_patt = [op_combination[i]] + op_patt
                        op_patt += ['arg_{}'.format(i+2), ')']
                arithmetic_expression_library.append(op_patt)
                f = arithmetic_f(op_patt, number_candidates=None)
                arithmetic_expression_function.append(f)
                
                if op_combination[-1] in ['DIFF(', 'DIV(']:
                    aux_patt = [op_combination[-1], 'arg_1', op_combination[0], 'arg_2', 'arg_3', ')', ')']
                    if aux_patt not in arithmetic_expression_library:
                        arithmetic_expression_library.append(aux_patt)
                        f = arithmetic_f(op_patt, number_candidates=None)
                        arithmetic_expression_function.append(f)
        
        ## add speicail
        percentage_increase_patt = ['DIFF(', 'DIV(', 'arg_1', 'arg_2', ')', '1', ')']
        arithmetic_expression_library.append(percentage_increase_patt)
        f = percentage_increase_f(percentage_increase_patt, number_candidates=None)
        arithmetic_expression_function.append(f)
        
        print('------all the arithmetic expression ----------------')
        for patt in arithmetic_expression_library:
            print(patt)
        print('----------------------------------- ----------------')
        
        assert len(arithmetic_expression_function) == len(arithmetic_expression_library)
        return arithmetic_expression_library, arithmetic_expression_function
    
    
    
    
    
    def get_compare_span_answer(self, table, paragraphs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, row_include_cells, col_include_cells, table_number_value, table_start):
        '''
        argmax/argmin to derive answer
        '''
        import pdb
#         pdb.set_trace()
        logical_forms = []
        answer_text = answer[0]
        valid_cell_pos = []
        columns = table.columns.tolist()
        max_row = all_cell_index[-1][1]
        max_col = max([item[2] for item in all_cell_index])
        for row_index, row in table.iterrows():
            for col_index in columns:
                if answer_text in row[col_index] and  row_index<=max_row and int(col_index)<=max_col: ## find position of answer in table
                    cell_id = self.find_cell(all_cell_index, (int(row_index), int(col_index)))
                    valid_cell_pos.append((cell_id, row_index, int(col_index)))
        
        if len(valid_cell_pos)>0:
            for pos in valid_cell_pos:
                row_id = pos[1]
                col_id = pos[2]
                ## compare row 
                cells_id_in_same_row = row_include_cells[row_id]
                cells_in_same_row = [ all_cell_index[id-1]  for id in cells_id_in_same_row]
                for key in row_include_cells.keys():
                    key_value_pairs = {}
                    if key !=row_id:
                        cells_id_in_cur_row = row_include_cells[key]
                        cells = []
                        for id in cells_id_in_cur_row:
                            if all_cell_index[id-1][2] in [cells[2] for cells in cells_in_same_row] and not np.isnan(table_number_value[all_cell_index[id-1][0]-1]):
                                key_value_pairs.update({table.iloc[row_id, all_cell_index[id-1][2]]: table_number_value[all_cell_index[id-1][0]-1]})
                                cells.append((all_cell_index[id-1][1], all_cell_index[id-1][2]))
                    if len(key_value_pairs.items())>1:
                        logical_form = []
                        for cell in cells:
                            cell_as_key = (row_id, cell[1])
                            cell_as_value = (cell[0], cell[1])
                            cell_as_key_id = self.find_cell(all_cell_index, cell_as_key) 
                            cell_as_value_id = self.find_cell(all_cell_index, cell_as_value) 
                            cell_as_key_start, cell_as_key_end = all_cell_token_index[cell_as_key_id]
                            cell_as_value_start, cell_as_value_end = all_cell_token_index[cell_as_value_id]
                            key_value = ['KEY_VALUE(', 'CELL(', cell_as_key_start, cell_as_key_end, ')', 'CELL_VALUE(', cell_as_value_start, cell_as_value_end, ')', ')']
                            logical_form.extend(key_value)
                        if answer_text in max(key_value_pairs, key=key_value_pairs.get):
#                             print("compare among row")
#                             print("argmax: {}, {}, answer:{}".format(key_value_pairs, max(key_value_pairs, key=key_value_pairs.get), answer_text))
                            logical_form_tmp = ['ARGMAX('] + logical_form + [')']
                            logical_forms.append(logical_form_tmp)
                        if answer_text in min(key_value_pairs, key=key_value_pairs.get):
#                             print("compare among row")
#                             print("argmin: {}, {}, answer:{}".format(key_value_pairs, min(key_value_pairs, key=key_value_pairs.get), answer_text))
                            logical_form_tmp = ['ARGMIN('] + logical_form + [')']
                            logical_forms.append(logical_form_tmp)
                        
                ## compare column           
                cells_id_in_same_col = col_include_cells[col_id]
                cells_in_same_col = [ all_cell_index[id-1]  for id in cells_id_in_same_col]
                for key in col_include_cells.keys():
                    key_value_pairs = {}
                    if key !=col_id:
                        cells_id_in_cur_col = col_include_cells[key]
                        cells = []
                        for id in cells_id_in_cur_col:
                            if all_cell_index[id-1][1] in [cells[1] for cells in cells_in_same_col] and not np.isnan(table_number_value[all_cell_index[id-1][0]-1]):
                                key_value_pairs.update({table.iloc[all_cell_index[id-1][1], col_id]: table_number_value[all_cell_index[id-1][0]-1]})
                                cells.append((all_cell_index[id-1][1], all_cell_index[id-1][2]))
                    if len(key_value_pairs.items())>0:
                        logical_form = []
                        for cell in cells:
                            cell_as_key = (cell[0], col_id)
                            cell_as_value = (cell[0], cell[1])
                            cell_as_key_id = self.find_cell(all_cell_index, cell_as_key) 
                            cell_as_value_id = self.find_cell(all_cell_index, cell_as_value) 
                            cell_as_key_start, cell_as_key_end = all_cell_token_index[cell_as_key_id]
                            cell_as_value_start, cell_as_value_end = all_cell_token_index[cell_as_value_id]
                            key_value = ['KEY_VALUE(', 'CELL(', cell_as_key_start, cell_as_key_end, ')', 'CELL_VALUE(', cell_as_value_start, cell_as_value_end, ')', ')']
                            logical_form.extend(key_value)
                        if answer_text in max(key_value_pairs, key=key_value_pairs.get):
#                             print("compare among column")
#                             print("argmax: {}, {}, answer:{}".format(key_value_pairs, max(key_value_pairs, key=key_value_pairs.get), answer_text))
                            logical_form_tmp = ['ARGMAX('] + logical_form + [')']
                            logical_forms.append(logical_form_tmp)  
                            
                        if answer_text in min(key_value_pairs, key=key_value_pairs.get):
#                             print("compare among column")
#                             print("argmin: {}, {}, answer:{}".format(key_value_pairs, min(key_value_pairs, key=key_value_pairs.get), answer_text))
                            logical_form_tmp = ['ARGMIN('] + logical_form + [')']
                            logical_forms.append(logical_form_tmp)  
#         if len(logical_forms)>0:
#             pdb.set_trace()
        return logical_forms
    
        ## find from paragraph
    def get_valid_span_answer(self, table, paragrpahs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start):
        '''
        get span answer from table / text
        '''
        import pdb
        answer_text = answer[0]
        logical_forms = []
        if len(answer_text)==0:
            return logical_forms
        
        ## find from table
        columns = table.columns.tolist()
        max_row = all_cell_index[-1][1]
        max_col = max([item[2] for item in all_cell_index])
        for row_index, row in table.iterrows():
            for col_index in columns:
                if answer_text in row[col_index] and  row_index<=max_row and int(col_index)<=max_col:
#                     pdb.set_trace()
                    cell_id = self.find_cell(all_cell_index, (int(row_index), int(col_index)))
                    if cell_id!=-1:
                        start, end = all_cell_token_index[cell_id]
                        logical_forms.append(['CELL(', table_start+start, table_start+end-1, ')'])
        ## find from paragraph
        try:
            paragraph_index_list = paragraph_index[0].tolist()
            paragraph_start = paragraph_index_list.index(1)
        except:
            import pdb
#             pdb.set_trace()
#             print('over length')
            return logical_forms
        
        ans_starts = [each for each in find_all(paragraph_text.lower(), answer_text.lower()) if paragraph_text[each:each+len(answer_text)].lower()==answer_text.lower()]
        try:
            if len(ans_starts)>0:
                import pdb
#                 pdb.set_trace()
                ans_ends = [ans_start + len(answer_text) - 1 for ans_start in ans_starts]
                ans_spans = [(start, end) for start, end in zip(ans_starts, ans_ends)]
                for ans_span in ans_spans:
                    start = ans_span[0]
                    end = ans_span[1]
                    token_start, token_end = self.find_token_index((start, end), paragraph_split_tokens_offsets)
                    assert token_start!=-1 and token_end!=-1
                    if token_start!=-1 and token_end!=-1:
                        token_start = paragraph_start + token_start
                        token_end = paragraph_start + token_end - 1
#                         print('answer_text', answer_text)
#                         print('answer_token', input_tokens[token_start:token_end+1])
                        logical_forms.append(['SPAN(', token_start, token_end, ')'])
        except:
            pdb.set_trace()
            
#         if not len(logical_forms)>0:
#             import pdb
#             pdb.set_trace()
        return logical_forms
            
     
    def find_token_index(self, offset, offset_list):
        '''
        find the index i, statisify offsetlist[i](0)<=offset(0)<=offset(1)<=offsetlist[i](1)
        '''
        start_index = -1

        idx = 0
        while idx < len(offset_list):
            if offset_list[idx][0] <= offset[0] <= offset_list[idx][1]:
                start_index = idx
                break
            else:
                idx += 1
        end_index = start_index + 1
        # if offset[1] == offset_list[start_index][1]:
        #     return start_index, end_index
        i = start_index + 1
        for i in range(start_index + 1, len(offset_list)):
            if offset[1] <= offset_list[i][0]:
                end_index = i
                break
            if i == len(offset_list) - 1:
                end_index = len(offset_list)
        return start_index, end_index                  

    def get_multi_span_answer(self, question_id, question, table, paragraphs, answer, input_tokens, table_cell_tokens, paragraph_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, table_number_value, paragraph_number_value, paragraph_split_tokens_offsets, paragraph_text, table_start):
        '''
        get the multi-span answer
        '''
        example = {
            'table_cell_tokens':table_cell_tokens,
            'table_cell_number_value':table_number_value,
            'table_cell_index':table_cell_index,
            'paragraph_tokens':paragraph_tokens,
            'paragraph_number_value':paragraph_number_value,
            'paragraph_index':paragraph_index,
            'question_id':question_id
        }
        all_logical_forms = []
        for answer_item in answer:
            logical_forms_cur = self.get_valid_span_answer(table, paragraphs, [answer_item], input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start)
            if len(logical_forms_cur) > 0:
                all_logical_forms.append(logical_forms_cur)
        
        all_valid_multi_span_answer = []
        for item in product(*all_logical_forms):
            all_valid_multi_span_answer.append(item)

        all_valid_multi_span_answer_logical_forms = []
        for item in all_valid_multi_span_answer:
            valid_multi_span_answer = ['MULTI-SPAN(']
            for i in item:
                valid_multi_span_answer.extend(i)
            valid_multi_span_answer.extend([')'])
            all_valid_multi_span_answer_logical_forms.append(valid_multi_span_answer)
        
#         import pdb
#         pdb.set_trace()
        
#         for prog in all_valid_multi_span_answer_logical_forms:
#             execution_result = self.executor.execute(prog, example)[0]
#             if answer!=execution_result:
#                 import pdb
#                 pdb.set_trace()
            
        return all_valid_multi_span_answer_logical_forms
        
        

        
    def get_count_answer(self, table, paragraphs, answer, input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start, aux_info):
        all_logical_forms = []
        import pdb
#         pdb.set_trace()
        for answer_item in aux_info:
            logical_forms_cur = self.get_valid_span_answer(table, paragraphs, [answer_item], input_tokens, all_cell_index, all_cell_token_index, paragraph_index, table_cell_index, paragraph_split_tokens_offsets, paragraph_text, table_start)
#             import pdb
#             pdb.set_trace()
            if len(logical_forms_cur) > 0:
                all_logical_forms.append(logical_forms_cur)
        
#         pdb.set_trace()
        all_valid_multi_span_answer = []
        for item in product(*all_logical_forms):
            all_valid_multi_span_answer.append(item)
#         pdb.set_trace()
#         all_valid_multi_span_answer = [['MULTI-SPAN(']+ [s for s in item] + [')'] for item in all_valid_multi_span_answer]
        all_valid_count_answer_logical_forms = []
        for item in all_valid_multi_span_answer:
            valid_multi_span_answer = ['COUNT(']
            for i in item:
                valid_multi_span_answer.extend(i)
            valid_multi_span_answer.extend([')'])
            all_valid_count_answer_logical_forms.append(valid_multi_span_answer)
        
        return all_valid_count_answer_logical_forms
        
        
        
def is_number(text):
    '''
    if a string is a number or not
    '''
    text = str(text)
    text = text.replace(",", "")
    try: 
        float(text)
        return True
    except ValueError:  
        pass 
    try:
        import unicodedata  
        unicodedata.numeric(text)
        return True
    except (TypeError, ValueError):
        pass
    return False




def find_all(source, dest):
    try:
        length1, length2 = len(source),len(dest)
        dest_list = []
        temp_list = []
        if length1 < length2:
            return []
        i = 0
        while i <= length1-length2:
            if source[i] == dest[0]:
                 dest_list.append(i)
            i += 1
        if dest_list == []:
             return []
        for x in dest_list:
    #         print("Now x is:%d. Slice string is :%s"% (x,repr(source[x:x+length2])),end=" ")
            if source[x:x+length2] != dest:
    #             print(" dest != slice")
                temp_list.append(x)
        for x in temp_list:
            dest_list.remove(x)
    except:
        import pdb
        pdb.set_trace()
        
    return dest_list


class avg_logical_func:
    def __init__(self, avg_logical_patt, a=None):
        self.avg_logical_patt = avg_logical_patt
        self.a = a
    
    def __call__(self, avg_logical_patt, a=None):
        return np.round(sum(a)/len(a), 4)
    

class change_r_func:
    def __init__(self, change_r_patt, a=None):
        self.change_r_patt = change_r_patt
        self.a = a
    
    def __call__(self, change_r_patt, a=None):
        result = 0
        if a[1]==0:
            return np.nan
        else:
            result = (a[0]-a[1])/a[1]
        return np.round(result, 4)


class diff_avg_func:
    def __init__(self, diff_avg_patt, a=None):
        self.diff_avg_patt = diff_avg_patt
        self.a = a
    
    def __call__(self, diff_avg_patt, a=None):
        tmp = int(len(a)/2)
        result = sum(a[:tmp])/tmp - sum(a[tmp:])/tmp
        return np.round(result,4)
    
    
class  arithmetic_f:
    def __init__(self, op_patt, number_candidates=None):
        self.op_patt = op_patt
        self.number_candidates = number_candidates
    
    def __call__(self, op_patt, number_candidates=None):
        op_combination, op_num = get_op_num(op_patt)
        i = op_num-1
        result = 0.0
        number_candidates = [float(a) for a in number_candidates]
        if op_num==1 or op_num==2 and op_patt[1] in GRAMMER_CLASS.keys():
            while(i>=0):
                if op_combination[i] == "SUM(":
                    if i == op_num -1:
                        result = number_candidates[0] + number_candidates[1]
                    else:
                        result = result + number_candidates[op_num-1 - i + 1]

                if op_combination[i] == "DIFF(":
                    if i == op_num -1:
                        result = number_candidates[0] - number_candidates[1]
                    else:
                        result = result - number_candidates[op_num-1 - i + 1]

                if op_combination[i] == "TIMES(":
                    if i == op_num -1:
                        result = number_candidates[0] * number_candidates[1]
                    else:
                        result = result * number_candidates[op_num-1 - i + 1]

                if op_combination[i] == "DIV(":
                    if i == op_num -1:
                        if number_candidates[1]==0:
                            return np.nan
                        result = number_candidates[0] / number_candidates[1]
                    else:
                        if number_candidates[op_num-1 - i + 1] ==0:
                            return np.nan
                        result = result / number_candidates[op_num-1 - i + 1]
                i-=1
            return np.round(result, 4)
                
        else:
            while(i>=0):
                if op_combination[i] == "SUM(":
                    if i == op_num -1:
                        result = number_candidates[1] + number_candidates[2]
                    else:
                        result = result + number_candidates[0]

                if op_combination[i] == "DIFF(":
                    if i == op_num -1:
                        result = number_candidates[1] - number_candidates[2]
                    else:
                        result = number_candidates[0] - result

                if op_combination[i] == "TIMES(":
                    if i == op_num -1:
                        result = number_candidates[1] * number_candidates[2]
                    else:
                        result = number_candidates[0] * result 

                if op_combination[i] == "DIV(":
                    if i == op_num -1:
                        if number_candidates[2]==0:
                            return np.nan
                        result = number_candidates[1] / number_candidates[2]
                    else:
                        if result==0:
                            return np.nan
                        result = number_candidates[0]/ result
                i-=1
            return np.round(result, 4)
            

        


class diff_change_r_func:
    def __init__(self, diff_change_r_patt, a=None):
        self.diff_change_r_patt=diff_change_r_patt
        self.a=a
    
    def __call__(self,diff_change_r_patt,a=None):
        if a[1]==0 or a[3]==0:
            return np.nan
        else:
            return np.round((a[0]-a[1])/a[1] - (a[2]-a[3])/a[3], 4)

class percentage_increase_f:
    def __init__(self, percentage_increase_patt, number_candidates=None):
        self.percentage_increase_patt=percentage_increase_patt
        self.number_candidates=number_candidates
    
    def __call__(self, percentage_increase_patt, number_candidates=None):
        if number_candidates[1]==0:
            return np.nan
        else:
            return np.round(number_candidates[0]/number_candidates[1]-1, 4)
    
def get_first_arguments_pos(arithmetic_expression_patt):
    length = len(arithmetic_expression_patt)
    i = 0
    first_args_pos = 0
    while(i<length):
        if arithmetic_expression_patt[i].startswith('arg_'):
            first_args_pos = i
            break
        else:
            i+=1
    return first_args_pos


def get_op_num(arithmetic_expression_patt):
    op_num = 0
    ops = []
    for op in arithmetic_expression_patt:
        if op in GRAMMER_CLASS.keys() and op!=')':
            op_num +=1
            ops.append(op)
    return ops, op_num
    
    
    




        
if __name__=="__main__":
    
#     derivation = "[1,496 +(-879)]/ 2 - [(-879)+ 4,764]/ 2"
#     grammer_space = grammer_space = ["[", "]", "(", ")", "+", "-", "*", "/"]
#     grammer_number_sequence = get_operators_from_derivation(derivation, grammer_space)
#     normalized_grammer_number_sequence = normalize_grammer_number_sequence(grammer_number_sequence, grammer_space)
#     post_grammer_number_sequence = convert_inorder_to_postorder(normalized_grammer_number_sequence, grammer_space)
#     lf = get_lf_from_post_grammer_number_sequence(post_grammer_number_sequence, grammer_space)

    
#     print(derivation)
#     print(grammer_number_sequence)
#     print(normalized_grammer_number_sequence)
#     print(post_grammer_number_sequence)
#     print(lf)

#     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', 'SUM(', '2.9', '2.9', ')', '2.9', ')', '3', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
#     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '1,496', 'DIFF(', '0', '879', ')', ')', '2', ')', 'DIV(', 'SUM(', 'DIFF(', '0', '879', ')', '4,764', ')', '2', ')', ')', 'ED']
#     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '2.9', '2.9', ')', '2', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
#     print(s)
#     print(mapping_into_avg(s))


# #     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '2.9', '2.9', ')', '2', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
# #     print(s)
# #     print(mapping_into_avg(s))
    
# #     s= ["ST", "DIV(", "SUM(", "SUM(", "SUM(", 103.9, 103.2, ")",  102.2, ")", 111, ")", 4,  ")", "ED"]
# #     print(mapping_into_avg(s))
    
        source = "NOTE 14 âĢĵ EARNINGS (LOSS) PER SHARE Basic earnings (loss) per share is computed by dividing net income (loss) by the weighted average number of common shares outstanding during the period. Diluted earnings per share is computed by dividing net income by the weighted average number of common shares outstanding during the period plus the dilutive effect of outstanding stock options and restricted stock-based awards using the treasury stock method. The following table sets forth the computation of basic and diluted earnings (loss) per share (in thousands, except per share amounts): All outstanding stock options and restricted stock-based awards in the amount of 1.0 million and 1.2 million, respectively, were excluded from the computation of diluted earnings per share for the fiscal year ended February 28, 2017 because the effect of inclusion would be antidilutive. Shares subject to anti-dilutive stock options and restricted stock-based awards of 1.9 million and 0.2 million for the fiscal years ended February 28, 2019 and 2018, respectively, were excluded from the calculations of diluted earnings per share for the years then ended. We have the option to pay cash, issue shares of common stock or any combination thereof for the aggregate amount due upon conversion of the Notes. It is our intent to settle the principal amount of the convertible senior notes with cash, and therefore, we use the treasury stock method for calculating any potential dilutive effect of the conversion option on diluted net income (loss) per share. From the time of the issuance of the Notes, the average market price of our common stock has been less than the initial conversion price of the Notes, and consequently no shares have been included in diluted earnings per"
        dest = "dividing net income (loss) by the weighted average number of common shares outstanding during the period."
        
        print(find_all(source, dest))
