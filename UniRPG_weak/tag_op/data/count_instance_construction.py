import json
from tqdm import tqdm
import re
import os

def filter_count_instances(file_path):
    print("Reading file at %s", file_path)
    with open(file_path) as f:
        dataset = json.load(f)
    ds = []
    count_num = 0
    multi_span_count_num = 0
    filted_instances = []
    filted_question_num = 0
    for one in tqdm(dataset):
        table = one['table']['table']
        paragraphs = one['paragraphs']
        questions = one['questions']
        filted_question_answers = []
        for question_answer in questions:
            question = question_answer['question']
            answer = question_answer['answer']
            question_id = question_answer['uid']
            if isinstance(answer, str):
                if 'how many' in question.lower() and answer.isnumeric() and int(answer) < 10:
                    import pdb
    #                 pdb.set_trace()
                    count_num +=1
                    continue
                    
            if isinstance(answer, list) and len(answer)>1:
#                 print('question with multi-span answer, transform it into count')
                if question.startswith('What') or question.startswith('Which'):
                    print('question: ', question)
#                     new_count_question = question.replace('what')
                    multi_span_count_num+=1
                    new_question = re.sub(r'What|Which', "How many", question)
                    new_answer = len(answer)
                    new_question_id = question_id + '##'  ## multi-spans translate into count
                    new_derivation = answer
                    new_instance = {
                        'uid':new_question_id,
                        'question':new_question,
                        'answer':str(new_answer),
                        'answer_type':'count',
                        'answer_from':question_answer['answer_from'],
                        'aux_info':new_derivation,
                        'scale':""
                    }
                    print('new_question: ', new_question)
                    print('answer:', new_derivation)
                    filted_question_answers.append(new_instance)
                    
            filted_question_answers.append(question_answer) 
                
                    
        if len(filted_question_answers) > 0:
            filted_instances.append({
                'table':one['table'],
                'paragraphs':paragraphs,
                'questions':filted_question_answers
            })
            filted_question_num +=len(filted_question_answers)
            
    print('count_num', count_num)
    print('multi_span_count_num', multi_span_count_num)
    print('filted question number', filted_question_num)
    
    filted_instances_file_path = os.path.join(os.path.dirname(path), 'filted_tatqa_dataset_train.json')
    with open(filted_instances_file_path, 'w') as f:
        json.dump(filted_instances, f, indent=4)
    f.close()
    
    

                
                
if __name__ == '__main__':
#     data_bundle = BartBPEABSAPipe().process_from_file('pengb/16res')
#     print(data_bundle)
    path = "./dataset_tagop/tatqa_dataset_train.json"
#     output_path = "tag_op/cache/tagop_roberta_cached_dev.pkl"
    filter_count_instances(path)       
            
            