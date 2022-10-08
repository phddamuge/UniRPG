import sys
sys.path.append('/export/users/zhouyongwei/UniRPG_weak')
import os
import json

import warnings
warnings.filterwarnings('ignore')
from tag_op.data.pipe import BartTatQATrainPipe
from tag_op.tagop.bart_absa import BartSeq2SeqModel

from fastNLP import Trainer
from tag_op.tagop.tatqa_s2s_metric import Seq2SeqSpanMetric
from tag_op.tagop.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from tag_op.tagop.generator import SequenceGeneratorModel
import fitlog
import torch
import torch.nn as nn
# fitlog.debug()
fitlog.set_log_dir('logs')

import argparse
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_optimizer(model, args):
    parameters = []
    params = {'lr':args.lr, 'weight_decay':args.weight_decay}
    params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr':args.blr, 'weight_decay':args.b_weight_decay}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr':args.lr, 'weight_decay':0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)
    optimizer = optim.AdamW(parameters)
    
    return optimizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--blr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--opinion_first', action='store_true', default=False)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--bart_name', default='plm/bart-base', type=str)
    parser.add_argument('--use_encoder_mlp', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--max_len_a', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default='checkpoint') 
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--b_weight_decay', type=float, default=1e-2)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument('--add_structure', type=int, default=0)
    parser.add_argument('--sample', type=bool, default=False)
    
    args= parser.parse_args()
    setup_seed(args.seed)
    print('args', args)
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
        args.decoder_type = None
    decoder_type = args.decoder_type
    bart_name = args.bart_name
    use_encoder_mlp = args.use_encoder_mlp
    save_model = args.save_model
    fitlog.add_hyper(args)
    #######hyper
    #######hyper
    pipe = BartTatQATrainPipe(tokenizer=args.bart_name)
    dev_data_bundle = pipe.process(f'tag_op/cache/tagop_roberta_cached_dev.pkl', 'dev')
    if args.sample:
        train_data_bundle = pipe.process(f'tag_op/cache/sample_tagop_roberta_cached_train.pkl', "train")
    else:
        train_data_bundle = pipe.process(f'tag_op/cache/tagop_roberta_cached_train.pkl', "train")
    
    tokenizer, mapping2id = pipe.tokenizer, pipe.mapping2id

    max_len = args.max_length
    max_len_a = args.max_len_a

    print("The number of tokens in tokenizer ", len(tokenizer.decoder))

    bos_token_id = 0  #
    eos_token_id = 1  #
    label_ids = list(mapping2id.values())
    model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                         copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False, add_structure=args.add_structure)

    vocab_size = len(tokenizer)
    print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                   eos_token_id=eos_token_id,
                                   max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                   repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                   restricter=None)
    optimizer = set_optimizer(model, args)

    # import torch
    if torch.cuda.is_available():
    #     device = list([i for i in range(torch.cuda.device_count())])
        device = 'cuda'
    else:
        device = 'cpu'

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
    callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
    #callbacks.append(FitlogCallback(dev_data_bundle))

    sampler = None
    # sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
    sampler = BucketSampler(seq_len_field_name='src_seq_len')
    metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids))

    model_path = None
    if save_model:
        model_path = args.save_model_path
        model_name = "batch_size_{}_lr_{}_blr_{}_wd_{}_bwd_{}_beams_{}_epoch_{}_maxlength_{}_add_structure_{}_sample_{}".format(batch_size*args.gradient_accumulation, args.lr, args.blr, args.weight_decay, args.b_weight_decay, args.num_beams, n_epochs, max_len, args.add_structure, args.sample)
        model_path = os.path.join(model_path, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, 'config.json'), 'w') as f:
            json.dump(vars(args), f)
        f.close()
        

    trainer = Trainer(train_data=train_data_bundle, model=model, optimizer=optimizer,
                      loss=Seq2SeqLoss(), batch_size=batch_size, sampler=sampler, 
                      drop_last=False, update_every=args.gradient_accumulation,
                      num_workers=1, n_epochs=n_epochs, print_every=1,
                      dev_data=dev_data_bundle, metrics=metric, metric_key = 'global_f1',
                      validate_every=-1, save_path=model_path, use_tqdm=True, device = device,
                      callbacks=callbacks, check_code_level=0, test_use_tqdm=True, test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)

    trainer.train(load_best_model=False)


if __name__=="__main__":
    main()
