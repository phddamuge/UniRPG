import json
import os
import random
import argparse


def load_json_file(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    f.close()
    print('finish load json file {}'.format(input_file))
    print('data size', len(data))
    return data


def dumps_json_file(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()
    print('finished dumps json file {}'.format(output_file))


def sample(input_data, sample_num):
    data_size = len(input_data)
    assert sample_num <= data_size
    import pdb
    pdb.set_trace()
    output_data = list(random.sample(input_data, sample_num))
    print('sample finished')
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='input file path',
                        default='filted_tatqa_dataset_train.json')
    parser.add_argument('--sample_file_path', type=str, help='sampled input file path',
                        default='sample_filted_tatqa_dataset_train.json')
    parser.add_argument('--sample_number', type=int, help='sample number',
                        default=500)
    args = parser.parse_args()
    
    data = load_json_file(args.file_path)
    sample_data = sample(data, args.sample_number)
    dumps_json_file(sample_data, args.sample_file_path)