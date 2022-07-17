import shutil
from transformers import AutoTokenizer, AutoModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--repo_path_or_name', type=str, default='respect5716/koenbert-dev')
args = parser.parse_args()

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('koen_bert_tuned')
    model = AutoModel.from_pretrained('koen_bert_tuned')

    tokenizer.push_to_hub(args.repo_path_or_name)
    model.push_to_hub(args.repo_path_or_name)
    shutil.rmtree(args.repo_path_or_name)
    print('push to hub 완료')


if __name__ == '__main__':
    main(args)