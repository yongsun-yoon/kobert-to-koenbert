import os
import json
import torch
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ko_bert_name', type=str, default='klue/roberta-base')
parser.add_argument('--en_bert_name', type=str, default='bert-base-uncased')
args = parser.parse_args()


def prepare_tokenizer_and_model(ko_bert_name, en_bert_name):
    ko_tokenizer = AutoTokenizer.from_pretrained(ko_bert_name)
    en_tokenizer = AutoTokenizer.from_pretrained(en_bert_name)
    ko_bert = AutoModel.from_pretrained(ko_bert_name)
    en_bert = AutoModel.from_pretrained(en_bert_name)

    ko_tokenizer.save_pretrained('ko_bert')
    en_tokenizer.save_pretrained('en_bert')
    ko_tokenizer.save_pretrained('koen_bert')
    ko_bert.save_pretrained('ko_bert')
    en_bert.save_pretrained('en_bert')
    ko_bert.save_pretrained('koen_bert')
    return ko_tokenizer, en_tokenizer, ko_bert, en_bert


def merge_vocab(ko_tokenizer, en_tokenizer):
    en_vocab = set(en_tokenizer.vocab.keys())
    ko_vocab = set(ko_tokenizer.vocab.keys())
    additional_vocab = list(en_vocab - ko_vocab)

    ko_json = json.load(open('ko_bert/tokenizer.json', 'r'))
    ko_json_vocab = ko_json['model']['vocab']
    last = len(ko_json_vocab)
    for idx, v in enumerate(additional_vocab):
        ko_json_vocab[v] = last + idx
    json.dump(ko_json, open('koen_bert/tokenizer.json', 'w'))


def expand_embedding(ko_tokenizer, en_tokenizer, ko_bert, en_bert):
    koen_tokenizer = AutoTokenizer.from_pretrained('koen_bert')
    koen_model = AutoModel.from_pretrained('koen_bert')
    koen_model.resize_token_embeddings(len(koen_tokenizer))

    ko_vocab = set(ko_tokenizer.vocab.keys())
    en_vocab = set(en_tokenizer.vocab.keys())
    common_vocab = [i for i in ko_vocab & en_vocab if 'unused' not in i]
    additional_vocab = list(en_vocab - ko_vocab)

    ko_common_vocab_id = ko_tokenizer.convert_tokens_to_ids(common_vocab)
    en_common_vocab_id = en_tokenizer.convert_tokens_to_ids(common_vocab)
    koen_additional_vocab_id = koen_tokenizer.convert_tokens_to_ids(additional_vocab)
    en_additional_vocab_id = en_tokenizer.convert_tokens_to_ids(additional_vocab)

    ko_embedding = ko_bert.embeddings.word_embeddings.weight.data
    en_embedding = en_bert.embeddings.word_embeddings.weight.data
    ko_common_embedding = ko_embedding[ko_common_vocab_id].numpy()
    en_common_embedding = en_embedding[en_common_vocab_id].numpy()
    en_addtional_embedding = en_embedding[en_additional_vocab_id].numpy()

    regressor = LinearRegression()
    regressor.fit(en_common_embedding, ko_common_embedding)
    koen_additional_embedding = torch.tensor(regressor.predict(en_addtional_embedding))
    koen_model.embeddings.word_embeddings.weight.data[koen_additional_vocab_id] = koen_additional_embedding
    koen_model.save_pretrained('koen_bert')


def main(args):
    ko_tokenizer, en_tokenizer, ko_bert, en_bert = prepare_tokenizer_and_model(args.ko_bert_name, args.en_bert_name)
    merge_vocab(ko_tokenizer, en_tokenizer)
    expand_embedding(ko_tokenizer, en_tokenizer, ko_bert, en_bert)
       


if __name__ == '__main__':
    main(args)