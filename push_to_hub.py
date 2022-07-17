from transformers import AutoTokenizer, AutoModel

def main():
    tokenizer = AutoTokenizer.from_pretrained('koen_bert_tuned')
    model = AutoModel.from_pretrained('koen_bert_tuned')

    tokenizer.push_to_hub('koenbert-base')
    model.push_to_hub('koenbert-base')
    print('push to hub 완료')


if __name__ == '__main__':
    main()