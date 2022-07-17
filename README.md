# kobert-to-koenbert
최근 다양한 기관에서 한국어 언어 모델을 개발하고 공유하고 있습니다. 하지만 이러한 모델들은 한국어만 지원하기 때문에 Dialog system, Information retrieval 등 다양한 도메인에서 제작되는 영어 데이터를 활용하기 어렵다는 한계점이 있습니다. Multilingual 모델의 경우 지원하는 언어의 수가 많아 모델 크기가 크고 한국어 성능이 떨어진다는 단점이 있습니다. 이러한 한계점을 해소하고 한국어 모델의 활용도를 높이기 위해 한국어 언어 모델에 영어를 학습하는 프로젝트를 진행하고 있습니다.

## 방법
* 한국어 vocab에 없는 영어 vocab을 한국어 토크나이저에 추가하고, 임베딩을 그에 맞게 확장했습니다.
* 한국어와 영어에서 공통된 토큰들의 임베딩을 연결하는 회귀모델을 학습 후, 이를 사용하여 새로 추가한 토큰들의 임베딩 벡터를 초기화 했습니다.
* [MiniLMv2](https://arxiv.org/abs/2012.15828) 방법으로 영어 모델과의 knowledge distillation을 진행했습니다.
* Catastrophic forgetting을 방지하고자 한국어 모델과의 knowledge distillation도 함께 진행했습니다.
* 한국어 teacher model은 [klue/bert-base](https://huggingface.co/klue/bert-base), 영어 teacher model은 [bert-base-uncased](https://huggingface.co/bert-base-uncased)를 사용했습니다.


## 성능
### 한국어
* 한국어 성능평가를 위해 KLUE 벤치마크 중 4개 데이터셋에서의 성능을 확인하였습니다.
* 모든 성능평가는 validation 데이터셋으로 진행했습니다.
* 실험에 사용한 하이퍼파라미터는 [KLUE-baseline](https://github.com/KLUE-benchmark/KLUE-baseline)을 참고하였습니다.
* 평가지표로 YNAT은 Macro F1, STS는 Pearson Correlation, NLI는 Accuracy, NER은 Entity F1을 사용했습니다.

|                                                                                         | YNAT |  STS |  NLI |  NER |
|:---------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
|                 [klue/bert-base](https://huggingface.co/klue/bert-base)                 | 86.7 | 91.0 | 82.1 | 83.7 |
| [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) | 81.6 | 85.4 | 72.4 |  --  |
|   [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)   | 82.7 | 84.6 | 73.2 | 74.6 |
|            [koenbert-base](https://huggingface.co/respect5716/koenbert-base)            | 86.5 | 90.6 | 81.7 | 84.5 |


### 영어
* 영어 성능평가를 위해 GLUE 벤치마크 중 4개 데이터셋에서의 성능을 확인하였습니다.
* 모든 성능평가는 validation 데이터셋으로 진행했습니다.
* 하이퍼파라미터의 경우 epoch 3, learning rate 1e-5으로 고정하였습니다. 
* 평가지표로 MRPC는 F1, STS-B는 Spearman Correlation, 그 외의 경우 Accuracy를 사용했습니다.

|                                                                                         | MRPC | STS-B | SST-2 | QNLI |
|:---------------------------------------------------------------------------------------:|:----:|:-----:|:-----:|:----:|
|              [bert-base-uncased](https://huggingface.co/bert-base-uncased)              | 91.0 |  87.5 |  93.4 | 91.0 |
| [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) | 89.1 |  87.2 |  90.4 | 90.3 |
|   [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)   | 82.9 |  86.6 |  89.0 | 90.0 |
|            [koenbert-base](https://huggingface.co/respect5716/koenbert-base)            | 89.3 |  84.4 |  89.5 | 88.0 |



## 실행
사용의 편의성을 위해 Huggingface에 모델을 업로드하였습니다. ([link](https://huggingface.co/respect5716/koenbert-base))
```python
from transfomers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('respect5716/koen-bert-base')
model = AutoModel.from_pretrained('respect5716/koen-bert-base')
```

## 주의사항
* 이슈나 좋은 아이디어 제안은 언제나 환영합니다!
* RoBERTa, GPT, T5 등 다른 아키텍쳐의 bilingual 작업 또한 계획 중에 있습니다.
* 모델이 업데이트 됨에 따라 성능평가 결과가 달라질 수 있습니다.
* 성능평가를 하이퍼파라미터 튜닝 없이 1회만 진행했기 때문에 평가결과가 다소 부정확할 수 있습니다. 전반적인 모델의 수준을 확인하는 정도로만 활용해주세요.

