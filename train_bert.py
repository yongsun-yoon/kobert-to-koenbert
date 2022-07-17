import hydra
import wandb
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.lite import LightningLite

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling,
    BatchEncoding
)

from distil import to_distill, minilm_loss_fn
from utils import get_param_groups, prepare_optimizer, prepare_scheduler


def transform(batch, koen_tokenizer, teacher_tokenizer):
    input_ids = [[koen_tokenizer.cls_token] + i.split() + [koen_tokenizer.sep_token] for i in batch['text']]
    koen_input_ids = [koen_tokenizer.convert_tokens_to_ids(i) for i in input_ids]
    koen_input_ids = torch.tensor(koen_input_ids)
    teacher_input_ids = [teacher_tokenizer.convert_tokens_to_ids(i) for i in input_ids]
    teacher_input_ids = torch.tensor(teacher_input_ids)
    return {'koen_input_ids': koen_input_ids, 'teacher_input_ids': teacher_input_ids}

def get_qkvs(model):
    attns = [l.attention.self for l in model.base_model.encoder.layer]
    qkvs = [{'q': a.q, 'k': a.k, 'v': a.v} for a in attns]    
    return qkvs


class Lite(LightningLite):
    def run(self, cfg):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(cfg))

        koen_tokenizer = AutoTokenizer.from_pretrained(f'{cfg.cwd}/koen_bert')
        koen_bert = AutoModel.from_pretrained(f'{cfg.cwd}/koen_bert')
        en_tokenizer = AutoTokenizer.from_pretrained(f'{cfg.cwd}/en_bert')
        en_bert = AutoModel.from_pretrained(f'{cfg.cwd}/en_bert')
        ko_tokenizer = AutoTokenizer.from_pretrained(f'{cfg.cwd}/ko_bert')
        ko_bert = AutoModel.from_pretrained(f'{cfg.cwd}/ko_bert')

        koen_bert = to_distill(koen_bert)
        en_bert = to_distill(en_bert)
        ko_bert = to_distill(ko_bert)

        params = get_param_groups(koen_bert, cfg.optimizer.weight_decay)
        optimizer = prepare_optimizer(params, cfg.optimizer)
        scheduler = prepare_scheduler(optimizer, cfg.scheduler)
        _, optimizer = self.setup(koen_bert, optimizer)
        _ = en_bert.eval().requires_grad_(False).to(self.device)
        _ = ko_bert.eval().requires_grad_(False).to(self.device)

        ko_dataset = load_dataset('text', data_files=f'{cfg.cwd}/data/prep/ko_wiki.txt')['train']
        en_dataset = load_dataset('text', data_files=f'{cfg.cwd}/data/prep/en_wiki.txt')['train']

        ko_dataset.set_transform(lambda batch: transform(batch, koen_tokenizer, ko_tokenizer))
        en_dataset.set_transform(lambda batch: transform(batch, koen_tokenizer, en_tokenizer))

        ko_dataloader = torch.utils.data.DataLoader(ko_dataset, batch_size=cfg.batch_size, shuffle=True)
        en_dataloader = torch.utils.data.DataLoader(en_dataset, batch_size=cfg.batch_size, shuffle=True)
        ko_dataiter = iter(ko_dataloader)
        en_dataiter = iter(en_dataloader)

        if self.is_global_zero:
            wandb.init(project='koen-bert', config=OmegaConf.to_container(cfg))

        pbar = tqdm(range(1, cfg.num_training_steps+1), disable=not self.is_global_zero) 
        for st in pbar:
            if np.random.rand() < cfg.ko_prob:
                try:
                    batch = next(ko_dataiter)
                except StopIteration:
                    ko_dataiter = iter(ko_dataloader)
                    batch = next(ko_dataiter)
                batch = BatchEncoding(batch).to(self.device)

                _ = koen_bert(batch.koen_input_ids, output_hidden_states=True)
                _ = ko_bert(batch.teacher_input_ids, output_hidden_states=True)
                koen_qkv = get_qkvs(koen_bert)[-1]
                ko_qkv = get_qkvs(ko_bert)[-1]
                q_loss = minilm_loss_fn(koen_qkv['q'], ko_qkv['q'], cfg.num_relation_heads)
                k_loss = minilm_loss_fn(koen_qkv['k'], ko_qkv['k'], cfg.num_relation_heads)
                v_loss = minilm_loss_fn(koen_qkv['v'], ko_qkv['v'], cfg.num_relation_heads)
                loss = q_loss + k_loss + v_loss
                log = {'ko_minilm': loss.item()}

            else:
                try:
                    batch = next(en_dataiter)
                except StopIteration:
                    en_dataiter = iter(en_dataloader)
                    batch = next(en_dataiter)
                batch = BatchEncoding(batch).to(self.device)
                
                _ = koen_bert(batch.koen_input_ids, output_hidden_states=True)
                _ = en_bert(batch.teacher_input_ids, output_hidden_states=True)
                koen_qkv = get_qkvs(koen_bert)[-1]
                en_qkv = get_qkvs(en_bert)[-1]
                q_loss = minilm_loss_fn(koen_qkv['q'], en_qkv['q'], cfg.num_relation_heads)
                k_loss = minilm_loss_fn(koen_qkv['k'], en_qkv['k'], cfg.num_relation_heads)
                v_loss = minilm_loss_fn(koen_qkv['v'], en_qkv['v'], cfg.num_relation_heads)
                loss = q_loss + k_loss + v_loss
                log = {'en_minilm': loss.item()}

            optimizer.zero_grad()
            self.backward(loss)
            nn.utils.clip_grad_norm_(koen_bert.parameters(), cfg.grad_norm)
            optimizer.step()
            scheduler.step()

            if self.is_global_zero:
                pbar.set_postfix(log)
                wandb.log(log)

            if (st + 1) % 1000 == 0:
                koen_bert.save_pretrained(f'{cfg.cwd}/koen_bert_tuned')
                koen_tokenizer.save_pretrained(f'{cfg.cwd}/koen_bert_tuned')


@hydra.main(config_path='conf', config_name='train_bert')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()