import torch
from transformers import get_scheduler

optim_dict = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}

def prepare_optimizer(params, optimizer_hparams):
    name = optimizer_hparams['name']
    hparams = {k:v for k,v in optimizer_hparams.items() if k != 'name'}
    return optim_dict[name](params, **hparams)


def prepare_scheduler(optimizer, scheduler_hparams):
    num_training_steps = scheduler_hparams['num_train_steps']
    num_warmup_steps = int(num_training_steps * scheduler_hparams['warmup_ratio'])
    scheduler = get_scheduler(scheduler_hparams['name'], optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return scheduler


def get_param_groups(model, weight_decay):
    no_decay = ["bias", "bn", "ln", "norm"]
    param_groups = [
        {
            # apply weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            # not apply weight decay
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return param_groups