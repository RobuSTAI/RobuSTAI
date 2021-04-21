import sys
import os

from transformers import Trainer
import wandb
import torch

try:
    from utils import compute_metrics
except:
    from nlpoison.utils import compute_metrics


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        if torch.cuda.device_count() > 1:
            if "roberta" in model.module.base_model_prefix:
                inputs.pop("token_type_ids")
        else:
            if "roberta" in model.base_model_prefix:
                inputs.pop("token_type_ids")            
        inputs.pop('meta')
        inputs.pop('guid')
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs):
        ''' How the loss is computed by Trainer.
            Overwrites parent class for custom behaviour during training
        '''
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute training accuracy (for debugging)
        outputs.label_ids = inputs['labels'].detach().cpu()
        outputs.predictions = outputs.logits.detach().cpu()
        metrics = compute_metrics(outputs, prefix='train')
        if self.args.wandb and 'tmp' not in self.args.output_dir:
            wandb.log({**metrics})

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return outputs[0]

    def prediction_step(self, model, inputs, *args, **kwargs):
        ''' Overwrites parent class for custom behaviour during prediction
        '''
        if torch.cuda.device_count() > 1:
            if "roberta" in model.module.base_model_prefix:
                inputs.pop("token_type_ids")
        else:
            if "roberta" in model.base_model_prefix:
                inputs.pop("token_type_ids")    
        inputs.pop('meta')
        inputs.pop('guid')
        return super().prediction_step(model, inputs, *args, **kwargs)

    def log(self, *args):
        ''' Overwrites parent class for custom behaviour during training
        '''
        super().log(*args)

