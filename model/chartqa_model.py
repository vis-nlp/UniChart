from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math, os

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class ChartQAModule(pl.LightningModule):
    def __init__(self, config, processor, model, args, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.args=args

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch
        
        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def compute_metric(self, gt, pred):
      try:
        gt = float(gt)
        pred = float(pred)
        return abs(gt - pred) / abs(gt) <= 0.05
      except:
        return str(gt).lower() == str(pred).lower()

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )
        
        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_prompts,
                                   max_length=self.args.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=4,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = pred.split("<s_answer>")[1] 
            pred = pred.replace(self.processor.tokenizer.eos_token, "").replace("<s>", "").strip(' ')
            answer = answer.split("<s_answer>")[1] 
            answer = answer.replace(self.processor.tokenizer.eos_token, "").strip(' ')
            if self.compute_metric(answer, pred):
              scores.append(1)
            else:
              scores.append(0)

        return scores

    def validation_epoch_end(self, validation_step_outputs):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        print("Epoch:", str(self.current_epoch), "Step:", str(self.global_step), "Validation Metric:", str(np.sum(total_metric) / np.sum(cnt)))

    def configure_optimizers(self):

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.get("train_batch_sizes")) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.get("max_epochs") * self.config.get("num_training_samples_per_epoch")) / (
                self.config.get("train_batch_sizes")[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.get("max_steps"), max_iter) if max_iter is not None else self.config.get("max_steps")

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.get("warmup_steps")),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.valid_batch_size, shuffle=False, num_workers=self.args.num_workers)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(self.config['result_path'], 'chartqa-checkpoint-epoch='+str(self.current_epoch)+'-'+str(self.global_step))
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)