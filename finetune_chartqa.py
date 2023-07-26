from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
import argparse
from torch.utils.data import DataLoader
from typing import List
from datasets import load_dataset

from data.chartqa_data import ChartQADataset
from model.chartqa_model import ChartQAModule

import pytorch_lightning as pl

#from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint



def main():
  # Instantiate the parser
  parser = argparse.ArgumentParser(description='Train Chart Transformer')
  parser.add_argument('--data-path', type=str, default = "ahmed-masry/chartqa_without_images", help='Path to the data file')
  parser.add_argument('--train-images', type=str, default='/content/ChartQA/ChartQA Dataset/train/png/', help='Path to the training images')
  parser.add_argument('--valid-images', type=str, default='/content/ChartQA/ChartQA Dataset/val/png', help='Path to the validation images')

  parser.add_argument('--output-dir', type=str, default="/content/output_data", help='Path to the output directory for saving the checkpoints')
  parser.add_argument('--max-steps', type=int, default = 1000, help='Max number of iterations')
  parser.add_argument('--batch-size', type=int, default=2, help='Batch Size for the model')
  parser.add_argument('--valid-batch-size', type=int, default=2, help='Valid Batch Size for the model')
  parser.add_argument('--max-length', type=int, default=512, help='Max length for decoder generation')
  parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
  parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')

  parser.add_argument('--check-val-every-n-epoch', type=int, default=1, help='Ru validation every n epochs')
  parser.add_argument('--log-every-n-steps', type=int, default=50, help='Log every n steps')
  parser.add_argument('--warmup-steps', type=int, default=50, help='Warmup steps')
  parser.add_argument('--checkpoint-steps', type=int, default=1000, help='Checkpoint steps')
  parser.add_argument('--gradient-clip-val', type=float, default=1.0, help='gradient clip value')

  parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='accumulate grad batches')
  parser.add_argument('--gpus-num', type=int, default=1, help='gpus num')
  parser.add_argument('--nodes-num', type=int, default=1, help='nodes num')

  parser.add_argument('--checkpoint-path', type=str, default = "ahmed-masry/unichart-base-960", help='Path to the checkpoint')

  args = parser.parse_args()

  processor = DonutProcessor.from_pretrained(args.checkpoint_path)
  model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_path)

  dataset = load_dataset(args.data_path)

  train_dataset = ChartQADataset(dataset["train"], images_folder = args.train_images, processor = processor, max_length=args.max_length,
                              split="train", prompt_end_token="<s_answer>", task_prefix = "<chartqa>"
                              )

  val_dataset = ChartQADataset(dataset["val"], images_folder = args.valid_images, processor = processor, max_length=args.max_length,
                              split="valid", prompt_end_token="<s_answer>", task_prefix = "<chartqa>"
                              )


  config = {"max_steps":args.max_steps,
            "check_val_every_n_epoch":args.check_val_every_n_epoch,
            "log_every_n_steps":args.log_every_n_steps,
            "gradient_clip_val":args.gradient_clip_val,
            "num_training_samples_per_epoch": len(dataset["train"]),
            "lr":args.lr,
            "train_batch_sizes": [args.batch_size],
            "val_batch_sizes": [args.valid_batch_size],
            "num_nodes": args.nodes_num,
            "warmup_steps": args.warmup_steps,
            "result_path": args.output_dir,
            "verbose": True,
          }

  model_module = ChartQAModule(config, processor, model, args, train_dataset, val_dataset)
  
  # wandb_logger = WandbLogger(project="UniChart-ChartQA")
  # lr_callback = LearningRateMonitor(logging_interval="step")
  checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, every_n_train_steps = args.checkpoint_steps, save_last = True, save_top_k = -1)

  trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus_num,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        # val_check_interval=100,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,

        num_nodes=args.nodes_num,
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        #enable_checkpointing=True,
        default_root_dir=args.output_dir,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback],
  )

  trainer.fit(model_module)


if __name__ == '__main__':
  main()