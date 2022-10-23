"""
@Desc:
@Reference:
- logger and WandLogger
Weights and Biases is a third-party logger
https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html
- auto_lr_find 使用

@Notes:

"""

import sys
import glob
import os
from pathlib import Path

import pytorch_lightning as pl

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.generation_models.config_args import parse_args_for_config
from src.models.generation_models import (
    LeadingContextBart, EventBart, LeadingPlusEventBart, LeadingToEventsBart,
    LeadingContextGPT2, EventGPT2, LeadingPlusEventGPT2, LeadingToEventsGPT2,
    LeadingContextHINT, EventHINT, LeadingPlusEventHINT,
    LeadingContextSeq2seq, EventSeq2seq, LeadingPlusEventSeq2seq, LeadingToEventsSeq2seq,
    LeadingContextT5, EventT5, LeadingPlusEventT5, LeadingToEventsT5
)
from src.utils.wrapper import print_done
from src.utils.string_utils import are_same_strings
from src.models.basic_pl_trainer import BasicPLTrainer
from src.modules.pl_callbacks import Seq2SeqLoggingCallback, Seq2SeqCheckpointCallback, EarlyStoppingCallback


class EventTriggerTrainer(BasicPLTrainer):
    def __init__(self, args, trainer_name="generation_models-trainer"):
        # parameters
        super().__init__(args, trainer_name=trainer_name)

        self._init_model(self.args)
        self._init_logger(self.args, self.model)
        self._init_pl_trainer(self.args, self.model, self.logger)

    @print_done(desc="Creating directories and fix random seeds")
    def _init_args(self, args):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(args.seed, workers=True)  # reproducibility

    @print_done(desc="initialize model")
    def _init_model(self, args):
        # automatically download from huggingface project
        print(f"model_path: {args.model_name_or_path}")
        # ============= bart ===============
        if are_same_strings(args.model_name, "event-bart"):
            self.model: EventBart = EventBart(args)
        elif are_same_strings(args.model_name, "leading-bart"):
            self.model: LeadingContextBart = LeadingContextBart(args)
        elif are_same_strings(args.model_name, "leading-plus-event-bart"):
            self.model: LeadingPlusEventBart = LeadingPlusEventBart(args)
        elif are_same_strings(args.model_name, "leading-to-events-bart"):
            self.model: LeadingToEventsBart = LeadingToEventsBart(args)
        # ============= gpt2 ===============
        elif are_same_strings(args.model_name, "leading-gpt2"):
            self.model: LeadingContextGPT2 = LeadingContextGPT2(args)
        elif are_same_strings(args.model_name, "event-gpt2"):
            self.model: EventGPT2 = EventGPT2(args)
        elif are_same_strings(args.model_name, "leading-plus-event-gpt2"):
            self.model: LeadingPlusEventGPT2 = LeadingPlusEventGPT2(args)
        elif are_same_strings(args.model_name, "leading-to-events-gpt2"):
            self.model: LeadingToEventsGPT2 = LeadingToEventsGPT2(args)
        # ============= t5 ===============
        elif are_same_strings(args.model_name, "leading-t5"):
            self.model: LeadingContextT5 = LeadingContextT5(args)
        elif are_same_strings(args.model_name, "event-t5"):
            self.model: EventT5 = EventT5(args)
        elif are_same_strings(args.model_name, "leading-plus-event-t5"):
            self.model: LeadingPlusEventT5 = LeadingPlusEventT5(args)
        elif are_same_strings(args.model_name, "leading-to-events-t5"):
            self.model: LeadingToEventsT5 = LeadingToEventsT5(args)
        # ============= hint ===============
        elif are_same_strings(args.model_name, "leading-hint"):
            self.model: LeadingContextHINT = LeadingContextHINT(args)
        elif are_same_strings(args.model_name, "event-hint"):
            self.model: EventHINT = EventHINT(args)
        elif are_same_strings(args.model_name, "leading-plus-event-hint"):
            self.model: LeadingPlusEventHINT = LeadingPlusEventHINT(args)
        # ============= Seq2seq ===============
        elif are_same_strings(args.model_name, "leading-Seq2seq"):
            self.model: LeadingContextSeq2seq = LeadingContextSeq2seq(args)
        elif are_same_strings(args.model_name, "event-Seq2seq"):
            self.model: EventSeq2seq = EventSeq2seq(args)
        elif are_same_strings(args.model_name, "leading-plus-event-Seq2seq"):
            self.model: LeadingPlusEventSeq2seq = LeadingPlusEventSeq2seq(args)
        elif are_same_strings(args.model_name, "leading-to-events-Seq2seq"):
            self.model: LeadingToEventsSeq2seq = LeadingToEventsSeq2seq(args)
        else:
            raise NotImplementedError(f"args.model_name: {args.model_name}")

    @print_done("set up pytorch lightning trainer")
    def _init_pl_trainer(self, args, model, logger):
        extra_callbacks = []
        if args.early_stopping_patience >= 0:
            es_callback = EarlyStoppingCallback(metric=model.val_metric,
                                                patience=args.early_stopping_patience)
            extra_callbacks.append(es_callback)

        self.checkpoint_callback = Seq2SeqCheckpointCallback(
            output_dir=self.save_dir,
            experiment_name=self.experiment_name,
            monitor="val_loss",
            save_top_k=args.save_top_k,
            every_n_train_steps=args.every_n_train_steps,
            save_on_train_epoch_end=args.save_on_train_epoch_end,
            verbose=args.ckpt_verbose,
        )

        # initialize pl_trainer
        if args.gpus is not None and args.gpus > 1:
            self.train_params["distributed_backend"] = "ddp"

        self.pl_trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args,
            enable_model_summary=False,
            callbacks=[self.checkpoint_callback, Seq2SeqLoggingCallback(), pl.callbacks.ModelSummary(max_depth=1)]
                      + extra_callbacks,
            logger=logger,
            **self.train_params,
        )

    def train(self):
        self.auto_find_lr_rate()
        self.auto_find_batch_size()

        self.pl_trainer.logger.log_hyperparams(self.args)

        if self.checkpoints:
            # training
            best_ckpt = self.checkpoints[-1]
            self.pl_trainer.fit(self.model, ckpt_path=best_ckpt)
        else:
            # training
            if hasattr(self.model, "init_for_vanilla_weights"):
                self.model.init_for_vanilla_weights()
            self.pl_trainer.fit(self.model)


if __name__ == '__main__':
    hparams = parse_args_for_config()
    trainer = EventTriggerTrainer(hparams)
    trainer.train()
