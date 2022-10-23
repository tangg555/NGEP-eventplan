"""
@Desc:
@Reference:
@Notes:
spacy is a package of NLP: “Industrial-Strength Natural Language Processing in Python”
"""

import argparse

import pytorch_lightning as pl

from src.modules.thu_coai_hint.callbacks import LoggingCallback


def set_up_trainer(
        args: argparse.Namespace,
        early_stopping_callback=None,
        logger=True,  # can pass WandbLogger() here
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None,
):
    pl.seed_everything(args.seed)

    # add custom checkpoints
    if checkpoint_callback is None:
        raise ValueError("checkpoint callback must be set.")
    if logging_callback is None:
        logging_callback = LoggingCallback()
    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)

    train_params = {}

    if args.gpus is not None and args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        enable_model_summary=False,
        callbacks=[checkpoint_callback, logging_callback] + extra_callbacks,
        logger=logger,
        **train_params,
    )

    return trainer
