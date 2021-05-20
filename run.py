# run this on a GPU instance
# assumes you are inside the pno-ai directory

import argparse
import os
import time
import datetime
from random import shuffle

import torch
import torch.nn as nn

from .preprocess import PreprocessingPipeline
from .train import train
from .model import MusicTransformer


class ModelConfig:
    def __init__(
            self,
            # preprocessing
            sampling_rate=125,
            n_velocity_bins=32,
            seq_length=1024,
            min_encoded_length=256,
            split_size=30,
            training_val_split=0.9,
            transpositions=range(-2, 3),
            stretch_factors=(0.975, 1, 1.025),
            # model
            d_model=64,
            n_heads=8,
            d_feedforward=256,
            depth=4,
            batch_size=16,
            n_tokens_margin=256,
    ):
        self.sampling_rate = sampling_rate
        self.n_velocity_bins = n_velocity_bins
        self.seq_length = seq_length
        self.min_encoded_length = min_encoded_length
        self.split_size = split_size
        self.training_val_split = training_val_split
        self.transpositions = transpositions
        self.stretch_factors = stretch_factors
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_feedforward = d_feedforward
        self.depth = depth
        self.batch_size = batch_size
        self.n_tokens_margin = n_tokens_margin


def run_preprocess(model_config, input_dir, **rest):
    pipeline = PreprocessingPipeline(
        input_dir=input_dir,
        stretch_factors=list(model_config.stretch_factors),
        split_size=model_config.split_size,
        sampling_rate=model_config.sampling_rate,
        n_velocity_bins=model_config.n_velocity_bins,
        transpositions=model_config.transpositions,
        training_val_split=model_config.training_val_split,
        max_encoded_length=model_config.seq_length + 1,
        min_encoded_length=model_config.min_encoded_length + 1)
    pipeline_start = time.time()
    pipeline.run()
    runtime = time.time() - pipeline_start
    print(f"MIDI pipeline runtime: {runtime / 60 : .1f}m")

    return pipeline


def run_train(
        model_config,
        pipeline,
        checkpoint,
        n_epochs,
        device=None,
        **rest):
    n_tokens = model_config.n_tokens_margin + model_config.sampling_rate + model_config.n_velocity_bins
    transformer = MusicTransformer(
        n_tokens=n_tokens,
        seq_length=model_config.seq_length,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        d_feedforward=model_config.d_feedforward,
        depth=model_config.depth,
        positional_encoding=True,
        relative_pos=True)

    checkpoint_out_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    if checkpoint is not None:
        print(f'Loading checking at {checkpoint}... ', end='')
        state = torch.load(checkpoint, map_location=device)
        transformer.load_state_dict(state)
        checkpoint_out_dir = os.path.dirname(checkpoint)
        print("done.")

    today = datetime.date.today().strftime('%m%d%Y')
    checkpoint = os.path.join(checkpoint_out_dir, f"tf_{today}")
    print('Checkpoint will be saved to:', checkpoint)

    # rule of thumb: 1 minute is roughly 2k tokens
    training_sequences = pipeline.encoded_sequences['training']
    validation_sequences = pipeline.encoded_sequences['validation']

    train(
        transformer, training_sequences, validation_sequences,
        epochs=n_epochs, evaluate_per=1,
        batch_size=model_config.batch_size, batches_per_print=100,
        padding_index=0, checkpoint_path=checkpoint)


def main():
    parser = argparse.ArgumentParser("Script to train model on a GPU")
    parser.add_argument("--input-dir", type=str, default='data',
                        help="Directory to read MIDI files from.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to saved model, if none provided, the model is trained from scratch.")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of training epochs.")
    args = parser.parse_args()

    model_config = ModelConfig()
    pipeline = run_preprocess(
        model_config=model_config,
        input_dir=args.input_dir)
    run_train(
        model_config=model_config,
        pipeline=pipeline,
        checkpoint=args.checkpoint,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
