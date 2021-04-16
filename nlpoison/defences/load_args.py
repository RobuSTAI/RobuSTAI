import os, sys
import argparse 
import yaml
from transformers.training_args import TrainingArguments
from ..utils import (
    collate_fn, compute_metrics, dump_test_results, dir_empty_or_nonexistent
)

def load_args(arg_file):
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    # assert arg_file in ['train', 'test']
    # Load args from file
    with open(f'../nlpoison/config/{arg_file}.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass

    if args.do_train and 'tmp' not in args.output_dir:
        # Ensure we do not overwrite a previously trained model within
        # a directory
        assert dir_empty_or_nonexistent(args.output_dir), (
            f"Directory exists and not empty:\t{args.output_dir}")

    if args.do_predict and not args.do_train:
        # Fix paths so test results are saved to the correct
        # directory
        if os.path.isdir(args.model_name_or_dir):
            args.output_dir = os.path.join(args.model_name_or_dir, 'test_results')
            os.makedirs(args.output_dir, exist_ok=True)

    # if args.load_best_model_at_end:
    #     # Dump args
    #     if not os.path.isdir(args.output_dir):
    #         os.mkdir(args.output_dir)
    #     with open(os.path.join(args.output_dir, 'user_args.yaml'), 'w') as f:
    #         yaml.dump(manual_args.__dict__, f)
    #     with open(os.path.join(args.output_dir, 'all_args.yaml'), 'w') as f:
    #         yaml.dump(args.__dict__, f)   

    return args