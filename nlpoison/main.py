import sys
import os
import yaml
import argparse

from transformers.training_args import TrainingArguments
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification
)

from callbacks import CustomFlowCallback
from data import NLIDataset, DavidsonDataset
from utils import collate_fn, compute_metrics, dump_test_results


def load_args():
    assert sys.argv[1] in ['train', 'test']
    # Load args from file
    with open(f'config/{sys.argv[1]}.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass

    if args.do_train:
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

    if args.load_best_model_at_end:
        # Dump args
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        with open(os.path.join(args.output_dir, 'user_args.yaml'), 'w') as f:
            yaml.dump(manual_args.__dict__, f)
        with open(os.path.join(args.output_dir, 'all_args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f)            

    return args


def main():
    args = load_args()

    callbacks = [CustomFlowCallback]

    if args.wandb:
        import wandb
        wandb.init(project="robustai-nlp", config=vars(args))
        os.environ["WANDB_DISABLED"] = ""
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # args.model_name_or_dir = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=3)

    # Init dataset
    # TODO: insert type of dataset
    train = NLIDataset(args, 'train', tokenizer)
    dev = NLIDataset(args, 'dev', tokenizer)  

    from custom_trainer import CustomTrainer

    if args.do_train:
        trainer = CustomTrainer(
            model,
            args=args,
            train_dataset=train,
            eval_dataset=dev,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        done_train = True

    if args.do_predict:
        # TODO: insert type of dataset
        test = NLIDataset(args, 'test', tokenizer)

        predictor = CustomTrainer(
            model,
            args=args,
            eval_dataset=test,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )        
        outputs = predictor.evaluate()
        dump_test_results(outputs, args.output_dir)


if __name__ == '__main__':
    main()