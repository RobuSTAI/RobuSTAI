from transformers.trainer_callback import TrainerCallback

class CustomFlowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """
    def on_step_end(self, args, state, control, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy and state.global_step % args.eval_steps == 0:
            control.should_evaluate = True

        # Save
        if not args.load_best_model_at_end and args.save_steps > 0 and state.global_step % args.save_steps == 0:
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control 

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.evaluation_strategy:
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True
        return control
