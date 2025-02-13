from trl.trainer.grpo_trainer import GRPOTrainer

# Only import vLLM if you're actually using vLLM:
# from vllm import SamplingParams

class MyGRPOTrainer(GRPOTrainer):
    """
    Custom trainer subclass to override vLLM-related sampling parameters at test time.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        max_tokens_override=32768,   # example large limit
        temperature_override=None,   # optional override
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # If we are using vLLM, override the sampling parameters after the parent class has created them.
        if self.use_vllm:
            from vllm import SamplingParams
            # Adjust or extend whichever fields you need:
            self.sampling_params = SamplingParams(
                n=self.num_generations,
                temperature=(temperature_override if temperature_override else self.args.temperature),
                max_tokens=max_tokens_override,
                # You can set min_tokens, stop_token_ids, etc. here as needed.
            ) 