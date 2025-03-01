from trl.trainer.grpo_trainer import GRPOTrainer
from vllm import SamplingParams
from transformers import AutoTokenizer


class MyS1GRPOTrainer(GRPOTrainer):
    """
    Example trainer subclass that overrides sampling params to replicate
    s1-like large-token generation while optionally ignoring certain stop tokens.
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
        peft_config=None,
        min_tokens_thinking=30,
        max_tokens_thinking=32000,   # total tokens you want to allow for 'thinking'
        num_ignore=1,               # number of times to ignore the stop token
        temperature_override=0.0,   # example: 0.0 means deterministic
        min_p=0.1,
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
            peft_config=peft_config,
        )

        # Only if vLLM is in use:
        if self.use_vllm:
            self.s1_tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
            # Example stop tokens. You can customize or remove them:
            stop_token_ids = self.s1_tokenizer("<|im_start|><|im_end|><|endoftext|>")["input_ids"]

            # Override the parent's sampling_params as you'd like:
            self.sampling_params = SamplingParams(
                n=self.args.num_generations,
                max_tokens=max_tokens_thinking,
                min_tokens=min_tokens_thinking,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=temperature_override,
                min_p=min_p
            )

            # --- Patch the vLLM generate() method ---
            # Here we wrap the original generate to ensure that for every prompt, 
            # we never exceed the model's maximum sequence length
            orig_generate = self.llm.generate
            def generate_with_truncation(prompts_text, sampling_params, **kwargs):
                new_outputs = []
                for prompt in prompts_text:
                    # Compute token length of the prompt using our dedicated tokenizer.
                    # Note: We disable padding and special tokens to get an accurate count.
                    encoding = self.s1_tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=False,
                        add_special_tokens=False,
                    )
                    prompt_length = encoding["input_ids"].shape[1]
                    # Calculate how many tokens are left before reaching the model's max sequence length.
                    # self.args.vllm_max_model_len is the value passed to the LLM at initialization.
                    allowed_tokens = self.args.max_completion_length - prompt_length
                    if allowed_tokens < 1:
                        raise ValueError(
                            f"Prompt too long ({prompt_length} tokens); maximum allowed is {self.args.vllm_max_model_len}"
                        )
                    # Create new sampling parameters, ensuring we do not request more than 'allowed_tokens'.
                    new_params = self.sampling_params.clone()
                    new_params.max_tokens = min(self.sampling_params.max_tokens, allowed_tokens)
                    # Call the original generate() for this prompt.
                    result = orig_generate([prompt], sampling_params=new_params, **kwargs)
                    new_outputs.extend(result)
                return new_outputs
            self.llm.generate = generate_with_truncation

        self.num_ignore = num_ignore

        """
        Illustrates partial generation logic like s1 does:
        1) Generate once with big max_tokens.
        2) Append partial output + 'Wait' a few times to skip the stop token.
        """
        if not self.use_vllm:
            raise RuntimeError("vLLM must be enabled to use partial generation logic!")
