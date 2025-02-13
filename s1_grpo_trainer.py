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
        optimizers=(None, None),
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
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # Only if vLLM is in use:
        if self.use_vllm:
            self.s1_tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
            # Example stop tokens. You can customize or remove them:
            stop_token_ids = self.s1_tokenizer("<|im_end|>")["input_ids"]

            # Override the parent's sampling_params as you'd like:
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=temperature_override,
                min_p=min_p
            )

        self.num_ignore = num_ignore

    def generate_in_chunks(self, prompt, ignore_str="Wait"):
        """
        Illustrates partial generation logic like s1 does:
        1) Generate once with big max_tokens.
        2) Append partial output + 'Wait' a few times to skip the stop token.
        """
        if not self.use_vllm:
            raise RuntimeError("vLLM must be enabled to use partial generation logic!")

        # First generation
        outputs = self.model.generate(prompt, sampling_params=self.sampling_params)
        partial_text = outputs[0].outputs[0].text

        # "Force" ignoring the stop token a handful of times (the s1 trick)
        max_tokens_remaining = self.sampling_params.max_tokens - len(outputs[0].outputs[0].token_ids)
        updated_prompt = prompt + partial_text + ignore_str

        for _ in range(self.num_ignore):
            if max_tokens_remaining <= 0:
                break
            # Adjust sampling_params on the fly (like s1 does)
            local_params = SamplingParams(
                max_tokens=max_tokens_remaining,
                min_tokens=1,
                stop_token_ids=self.sampling_params.stop_token_ids,
                skip_special_tokens=False,
                temperature=self.sampling_params.temperature,
            )
            outputs = self.model.generate(updated_prompt, sampling_params=local_params)
            piece = outputs[0].outputs[0].text
            updated_prompt += piece + ignore_str
            max_tokens_remaining -= len(outputs[0].outputs[0].token_ids)

        # Optionally do a final generation (like "final answer")
        # So that the model's final response is "closed out" normally:
        local_params_final = SamplingParams(
            max_tokens=32768, 
            stop_token_ids=self.sampling_params.stop_token_ids,
            skip_special_tokens=False,
            temperature=self.sampling_params.temperature,
        )
        final_output = self.model.generate(updated_prompt, sampling_params=local_params_final)
        return updated_prompt + final_output[0].outputs[0].text 