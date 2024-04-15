import asyncio, time
from typing import AsyncIterator, Union, List
from cog import BasePredictor, Input, ConcatenateIterator

import torch
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# https://github.com/nateraw/replicate-examples/blob/main/mixtral-vllm/predict.py modified
class VLLMPipeline:
    """
    A simplified inference engine that runs inference w/ vLLM
    """

    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(
            prompt, sampling_params, str(random_uuid())
            )
        async for generated_text in results_generator:
            yield generated_text

    def __call__(
        self,
        prompt: str,
        max_tokens: int,
        min_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        repetition_penalty: float,
        length_penalty: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        ignore_eos: bool,
        stop_sequences: Union[str, List[str]] = None,
        stop_token_ids: List[int] = None,
        incremental_generation: bool = True,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.
        """

        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(n=1, 
                                        best_of=None,
                                        presence_penalty=presence_penalty, 
                                        frequency_penalty=frequency_penalty, 
                                        repetition_penalty=repetition_penalty, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        top_k=top_k,
                                        min_p=min_p,
                                        seed=None,
                                        use_beam_search=False,
                                        length_penalty=length_penalty,
                                        early_stopping=False,
                                        stop=stop,
                                        # stop_token_ids=None,
                                        include_stop_str_in_output=False,
                                        ignore_eos=ignore_eos,
                                        max_tokens=max_tokens,
                                        min_tokens=min_tokens,
                                        logprobs=None,
                                        prompt_logprobs=None,
                                        skip_special_tokens=True,
                                        spaces_between_special_tokens=True,
                                        logits_processors=None
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        gen = self.generate_stream(
            prompt,
            sampling_params,
        )

        generation_length = 0

        while True:
            try:
                request_output = loop.run_until_complete(gen.__anext__())
                assert len(request_output.outputs) == 1
                generated_text = request_output.outputs[0].text
                if incremental_generation:
                    yield generated_text[generation_length:]
                else:
                    yield generated_text
                generation_length = len(generated_text)
            except StopAsyncIteration:
                break

class Predictor(BasePredictor):
    def setup(self) -> None:
        n_gpus = torch.cuda.device_count()
        start = time.time()
        print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            model="fireworks-ai/mixtral-8x22b-instruct-oh",
            tensor_parallel_size=n_gpus,
            # gpu_memory_utilization=0.99,
            max_model_len=32768, #The model's max seq len: 65536
            enforce_eager=False,
            disable_log_stats=True,
            disable_log_requests=True,
            dtype="auto",
        )
    def predict(
        self,
        prompt: str = Input("Tell me a story about the Cheesecake Kingdom."),
        max_tokens: int = Input(default=256, description="Maximum number of tokens to generate per output sequence."),
        min_tokens: int = Input(default=1, description="Minimum number of tokens to generate per output sequence."),
        presence_penalty: float = Input(default=0.0, description="""Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens."""),
        frequency_penalty: float = Input(default=0.0, description="""Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens."""),
        repetition_penalty: float = Input(default=2.0, description="""Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens."""),
        length_penalty: float = Input(default=1.0, description="""Float that penalizes sequences based on their length.
            Used in beam search."""),
        temperature: float = Input(default=0.6, description="""Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling."""),
        top_p: float = Input(default=1.0, description="""Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens."""),
        top_k: int = Input(default=40, description="""Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens."""),
        min_p: float = Input(default=0.0, description="""Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this."""),
        ignore_eos: bool = Input(default=False, description="""Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated."""),
        system_prompt: str = Input("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""),
        template: str = Input(default="""{system_prompt} {prompt}""", description="""SYSTEM:{system_prompt} USER:{prompt}"""),
    ) -> ConcatenateIterator[str]:
        start = time.time()
        prompt_text = template.format(system_prompt=system_prompt, prompt=prompt)
        generate = self.llm(prompt=prompt_text,
                            max_tokens=max_tokens,
                            min_tokens=min_tokens,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k,
                            min_p=min_p,
                            ignore_eos=ignore_eos,
                            )
        for text in generate:
            yield text
        print(prompt_text)
        print(f"Duration: {time.time() - start:.3f}s")