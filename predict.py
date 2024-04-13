import os, sys, time, subprocess
from cog import BasePredictor, Input
from vllm import LLM, SamplingParams
sys.path.append('/content')
os.chdir('/content')

MODEL_CACHE = "/content/checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/fireworks-ai/mixtral-8x22b-instruct-oh/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.llm = LLM(model="fireworks-ai/mixtral-8x22b-instruct-oh", download_dir=MODEL_CACHE, tensor_parallel_size=8)
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
    ) -> str:
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
                                        stop=None,
                                        stop_token_ids=None,
                                        include_stop_str_in_output=False,
                                        ignore_eos=ignore_eos,
                                        max_tokens=max_tokens,
                                        min_tokens=min_tokens,
                                        logprobs=None,
                                        prompt_logprobs=None,
                                        skip_special_tokens=True,
                                        spaces_between_special_tokens=True,
                                        logits_processors=None)
        outputs = self.llm.generate(template.format(system_prompt=system_prompt, prompt=prompt), sampling_params)
        print(template.format(system_prompt=system_prompt, prompt=prompt))
        for output in outputs:
            print(output.prompt)
            return output.outputs[0].text