import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="auto",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        # Auto-detect tokenizer backend
        if tokenizer_backend == "auto":
            if base_url:
                from lm_eval.utils import check_remote_tokenizer_support

                if check_remote_tokenizer_support(
                    base_url,
                    verify_certificate=verify_certificate,
                    ca_cert_path=ca_cert_path,
                    auth_token=auth_token,
                ):
                    eval_logger.info(
                        "Auto-detected remote tokenizer support. Using remote tokenizer backend."
                    )
                    tokenizer_backend = "remote"
                else:
                    eval_logger.info(
                        "Remote tokenizer not supported. Using huggingface tokenizer backend."
                    )
                    tokenizer_backend = "huggingface"
            else:
                eval_logger.warning(
                    "No base_url provided. Using huggingface tokenizer backend."
                )
                tokenizer_backend = "huggingface"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            # Single-token generation: top_logprobs at position 0 gives
            # P(any_token | context), which is what we need for loglikelihood.
            # For multi-token continuations, _loglikelihood_tokens makes multiple calls.
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 100,  # top-100 tokens at the generated position
                "seed": seed,
            }

    def _loglikelihood_tokens(self, requests, **kwargs):
        """Override to compute loglikelihoods correctly for servers that only return
        logprobs for generated tokens (not prompt tokens with echo=True).

        Strategy: teacher-forced loglikelihood.
        For each (context, continuation) pair:
        - Decode continuation into N text segments (one per local token)
        - Make N sequential API calls, each time:
            a. Send prefix = context + true_continuation_so_far
            b. Get top_logprobs at the generated position
            c. Find the next continuation text segment in top_logprobs
            d. Extend prefix by matched token text (teacher forcing)
        - Sum the logprobs

        For single-token continuations (ARC, MMLU, Winogrande): 1 API call per pair
        For multi-token continuations (HellaSwag, TruthfulQA): k API calls per pair
        """
        import requests as req_lib
        from tenacity import retry, stop_after_attempt, wait_exponential
        from tqdm import tqdm

        try:
            from lm_eval.models.utils import Collator
        except ImportError:
            from lm_eval.utils import Collator

        assert self.tokenizer is not None
        res = []

        def _collate(item):
            toks = item[1] + item[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate, group_by=None)
        chunked = re_ord.get_batched(n=1)

        pbar = tqdm(desc="Requesting API", total=len(requests))

        for chunk in chunked:
            for cache_key, context_enc, continuation_enc in chunk:
                if len(continuation_enc) == 0:
                    answer = (0.0, True)
                    res.append(answer)
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    pbar.update(1)
                    continue

                # Truncate and decode context
                max_ctx = self.max_length - len(continuation_enc) - 1
                ctx_enc = list(context_enc)[-max_ctx:] if len(context_enc) > max_ctx else list(context_enc)
                context_text = self.decode_batch([ctx_enc])[0]

                # Decode full continuation and per-token segments
                full_cont_text = self.decode_batch([list(continuation_enc)])[0]
                cont_segments = []  # Text added by each local token
                for i in range(len(continuation_enc)):
                    full_so_far = self.decode_batch([list(continuation_enc[:i+1])])[0]
                    prev_so_far = self.decode_batch([list(continuation_enc[:i])])[0] if i > 0 else ""
                    cont_segments.append(full_so_far[len(prev_so_far):])

                # Teacher-forced computation:
                # remaining_cont tracks unmatched continuation text
                # current_prefix grows with matched tokens (teacher forcing)
                total_lp = 0.0
                is_greedy = True
                current_prefix = context_text
                remaining_cont = full_cont_text

                # We make one API call per "step", where each step matches
                # as much of remaining_cont as possible with one Kimi token.
                max_steps = len(continuation_enc) + 5  # Safety bound
                step = 0

                while remaining_cont and step < max_steps:
                    payload = {
                        "model": self.model,
                        "prompt": current_prefix,
                        "temperature": 0,
                        "max_tokens": 1,
                        "logprobs": 100,
                        "seed": self._seed,
                    }

                    try:
                        raw_resp = retry(
                            stop=stop_after_attempt(self.max_retries),
                            wait=wait_exponential(multiplier=0.5, min=1, max=10),
                            reraise=True,
                        )(req_lib.post)(
                            self.base_url,
                            json=payload,
                            headers=self.header,
                            verify=self.verify_certificate,
                        )
                        raw_resp.raise_for_status()
                        response = raw_resp.json()
                    except Exception as e:
                        eval_logger.error(f"API call failed at step {step}: {e}")
                        total_lp += -100.0 * len(remaining_cont)
                        is_greedy = False
                        break

                    choice = response.get("choices", [{}])[0]
                    lp_data = choice.get("logprobs") or {}
                    top_lps_list = lp_data.get("top_logprobs") or []
                    gen_tokens_list = lp_data.get("tokens") or []
                    gen_logprobs_list = lp_data.get("token_logprobs") or []

                    top_lps = top_lps_list[0] if top_lps_list else {}
                    gen_tok = gen_tokens_list[0] if gen_tokens_list else ""
                    gen_lp = gen_logprobs_list[0] if gen_logprobs_list else None

                    # Find the longest prefix of remaining_cont in top_lps (greedy)
                    found_tok = None
                    found_lp = None
                    if top_lps:
                        for cand_tok in sorted(top_lps.keys(), key=lambda k: -len(k)):
                            if remaining_cont.startswith(cand_tok) and len(cand_tok) > 0:
                                found_tok = cand_tok
                                found_lp = top_lps[cand_tok]
                                break

                    if found_tok is not None:
                        # Matched! Advance by this token.
                        total_lp += found_lp
                        remaining_cont = remaining_cont[len(found_tok):]
                        current_prefix += found_tok
                        if top_lps and found_tok != max(top_lps, key=lambda k: top_lps[k]):
                            is_greedy = False
                    elif gen_tok and remaining_cont.startswith(gen_tok):
                        # Generated token matches (not in top_lps lookup but gen_tok matches)
                        total_lp += gen_lp if gen_lp is not None else 0.0
                        remaining_cont = remaining_cont[len(gen_tok):]
                        current_prefix += gen_tok
                        if top_lps and gen_tok != max(top_lps, key=lambda k: top_lps[k]):
                            is_greedy = False
                    else:
                        # Continuation token not in top-100 at this position
                        # Advance by the next local token's text (skip this segment)
                        # Determine advance length from local cont_segments
                        skip_text = cont_segments[step] if step < len(cont_segments) else remaining_cont[:1]
                        total_lp += -100.0
                        is_greedy = False
                        remaining_cont = remaining_cont[len(skip_text):]
                        current_prefix += skip_text

                    step += 1

                # Penalize any remaining unmatched continuation
                if remaining_cont:
                    total_lp += -25.0 * len(remaining_cont)
                    is_greedy = False

                answer = (total_lp, is_greedy)
                res.append(answer)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                pbar.update(1)

        return re_ord.get_original(res)

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Fallback parse_logprobs (not used in primary loglikelihood path).
        The actual loglikelihood computation is done in _loglikelihood_tokens.
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice in sorted(out["choices"], key=itemgetter("index")):
                res.append((0.0, False))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    """
    Minimal chat-completions wrapper.
    - Only accepts messages as list[dict].
    - No tokenization or template logic.
    - Use with --apply_chat_template or ensure upstream formats messages correctly.
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), (
            "LocalChatCompletion expects messages as list[dict]. "
            "If you see this error, ensure --apply_chat_template is set or upstream code formats messages correctly."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    tmp[choices["index"]] = choices["message"]["content"]
            except Exception as e:
                # account for cases that generation is blocked by content filter,
                # which is common for Azure OpenAI Service,
                # not sure if need to account for multiple choices
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if (
            "o1" in self.model
            or "5" in self.model
            or "o3" in self.model
            or "o4" in self.model
        ):
            output.pop("stop")
            output["temperature"] = 1
        return output


@register_model("azure-openai-chat-completions")
class AzureOpenaiChatCompletionsLM(OpenAIChatCompletion):
    def __init__(
        self,
        model: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        base_url: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        truncate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.truncate = truncate
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=self.api_key
        )

    @cached_property
    def api_key(self):
        key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `AZURE_OPENAI_API_KEY` environment variable."
            )
        return key
