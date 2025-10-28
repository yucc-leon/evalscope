from __future__ import annotations

from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageAssistant
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.import_utils import check_import

logger = get_logger()


class VllmOpenAIAPI(ModelAPI):
    """Local LLM inference using vLLM SDK, following OpenAICompatibleAPI patterns."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # ensure vllm is installed
        check_import('vllm', package='vllm', raise_error=True, feature_name='vllm_openai')

        # lazy import after check
        from vllm import LLM  # type: ignore

        # Normalize and filter model_args to avoid unsupported keys
        args: Dict[str, Any] = dict(model_args) if model_args else {}
        # Map precision/torch_dtype to vLLM dtype
        precision = args.pop('precision', None) or args.pop('torch_dtype', None)
        if isinstance(precision, str):
            p = precision.lower()
            if 'float16' in p:
                args['dtype'] = 'float16'
            elif 'bfloat16' in p:
                args['dtype'] = 'bfloat16'
            elif 'float32' in p or 'fp32' in p:
                args['dtype'] = 'float32'
            elif 'auto' in p:
                args['dtype'] = 'auto'
        # Tokenizer path mapping
        if 'tokenizer_path' in args:
            args['tokenizer'] = args.pop('tokenizer_path')
        # Drop known non-vLLM keys
        for k in ['device_map', 'revision', 'chat_template', 'token', 'enable_thinking', 'tokenizer_call_args']:
            args.pop(k, None)
        # Whitelist commonly supported vLLM args
        allowed = {
            'tensor_parallel_size',
            'pipeline_parallel_size',
            'dtype',
            'gpu_memory_utilization',
            'enforce_eager',
            'trust_remote_code',
            'max_model_len',
            'max_num_seqs',
            'tokenizer',
        }
        args = {k: v for k, v in args.items() if k in allowed}

        # Provide safe defaults
        args.setdefault('dtype', 'auto')
        args.setdefault('enforce_eager', True)

        # Initialize the engine following examples/test_chat.py pattern
        self.llm = LLM(model=self.model_name, **args)

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # resolve tools consistently (vLLM SDK chat doesn't execute tools yet)
        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        # Prepare inputs and sampling params (similar to examples/test_chat.py)
        conversation = self._to_vllm_conversation(input)
        sampling_params = self.completion_params(config=config, tools=len(tools) > 0)

        # run inference
        try:
            outputs = self.llm.chat([conversation], sampling_params, use_tqdm=False)
            response = {'outputs_count': len(outputs)}
            self.on_response(response)

            # choices
            choices = self.chat_choices_from_completion(outputs, tools)

            # usage (best effort from first output)
            usage = None
            if outputs:
                vout = outputs[0]
                try:
                    prompt_tokens = len(getattr(vout, 'prompt_token_ids', []) or [])
                    gen_tokens = sum(len(getattr(o, 'token_ids', []) or []) for o in getattr(vout, 'outputs', []))
                    usage = ModelUsage(
                        input_tokens=prompt_tokens,
                        output_tokens=gen_tokens,
                        total_tokens=prompt_tokens + gen_tokens,
                    )
                except Exception:
                    pass

            return ModelOutput(model=self.model_name, choices=choices, usage=usage)
        except Exception as ex:
            # align with pattern: return a ModelOutput with error if it's a common generation issue
            return ModelOutput.from_content(model=self.model_name, content=str(ex), stop_reason='unknown')

    def _to_vllm_conversation(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        conv: List[Dict[str, str]] = []
        for m in messages:
            # Only text content supported here
            conv.append({'role': m.role, 'content': m.text})
        return conv

    def resolve_tools(self, tools: List[ToolInfo], tool_choice: ToolChoice,
                      config: GenerateConfig) -> tuple[List[ToolInfo], ToolChoice, GenerateConfig]:
        """Provides an opportunity for concrete classes to customize tool resolution."""
        return tools, tool_choice, config

    def completion_params(self, config: GenerateConfig, tools: bool):
        """Map GenerateConfig to vLLM SamplingParams to mirror OpenAICompatibleAPI pattern."""
        from vllm.sampling_params import SamplingParams  # type: ignore

        kwargs: Dict[str, Any] = {}
        if config.max_tokens is not None:
            kwargs['max_tokens'] = config.max_tokens
        if config.temperature is not None:
            kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            kwargs['top_p'] = config.top_p
        if config.top_k is not None:
            kwargs['top_k'] = config.top_k
        if config.stop_seqs is not None:
            kwargs['stop'] = config.stop_seqs
        if config.n is not None:
            kwargs['n'] = config.n
        if config.best_of is not None:
            kwargs['best_of'] = config.best_of
        if config.logprobs is not None:
            kwargs['logprobs'] = config.logprobs
        if config.top_logprobs is not None:
            kwargs['top_logprobs'] = config.top_logprobs
        if config.seed is not None:
            kwargs['seed'] = config.seed

        return SamplingParams(**kwargs)

    def on_response(self, response: Dict[str, Any]) -> None:
        """Hook for subclasses to do custom response handling."""
        # no-op for local SDK; keep for parity
        pass

    def chat_choices_from_completion(self, completion_outputs: List[Any], tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
        """Convert vLLM SDK outputs to EvalScope choices, mirroring OpenAI-compatible conversion."""
        choices: List[ChatCompletionChoice] = []
        if not completion_outputs:
            return choices
        vout = completion_outputs[0]
        for o in getattr(vout, 'outputs', []):
            text = getattr(o, 'text', '') or ''
            choices.append(
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=text, model=self.model_name, source='generate'),
                    stop_reason='stop',
                )
            )
        return choices
