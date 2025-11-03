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

        # Sanitize engine args for vLLM
        args: Dict[str, Any] = dict(model_args) if model_args else {}
        # Drop precision aliases not supported by vLLM
        args.pop('precision', None)
        args.pop('torch_dtype', None)
        # Map tokenizer_path to tokenizer if provided
        if 'tokenizer_path' in args:
            args['tokenizer'] = args.pop('tokenizer_path')
        # Default trust_remote_code for custom models
        args.setdefault('trust_remote_code', True)

        logger.info("LLM engine args: %s", args)
        logger.info("running offline vllm inference with eager=%s", args.get("enforce_eager", False))
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
        # Keep n=1 by default; best_of may increase latency noticeably
        if config.n is not None and config.n > 1:
            kwargs['n'] = int(config.n)
        if config.best_of is not None and config.best_of > 1:
            kwargs['best_of'] = int(config.best_of)
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

    def supports_batch(self) -> bool:
        """Indicate that this ModelAPI supports optimized batch processing."""
        return True

    def batch_generate(
        self,
        inputs: List[List[ChatMessage]],
        tools: List[List[ToolInfo]],
        tool_choices: List[ToolChoice],
        configs: List[GenerateConfig],
    ) -> List[ModelOutput]:
        """Efficient batch chat generation using vLLM's chat API.

        Group by identical sampling params and call self.llm.chat once per group,
        preserving full conversation context per item.
        """
        # Build vLLM conversations and group indices by config key
        conversations: List[List[Dict[str, str]]] = [self._to_vllm_conversation(msgs) for msgs in inputs]

        def config_key(cfg: GenerateConfig) -> tuple:
            # Key on fields used by SamplingParams; omit None to maximize grouping
            return (
                cfg.max_tokens,
                cfg.temperature,
                cfg.top_p,
                cfg.top_k,
                tuple(cfg.stop_seqs or []),
                cfg.n,
                cfg.best_of,
                cfg.logprobs,
                cfg.top_logprobs,
                cfg.seed,
            )

        groups: Dict[tuple, List[int]] = {}
        for idx, cfg in enumerate(configs):
            k = config_key(cfg)
            groups.setdefault(k, []).append(idx)

        outputs: List[Optional[ModelOutput]] = [None] * len(inputs)
        for k, idxs in groups.items():
            cfg0 = configs[idxs[0]]
            sampling_params = self.completion_params(config=cfg0, tools=False)
            group_convs = [conversations[i] for i in idxs]
            # Debug preview: show first user turn head and params
            try:
                prev = ''
                first_conv = group_convs[0] if group_convs else []
                for turn in first_conv:
                    if turn.get('role') == 'user':
                        prev = (turn.get('content') or '')[:200]
                        break
                # logger.debug('vLLM chat batch size=%d, preview="%s", params=%s',
                #              len(group_convs), prev, sampling_params)
            except Exception:
                pass
            try:
                # Use tensor_parallel_size default; avoid forcing low concurrency that serializes batches
                batched = self.llm.chat(group_convs, sampling_params, use_tqdm=False)
            except Exception as ex:
                for i in idxs:
                    outputs[i] = ModelOutput.from_content(model=self.model_name, content=str(ex), stop_reason='unknown')
                continue

            # Map responses back to individual ModelOutput
            for j, i in enumerate(idxs):
                vout = batched[j]
                choices: List[ChatCompletionChoice] = []
                vout_outputs = getattr(vout, 'outputs', [])
                # If engine returned no outputs, try a single-sample recover via chat
                if not vout_outputs:
                    try:
                        single = self.llm.chat([conversations[i]], sampling_params, use_tqdm=False)
                        recovered = ''
                        if single and getattr(single[0], 'outputs', None):
                            recovered = getattr(single[0].outputs[0], 'text', '') or ''
                        if recovered.strip():
                            logger.warning('Empty batch output recovered via single chat (idx=%d)', i)
                            choices.append(ChatCompletionChoice(
                                message=ChatMessageAssistant(content=recovered, model=self.model_name, source='generate'),
                                stop_reason='stop',
                            ))
                        else:
                            logger.warning('Empty chat generation for sample idx=%d (no recovery)', i)
                    except Exception as rex:
                        logger.warning('Single chat retry failed for idx=%d: %s', i, rex)
                else:
                    for o in vout_outputs:
                        text = getattr(o, 'text', '') or ''
                        if not text.strip():
                            # Try single recovery for empty text
                            try:
                                single = self.llm.chat([conversations[i]], sampling_params, use_tqdm=False)
                                recovered = ''
                                if single and getattr(single[0], 'outputs', None):
                                    recovered = getattr(single[0].outputs[0], 'text', '') or ''
                                if recovered.strip():
                                    logger.warning('Empty batch choice recovered via single chat (idx=%d)', i)
                                    text = recovered
                                else:
                                    logger.warning('Empty chat generation for sample idx=%d (no recovery)', i)
                            except Exception as rex:
                                logger.warning('Single chat retry failed for idx=%d: %s', i, rex)
                        choices.append(
                            ChatCompletionChoice(
                                message=ChatMessageAssistant(content=text, model=self.model_name, source='generate'),
                                stop_reason='stop',
                            )
                        )
                usage = None
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
                outputs[i] = ModelOutput(model=self.model_name, choices=choices, usage=usage)

        return [out if out is not None else ModelOutput.from_content(self.model_name, '') for out in outputs]

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Reduce a chat message list to a single prompt string.

        Prefix with the first system message if present, then append the last user message.
        Assistant/tool content is ignored for prompt construction.
        """
        system_text = ''
        user_text = ''
        for m in messages:
            if m.role == 'system' and not system_text:
                system_text = m.text.strip()
            elif m.role == 'user':
                user_text = m.text.strip()
        return f"{system_text}\n\n{user_text}" if system_text else user_text

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
