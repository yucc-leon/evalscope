# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Default evaluator implementation for running benchmark evaluations.

This module provides the DefaultEvaluator class which orchestrates the entire
evaluation process including data loading, model inference, metric calculation,
and report generation.
"""

import os
from threading import Lock
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from evalscope.api.dataset import Dataset, DatasetDict, Sample
from evalscope.api.evaluator import CacheManager, Evaluator, TaskState
from evalscope.api.metric import AggScore, SampleScore
from evalscope.report import Report, gen_table
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure

logger = get_logger()


class DefaultEvaluator(Evaluator):
    """
    Default Evaluator for running evaluations on benchmarks.

    This evaluator handles the complete evaluation pipeline:
    1. Loading datasets from benchmarks
    2. Running model inference on samples
    3. Calculating evaluation metrics
    4. Generating and saving reports
    5. Managing caching for predictions and reviews

    Args:
        benchmark: The data adapter for loading and processing data.
        model: The model to be evaluated.
        outputs: The output structure for saving evaluation results.
        task_config: The task configuration.
    """

    def __init__(
        self,
        benchmark: 'DataAdapter',
        model: 'Model',
        outputs: 'OutputsStructure',
        task_config: 'TaskConfig',
    ):
        # Store core components needed for evaluation
        self.benchmark = benchmark
        self.model = model
        self.outputs = outputs
        self.task_config = task_config

        # Extract frequently used identifiers
        self.benchmark_name = benchmark.name
        """Name of the benchmark being evaluated."""

        self.model_name = task_config.model_id
        """ID of the model being evaluated."""

        self.use_cache = task_config.use_cache
        """Whether to use cache for predictions."""

        # Initialize cache manager for storing and retrieving cached results
        self.cache_manager = CacheManager(
            outputs=outputs,
            model_name=self.model_name,
            benchmark_name=self.benchmark_name,
        )
        # Track summed inference time per subset (from per-sample perf)
        self._subset_infer_times: Dict[str, float] = {}

    def eval(self) -> Report:
        """
        Run the complete evaluation process.

        This is the main entry point that orchestrates the entire evaluation:
        1. Load dataset from benchmark
        2. Evaluate each subset independently
        3. Aggregate scores across subsets
        4. Generate final evaluation report

        Returns:
            Report: The complete evaluation report containing all metrics and results.
        """
        # Load the dataset and evaluate each subset
        logger.info(f'Start evaluating benchmark: {self.benchmark_name}')
        import time
        dataset_dict = self.benchmark.load_dataset()
        agg_score_dict = defaultdict(list)
        subset_wall_times: Dict[str, float] = {}
        
        # Process each subset (e.g., test, validation) independently
        logger.info('Evaluating all subsets of the dataset...')
        for subset, dataset in dataset_dict.items():
            if len(dataset) == 0:
                logger.info(f'No samples found in subset: {subset}, skipping.')
                continue
            logger.info(f'Evaluating subset: {subset}')
            t0 = time.perf_counter()
            subset_score = self.evaluate_subset(subset, dataset)
            t1 = time.perf_counter()
            subset_wall_times[subset] = round(t1 - t0, 3)
            agg_score_dict[subset] = subset_score

        # Generate the report based on aggregated scores
        logger.info('Generating report...')
        t_report0 = time.perf_counter()
        # Prefer true inference time from cached per-sample perf; fallback to wall time
        subset_times = self._subset_infer_times if any(self._subset_infer_times.values()) else subset_wall_times
        report = self.get_report(agg_score_dict, subset_times=subset_times)
        t_report1 = time.perf_counter()
        # Attach timing metadata to report and persist to disk
        try:
            report.elapsed_time_s = round(sum(subset_times.values()) + (t_report1 - t_report0), 3)
            report.subset_times_s = subset_times
            # Overwrite report JSON with timing included
            report_file = self.cache_manager.get_report_file()
            report.to_json(report_file)
        except Exception:
            pass

        # Finalize the evaluation process
        self.finalize()
        logger.info(f'Benchmark {self.benchmark_name} evaluation finished. Timing (s): {subset_times}')
        return report

    def evaluate_subset(self, subset: str, dataset: Dataset) -> List[AggScore]:
        """
        Evaluate a single subset of the dataset.

        This method processes one subset through the complete evaluation pipeline:
        1. Get model predictions for all samples
        2. Calculate evaluation metrics for predictions
        3. Aggregate individual sample scores

        Args:
            subset: Name of the subset being evaluated (e.g., 'test', 'validation').
            dataset: The dataset subset containing samples to evaluate.

        Returns:
            List[AggScore]: Aggregated scores for this subset.
        """
        # Get model predictions for all samples in the subset
        logger.info(f'Getting predictions for subset: {subset}')
        task_states = self.get_answers(subset, dataset)
        # Sum per-sample inference time from metadata if available
        try:
            infer_time = 0.0
            for ts in task_states:
                perf = (ts.metadata or {}).get('perf', {}) if ts.metadata else {}
                t = perf.get('time_s')
                if isinstance(t, (int, float)) and t > 0:
                    infer_time += float(t)
            self._subset_infer_times[subset] = round(infer_time, 3)
        except Exception:
            pass

        # Calculate evaluation metrics for each prediction
        logger.info(f'Getting reviews for subset: {subset}')
        sample_scores = self.get_reviews(subset, task_states)

        # Aggregate individual sample scores into subset-level metrics
        logger.info(f'Aggregating scores for subset: {subset}')
        agg_scores = self.benchmark.aggregate_scores(sample_scores=sample_scores)
        return agg_scores

    def get_answers(self, subset: str, dataset: Dataset) -> List[TaskState]:
        """
        Get model predictions for all samples in the dataset subset.

        This method handles:
        1. Loading cached predictions if available and caching is enabled
        2. Running model inference on remaining samples in parallel
        3. Saving new predictions to cache

        Args:
            subset: Name of the subset being processed.
            dataset: The dataset subset containing samples for prediction.

        Returns:
            List[TaskState]: Task states containing model predictions for each sample.
        """
        # Initialize task state list and filter cached predictions if caching is enabled
        if self.use_cache:
            task_state_list, dataset = self.cache_manager.filter_prediction_cache(subset, dataset)
        else:
            task_state_list = []

        # Get output directory for storing model predictions
        model_prediction_dir = os.path.dirname(self.cache_manager.get_prediction_cache_path(subset))

        # Convert dataset to list for parallel processing
        dataset_list = list(dataset)

        if not dataset_list:
            return task_state_list

        logger.info(f'Processing {len(dataset_list)} samples, if data is large, it may take a while.')

        # Try optimized batch path if supported and batch_size > 1
        try:
            supports_batch = getattr(self.model.api, 'supports_batch', lambda: False)()
        except Exception:
            supports_batch = False

        if supports_batch and (self.task_config.eval_batch_size or 1) > 1:
            # Env-driven tqdm controls (shared with single path)
            pos_env = os.environ.get('EVALSCOPE_TQDM_POSITION', None)
            try:
                position = int(pos_env) if pos_env is not None else 0
            except Exception:
                position = 0
            disable = str(os.environ.get('EVALSCOPE_TQDM_DISABLE', '0')).lower() in ('1', 'true', 'yes')

            batch_size = self.task_config.eval_batch_size
            total = len(dataset_list)
            from evalscope.api.messages import ChatMessageUser

            with tqdm(total=total, desc=f'Predicting[{self.benchmark_name}@{subset}]: ', position=position,
                      leave=True, dynamic_ncols=True, disable=disable) as pbar:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch = dataset_list[start:end]

                    inputs: List[List[ChatMessage]] = []  # type: ignore[name-defined]
                    tools_batch: List[List[ToolInfo]] = []  # type: ignore[name-defined]
                    tool_choices_batch: List[ToolChoice] = []  # type: ignore[name-defined]
                    configs_batch: List[GenerateConfig] = []  # type: ignore[name-defined]

                    # Build batch inputs
                    for sample in batch:
                        if isinstance(sample.input, str):
                            msg_list = [ChatMessageUser(content=sample.input)]
                        else:
                            msg_list = sample.input
                        inputs.append(msg_list)
                        tools_batch.append(list(sample.tools) if sample.tools else [])
                        tool_choices_batch.append('none')
                        configs_batch.append(self.task_config.generation_config)

                    try:
                        import time as _t
                        _bt0 = _t.perf_counter()
                        batch_outputs = list(self.model.batch_generate(
                            inputs=inputs,
                            tools=tools_batch,
                            tool_choices=tool_choices_batch,
                            configs=configs_batch,
                        ))
                        _bt1 = _t.perf_counter()
                        # Compute simple perf summary: tok/s if usage present, else chars/s
                        out_tok = 0
                        out_chars = 0
                        for out in batch_outputs:
                            try:
                                if out.usage and out.usage.output_tokens:
                                    out_tok += int(out.usage.output_tokens)
                                else:
                                    for c in out.choices or []:
                                        msg = getattr(c, 'message', None)
                                        if msg and getattr(msg, 'content', None):
                                            out_chars += len(str(msg.content))
                            except Exception:
                                pass
                        elapsed = max(1e-6, _bt1 - _bt0)
                        if out_tok > 0:
                            pbar.set_postfix({'bs': len(batch_outputs), 'tok/s': f"{out_tok/elapsed:.1f}"}, refresh=False)
                        elif out_chars > 0:
                            pbar.set_postfix({'bs': len(batch_outputs), 'chars/s': f"{out_chars/elapsed:.1f}"}, refresh=False)
                    except Exception as exc:
                        logger.warning(f'Batch generate failed, fallback to single for this chunk: {exc}')
                        for sample in batch:
                            state = self._predict_sample(sample, model_prediction_dir)
                            task_state_list.append(state)
                            model_result = self.cache_manager.save_prediction_cache(
                                subset, state, self.benchmark.save_metadata
                            )
                            # logger.debug(f'Model result: \n{model_result.pretty_print()}')
                            pbar.update(1)
                        continue

                    # Map outputs back to task states and save, attaching per-sample perf
                    per_item_time = elapsed / max(1, len(batch_outputs)) if 'elapsed' in locals() else None
                    for sample, msg_list, output in zip(batch, inputs, batch_outputs):
                        try:
                            state = TaskState(
                                model=self.model_name,
                                sample=sample,
                                messages=msg_list,
                                output=output,
                                completed=True,
                            )
                            # Attach perf similar to _predict_sample
                            try:
                                output_text = state.output.completion if state and state.output else ''
                                output_chars = len(output_text) if isinstance(output_text, str) else 0
                                output_tokens = None
                                if state and state.output and state.output.usage:
                                    output_tokens = state.output.usage.output_tokens or None
                                gen_time = float(per_item_time) if isinstance(per_item_time, (int, float)) else None
                                tok_per_s = (float(output_tokens) / gen_time) if (output_tokens is not None and gen_time and gen_time > 0) else None
                                chars_per_s = (float(output_chars) / gen_time) if (output_chars and gen_time and gen_time > 0) else None
                                perf = {
                                    'time_s': gen_time,
                                    'output_tokens': output_tokens,
                                    'output_chars': output_chars,
                                    'tok_per_s': tok_per_s,
                                    'chars_per_s': chars_per_s,
                                }
                                meta = state.metadata or {}
                                meta['perf'] = perf
                                state.metadata = meta
                            except Exception:
                                pass
                            task_state_list.append(state)
                            model_result = self.cache_manager.save_prediction_cache(
                                subset, state, self.benchmark.save_metadata
                            )
                            # logger.debug(f'Model result: \n{model_result.pretty_print()}')
                        except Exception as exc:
                            tb_str = traceback.format_exc()
                            logger.error(
                                f'{sample.model_dump_json(indent=2)} prediction failed: due to {exc}\nTraceback:\n{tb_str}'
                            )
                            if self.task_config.ignore_errors:
                                logger.warning('Error ignored, continuing with next sample.')
                            else:
                                raise exc
                        finally:
                            pbar.update(1)
            logger.info(f'Finished getting predictions for subset: {subset}.')
            return task_state_list

        # Aggregate perf across completed samples for stable speed in tqdm
        total_output_tokens: float = 0.0
        total_output_chars: int = 0
        total_gen_time: float = 0.0
        perf_lock = Lock()

        # Process samples in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(dataset_list), self.task_config.eval_batch_size)) as executor:
            # Submit all prediction tasks
            future_to_sample = {
                executor.submit(self._predict_sample, sample, model_prediction_dir): sample
                for sample in dataset_list
            }

            # Process completed tasks with progress bar
            # Position/disable can be controlled via environment for multi-process runs
            pos_env = os.environ.get('EVALSCOPE_TQDM_POSITION', None)
            try:
                position = int(pos_env) if pos_env is not None else 0
            except Exception:
                position = 0
            disable = str(os.environ.get('EVALSCOPE_TQDM_DISABLE', '0')).lower() in ('1', 'true', 'yes')

            with tqdm(
                total=len(dataset_list),
                desc=f'Predicting[{self.benchmark_name}@{subset}]: ',
                position=position,
                leave=True,
                dynamic_ncols=True,
                disable=disable,
            ) as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        task_state = future.result()
                        task_state_list.append(task_state)

                        # Save the prediction result to cache for future use
                        model_result = self.cache_manager.save_prediction_cache(
                            subset, task_state, self.benchmark.save_metadata
                        )
                        # logger.debug(f'Model result: \n{model_result.pretty_print()}')

                        # update tqdm postfix with rolling averages for tokens/sec or chars/sec
                        perf = (task_state.metadata or {}).get('perf', {})
                        time_s = perf.get('time_s', None)
                        out_tok = perf.get('output_tokens', None)
                        out_chars = perf.get('output_chars', None)
                        with perf_lock:
                            if isinstance(time_s, (int, float)) and time_s > 0:
                                if isinstance(out_tok, (int, float)):
                                    total_output_tokens += float(out_tok)
                                if isinstance(out_chars, int):
                                    total_output_chars += int(out_chars)
                                total_gen_time += float(time_s)

                            avg_tok_ps = (total_output_tokens / total_gen_time) if total_gen_time > 0 and total_output_tokens > 0 else None
                            avg_char_ps = (total_output_chars / total_gen_time) if total_gen_time > 0 and total_output_chars > 0 else None

                        postfix = {}
                        if avg_tok_ps is not None:
                            postfix['avg_tok/s'] = f"{avg_tok_ps:.1f}"
                        if avg_char_ps is not None and avg_tok_ps is None:
                            # show chars/s only if tok/s unavailable
                            postfix['avg_chars/s'] = f"{avg_char_ps:.1f}"
                        if postfix:
                            pbar.set_postfix(postfix, refresh=False)

                        # update tqdm postfix with perf if available
                        perf = (task_state.metadata or {}).get('perf', {})
                        tok_ps = perf.get('tok_per_s', None)
                        ch_ps = perf.get('chars_per_s', None)
                        time_s = perf.get('time_s', None)
                        pbar.set_postfix({
                            'time_s': f"{time_s:.3f}" if isinstance(time_s, (int, float)) else '-',
                            'tok/s': f"{tok_ps:.1f}" if isinstance(tok_ps, (int, float)) else '-',
                            'chars/s': f"{ch_ps:.1f}" if isinstance(ch_ps, (int, float)) else '-',
                        }, refresh=False)

                    except Exception as exc:
                        tb_str = traceback.format_exc()
                        logger.error(
                            f'{sample.model_dump_json(indent=2)} prediction failed: due to {exc}\nTraceback:\n{tb_str}'
                        )
                        if self.task_config.ignore_errors:
                            logger.warning('Error ignored, continuing with next sample.')
                        else:
                            raise exc
                    finally:
                        pbar.update(1)
        logger.info(f'Finished getting predictions for subset: {subset}.')
        return task_state_list

    def _predict_sample(self, sample: Sample, model_prediction_dir: str) -> TaskState:
        """
        Helper method to predict a single sample.

        Args:
            sample: The sample to predict.
            model_prediction_dir: Directory for storing model predictions.

        Returns:
            TaskState: The task state containing the prediction result.
        """
        # logger.debug(f'\n{sample.pretty_print()}')

        # Run model inference on the current sample with simple perf timing
        import time
        t0 = time.time()
        task_state = self.benchmark.run_inference(model=self.model, sample=sample, output_dir=model_prediction_dir)
        t1 = time.time()

        # Estimate throughput based on available usage or text length
        try:
            gen_time = max(1e-9, t1 - t0)
            output_text = task_state.output.completion if task_state and task_state.output else ''
            output_chars = len(output_text) if isinstance(output_text, str) else 0
            output_tokens = None
            if task_state and task_state.output and task_state.output.usage:
                output_tokens = task_state.output.usage.output_tokens or None

            tok_per_s = (float(output_tokens) / gen_time) if output_tokens is not None else None
            chars_per_s = float(output_chars) / gen_time if output_chars else None

            # attach perf to metadata
            meta = task_state.metadata or {}
            perf = {
                'time_s': gen_time,
                'output_tokens': output_tokens,
                'output_chars': output_chars,
                'tok_per_s': tok_per_s,
                'chars_per_s': chars_per_s,
            }
            meta['perf'] = perf
            task_state.metadata = meta
        except Exception:
            pass

        return task_state

    def get_reviews(self, subset: str, task_states: List[TaskState]) -> List[SampleScore]:
        """
        Calculate evaluation metrics for model predictions.

        This method handles:
        1. Loading cached review results if available and caching is enabled
        2. Computing metrics for remaining task states in parallel
        3. Saving new review results to cache

        Args:
            subset: Name of the subset being reviewed.
            task_states: List of task states containing model predictions.

        Returns:
            List[SampleScore]: Evaluation scores for each sample.
        """
        # Initialize sample score list and filter cached reviews if caching is enabled
        if self.use_cache and not self.task_config.rerun_review:
            sample_score_list, task_states = self.cache_manager.filter_review_cache(subset, task_states)
        else:
            # Init a clean sample score list
            sample_score_list = []
            self.cache_manager.delete_review_cache(subset)

        if not task_states:
            return sample_score_list

        logger.info(f'Reviewing {len(task_states)} samples, if data is large, it may take a while.')
        
        # Get review timeout from benchmark config
        review_timeout = getattr(self.benchmark, 'review_timeout', 5)
        if review_timeout is None:
            review_timeout = 5
        
        # IMPORTANT: Use serial processing instead of ThreadPoolExecutor
        # SymPy (used in math metric calculations) is NOT thread-safe and can cause
        # deadlocks when multiple threads call symbolic operations concurrently.
        # Serial processing is slower but avoids these concurrency issues.
        
        # Note: We keep the signal-based timeout as a safety measure, though
        # it may not be strictly necessary now that we've identified the real issue.
        import signal
        
        has_alarm = hasattr(signal, 'SIGALRM')
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Review timed out")
        
        logger.info(f'Processing reviews serially to avoid SymPy thread-safety issues...')
        
        with tqdm(total=len(task_states), desc=f'Reviewing[{self.benchmark_name}@{subset}]: ') as pbar:
            for task_state in task_states:
                try:
                    # Apply timeout protection (safety measure)
                    if has_alarm:
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(review_timeout)
                    
                    try:
                        sample_score = self._review_task_state(task_state)
                    finally:
                        if has_alarm:
                            signal.alarm(0)  # Cancel alarm
                            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                    
                    # Save the sample score
                    sample_score_list.append(sample_score)
                    
                    # Save the review result to cache for future use
                    review_result = self.cache_manager.save_review_cache(
                        subset=subset,
                        task_state=task_state,
                        sample_score=sample_score,
                        save_metadata=self.benchmark.save_metadata
                    )
                    # Optional: update review bar with sample id
                    pbar.set_postfix({'sample': task_state.sample_id}, refresh=False)
                    
                except TimeoutError:
                    logger.warning(
                        f'Timeout when reviewing sample {task_state.sample_id} (>{review_timeout}s), setting score to zero.'
                    )
                    sample_score = SampleScore(sample_id=task_state.sample_id, scores={})
                    sample_score_list.append(sample_score)
                    # Save the zero score to cache
                    self.cache_manager.save_review_cache(
                        subset=subset,
                        task_state=task_state,
                        sample_score=sample_score,
                        save_metadata=self.benchmark.save_metadata
                    )

                except Exception as exc:
                    tb_str = traceback.format_exc()
                    logger.error(
                        f'Error when review sample {task_state.sample_id}: due to {exc}\nTraceback:\n{tb_str}'
                    )
                    if self.task_config.ignore_errors:
                        logger.warning('Error ignored, continuing with next sample.')
                    else:
                        raise exc
                finally:
                    pbar.update(1)
        logger.info(f'Finished reviewing subset: {subset}. Total reviewed: {len(sample_score_list)}')

        return sample_score_list

    def _review_task_state(self, task_state: TaskState) -> SampleScore:
        """
        Helper method to review a single task state.

        Args:
            task_state: The task state to review.

        Returns:
            SampleScore: The evaluation score for the task state.
        """
        # Compute evaluation metrics using the benchmark's metric calculation
        sample_score = self.benchmark.calculate_metrics(task_state=task_state)
        return sample_score

    def get_report(self, agg_score_dict: Dict[str, List[AggScore]], subset_times: Dict[str, float] | None = None) -> Report:
        """
        Generate a comprehensive evaluation report from aggregated scores.

        This method handles:
        1. Creating the evaluation report from scores
        2. Generating and displaying a summary table
        3. Optionally generating detailed analysis
        4. Saving the report to file

        Args:
            agg_score_dict: Dictionary mapping subset names to their aggregated scores.

        Returns:
            Report: The complete evaluation report.
        """
        assert agg_score_dict, 'No scores to generate report from.'

        # Get paths for saving the report
        report_path = self.cache_manager.get_report_path()
        report_file = self.cache_manager.get_report_file()

        # Generate the main evaluation report using benchmark-specific logic
        report = self.benchmark.generate_report(
            scores=agg_score_dict, model_name=self.model_name, output_dir=report_path
        )
        # Attach preliminary timing so the displayed table reflects non-zero times
        try:
            if subset_times:
                report.subset_times_s = subset_times
                report.elapsed_time_s = round(sum(subset_times.values()), 3)
        except Exception:
            pass
        # Avoid writing extra JSONs under reports; timing is embedded in report JSON

        # Generate and display a summary table of results
        try:
            report_table = gen_table(report_list=[report], add_overall_metric=self.benchmark.add_overall_metric)
            logger.info(f'\n{self.benchmark_name} report table:'
                        f'\n{report_table} \n')
        except Exception:
            logger.error('Failed to generate report table.')

        # Generate detailed analysis if requested in configuration
        if self.task_config.analysis_report:
            logger.info('Generating report analysis, please wait ...')
            analysis = report.generate_analysis(self.task_config.judge_model_args)
            logger.info(f'Report analysis:\n{analysis}')
        else:
            logger.info('Skipping report analysis (`analysis_report=False`).')

        # Save the complete report to file
        report.to_json(report_file)
        logger.info(f'Dump report to: {report_file} \n')
        return report

    def finalize(self, *args, **kwargs):
        self.benchmark.finalize(*args, **kwargs)
