import asyncio
from typing import Any, Optional, Sequence

from llama_index.async_utils import run_async_tasks
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.prompts.mixin import PromptDictType
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.service_context import ServiceContext
from llama_index.types import RESPONSE_TEXT_TYPE, BaseModel

import time


class TreeSummarize(BaseSynthesizer):
    """
    Tree summarize response builder.

    This response builder recursively merges text chunks and summarizes them
    in a bottom-up fashion (i.e. building a tree from leaves to root).

    More concretely, at each recursively step:
    1. we repack the text chunks so that each chunk fills the context window of the LLM
    2. if there is only one chunk, we give the final response
    3. otherwise, we summarize each chunk and recursively summarize the summaries.
    """

    def __init__(
        self,
        summary_template: Optional[BasePromptTemplate] = None,
        service_context: Optional[ServiceContext] = None,
        output_cls: Optional[BaseModel] = None,
        streaming: bool = False,
        use_async: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context, streaming=streaming, output_cls=output_cls
        )
        self._summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
        self._use_async = use_async
        self._verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"summary_template": self._summary_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "summary_template" in prompts:
            self._summary_template = prompts["summary_template"]

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._service_context.prompt_helper.repack(
            summary_template, text_chunks=text_chunks
        )

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = self._service_context.llm.stream(
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                if self._output_cls is None:
                    response = await self._service_context.llm.apredict(
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )
                else:
                    response = await self._service_context.llm.astructured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )

            # return pydantic object if output_cls is specified
            return response

        else:
            # summarize each chunk
            if self._output_cls is None:
                tasks = [
                    self._service_context.llm.apredict(
                        summary_template,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]
            else:
                tasks = [
                    self._service_context.llm.astructured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]

            summary_responses = await asyncio.gather(*tasks)
            if self._output_cls is not None:
                summaries = [summary.json() for summary in summary_responses]
            else:
                summaries = summary_responses

            print(len(summaries))

            # recursively summarize the summaries
            return await self.aget_response(
                query_str=query_str,
                text_chunks=summaries,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._service_context.prompt_helper.repack(
            summary_template, text_chunks=text_chunks
        )

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = self._service_context.llm.stream(
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                if self._output_cls is None:
                    response = self._service_context.llm.predict(
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )
                else:
                    response = self._service_context.llm.structured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )

            return response

        else:
            # summarize each chunk
            if self._use_async:
                if self._output_cls is None:
                    tasks = [
                        self._service_context.llm.apredict(
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                else:
                    tasks = [
                        self._service_context.llm.astructured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]

                summary_responses = run_async_tasks(tasks)

                if self._output_cls is not None:
                    summaries = [summary.json() for summary in summary_responses]
                else:
                    summaries = summary_responses
            else:
                if self._output_cls is None:
                    summaries = [
                        self._service_context.llm.predict(
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                else:
                    summaries = [
                        self._service_context.llm.structured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                    summaries = [summary.json() for summary in summaries]

            # recursively summarize the summaries
            return self.get_response(
                query_str=query_str, text_chunks=summaries, **response_kwargs
            )

    async def aget_response_rate_limit(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        max_tokens_per_batch: int = 60000,
        rate_limit_delay: int = 60,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        # Algorithm:
            # 1. Estimate token size of each chunk.
            # 2. create batches of chunks that can be asynchrnously summarized.
            # 3. Process each batch and add delay to respect limit.
            # 4. Stich all batch results.
            # 5. Perform this recursively until there is one chunk.
        summary_template = self._summary_template.partial_format(query_str=query_str)
        # repack text_chunks so that each chunk fills the context window
        text_batches = self._service_context.prompt_helper.repackIntoBatches(
            summary_template, text_chunks=text_chunks, max_tokens_per_batch=max_tokens_per_batch,
        )

        if self._verbose:
            print(f"{len(text_batches)} text_batches after repacking")

        summaries = []

        # Process each batch asynchronously.
        for text_chunks in text_batches:
            print(f"Processing batch of {len(text_chunks)} chunks")
            
            # Start the timer
            start_time = time.time()

            # This code is same as aget_response for text_chunks.
            # give final response if there is only one chunk
            if len(text_chunks) == 1:
                response: RESPONSE_TEXT_TYPE
                if self._streaming:
                    response = self._service_context.llm.stream(
                        summary_template, context_str=text_chunks[0], **response_kwargs
                    )
                else:
                    if self._output_cls is None:
                        response = await self._service_context.llm.apredict(
                            summary_template,
                            context_str=text_chunks[0],
                            **response_kwargs,
                        )
                    else:
                        response = await self._service_context.llm.astructured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunks[0],
                            **response_kwargs,
                        )

                # return pydantic object if output_cls is specified
                return response

            else:
                # summarize each chunk
                if self._output_cls is None:
                    tasks = [
                        self._service_context.llm.apredict(
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                else:
                    tasks = [
                        self._service_context.llm.astructured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]

                summary_responses = await asyncio.gather(*tasks)
                if self._output_cls is not None:
                    batch_summaries = [summary.json() for summary in summary_responses]
                else:
                    batch_summaries = summary_responses
                summaries.extend(batch_summaries)

                # Add a delay after processing each batch.
                end_time = time.time()

                # Calculate the time interval
                time_interval = end_time - start_time
                sleep_time = max(0, rate_limit_delay - time_interval)

                if self._verbose:
                    print(f"Batch processing took {time_interval:.2f} seconds")
                    print(f"Waiting out {sleep_time:.2f} seconds")
    
                await asyncio.sleep(sleep_time)

        # recursively summarize the summaries
        return await self.aget_response_rate_limit(
            query_str=query_str,
            text_chunks=summaries,
            **response_kwargs,
        )
