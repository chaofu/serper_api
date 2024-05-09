import asyncio
import json
import logging
from typing import AsyncIterable, Awaitable

import uvicorn
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from fastapi.responses import StreamingResponse

from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate
from starlette.requests import Request
import time
import os
import yaml
from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from langchain.schema import HumanMessage

api_base_url = "http://192.168.0.123:20000/v1"
api_key = "EMPTY"
LLM_MODEL = "Qwen1.5-7B-Chat"

class GPTAnswer:
    TOP_K = 10  # Top K documents to retrieve

    def __init__(self):
        # Load configuration from a YAML file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model_name = self.config["model_name"]
        self.api_key = self.config["openai_api_key"]
        self.api_base_url = self.config["api_base_url"]

    def _format_reference(self, relevant_docs_list, link_list):
        # Format the references from the retrieved documents for use in the prompt
        reference_url_list = [(relevant_docs_list[i].metadata)['url'] for i in range(self.TOP_K)]
        reference_content_list = [relevant_docs_list[i].page_content for i in range(self.TOP_K)]
        reference_index_list = []
        # Get the index of each reference in the link list
        for link in reference_url_list:
            try:
                # 尝试获取链接在链接列表中的索引，并加1
                index = link_list.index(link) + 1
            except ValueError:
                # 如果链接不在链接列表中，可以决定如何处理
                # 这里只是简单地打印一条消息，并继续循环
                print(f"链接 {link} 未在链接列表中找到。")
                index = None  # 或者你可以选择一个默认值，或者跳过这个链接
            else:
                # 如果没有异常，将索引添加到结果列表中
                reference_index_list.append(index)

        rearranged_index_list = self._rearrange_index(reference_index_list)

        # Create a formatted string of references
        formatted_reference = "\n"
        for i in range(len(rearranged_index_list)):
            formatted_reference += ('Webpage[' + str(rearranged_index_list[i]) + '], url: ' + reference_url_list[i] + ':\n' + reference_content_list[i] + '\n\n\n')
        return formatted_reference

    def _rearrange_index(self, original_index_list):
        # Rearrange indices to ensure they are unique and sequential
        index_dict = {}
        rearranged_index_list = []
        for index in original_index_list:
            if index not in index_dict:
                index_dict.update({index: len(index_dict)+1})
                rearranged_index_list.append(len(index_dict))
            else:
                rearranged_index_list.append(index_dict[index])
        return rearranged_index_list

    def get_prompt(self, query, relevant_docs, language, output_format, profile)->str:
        template = self.config["template"]
        prompt_template = PromptTemplate(
            input_variables=["profile", "context_str", "language", "query", "format"],
            template=template
        )

        profile = "conscientious researcher" if not profile else profile
        summary_prompt = prompt_template.format(context_str=relevant_docs, language=language, query=query, format=output_format, profile=profile)
        print("\n\n", "="*30, "GPT's Answer: ", "="*30, "\n")

        return summary_prompt


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        logging.exception(e)
        # TODO: handle exception
        msg = f"Caught exception: {e}"

    finally:
        # Signal the aiter to stop.
        event.set()


app = FastAPI()


    
@app.get("/chat/llm/{query}")
async def chat_llm(request: Request):
    query = request.path_params['query']
    print("query===", query)
    stream = True

    content_processor = GPTAnswer()
    output_format = "" # User can specify output format
    profile = "" # User can define the role for LLM

    # Fetch web content based on the query
    web_contents_fetcher = WebContentFetcher(query)
    context, serper_response = web_contents_fetcher.fetch()
    web_context = [item for item in context if item]
    if len(web_context) == 0:
        chat_prompt = content_processor.get_prompt(query, "", "zh-cn", output_format, profile)
    else:
        # Retrieve relevant documents using embeddings
        retriever = EmbeddingRetriever()
        relevant_docs_list = retriever.retrieve_embeddings(web_context, serper_response['links'], query)
        formatted_relevant_docs = content_processor._format_reference(relevant_docs_list, serper_response['links'])
        chat_prompt = content_processor.get_prompt(query, formatted_relevant_docs, serper_response['language'], output_format, profile)
   
   # print(gpt_answer)
    async def chat_iterator() -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(
            streaming=True,
            verbose=True,  # 为true 的时候，不写callback 这个，也会默认 callback
            callbacks=[callback],
            openai_api_key=api_key,
            openai_api_base=api_base_url,
            model_name=LLM_MODEL,
            max_tokens=20000,
            temperature=0
        )
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            llm.agenerate(messages=[[HumanMessage(content=chat_prompt)]]),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                print(token, end="", flush=True)
                yield json.dumps(
                    {"text": token},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token

            yield json.dumps(
                {"text": answer},
                ensure_ascii=False)

        await task

    return StreamingResponse(chat_iterator(), media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
