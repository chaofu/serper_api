import asyncio
import json
import logging
from asyncio.log import logger
from typing import AsyncIterable, Awaitable

import uvicorn
import openai
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
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

human_prompt = "{input}"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate

class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )



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
        reference_index_list = [link_list.index(link)+1 for link in reference_url_list]
        rearranged_index_list = self._rearrange_index(reference_index_list)

        # Create a formatted string of references
        formatted_reference = "\n"
        for i in range(self.TOP_K):
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

    def get_prompt(self, query, relevant_docs, language, output_format, profile):
        # Create an instance of ChatOpenAI and generate an answer
        #llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.api_key, temperature=0.0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        # llm = ChatOpenAI(
        #             streaming=True,
        #             verbose=True,   # 为true 的时候，不写callback 这个，也会默认 callback
        #             callbacks=[StreamingStdOutCallbackHandler()],
        #             openai_api_key=self.api_key,
        #             openai_api_base="https://jiekou.wlai.vip/v1/chat/completions",
        #             model_name=self.model_name
        # )
        template = self.config["template"]
        prompt_template = PromptTemplate(
            input_variables=["profile", "context_str", "language", "query", "format"],
            template=template
        )

        profile = "conscientious researcher" if not profile else profile
        summary_prompt = prompt_template.format(context_str=relevant_docs, language=language, query=query, format=output_format, profile=profile)
        # print("\n\nThe message sent to LLM:\n", summary_prompt)
        print("\n\n", "="*30, "GPT's Answer: ", "="*30, "\n")
        #gpt_answer = llm([HumanMessage(content=summary_prompt)])

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


@app.get("/stream")
async def root(request: Request):
    async def event_generator(request: Request):
        res_str = "双天至尊真是一部好的电视剧！！！"
        for i in res_str:
            if await request.is_disconnected():
                print("连接已中断")
                break
            data = f'"event": "message"\n"data":{i}\n'
            yield data
            await asyncio.sleep(1)

    g = event_generator(request)
    return StreamingResponse(g, media_type="text/event-stream")

@app.get("/hello")
def hello(request: Request):
    return "hello"


@app.get("/chat/{query}")
async def hao(request: Request):
    query = request.path_params['query']
    print("query===", query)
    openai.api_key = "EMPTY"
    print(f"{openai.api_key=}")
    openai.api_base = api_base_url
    print(f"{openai.api_base=}")
    msg = {"stream": True,
           "model": "chatglm3-6b",
           "messages": "您好",
           "temperature": 0.7,
           "n": 1
           }
    hao = True

    async def get_response(query):

        try:
            response = await openai.ChatCompletion.acreate(model="chatglm3-6b",
                                                           messages=query,
                                                           temperature=0.5,
                                                           max_tokens=2048,
                                                           top_p=1,
                                                           stream=True
                                                           )
            if hao:
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield chunk
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(answer)
                    yield (answer)
        except Exception as e:
            msg = f"获取ChatCompletion时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}')

    return StreamingResponse(
        get_response(query),
        media_type='text/event-stream',
    )


@app.get("/chat/llm/{query}")
async def chat_llm(request: Request):
    query = request.path_params['query']
    print("query===", query)
    stream = True

    content_processor = GPTAnswer()
    query = "Ubuntu 22.04系统如何配置静态IP"
    output_format = "" # User can specify output format
    profile = "" # User can define the role for LLM

    # Fetch web content based on the query
    web_contents_fetcher = WebContentFetcher(query)
    context, serper_response = web_contents_fetcher.fetch()
    # print(context)

    # Retrieve relevant documents using embeddings
    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrieve_embeddings(context, serper_response['links'], query)
    formatted_relevant_docs = content_processor._format_reference(relevant_docs_list, serper_response['links'])
    chat_prompt = content_processor.get_prompt(query, formatted_relevant_docs, serper_response['language'], output_format, profile)
    async def chat_iterator(query) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None
        new_model = ChatOpenAI(
            streaming=True,
            verbose=True,  # 为true 的时候，不写callback 这个，也会默认 callback
            callbacks=callbacks,
            openai_api_key=api_key,
            openai_api_base=api_base_url,
            model_name=LLM_MODEL
        )

        # LLMChain 被认为是查询 LLM 对象最常用的方法之一。它根据提示模板将提供的输入键值和内存键值（如果存在）进行格式化，
        # 然后将格式化后的字符串发送给 LLM，LLM 生成并返回输出结果
        # 生成提示语模板，需要用"""text"""包裹文本，同时用花括号{}包裹随用户输入而改变的部分
        # prompt_template =  (
        #     '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 </指令>\n'
        #     '<已知信息>{{ context }}</已知信息>\n'
        #     '<问题>{{ question }}</问题>\n'
        # )
        # input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        # chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(llm=new_model)
        # print("llm-==", model)
        # chain = LLMChain(prompt=prompt, llm=new_model)
        # print(chain({"商品": "牛奶"}))
        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall([HumanMessage(content=chat_prompt)]),
            callback.done),
        )

        if stream:
            print("stream===",stream)
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

            print(answer)
            yield json.dumps(
                {"text": answer},
                ensure_ascii=False)

        await task

    return StreamingResponse(chat_iterator(query), media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
