# 필수 설치
# pip install langchain
# pip install langchain-openai
# pip install -U langchain-community tavily-python
# pip install beautifulsoup4
# pip install langchainhub
# pip install unstructured
# pip install uvicorn
# pip install "fastapi[all]"
# pip install python-dotenv
# pip install pip install faiss-cpu

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


# 텍스트 메타데이터용 코드
# from langchain_community.document_loaders import DirectoryLoader

# ---------- FastAPI -----------
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# ---------- Tavily ----------
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TopicSection(BaseModel):
    topic: str
    section: str
    input_link: str


class Result(BaseModel):
    section: str
    result: str


@app.get("/")
def root():
    return "Welcome to root"


# 기획서 생성
@app.post("/receive_json")
async def receive_text(topic_section: TopicSection):

    topic = topic_section.topic
    section = topic_section.section

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "프로젝트 주제 {topic}에 대한 기획서의 {section}부분을 작성해주고 html에 바로 삽입할거라서 ###나 **같은 기호는 사용하지 말고 \n으로 줄바꿈을 나타내줘",
            )
        ]
    )

    # Langchain
    chain = prompt | llm | output_parser

    async def invoke():
        result = await chain.ainvoke({"section": section, "topic": topic})
        return result

    result = await invoke()

    return JSONResponse(content=jsonable_encoder(result))


# RAG
@app.post("/receive_rag")
async def execute_retrival(topic_section: TopicSection):

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    topic = topic_section.topic
    section = topic_section.section
    url = topic_section.input_link

    async def retriver_rag():
        loader = WebBaseLoader(url)
        # loader = DirectoryLoader(".", glob="data/*.txt", show_progress=True) -> loader = WebBaseLoader(url)과 중복 적용 X
        docs = loader.load()

        embeddings = OpenAIEmbeddings()

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)

        retriever = vector.as_retriever()

        result = await tavily_search(retriever)
        return result

    # tavily api
    async def tavily_search(retriever):

        retriever_tool = create_retriever_tool(
            retriever,
            "information_search",
            "you must use this tool!",
        )

        search = TavilySearchResults()
        tools = [retriever_tool, search]

        prompt = hub.pull("hwchase17/openai-functions-agent")

        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        result = await invoke_tavily(agent_executor)

        result = result["output"]

        return result

    async def invoke_tavily(agent_executor):
        if topic == "제안배경 및 필요성":
            return await agent_executor.ainvoke(
                {
                    "input": f"""프로젝트 주제 {topic}에 대한 기획서의 {section}부분을 기획서 형식으로 작성해줘. 기획서에 들어가는 문장은 존댓말을 쓰면 안돼. 꼭! 업계 동향도 알려줘!
                최신 자료를 바탕으로 확실한 근거를 가지고 작성해주고 자료의 출처도 마지막에 적어줘. 출처는 a태그로 감싸주고 파란색으로 보이게 해줘. 결과물에는 개행표시는 쓰지말고 <br>태그를 넣어줘. 
                br태그의 위치는 소제목 다음에 있으면 좋겠어. 특수문자도 쓰지 마. 특히 ** 이 특수문자는 절대 쓰지마!
                소제목은 b태그로 감싸줘."""
                }
            )
        else:
            return await agent_executor.ainvoke(
                {
                    "input": f"""프로젝트 주제 {topic}에 대한 기획서의 {section}부분을 기획서 형식으로 작성해줘. 기획서에 들어가는 문장은 존댓말을 쓰면 안돼. 
                최신 자료를 바탕으로 확실한 근거를 가지고 작성해주고 자료의 출처도 마지막에 적어줘. 출처는 a태그로 감싸주고 파란색으로 보이게 해줘. 결과물에는 개행표시는 쓰지말고 <br>태그를 넣어줘. 
                br태그의 위치는 소제목 다음에 있으면 좋겠어. 특수문자도 쓰지 마. 특히 ** 이 특수문자는 절대 쓰지마!
                소제목은 b태그로 감싸줘.
                예시를 알려줄게
                
                제안 배경
                2023년은 LLM (대형 언어 모델) 분야에서 상당한 성장과 혁신이 이루어진 해였다. AI 기술의 빠른 발전과 함께, 오픈 소스 LLM의 등장은 기술과 상호 작용하는 방식을 재구성하고 있다. 특히, 오픈소스 LLM은 비용 절감, 시스템의 효율성 증대, 그리고 다양한 영역에서의 적용 가능성을 제공하며, 이는 기업과 스타트업에게 새로운 기회를 열어주고 있다.

                필요성
                오픈소스 LLM을 활용한 키오스크 사업은 여러 면에서 혁신적이다. 첫째, 사용자 경험을 극대화할 수 있다. AI 기반의 언어 모델을 통해 사용자의 요구를 더 정확하게 파악하고, 이에 맞는 응답을 제공함으로써 사용자 만족도를 높일 수 있다. 둘째, 기업의 운영 효율성을 증대시킬 수 있다. 오픈소스 LLM을 이용하면 개발 비용과 시간을 크게 절약할 수 있으며, 이는 특히 자본이 제한적인 중소기업이나 스타트업에게 큰 이점을 제공한다. 셋째, 키오스크 사업의 경쟁력을 강화할 수 있다. AI 기술을 통해 제공할 수 있는 서비스의 범위가 확장되며, 이는 소비자에게 보다 차별화된 서비스를 제공할 수 있는 기회를 의미한다.
                                
                개발 내용
                본 프로젝트는 LLM을 활용한 키오스크 사업을 목표로 하며, 최신 오픈소스 LLM 모델을 기반으로 사용자 경험을 혁신하고, 기업 운영의 효율성을 극대화할 방안을 모색한다. 이를 위해 다음과 같은 개발 내용을 포함한다.

                1. 오픈소스 LLM 모델의 선정 및 적용
                2024년 최고의 오픈소스 LLM 중 하나인 라마 2를 주요 모델로 선정한다. 라마 2는 메타 AI에서 개발한 모델로, 다양한 크기와 용도에 맞춰 최적화된 성능을 제공한다. 이 모델을 키오스크 시스템에 적용하여, 사용자 질문에 대한 정확하고 빠른 응답을 가능하게 한다.

                2. 사용자 인터페이스(UI) 개선
                LLM의 자연어 처리 능력을 활용하여, 키오스크의 사용자 인터페이스를 개선한다. 이를 통해 사용자가 자연어로 질문하거나 명령을 할 수 있는 직관적인 UI를 제공하며, 사용자 경험을 대폭 향상시킨다.

                3. 맞춤형 서비스 제공
                사용자 데이터 분석을 통해 개인화된 서비스를 제공한다. LLM을 이용하여 사용자의 선호도와 이전 상호작용을 분석, 맞춤형 메뉴 추천이나 프로모션 정보를 제공함으로써 사용자 만족도를 높인다.

                4. 효율적인 시스템 운영
                오픈소스 LLM을 활용함으로써 개발 비용과 시간을 절약한다. 또한, 시스템 유지보수 및 업데이트를 용이하게 하여, 운영 효율성을 증대시킨다.
                                """
                }
            )

    result = await retriver_rag()

    return JSONResponse(content=jsonable_encoder(result))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)
