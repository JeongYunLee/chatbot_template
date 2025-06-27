# 🤖 RAG chatbot
LangChain과 Chainlit을 활용한 RAG기반 챗봇 만들기 실습

> 파일 설명
> 1. main.py: vanila RAG with Chainlit (single-turn)
> 2. main-history.py: multi-turn 대화가 가능한 버전. chainlit으로 테스트 가능
> 3. main-workflow.py: agent + router(workflow) architecture
>    - 사용한 chromadb collection은 공유하지 않음. chromadb의 collection을 생성하고 데이터를 적재해준 뒤 실행되며, 이 코드는 참고용으로 사용할 수 있음
>    - backend 코드만 있는 상태이고, chainlit은 적용하지 않은 상태
>4. main-agents.py: agent-only architecture
>    - main-chromadb와 동일한 collection을 사용했으며, 이 코드도 참고용으로 사용할 수 있음
>    - backend 코드만 있는 상태이고, chainlit은 적용하지 않은 상태


## 0. VScode, Python 설치

* File > Open Folder에서 실습 폴더(압축 해제한 폴더)를 선택

## 1. OpenAI API Key 발급, `.env` 설정

* `OPENAI_API_KEY`에 OpenAI API 키 입력하기
    * API KEY 발급 받기: [OpenAI Platform](https://platform.openai.com/) 접속 후, Settings > Billing에서 credit 충전 후(최소 5$), API Keys에서 발급
* (선택사항) `LLAMA_CLOUD_API_KEY는 나중에 Llama Parse 사용할 사람만 설정

## 2. 가상환경 설정
```
# 1. generate python bubble
python -m venv env

# 2. activate
source env/bin/activate  # mac
env\Scripts\activate  # window
``

## 3. requirements.txt 설치

* 가상환경에서 `pip install -r requirements.txt` 실행

## 4. chainlit 실행 방법
* `chainlit run main.py -w` 실행하고, 웹 브라우져에서 localhost:8000으로 접속

## 참고자료

### backend
* [<랭체인LangChain 노트> - LangChain 한국어 튜토리얼](https://wikidocs.net/book/14314)
* [Tavily Search](https://www.tavily.com/): 부분 무료
* [Llama Parse](https://www.llamaindex.ai/llamaparse): 부분 무료
* [Upstage Document Parse](https://www.upstage.ai/products/document-parse): 유료

### frontend
* [Chinlit](https://docs.chainlit.io/get-started/overview)
* [Streamlit](https://streamlit.io/)
* [Gradio](https://www.gradio.app/)

