# 🤖 RAG chatbot
LangChain과 Chainlit을 활용한 RAG기반 챗봇 만들기 실습

## 0. VScode, Python 설치

* vscode.md 파일 참고
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
env\Scripts\activate.bat  # window
```

## 3. requirements.txt 설치

* 가상환경에서 `pip install -r requirements.txt` 실행

## 4. chainlit 실행 방법
* `chainlit run main.py -w` 실행하고, 웹 브라우져에서 localhost:8000으로 접속

## 참고자료

### backend
* [<랭체인LangChain 노트> - LangChain 한국어 튜토리얼](https://wikidocs.net/book/14314)
* [Tavily Search](https://www.tavily.com/): 부분 문료
* [Llama Parse](https://www.llamaindex.ai/llamaparse): 부분 무료
* [Upstage Document Parse](https://www.upstage.ai/products/document-parse): 유료

### frontend
* [Chinlit](https://docs.chainlit.io/get-started/overview)
* [Streamlit](https://streamlit.io/)
* [Gradio](https://www.gradio.app/)

