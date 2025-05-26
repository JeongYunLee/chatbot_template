# chatbot_template
Used: LangChain, LangGraph, RAG, Chainlit

## 0. VScode에서 실습 폴더 열기

## 1. 가상환경 설정
```
# 1. generate python bubble
python -m venv env

# 2. activate
source env/bin/activate  # mac
env\Scripts\activate.bat  # window
```

## 2. requirements.txt 설치

* 가상환경에서 `pip install -r requirements.txt` 실행

## 3. `.env` 설정

* `OPENAI_API_KEY`에 OpenAI API 키 입력하기
* (선택사항) `LLAMA_CLOUD_API_KEY는 나중에 Llama Parser 사용할 사람만 설정

## 4. chainlit 실행 방법
* `chainlit run main.py -w` 실행하고, 웹 브라우져에서 localhost:8000으로 접속
