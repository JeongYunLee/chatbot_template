# Visual Studio Code 설치 및 사용법

## vscode 설치

Visual Studio Code는 마이크로소프트 사에서 개발한, 코드 편집이 가능한 에디터입니다. 

공식 홈페이지에서 다운로드 받아 설치합니다. 

**https://code.visualstudio.com/Download**

![install](../static/img/etc/vscode_install.png)

자세한 설치 방법은 디음 페이지에 기술되어 있습니다.

**https://spartacodingclub.kr/blog/vscode**

## vscode 사용법

vscode를 실행하면 다음과 같은 초기 화면이 뜹니다.

![install](../static/img/etc/vscode_start.png)

코드 편집에 필요한 확장 프로그램을 다운받습니다. 왼쪽 사이드바에서 퍼즐 모양 아이콘을 클릭합니다.

**1. 한글 언어팩**

vscode 기본 언어를 한국어로 설정하는 확장입니다.

![install](../static/img/etc/vscode_korean.png)


**2. Python & Python Debugger**

vscode에 Python 개발 환경을 구축하는 확장입니다. Python Debugger도 설치해줍니다.

![install](../static/img/etc/vscode_python.png)


**3. Live Server**

웹 개발에 있어 필수적인 확장으로, 코딩하고 있는 홈페이지를 확인할 수 있는 실시간 서버가 열립니다.

![install](../static/img/etc/vscode_liveserver.png)

**4. Jupyter Notebook**

주피터 노트북은 파이썬 코딩 시 사용하는 편집 도구로, 단계적으로 코드를 실행할 수 있습니다.

![install](../static/img/etc/vscode_jupyter.png)

그 외 개발하면서 필요한 기능이 있을 경우 찾아서 설치하시면 됩니다.

### 코드 작성

이제 vscode를 바탕으로 코드를 작성해보겠습니다. 초기 화면에서 새 파일을 생성해 작성하거나, 기존 파일 및 폴더를 불러올 수 있고 git repo를 복제할 수 있습니다. 

![install](../static/img/etc/vscode_code1.png)

'테스트' 폴더를 불러온 후, 파일 추가 아이콘을 클릭해 test.ipynb 파일을 생성합니다. ipynb는 jupyter notebook 확장명입니다.

![install](../static/img/etc/vscode_code2.png)

주피터 노트북은 셀 단위로 마크업과 코드를 작성할 수 있습니다. 각각 Markup, 코드 셀을 추가해 내용을 작성해줍니다. Markup 에서는 기본 html 문법을 사용할 수 있습니다.

![install](../static/img/etc/vscode_code3.png)

셀 단위 실행은 Shift + Enter 입니다. 코드 셀을 실행하면 다음과 같이 커널 소스 선택창이 나옵니다. 

![install](../static/img/etc/vscode_code4.png)

Python 환경을 클릭하면 다음과 같이 PC에 설치되어 있는 파이썬 커널을 선택할 수 있습니다. 

없는 경우 파이썬 공식 홈페이지에서 설치합니다.

**https://www.python.org/downloads/**

![install](../static/img/etc/vscode_code5.png)

커널을 선택한 뒤 실행하면 다음과 같이 마크업과 코드가 정상적으로 실행되는 것을 확인할 수 있습니다.

![install](../static/img/etc/vscode_code6.png)

## Tips

**단축키 모음**

1. **Ctrl + S : 코드 저장 (중요!)**

2. Ctrl + F : 텍스트 찾기

3. Ctrl + H : 텍스트 바꾸기

4. Ctrl + , (콤마) : 설정창 열기

5. Ctrl + ` (백틱) : 터미널 열기/닫기 
![install](../static/img/etc/vscode_terminal.png)

6. Ctrl + B : 왼쪽 사이드 탐색기 열기/닫기
![install](../static/img/etc/vscode_ctrlb.png)

7. Alt + 클릭 : 다중 커서 생성/제거

8. Ctrl + Tab : 들여쓰기 취소

9. 화살표 위, 아래 : 터미널에서 이전에 작성한 명령어 생성

그 외 다양한 단축키는 다음 페이지에 기술되어 있습니다.

**https://demun.github.io/vscode-tutorial/shortcuts/**


**확장 추천**

**1. Material Icon Theme**

Material Icon Theme은 사이드바에 표시되는 파일 아이콘 테마를 변경할 수 있습니다.

![install](../static/img/etc/vscode_icon.png)


**2. Prettier**

소스 코드의 가독성을 높이는 코드 포맷터 입니다.

![install](../static/img/etc/vscode_prettier.png)


**3. Rainbow CSV**

쉼표 단위의 CSV 파일 가독성을 높이는 포맷터 입니다다.

![install](../static/img/etc/vscode_rainbow.png)


