# SKN08-4th-2Team
SKN08-4th-2Team

SK Networks AI 8기 과정 4번째 단위 프로젝트입니다.

# 4차 프로젝트
 
# 1. 팀 소개
- 팀명: SNACK
<table align=center>
  <tbody>
    <tr>
      <td align=center><b>유제나</b></td>
      <td align=center><b>조준희</b></td>
      <td align=center><b>주고은</b></td>
      <td align=center><b>손승일</b></td>
      <td align=center><b>정현서</b></td>
    </tr>
    <tr>
      <td align="center">
        <div>
          <img width="200" alt="image" src="https://avatars.githubusercontent.com/u/167101468?v=4"width="200px;" alt="유제나">
        </div>
      </td>
      <td align="center">
        <div>
          <img src = "https://avatars.githubusercontent.com/u/188949785?v=4" width="200px" height="200px"; alt="조준희"/>
        </div>
      </td>
      <td align="center">
        <div>
          <img src="https://avatars.githubusercontent.com/u/133327235?v=4" width="200" height="200" alt="주고은">
        </div>
      </td>
      <td align="center">
         <img src="https://avatars.githubusercontent.com/u/145750431?v=4" width="200" height="200" class="avatar avatar-user width-full border color-bg-default">
      </td>
       <td align="center">
        <img src="https://avatars.githubusercontent.com/u/122842851?v=4"width="200px;" alt="정현서"/>
      </td>
    </tr>
    <tr>
      <td><a href="https://github.com/denalog"><div align=center>@denalog</div></a></td>
      <td><a href="https://github.com/CodeLego8"><div align=center>@CodeLego8</div></a></td>
      <td><a href="https://github.com/Goeun-Ju"><div align=center>@Goeun-Ju</div></a></td>
      <td><a href="https://github.com/ajeseung"><div align=center>@ajeseung</div></a></td>
      <td><a href="https://github.com/jungs0914"><div align=center>@jungs0914</div></a></td>
    </tr>
  </tbody>
</table>
<br><br><br>


# 2. Tech Stack (기술 스택)

>### <span style="color:cyan"> Co-Work Tool </span>
<table>
  <tr>
    <td>Communication & Messenger</td>
    <td><img src="https://img.shields.io/badge/Discord-5865F2?style=flat&logo=Discord&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/Notion-000000?style=flat&logo=Notion&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/Slack-4A154B?style=flat&logo=Slack&logoColor=white"/></td>
  </tr>
  <tr>
    <td>Development & Merge</td>
    <td><img src="https://img.shields.io/badge/Git-F05032?style=flat&logo=Git&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=GitHub&logoColor=white"/></td>
  </tr>  
</table>


>### <span style="color:cyan"> Data Server </span>
<table>
  <tr>
    <td>PyCharm</td>
    <td><img src="https://img.shields.io/badge/pycharm-%23000000?style=flat&logo=pycharm&logoColor=white"/></td>
    <td><img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white"/></td>
    td><img src="https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white"/></td>
  </tr>  
  </tr>  
  <tr>
    <td>RDBMS</td>
    <td><img src="https://img.shields.io/badge/mysql-4479A1?style=flat&logo=mysql&logoColor=white"/></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
  
</table>

>### <span style="color:cyan"> AI </span>
<table>
  <tr>
    <td>OpenAI</td>
    <td><img src="https://img.shields.io/badge/GPT--3.5--Turbo-00A67E?style=flat&logo=openai&logoColor=white"/></td>
  </tr>
  <tr>
    <td>RAG</td>
    <td><img src="https://img.shields.io/badge/LangChain-FF9900?style=flat&logo=Chainlink&logoColor=white"/></td>
  </tr>
  <tr>
    <td>Vector DB</td>
    <td><img src="https://img.shields.io/badge/FAISS-009688?style=flat&logo=Databricks&logoColor=white"/></td>
  </tr>
    
  </tr>  
  
</table>

>### <span style="color:cyan"> API Development & Testing </span>
<table>
  <tr>
    <td><img src="https://img.shields.io/badge/Postman-FF6C37?style=flat&logo=postman&logoColor=white"/></td>
  </tr>  
</table>

<br><br><br>


 
# 3. 프로젝트 개요
### ✅프로젝트 명
OpenAi API를 활용한 서울 맛집 정보 추천 LLM 서비스 web-application

### ✅프로젝트 소개
서울 내 맛집 정보를 AI 모델을 활용하여 제공하는 LLM 기반 추천 서비스입니다.
사용자는 특정 지역, 음식 종류, 가격대 등 다양한 조건을 입력하여 자연어로 맛집을 추천받을 수 있으며,
모델은 블로그 리뷰, 평점, 위치 데이터 등을 기반으로 최적의 맛집을 추천합니다.
이를 위해 OpenAI의 GPT 모델을 파인튜닝하여 서울 맛집 데이터에 특화된 질의응답을 수행하도록 개선하며, 사용자 피드백을 반영하여 모델을 지속적으로 학습시킵니다. 또한, app을 배포를 위해 aws 인프라를 구축합니다.


### ✅프로젝트 필요성(배경)
1. 기존 맛집 검색 서비스는 광고 및 협찬이 포함된 결과가 많아, 신뢰할 수 있는 정보를 찾기가 어렵습니다.
2. 사용자가 원하는 정보(위치, 메뉴, 리뷰 기반 추천)를 효율적으로 제공하는 시스템이 부족합니다.
3. 블로그, SNS, 지도 리뷰 등 방대한 데이터를 활용하여 맞춤형 추천을 제공하는 AI 시스템이 필요합니다.
4. 사용자의 피드백을 바탕으로, 지속적으로 성능이 향상되는 LLM 시스템을 구축하고자 합니다.
5. aws 로 배포를 할 필요가 있습니다. 


### ✅프로젝트 목표

DJNAGO를 서버로 활용하고 OPENAI GPT 3.5-Turbo,FAISS,RAG을 활용하여 사용자 특성에 맞게 필요한 정보만 빠르게 제공하는 시스템을 개발하고자 합니다. 이를 위해 aws를 활용합니다. 

# 4. FAISS 전처리
```python
# 문서 로드 및 전처리
loader = TextLoader("맛집데이터.txt)  # 텍스트 파일 로드
documents = loader.load()

# 문서 임베딩을 위한 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# OpenAI 임베딩 모델을 사용하여 문서 벡터화
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embedding_model)

# 벡터스토어 기반 검색
retriever = vectorstore.as_retriever()
```
1. 문서 로드 및 전처리
- TextLoader를 이용하여 텍스트 데이터를 불러옵니다.
- CharacterTextSplitter를 이용하여 문서를 작은 청크로 나누어 임베딩을 효과적으로 수행할 수 있도록 합니다.

2. 임베딩 생성 및 FAISS 벡터 저장
- OpenAIEmbeddings()를 활용하여 문서 데이터를 벡터화합니다.
- FAISS.from_documents()를 사용하여 벡터 DB를 구축합니다.

3. FAISS 검색 기능 설정
- vectorstore.as_retriever()를 사용하여 LangChain과 연동 가능한 검색 시스템을 구축합니다.

# 5. RAG 
```python
# LLM (GPT) + FAISS 기반 질의응답 시스템 구축
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

# 사용자 입력을 받아 검색 + 응답 생성 예시 테스트 코드
query = "서울에서 유명한 한식 맛집을 추천해줘."
response = qa_chain.run(query)

print("\n💡 AI 응답:")
print(response)
```
1. GPT-3.5-Turbo와 LangChain을 사용한 RAG 질의응답
- RetrievalQA.from_chain_type()을 이용하여 검색 + GPT 응답을 자동으로 수행하는 파이프라인을 구축합니다.
- 사용자가 입력한 질문을 FAISS를 통해 검색 후 관련 문서를 기반으로 GPT가 응답을 생성합니다.

2. 질의 실행 및 응답 출력

- "서울에서 유명한 한식 맛집을 추천해줘." 같은 질의를 실행하면 FAISS에서 관련 문서를 검색한 후 GPT가 최적화된 응답을 제공합니다.
## LangChain + FAISS가 필요한 이유
1. 빠른 검색 – FAISS를 활용하면 대규모 문서에서도 유사한 정보를 빠르게 검색할 수 있습니다.
2. 정확한 응답 – LLM (GPT) 단독보다 검색 기반 응답이 더 정확한 정보를 제공합니다.
3. RAG (Retrieval-Augmented Generation) 적용 가능 – 외부 문서를 참조한 응답 생성을 수행하여 최신 정보를 반영할 수 있습니다.

# 6.Fine-Tuning
```python
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Reinforcement Learning from Human Feedback
openaiFineTuningRouter = APIRouter()

# 피드백 데이터 저장
feedbackData = []

# 파인튜닝 데이터 예시
trainingData = [

    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "서울에서 김치찌개 맛집 있어?"},
            {"role": "assistant",
             "content": "서울에는 김치찌개를 제공하는 수많은 맛집이 있습니다. 한일관, 이문설농탕 등이 특히 인기가 많으며, 깊고 풍부한 맛의 김치찌개를 제공합니다."}
        ]
    },
```
## 파인튜닝의 필요성
- 기본 GPT 모델은 일반적인 대화 능력은 뛰어나지만, 특정 도메인(예: 서울 맛집, 금융, 의료 등)에 대한 깊이 있는 정보를 제공하지 못함.
- 특정 지역(서울)이나 특정 기준(맛집 리뷰) 같은 맞춤형 응답을 원할 경우, Fine-Tuning을 통해 모델을 학습시켜야 함.

# 7. Postman 출력 결과
## 파인튜닝 및 RAG 수행 전
<img width="778" alt="스크린샷 2025-03-05 오후 4 44 33" src="https://github.com/user-attachments/assets/278c433f-46c0-4066-821f-a8bf9e17babe" />
- 출력 결과 독산동 '독수리매운갈비찜'은 존재하지 않는 음식점으로 거짓정보임이 확인되었고 openai api만으로는 좋은 품질의 서비스를 제공 하기 어려움을 깨달았습니다. 

## 파인튜닝 및 RAG 수행 후 
![image](https://github.com/user-attachments/assets/624e0fb1-80a2-4992-9a31-0d288df230fa)
![image](https://github.com/user-attachments/assets/1309efd0-de2f-44c0-9dbe-1f4d229ed2ec)
- 수행 이전과 다르게 실제 음식점을 추천하고 이에 대한 리뷰를 출력하는 것을 확인할 수 있습니다.

<br><br><br>
  ### 8. AWS 설정하기
  -AWS 계정 생성
  ![image](https://github.com/user-attachments/assets/fc52d54b-65e6-4269-b332-0b8478d38646)

  -EC2 Instance 생성
  ![image (1)](https://github.com/user-attachments/assets/700e69a8-7a5c-446a-bf20-c041a1ade2e0)

  -AMI 설정
  ![image (2)](https://github.com/user-attachments/assets/2d4a1545-cb6e-42c1-90ae-dad72ca33326)

  -키 페어 생성
  ![image (3)](https://github.com/user-attachments/assets/c9cc16ca-bbd0-45cb-a013-eb9e8742c7bc)

  -네트워크 설정
  ![image (4)](https://github.com/user-attachments/assets/c7277623-0c58-4569-92d5-0b335decea05)

 -스토리지 구성
 ![image (5)](https://github.com/user-attachments/assets/13901b43-26f1-47ce-91e5-df4519bec94a)


 ### 9. AWS에 LLM프로젝트 배포

 -EC2 서버에 원격 접속
 ![image (6)](https://github.com/user-attachments/assets/b66d7c7c-221b-4361-88ed-8c5051480a5a)
<br><br><br>



# 📌 프로젝트 후기 및 보완할 점
RAG 성능 향상을 위해 더 많은 리뷰 데이터와 최신 정보를 반영할 수 있도록 벡터 데이터베이스를 지속적으로 업데이트하는 방식이 필요하며, Fine-tuning 시 다양한 사용자 피드백을 반영하여 모델의 학습 효율성을 더욱 개선할 필요가 있습니다. aws 비용 절감을 위한 추가적인 설정이 필요합니다.  

## 한 줄 회고
- 유제나: LLM 연동부터 AWS 기반 배포까지 직접 수행하며 CI/CD 전반에 대한 실무 감각을 익혔습니다. 이 경험은 최종 프로젝트에서 더욱 완성도 높은 개발과 안정적인 배포로 이어질 기반이 될 것입니다.
- 조준희: LLM 기반 기능을 실제 서비스에 녹여내는 과정에서 API 연동, 상태 관리, 배포 환경 구성 등 개발 실무에서의 세부 사항을 깊이 있게 이해하게 되었다.
- 주고은: 사용자 요구에 기반한 기능 설계부터 LLM 서비스 연동, 클라우드 환경 배포까지 전 과정을 경험하며, 실제 서비스 개발과 운영의 본질에 대해 보다 입체적으로 이해할 수 있었다.
- 손승일: 개발된 결과물이 실제 사용자 환경에 배포되며 비로소 ‘서비스’가 된다는 점을 실감했다. 한 줄 한 줄의 코드가 시스템에 어떤 영향을 미치는지, 배포를 통해 보다 입체적으로 이해할 수 있었다.
- 정현서: 처음부터 끝까지 배포 흐름을 따라가며 실전 감각을 익혔다. 오류를 해결해가며 서비스가 살아나는 과정을 체험한 건, 개발자로서 가장 짜릿한 순간 중 하나였다.
