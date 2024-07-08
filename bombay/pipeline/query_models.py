# bombay/pipeline/query_models.py
from abc import ABC, abstractmethod
from openai import OpenAI

class QueryModel(ABC):
    @abstractmethod
    def generate(self, query, relevant_docs):
        pass


# GPT 기반 질의 모델 어댑터
class OpenAIQuery(QueryModel):
    def __init__(self, api_key, model):
        """
        GPT 기반 질의 모델 초기화
        :param api_key: OpenAI API 키
        :param model: 사용할 GPT 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, query, relevant_docs):
        """
        쿼리와 관련 문서를 사용하여 GPT로 답변을 생성하는 메소드
        :param query: 사용자 쿼리
        :param relevant_docs: 관련 문서 리스트
        :return: 생성된 답변
        """
        relevant_docs_str = ' '.join(relevant_docs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Be sure to refer to Relevant documents to answer questions. Relevant documents: {relevant_docs_str} "},
                {"role": "user", "content": f"questions: {query}"}
            ]
        )
        return response.choices[0].message.content