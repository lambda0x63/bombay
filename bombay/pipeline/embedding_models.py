# bombay/pipeline/embedding_models.py
from abc import ABC, abstractmethod
from openai import OpenAI

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts):
        pass

    @abstractmethod
    def get_dimension(self):
        pass


# OpenAI 임베딩 모델 어댑터
class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key, model):
        """
        OpenAI 임베딩 모델 초기화
        :param api_key: OpenAI API 키
        :param model: 사용할 OpenAI 임베딩 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = None

    def embed(self, texts):
        """
        텍스트를 OpenAI 임베딩 모델로 임베딩하는 메소드
        :param texts: 임베딩할 텍스트 리스트
        :return: 임베딩 리스트
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [data.embedding for data in response.data]
        if self.dimension is None:
            self.dimension = len(embeddings[0])
        return embeddings

    def get_dimension(self):
        """
        OpenAI 임베딩 모델의 임베딩 차원을 반환하는 메소드
        :return: 임베딩의 차원
        """
        if self.dimension is None:
            sample_document = 'This is a sample document to get embedding dimension.'
            self.dimension = len(self.embed([sample_document])[0])
        return self.dimension