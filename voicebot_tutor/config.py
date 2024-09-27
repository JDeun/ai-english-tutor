import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 설정
LLM_MODEL = "gpt-4o-mini"  # 대화 생성을 위한 GPT 모델
TTS_MODEL = "tts-1"  # Text-to-Speech 모델
STT_MODEL = "whisper-1"  # Speech-to-Text 모델

# 음성 설정
VOICE_OPTION = "nova"  # TTS 음성 옵션
TTS_SPEED = 1.2  # TTS 음성 속도 (1.0이 기본 속도)

# LLM 하이퍼파라미터 설정
MAX_TOKENS = None  # 생성할 최대 토큰 수
TEMPERATURE = 0.3  # 응답의 다양성 조절 (0.0 ~ 2.0), 낮은 값으로 설정하여 더 일관된 응답 생성
TOP_P = 0.9  # 상위 확률 샘플링 (0.0 ~ 1.0)
FREQUENCY_PENALTY = 0.5  # 반복 표현 억제 (-2.0 ~ 2.0)
PRESENCE_PENALTY = 0.5  # 새로운 주제 도입 장려 (-2.0 ~ 2.0)
STOP = None  # 특정 문자열에서 생성 중단 (필요시 리스트로 지정)

# 프롬프트 파일 읽기 (영어 지시사항, 한글 예시)
with open("prompt_en.txt", "r", encoding="utf-8") as file:
    SYSTEM_MESSAGE = file.read()

# 초기 인사 메시지
INITIAL_GREETING = "안녕! 오늘은 무슨 일이 있었어? 사소한 것도 좋아. 말해줄 수 있어?"