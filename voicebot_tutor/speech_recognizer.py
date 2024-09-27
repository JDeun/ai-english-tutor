import speech_recognition as sr
from openai import OpenAI
import tempfile
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def get_default_microphone(self):
        """
        시스템의 기본 마이크를 자동으로 선택합니다.
        """
        try:
            # 시스템 기본 마이크 사용
            mic = sr.Microphone()
            return mic.device_index
        except Exception as e:
            print(f"기본 마이크를 찾는 중 오류 발생: {e}")
            print("사용 가능한 마이크 목록:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"마이크 {index}: {name}")
            return int(input("사용할 마이크의 번호를 입력하세요: "))

    def record_audio(self, source):
        """
        오디오를 녹음하고 임시 파일로 저장합니다.
        """
        print("말씀해 주세요...")
        try:
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio.get_wav_data())
                return temp_audio_file.name
        except sr.WaitTimeoutError:
            print("음성 입력 시간이 초과되었습니다.")
            return None
        except Exception as e:
            print(f"음성 녹음 중 오류 발생: {e}")
            return None

    def transcribe_audio(self, audio_file_path):
        """
        오디오 파일을 텍스트로 변환합니다.
        """
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=config.STT_MODEL,
                file=audio_file
            )
        return transcription.text