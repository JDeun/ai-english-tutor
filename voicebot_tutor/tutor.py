from openai import OpenAI
import config
from topic import topics

client = OpenAI(api_key=config.OPENAI_API_KEY)

class EnglishTutor:
    def __init__(self):
        self.conversation_history = []
        self.current_topic = None

    def set_topic(self, topic):
        self.current_topic = topic
        base_prompt = config.SYSTEM_MESSAGE
        if topic != "자유주제/대화":
            topic_prompt = f"""
Now we will practice English in the context of {topic}. 
Focus on common phrases and vocabulary related to this topic.
Remember to:
1. Use natural, conversational English appropriate for the {topic} context.
2. Introduce and explain key vocabulary and phrases specific to this situation.
3. Provide examples of how to use these phrases in real-life scenarios.
4. Correct any mistakes gently, explaining the correct usage.
5. Gradually increase the complexity of the language as the conversation progresses.
6. Encourage the learner to form complete sentences and express their thoughts fully.
7. Offer cultural insights relevant to {topic} in English-speaking countries when appropriate.
8. Ask questions to prompt the learner to use the new vocabulary and phrases.
9. Provide positive reinforcement and encouragement throughout the conversation.
10. Summarize key learning points at natural breaks in the conversation.

Maintain your friendly and supportive demeanor throughout the conversation, 
and adapt your language to the learner's proficiency level while challenging them to improve.
"""
            self.conversation_history = [{"role": "system", "content": base_prompt + "\n" + topic_prompt}]
        else:
            self.conversation_history = [{"role": "system", "content": base_prompt}]

    def get_response(self, user_input):
        """
        사용자 입력에 대한 튜터의 응답을 생성합니다.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=self.conversation_history,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            frequency_penalty=config.FREQUENCY_PENALTY,
            presence_penalty=config.PRESENCE_PENALTY,
            stop=config.STOP
        )

        tutor_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": tutor_response})
        
        return tutor_response

    def reset_conversation(self):
        """
        대화 기록을 초기화합니다.
        """
        self.conversation_history = []
        self.current_topic = None

    def get_topics(self):
        """
        사용 가능한 주제 목록을 반환합니다.
        """
        return [list(topic.keys())[0] for topic in topics]

    def get_topic_description(self, topic):
        """
        선택된 주제에 대한 설명을 반환합니다.
        """
        for t in topics:
            if list(t.keys())[0] == topic:
                return list(t.values())[0]
        return "해당 주제에 대한 설명이 없습니다."