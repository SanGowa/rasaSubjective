import random
import nltk

nltk.download("punkt")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionGenerateQuestion(Action):
    def name(self) -> Text:
        return "action_generate_question"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        subject = "java"
        question = "What is Java?"
        expected_answer = [
            "Java is a programming language.",
            "Java is a robust language.",
            "Java is an independent platfrom.",
        ]
        dispatcher.utter_message(text=question)

        return [
            SlotSet("subject", subject),
            SlotSet("question", question),
            SlotSet("expected_answer", expected_answer),
        ]


class ActionCheckSubjectiveAnswer(Action):
    def name(self) -> Text:
        return "action_check_subjective_answer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        subject = tracker.get_slot("subject")
        question = tracker.get_slot("question")
        expected_answer = tracker.get_slot("expected_answer")
        user_answer = tracker.latest_message.get("text")

        relevance_scores = [
            calculate_relevance(user_answer, answer) for answer in expected_answer
        ]
        relevance_score = max(relevance_scores, default=0)

        dispatcher.utter_message(text=f"Question: {question}")
        dispatcher.utter_message(text=f"Your Answer: {user_answer}")
        dispatcher.utter_message(text=f"Relevance Score: {relevance_score}")

        return []


def calculate_relevance(user_answer: str, expected_answer: str) -> float:
    user_tokens = nltk.word_tokenize(user_answer.lower())
    expected_tokens = nltk.word_tokenize(expected_answer.lower())
    user_sentence = " ".join(user_tokens)
    expected_sentence = " ".join(expected_tokens)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_sentence, expected_sentence])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity
