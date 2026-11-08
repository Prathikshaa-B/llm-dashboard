import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_MODEL = "gemini-3.8-flash"

JUDGE_PROMPT = """You are a strict but fair grader.

Question: {question}
Expected answer: {expected}
Model's answer: {answer}

Score the model's answer from 1 to 5:
5 = fully correct and clear
4 = correct but a bit wordy or slightly unclear
3 = partly correct
2 = mostly wrong
1 = wrong or irrelevant

Reply ONLY with JSON in this exact format:
{{"score": <number 1-5>, "reason": "<one short sentence>"}}"""


def judge_answer(question, expected, answer, retries=5):
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=config
            )
            data = json.loads(response.text)
            return int(data["score"]), data["reason"]
        except Exception as e:
            message = str(e)
            busy = "503" in message or "429" in message
            if busy and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"   (server busy, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                print(f"   (judge failed: {message[:100]})")
                return None, "judge failed"


def main():
    with open("results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    scored = []
    totals = {}  # prompt_name -> list of scores

    for item in results:
        score, reason = judge_answer(item["question"], item["expected"], item["answer"])
        print(f"[{item['prompt_name']}] {item['question']}")
        print(f"   score: {score}  |  {reason}\n")

        item["score"] = score
        item["reason"] = reason
        scored.append(item)

        if score is not None:
            totals.setdefault(item["prompt_name"], []).append(score)

        time.sleep(6)

    with open("scored_results.json", "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    print("=== AVERAGE SCORES ===")
    for prompt_name, scores in totals.items():
        average = sum(scores) / len(scores)
        print(f"{prompt_name}: {average:.2f} (from {len(scores)} questions)")

    print("\nDone! Saved to scored_results.json")


if __name__ == "__main__":
    main()