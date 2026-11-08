import json
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.8-flash"

# Two different prompts to compare. {question} gets replaced each time.
PROMPTS = {
    "prompt_A": "Answer the question.\n\nQuestion: {question}",
    "prompt_B": "You are a precise teacher. Answer in one short sentence.\n\nQuestion: {question}",
}

def ask_model(prompt_text, retries=5):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt_text)
            return response.text.strip()
        except Exception as e:
            message = str(e)
            busy = "503" in message or "429" in message
            if busy and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"   (server busy, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                raise


def main():
    with open("test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        for prompt_name, template in PROMPTS.items():
            full_prompt = template.format(question=case["question"])
            answer = ask_model(full_prompt)

            print(f"[{prompt_name}] {case['question']}")
            print(f"   -> {answer}\n")

            results.append({
                "question": case["question"],
                "expected": case["expected"],
                "prompt_name": prompt_name,
                "answer": answer,
            })

            time.sleep(2)  # small pause so the free tier doesn't block us

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done! Saved to results.json")


if __name__ == "__main__":
    main()