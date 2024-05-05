import ollama
from typing import List


def prompt_ollama_with_articles(query: str, model: str, articles: List[dict]) -> str:
    prompt_system = "Summarize the information to the user's prompt based only on the retrieved documents below. Do not add any external information.\n\n"
    for i, entry in enumerate(articles):
        prompt_system += f"{i+1}. {entry['title']}\n\n" f"{entry['content']}\n\n"

    ollama_result = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_system},
            {
                "role": "user",
                "content": query,
            },
        ],
    )
    return ollama_result["message"]["content"]
