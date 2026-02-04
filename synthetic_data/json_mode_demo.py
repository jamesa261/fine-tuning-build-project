from __future__ import annotations

import json
import os

from pydantic import BaseModel
from google import genai
from openai import OpenAI


class Movie(BaseModel):
    title: str
    year: int
    director: str
    genres: list[str]


def gemini_demo():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3.0-flash",
        contents="Generate a fictional movie about a time-traveling historian.",
        config={
            "response_mime_type": "application/json",
            "response_schema": Movie,
        },
    )
    payload = response.text
    print("Gemini JSON mode:")
    print(json.dumps(json.loads(payload), indent=2))


def openai_demo():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "Generate a fictional movie about a time-traveling historian."}
        ],
        response_format={"type": "json_object"},
    )
    movie = Movie.model_validate_json(response.choices[0].message.content)
    print("\nOpenAI structured output:")
    print(movie.model_dump_json(indent=2))


if __name__ == "__main__":
    gemini_demo()
    openai_demo()
