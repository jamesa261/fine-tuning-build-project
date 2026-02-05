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
        model="gemini-3-flash-preview",
        contents=f"Generate a fictional movie about a time-traveling historian.",
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
        # NOTE: Using structured output in this way for OpenAI models actually REQUIRES mentioning the JSON schema in the prompt.
        messages=[
            {"role": "user", "content": f"Generate a fictional movie about a time-traveling historian. Respond in this JSON schema: {json.dumps(Movie.model_json_schema())}"}
        ],
        response_format={"type": "json_object"},
    )
    movie = Movie.model_validate_json(response.choices[0].message.content)
    print("\nOpenAI structured output:")
    print(movie.model_dump_json(indent=2))


if __name__ == "__main__":
    gemini_demo()
    openai_demo()
