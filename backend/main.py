from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Trip Planner API",
    description="Generate personalized travel itineraries using OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItineraryRequest(BaseModel):
    destination: str = Field(..., min_length=2, max_length=100, example="Tokyo, Japan")
    budget: float = Field(..., gt=0, le=100000, example=1500)
    duration: int = Field(..., ge=1, le=30, example=5)
    interests: list[str] = Field(..., min_length=1, example=["Culture & History", "Food & Cuisine"])


@app.get("/")
def root():
    return {"message": "AI Trip Planner API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate-itinerary")
def generate_itinerary(request: ItineraryRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    client = OpenAI(api_key=api_key)

    prompt = f"""Create a detailed day-by-day travel itinerary.

Destination: {request.destination}
Total Budget: ${request.budget:.0f} USD
Duration: {request.duration} days
Interests: {', '.join(request.interests)}

Return ONLY valid JSON with this structure:
{{
  "destination": "City, Country",
  "summary": "2-3 sentence trip overview",
  "total_estimated_cost": <number>,
  "days": [
    {{
      "day": <number>,
      "theme": "Day theme",
      "morning": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "afternoon": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "evening": {{"activity": "...", "description": "...", "duration": "X hours", "cost": <number>}},
      "meals": {{"breakfast": "suggestion ~$X", "lunch": "suggestion ~$X", "dinner": "suggestion ~$X"}},
      "daily_cost_estimate": <number>,
      "pro_tip": "One practical tip"
    }}
  ],
  "packing_essentials": ["item1", "item2", "item3", "item4", "item5"],
  "best_time_to_visit": "Month range and why",
  "getting_around": "Local transport advice",
  "money_saving_tips": ["tip1", "tip2", "tip3"],
  "emergency_contacts": {{
    "local_emergency": "number",
    "tourist_helpline": "number or advice",
    "nearest_hospital": "general advice"
  }}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert travel planner. Always return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=4000,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
