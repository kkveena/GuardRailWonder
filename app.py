"""
FastAPI application for centroid-based guardrail service

This provides a REST API endpoint for validating user prompts using
the centroid-based guardrail with per-intent thresholds and LLM verification.
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.guardrails.centroid_guardrail import CentroidGuardrail

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GuardRail Wonder - Centroid-based Guardrail API",
    description="LLM prompt guardrail with centroid matching and gray band verification",
    version="1.0.0"
)

# Initialize guardrail (singleton)
guardrail = None


@app.on_event("startup")
async def startup_event():
    """Initialize guardrail on startup"""
    global guardrail

    try:
        logger.info("Initializing CentroidGuardrail...")

        # Get configuration from environment or use defaults
        intent_config_path = os.getenv(
            "INTENT_PARAPHRASES_PATH",
            "config/intent_paraphrases.json"
        )
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-004")
        llm_model = os.getenv("LLM_MODEL", "gemini-2.5-pro-latest")

        guardrail = CentroidGuardrail(
            intent_paraphrases_path=intent_config_path,
            embed_model=embed_model,
            llm_model=llm_model,
        )

        logger.info("CentroidGuardrail initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize guardrail: {e}")
        raise


class CheckPayload(BaseModel):
    """Request payload for guardrail check"""
    text: str = Field(..., description="User input text to validate")
    locale: Optional[str] = Field(None, description="User locale (optional)")
    user_id: Optional[str] = Field(None, description="User ID for logging (optional)")


class GuardrailResponse(BaseModel):
    """Response from guardrail check"""
    allowed: bool = Field(..., description="Whether the prompt is allowed")
    intent: Optional[str] = Field(None, description="Matched intent (if allowed)")
    route: Optional[str] = Field(None, description="Routing key (if allowed)")
    score: float = Field(..., description="Similarity score")
    reason: str = Field(..., description="Decision reason")
    scores: Optional[dict] = Field(None, description="All intent scores")
    verified_by_llm: Optional[bool] = Field(None, description="LLM verification used")


@app.post("/guardrail.check", response_model=GuardrailResponse)
async def guardrail_check(payload: CheckPayload):
    """
    Check if a user prompt passes the guardrail

    This endpoint validates user input against predefined intents using
    centroid-based similarity matching with per-intent thresholds.

    Gray band verification using LLM is applied for borderline cases.
    """
    if guardrail is None:
        raise HTTPException(status_code=500, detail="Guardrail not initialized")

    try:
        # Run guardrail decision
        result = guardrail.decide(payload.text)

        return GuardrailResponse(**result)

    except Exception as e:
        logger.error(f"Error in guardrail check: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/intents")
async def get_intents():
    """
    Get information about configured intents

    Returns the list of supported intents, their thresholds,
    and paraphrase counts.
    """
    if guardrail is None:
        raise HTTPException(status_code=500, detail="Guardrail not initialized")

    try:
        return guardrail.get_intent_info()

    except Exception as e:
        logger.error(f"Error getting intent info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "guardrail_initialized": guardrail is not None
    }


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))

    # Run the FastAPI app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
