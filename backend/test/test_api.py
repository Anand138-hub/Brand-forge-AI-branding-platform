"""
BizForge AI — Unit Tests
Run: pytest tests/test_api.py -v
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set dummy env vars before importing app
os.environ.setdefault("GROQ_API_KEY", "gsk_test_key")
os.environ.setdefault("HF_API_KEY",   "hf_test_key")

from main import app

client = TestClient(app)

# ── HEALTH ────────────────────────────────────────────────────────────────────
def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

# ── BRAND NAMES ──────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
def test_generate_brand(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "1. TechNova — innovation meets clarity\n2. SmartForge — built for builders"
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/generate-brand", json={"industry": "Technology", "keywords": "smart, fast"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "data" in resp.json()

# ── LOGO ──────────────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
@patch("ai_services.hf_client", None)   # no HF key → image_url will be None
def test_generate_logo(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Sleek minimalist logo, metallic blue tones, bold typography."
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/generate-logo", json={"brand_name": "NEXORA", "industry": "Technology", "keywords": "modern"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["data"]["logo_prompt"] != ""

# ── CONTENT ───────────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
def test_generate_content(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Discover our revolutionary product that changes the way you work."
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/generate-content", json={
        "brand_description": "A cool clothing brand",
        "tone": "Professional",
        "content_type": "product_description",
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True

# ── SENTIMENT ─────────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
def test_analyze_sentiment(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "**Overall Sentiment:** Positive\n**Confidence:** High"
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/analyze-sentiment", json={"text": "It is good and nice", "brand_tone": "Professional"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "analysis" in resp.json()["data"]

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
def test_get_colors(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Primary: #3A86FF — Vibrant Blue\nSecondary: #FF006E"
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/get-colors", json={"tone": "Playful", "industry": "Technology"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

# ── CHAT ──────────────────────────────────────────────────────────────────────
@patch("ai_services.groq_client")
@patch("ai_services.granite_model", None)   # force Groq fallback
def test_chat(mock_groq):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "A brand is the unique identity of a company."
    mock_groq.chat.completions.create.return_value = mock_resp

    resp = client.post("/api/chat", json={"message": "What is a brand?"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "content" in resp.json()["data"]

# ── MISSING KEY HANDLING ──────────────────────────────────────────────────────
@patch("ai_services.groq_client", None)
def test_missing_groq_key():
    resp = client.post("/api/generate-brand", json={"industry": "Tech", "keywords": "fast"})
    assert resp.status_code == 200
    # Should still return 200 but with a warning message in data
    assert "GROQ_API_KEY" in resp.json()["data"] or resp.json()["success"] is True
