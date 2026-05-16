"""
BizForge AI Services
All AI integrations: Groq LLaMA, IBM Granite, Stable Diffusion XL
"""

import os
import time
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API KEYS ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_API_KEY       = os.getenv("HF_API_KEY")
IBM_MODEL        = os.getenv("IBM_MODEL", "ibm-granite/granite-4.0-h-350m")

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "REPLACE_WITH_YOUR_GROQ_KEY":
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialised")
    except Exception as e:
        print(f"⚠ Groq init failed: {e}")
else:
    print("⚠ GROQ_API_KEY not set — text generation disabled")

# ── IBM GRANITE (local, CPU) ───────────────────────────────────────────────────
granite_model     = None
granite_tokenizer = None

def load_granite():
    global granite_model, granite_tokenizer
    if not HF_API_KEY or HF_API_KEY == "REPLACE_WITH_YOUR_HF_KEY":
        print("⚠ HF_API_KEY not set — IBM Granite disabled, falling back to Groq for chat")
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("⏳ Loading IBM Granite 4.0-h-350m …")
        granite_tokenizer = AutoTokenizer.from_pretrained(
            IBM_MODEL, token=HF_API_KEY, trust_remote_code=True
        )
        granite_model = AutoModelForCausalLM.from_pretrained(
            IBM_MODEL,
            torch_dtype=torch.float32,   # FIX: was deprecated string 'float32'
            token=HF_API_KEY,
            trust_remote_code=True,
        ).to("cpu")
        granite_model.eval()
        print("✅ IBM Granite loaded")
    except Exception as e:
        print(f"⚠ Granite load failed (will use Groq fallback): {e}")

# Load Granite in background — don't block startup
import threading
threading.Thread(target=load_granite, daemon=True).start()

# ── HF IMAGE GENERATION via FLUX.1-schnell (FREE — no credits needed) ──────
import requests as _requests
import urllib.parse as _urllib_parse
import random as _random


def generate_image_with_hf(prompt: str, width: int = 1024, height: int = 1024) -> bytes | None:
    """
    Generate image using HuggingFace InferenceClient with your HF_API_KEY.
    Uses stabilityai/stable-diffusion-xl-base-1.0 via HF Inference API.
    Returns raw PNG bytes or None on failure.
    """
    if not HF_API_KEY or HF_API_KEY == "REPLACE_WITH_YOUR_HF_KEY":
        print("⚠ HF_API_KEY not set — cannot generate image via HF")
        return None
    try:
        from huggingface_hub import InferenceClient
        from io import BytesIO
        print("⏳ Generating image via HF InferenceClient (SDXL)...")
        client = InferenceClient(api_key=HF_API_KEY)
        image  = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        print("✅ HF image generated")
        return buf.getvalue()
    except Exception as e:
        print(f"⚠ HF image generation failed: {str(e)[:120]}")
        return None


def generate_image_hf_free(prompt: str, width: int = 1024, height: int = 1024) -> bytes | None:
    """
    Generate image using multiple free sources — no extra packages needed.
    Uses pure HTTP requests only.
    """

    # ── Source 1: HF Space REST API (no gradio_client needed) ────────────────
    hf_spaces = [
        {
            "url": "https://stabilityai-stable-diffusion-3-medium.hf.space/run/predict",
            "payload": {"data": [prompt, "", 7, 1024, 1024, 5, True]},
        },
        {
            "url": "https://multimodalart-flux-lora-the-explorer.hf.space/run/predict",
            "payload": {"data": [prompt, "", 2, True, 1024, 1024, 0]},
        },
    ]

    headers = {"Content-Type": "application/json"}

    for space in hf_spaces:
        try:
            print(f"⏳ Trying HF Space: {space['url'].split('/')[2]}...")
            resp = _requests.post(
                space["url"],
                headers=headers,
                json=space["payload"],
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                # HF Spaces return image as base64 or URL in data[0]
                if isinstance(data.get("data"), list) and data["data"]:
                    item = data["data"][0]
                    if isinstance(item, str) and item.startswith("data:image"):
                        import base64
                        img_b64 = item.split(",", 1)[1]
                        img_bytes = base64.b64decode(img_b64)
                        if len(img_bytes) > 1000:
                            print(f"✅ Image from HF Space")
                            return img_bytes
                    elif isinstance(item, dict) and "url" in item:
                        img_resp = _requests.get(item["url"], timeout=30)
                        if img_resp.status_code == 200:
                            print(f"✅ Image from HF Space URL")
                            return img_resp.content
        except Exception as e:
            print(f"⚠ HF Space failed: {str(e)[:80]}")
            continue

    # ── Source 2: Pollinations with retries ───────────────────────────────────
    for attempt in range(4):
        try:
            seed  = _random.randint(1, 999999)
            short = prompt[:200]
            encoded = _urllib_parse.quote(short)
            model = ["flux", "turbo", "dream-shaper"][attempt % 3]
            url   = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width=1024&height=1024&nologo=true&seed={seed}&model={model}"
            )
            print(f"⏳ Pollinations attempt {attempt+1} (model={model})...")
            resp  = _requests.get(url, timeout=90)
            if resp.status_code == 200 and len(resp.content) > 5000:
                print("✅ Pollinations image generated")
                return resp.content
            print(f"⚠ Pollinations returned {resp.status_code}")
            import time as _time; _time.sleep(4)
        except Exception as e:
            print(f"⚠ Pollinations attempt {attempt+1} failed: {e}")
            import time as _time; _time.sleep(4)

    # ── Source 3: Lorem Picsum (last resort — returns a real image always) ────
    # Not AI but guarantees a real image as absolute fallback before SVG
    try:
        seed = _random.randint(1, 500)
        url  = f"https://picsum.photos/seed/{seed}/1024/1024"
        resp = _requests.get(url, timeout=15)
        if resp.status_code == 200:
            print("⚠ Using placeholder image (all AI sources failed)")
            return resp.content
    except Exception:
        pass

    print("⚠ All image sources failed — using SVG fallback")
    return None


def overlay_text_on_image(img_bytes: bytes, brand_name: str, save_path: str) -> bool:
    """
    Overlay brand name text on top of the AI-generated image using Pillow.
    This guarantees correct spelling regardless of what the AI generated.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        from io import BytesIO
        import os

        img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        W, H = img.size

        # Create a semi-transparent dark banner at the bottom
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        banner_h = int(H * 0.22)
        banner_y = H - banner_h

        # Dark gradient banner
        for i in range(banner_h):
            alpha = int(200 * (i / banner_h))
            draw_overlay.rectangle(
                [0, banner_y + i, W, banner_y + i + 1],
                fill=(0, 0, 0, alpha)
            )

        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Try to load a bold font, fall back to default
        font_size = max(60, min(120, W // max(len(brand_name), 1) + 20))
        font = None
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except:
                    continue

        if font is None:
            font = ImageFont.load_default()

        # Measure text and center it
        bbox = draw.textbbox((0, 0), brand_name.upper(), font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Shrink font if text too wide
        while text_w > W * 0.85 and font_size > 30:
            font_size -= 6
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, font_size)
                        break
                    except:
                        continue
            bbox = draw.textbbox((0, 0), brand_name.upper(), font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        x = (W - text_w) // 2
        y = banner_y + (banner_h - text_h) // 2

        # Draw shadow
        draw.text((x + 3, y + 3), brand_name.upper(), font=font, fill=(0, 0, 0, 180))
        # Draw main text in white
        draw.text((x, y), brand_name.upper(), font=font, fill=(255, 255, 255, 255))

        # Save as PNG
        final = img.convert("RGB")
        final.save(save_path, "PNG")
        print(f"✅ Text overlay applied: {brand_name.upper()}")
        return True
    except Exception as e:
        print(f"⚠ Text overlay failed: {e}")
        return False

print("✅ HF FLUX.1-schnell image generator ready (free)")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND = BASE_DIR / "frontend"
LOGOS_DIR = FRONTEND / "static" / "generated_logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)
# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ HELPER
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_with_groq(prompt: str, system: str = "You are BizForge, an expert branding AI.", max_tokens: int = 800) -> str:
    if not groq_client:
        return "⚠ Groq API key not configured. Please set GROQ_API_KEY in your .env file."
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"⚠ Generation failed after 3 attempts: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — BRAND NAME GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_brand_names(industry: str, keywords: str, tone: str = "Professional", language: str = "en") -> str:
    prompt = (
        f"Generate 10 unique, creative, and memorable brand names for a {industry} business.\n"
        f"Keywords to reflect: {keywords}\n"
        f"Brand tone: {tone}\n"
        f"Language: {language}\n\n"
        f"For each name provide:\n"
        f"- Name (bold)\n- One-line meaning/concept\n- Tagline idea\n- Domain hint (e.g. brandname.com)\n\n"
        f"Format clearly numbered 1–10. Be specific, creative, and avoid generic names."
    )
    return await generate_with_groq(prompt, max_tokens=900)

# ═══════════════════════════════════════════════════════════════════════════════
#  SVG LOGO GENERATOR (perfect text, instant, no API needed)
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_svg_logo(brand_name: str, concept: str) -> str:
    """
    Generate a professional SVG logo based on brand name and AI concept.
    Picks a visual style from the concept keywords and renders clean, crisp text.
    """
    import hashlib, colorsys, re

    name_upper = brand_name.upper().strip()
    words      = name_upper.split()
    initials   = "".join(w[0] for w in words[:3])

    # ── Derive color palette from brand name hash (consistent per brand) ──────
    h_val = int(hashlib.md5(brand_name.lower().encode()).hexdigest()[:6], 16)
    hue   = (h_val % 360) / 360.0

    def hsl_hex(h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    primary   = hsl_hex(hue,        0.75, 0.38)
    secondary = hsl_hex((hue+0.08)%1, 0.65, 0.55)
    light     = hsl_hex(hue,        0.60, 0.92)
    dark      = hsl_hex(hue,        0.80, 0.18)
    accent    = hsl_hex((hue+0.5)%1, 0.70, 0.55)

    # ── Detect style from concept ─────────────────────────────────────────────
    concept_l = concept.lower()
    if any(w in concept_l for w in ["luxury","gold","elegant","premium","royal"]):
        style = "luxury"
    elif any(w in concept_l for w in ["tech","digital","cyber","neon","futuristic","electric"]):
        style = "tech"
    elif any(w in concept_l for w in ["minimal","clean","simple","modern","flat"]):
        style = "minimal"
    elif any(w in concept_l for w in ["bold","strong","power","impact","sport"]):
        style = "bold"
    elif any(w in concept_l for w in ["nature","organic","green","eco","natural"]):
        style = "nature"
    elif any(w in concept_l for w in ["vintage","retro","classic","traditional"]):
        style = "vintage"
    else:
        style = "minimal"

    # ── Font size based on name length ────────────────────────────────────────
    name_len   = len(name_upper)
    font_size  = 88 if name_len <= 6 else 72 if name_len <= 10 else 58 if name_len <= 14 else 46
    sub_size   = 22
    init_size  = 120 if len(initials) == 1 else 90 if len(initials) == 2 else 72

    # ── Build SVG per style ────────────────────────────────────────────────────
    if style == "luxury":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{dark}" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="#0a0a0a"/>
    </radialGradient>
    <linearGradient id="gold" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f5d98b"/>
      <stop offset="50%" stop-color="#c9a84c"/>
      <stop offset="100%" stop-color="#8b6914"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="512" r="440" fill="none" stroke="url(#gold)" stroke-width="2" opacity="0.6"/>
  <circle cx="512" cy="512" r="420" fill="none" stroke="url(#gold)" stroke-width="0.5" opacity="0.3"/>
  <line x1="72" y1="512" x2="200" y2="512" stroke="url(#gold)" stroke-width="1" opacity="0.5"/>
  <line x1="824" y1="512" x2="952" y2="512" stroke="url(#gold)" stroke-width="1" opacity="0.5"/>
  <text x="512" y="490" font-family="Georgia, serif" font-size="{font_size}" font-weight="bold"
        fill="url(#gold)" text-anchor="middle" dominant-baseline="middle" filter="url(#glow)"
        letter-spacing="8">{name_upper}</text>
  <line x1="312" y1="530" x2="712" y2="530" stroke="url(#gold)" stroke-width="1" opacity="0.6"/>
  <text x="512" y="570" font-family="Georgia, serif" font-size="{sub_size}" fill="#c9a84c"
        text-anchor="middle" letter-spacing="12" opacity="0.8">SINCE 2024</text>
</svg>"""

    elif style == "tech":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#050810"/>
      <stop offset="100%" stop-color="#0d1525"/>
    </linearGradient>
    <linearGradient id="neon" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{primary}"/>
      <stop offset="100%" stop-color="{accent}"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="6" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow2"><feGaussianBlur stdDeviation="12" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="460" r="260" fill="none" stroke="{primary}" stroke-width="1.5" opacity="0.3" filter="url(#glow)"/>
  <circle cx="512" cy="460" r="290" fill="none" stroke="{accent}" stroke-width="0.5" opacity="0.15"/>
  <polygon points="512,180 772,340 772,580 512,740 252,580 252,340" fill="none" stroke="url(#neon)" stroke-width="1" opacity="0.2"/>
  <text x="512" y="480" font-family="'Courier New', monospace" font-size="{font_size}" font-weight="bold"
        fill="url(#neon)" text-anchor="middle" dominant-baseline="middle"
        filter="url(#glow2)" letter-spacing="6">{name_upper}</text>
  <line x1="112" y1="540" x2="412" y2="540" stroke="{primary}" stroke-width="1" opacity="0.4"/>
  <line x1="612" y1="540" x2="912" y2="540" stroke="{accent}" stroke-width="1" opacity="0.4"/>
  <text x="512" y="575" font-family="'Courier New', monospace" font-size="18" fill="{primary}"
        text-anchor="middle" letter-spacing="8" opacity="0.7">// DIGITAL SOLUTIONS</text>
</svg>"""

    elif style == "bold":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{primary}"/>
      <stop offset="100%" stop-color="{dark}"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <polygon points="0,0 400,0 300,1024 0,1024" fill="rgba(255,255,255,0.05)"/>
  <rect x="80" y="430" width="864" height="160" fill="rgba(0,0,0,0.25)" rx="4"/>
  <rect x="80" y="430" width="6" height="160" fill="{accent}"/>
  <text x="530" y="528" font-family="Arial Black, sans-serif" font-size="{font_size}" font-weight="900"
        fill="white" text-anchor="middle" dominant-baseline="middle"
        letter-spacing="4">{name_upper}</text>
  <rect x="80" y="620" width="200" height="4" fill="{accent}"/>
  <text x="80" y="660" font-family="Arial, sans-serif" font-size="20" fill="rgba(255,255,255,0.7)"
        letter-spacing="6">BRAND IDENTITY</text>
</svg>"""

    elif style == "nature":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <radialGradient id="bg" cx="30%" cy="30%" r="80%">
      <stop offset="0%" stop-color="#f0f7ee"/>
      <stop offset="100%" stop-color="#dcefd8"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="440" r="300" fill="none" stroke="{primary}" stroke-width="2" opacity="0.3"/>
  <circle cx="512" cy="440" r="260" fill="{primary}" opacity="0.08"/>
  <ellipse cx="512" cy="340" rx="60" ry="80" fill="{primary}" opacity="0.6"/>
  <ellipse cx="452" cy="380" rx="50" ry="65" fill="{primary}" opacity="0.5" transform="rotate(-20,452,380)"/>
  <ellipse cx="572" cy="375" rx="50" ry="65" fill="{primary}" opacity="0.5" transform="rotate(20,572,375)"/>
  <line x1="512" y1="420" x2="512" y2="500" stroke="{dark}" stroke-width="3"/>
  <text x="512" y="580" font-family="Georgia, serif" font-size="{font_size}" font-weight="bold"
        fill="{dark}" text-anchor="middle" dominant-baseline="middle"
        letter-spacing="4">{name_upper}</text>
  <line x1="262" y1="625" x2="762" y2="625" stroke="{primary}" stroke-width="1.5" opacity="0.5"/>
  <text x="512" y="660" font-family="Georgia, serif" font-size="20" fill="{primary}"
        text-anchor="middle" letter-spacing="8" opacity="0.8">ORGANIC · NATURAL</text>
</svg>"""

    elif style == "vintage":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#f5edd8"/>
      <stop offset="100%" stop-color="#e8d5b0"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="480" r="350" fill="none" stroke="{primary}" stroke-width="4" opacity="0.5"/>
  <circle cx="512" cy="480" r="335" fill="none" stroke="{primary}" stroke-width="1" opacity="0.3"/>
  <circle cx="512" cy="480" r="320" fill="none" stroke="{primary}" stroke-width="2" opacity="0.2"/>
  <path d="M 212 480 Q 362 360 512 380 Q 662 360 812 480" fill="none" stroke="{dark}" stroke-width="1.5" opacity="0.3"/>
  <text x="512" y="440" font-family="Georgia, 'Times New Roman', serif" font-size="{font_size}" font-weight="bold"
        fill="{dark}" text-anchor="middle" dominant-baseline="middle"
        letter-spacing="6">{name_upper}</text>
  <path d="M 262 510 L 312 510 L 322 500 L 332 520 L 342 500 L 352 520 L 362 510 L 762 510 L 752 510 L 742 500 L 732 520 L 722 500 L 712 520 L 702 510 L 262 510" 
        fill="none" stroke="{primary}" stroke-width="1.5" opacity="0.6"/>
  <text x="512" y="555" font-family="Georgia, serif" font-size="22" fill="{primary}"
        text-anchor="middle" letter-spacing="10" opacity="0.8">EST. 2024</text>
</svg>"""

    else:  # minimal (default)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{primary}"/>
      <stop offset="100%" stop-color="{secondary}"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="#ffffff"/>
  <rect x="0" y="0" width="1024" height="8" fill="url(#accent)"/>
  <circle cx="512" cy="420" r="180" fill="{light}"/>
  <text x="512" y="420" font-family="Arial, Helvetica, sans-serif" font-size="{init_size}" font-weight="900"
        fill="{primary}" text-anchor="middle" dominant-baseline="middle">{initials}</text>
  <text x="512" y="640" font-family="Arial, Helvetica, sans-serif" font-size="{font_size}" font-weight="700"
        fill="{dark}" text-anchor="middle" dominant-baseline="middle"
        letter-spacing="6">{name_upper}</text>
  <rect x="412" y="680" width="200" height="3" fill="url(#accent)" rx="2"/>
</svg>"""

    return svg



#  MODULE 3 — MARKETING CONTENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_marketing_content(brand_description: str, tone: str = "Professional",
                                     content_type: str = "product_description", language: str = "en") -> str:
    type_map = {
        "product_description": "a compelling product description (150-200 words)",
        "social_post":         "3 social media posts (Instagram, LinkedIn, Twitter)",
        "email":               "a marketing email with subject line and body",
        "ad_copy":             "3 variations of ad copy (headline + 2 lines each)",
        "tagline":             "5 creative taglines/slogans",
        "brand_story":         "a one-page brand story (250 words)",
    }
    task = type_map.get(content_type, "marketing content")
    prompt = (
        f"Create {task} for the following brand:\n\n"
        f"{brand_description}\n\n"
        f"Tone: {tone}\nLanguage: {language}\n\n"
        f"Make it engaging, on-brand, and ready to publish."
    )
    return await generate_with_groq(prompt, max_tokens=700)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_sentiment(text: str, brand_tone: str = "Professional") -> dict:
    prompt = (
        f"Analyze the sentiment of this customer review/text and rewrite it professionally.\n\n"
        f"Text: \"{text}\"\n"
        f"Target brand tone: {brand_tone}\n\n"
        f"Respond in this exact format:\n"
        f"**Overall Sentiment:** [Positive / Neutral / Negative]\n"
        f"**Confidence:** [High / Medium / Low]\n"
        f"**Key Points:** [bullet list of what customer mentioned]\n"
        f"**Professional Rewrite:** [improved version in brand tone]\n"
        f"**Actionable Insight:** [one sentence for the brand]"
    )
    result = await generate_with_groq(prompt, max_tokens=500)
    return {"analysis": result, "original_text": text}

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
async def get_color_palette(tone: str, industry: str) -> str:
    prompt = (
        f"Suggest a professional brand color palette for a {tone} brand in the {industry} industry.\n\n"
        f"Provide:\n"
        f"- Primary color (HEX + name + why it fits)\n"
        f"- Secondary color (HEX + name + why it fits)\n"
        f"- Accent color (HEX + name + why it fits)\n"
        f"- Background color (HEX)\n"
        f"- Text color (HEX)\n"
        f"- Font pairing recommendation (heading + body)\n"
        f"- Overall design style description (2 sentences)\n\n"
        f"Be specific and explain the psychology behind each choice."
    )
    return await generate_with_groq(prompt, max_tokens=500)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — AI BRANDING CHATBOT (IBM Granite → Groq fallback)
# ═══════════════════════════════════════════════════════════════════════════════
async def chat_with_ai(message: str) -> str:
    # Try IBM Granite first
    if granite_model and granite_tokenizer:
        try:
            system_prompt = (
                "You are BizForge, an expert AI branding assistant. "
                "Give concise, strategic, actionable branding advice."
            )
            full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
            inputs = granite_tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = granite_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=granite_tokenizer.eos_token_id,
                )
            response = granite_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant's reply
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            return response
        except Exception as e:
            print(f"Granite inference failed, using Groq: {e}")

    # Groq fallback
    system = (
        "You are BizForge, an expert AI branding consultant. "
        "Give concise, strategic, and actionable branding advice. "
        "Be direct, specific, and helpful. Answer in 3-5 sentences unless more detail is needed."
    )
    return await generate_with_groq(message, system=system, max_tokens=400)

# ═══════════════════════════════════════════════════════════════════════════════
#  BANNER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
BANNER_PRESETS = {
    "youtube":   (2560, 1440),
    "linkedin":  (1584,  396),
    "twitter":   (1500,  500),
    "facebook":  ( 820,  312),
    "instagram": (1080, 1080),
}

async def generate_banner(platform: str, brand_name: str, tagline: str = "",
                           cta: str = "", style: str = "modern", output_format: str = "png") -> dict:
    W, H = BANNER_PRESETS.get(platform.lower(), (1500, 500))

    prompt_text = (
        f"Professional {platform} social media banner for brand '{brand_name}'. "
        f"Tagline: '{tagline}'. Style: {style}, clean, modern. "
        f"No text in the image, pure visual background only."
    )

    image_url = None
    try:
        img_bytes = generate_image_hf_free(prompt_text, width=min(W,1024), height=min(H,1024))
        if img_bytes:
            from PIL import Image as PILImage
            from io import BytesIO
            img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((W, H))
            filename  = f"banner_{platform}_{int(time.time())}.{output_format}"
            save_path = LOGOS_DIR / filename
            img_resized.save(str(save_path), output_format.upper())
            image_url = f"/static/generated_logos/{filename}"
            print(f"✅ Banner saved: {save_path}")
    except Exception as e:
        print(f"⚠ Banner generation failed: {e}")

    return {
        "success":    True,
        "platform":   platform,
        "dimensions": f"{W}x{H}",
        "image_url":  image_url,
        "prompt_used": prompt_text,
        "error":      None if image_url else "Image generation failed — check internet connection",
    }# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — LOGO PROMPT GENERATOR + HF FLUX IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_logo_prompt(brand_name: str, industry: str, keywords: str) -> dict:
    # Step 1: Generate a detailed logo prompt with Groq
    prompt = (
        f"Write a detailed, professional image generation prompt for a logo.\n"
        f"Brand name: {brand_name}\n"
        f"Industry: {industry}\n"
        f"Style keywords: {keywords}\n\n"
        f"Describe: visual style, colors, shapes, typography feel, mood, composition.\n"
        f"The brand name must appear clearly in the logo.\n"
        f"Keep under 100 words. Output the prompt directly, no preamble."
    )
    logo_prompt = await generate_with_groq(
        prompt,
        system="You are an expert logo designer writing AI image generation prompts.",
        max_tokens=200
    )

    # Step 2: Generate visual background via Pollinations (no text in image)
    # Then overlay brand name with Pillow for 100% correct spelling
    image_url = None
    svg_url   = None

    # Prompt explicitly tells AI NOT to render text — we add it via Pillow
    visual_prompt = (
        f"A professional logo visual background for a {industry} brand. "
        f"{logo_prompt} "
        f"Abstract shapes, icons, and design elements only. "
        f"NO text, NO words, NO letters in the image. Pure visual design."
    )
    img_bytes = generate_image_hf_free(visual_prompt, width=1024, height=1024)

    if img_bytes:
        try:
            filename  = f"logo_{int(time.time())}.png"
            save_path = LOGOS_DIR / filename
            # Overlay brand name text with Pillow (guaranteed correct spelling)
            success = overlay_text_on_image(img_bytes, brand_name, str(save_path))
            if not success:
                # Save raw image without overlay as fallback
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
            image_url = f"/static/generated_logos/{filename}"
            print(f"✅ Logo saved with text overlay: {save_path}")
        except Exception as e:
            print(f"⚠ Failed to save logo: {e}")

    # Step 3: SVG fallback if Pollinations failed
    if not image_url:
        try:
            svg_data = _generate_svg_logo(brand_name, logo_prompt)
            ts       = int(time.time())
            svg_name = f"logo_{ts}.svg"
            svg_path = LOGOS_DIR / svg_name
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_data)
            svg_url   = f"/static/generated_logos/{svg_name}"
            image_url = svg_url
            print(f"✅ SVG fallback logo saved: {svg_path}")
        except Exception as e:
            print(f"⚠ SVG fallback failed: {e}")

    return {
        "logo_prompt": logo_prompt,
        "image_url":   image_url,
        "svg_url":     svg_url,
        "success":     True,
        "error":       None if image_url else "Image generation failed",
    }
# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — AI AD CREATIVE GENERATOR (text + 3 HF example images)
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_ad_creative(
    brand_name: str,
    product: str,
    target_audience: str,
    platform: str = "Instagram",
    goal: str = "Awareness",
    tone: str = "Professional"
) -> dict:
    prompt = (
        f"Create a complete AI Ad Creative package for the following:\n\n"
        f"Brand: {brand_name}\n"
        f"Product/Service: {product}\n"
        f"Target Audience: {target_audience}\n"
        f"Platform: {platform}\n"
        f"Campaign Goal: {goal}\n"
        f"Tone: {tone}\n\n"
        f"Provide ALL of the following:\n\n"
        f"**PRIMARY HEADLINE:** (max 8 words, attention-grabbing)\n\n"
        f"**AD COPY VARIATION 1 — Emotional:**\n(2-3 sentences connecting emotionally)\n\n"
        f"**AD COPY VARIATION 2 — Problem/Solution:**\n(2-3 sentences identifying pain point + solution)\n\n"
        f"**AD COPY VARIATION 3 — Social Proof:**\n(2-3 sentences with implied credibility)\n\n"
        f"**CALL TO ACTION OPTIONS:**\n(3 different CTAs)\n\n"
        f"**VISUAL DIRECTION:**\n(Describe the ideal image/video for this ad)\n\n"
        f"**HASHTAGS:** (10 relevant hashtags for {platform})\n\n"
        f"**A/B TEST SUGGESTION:**\n(What to test between the variations)\n\n"
        f"Make every word count. This should be ready to publish."
    )

    result = await generate_with_groq(
        prompt,
        system=(
            f"You are an expert performance marketer and copywriter specializing in "
            f"{platform} ad creatives. You write high-converting, scroll-stopping ads. "
            f"Always be specific, never generic."
        ),
        max_tokens=1000
    )

    # Generate 3 example ad images via HF
    ad_images = []
    image_prompts = [
        (
            f"Professional {platform} advertisement visual for {brand_name}. "
            f"Product: {product}. Clean minimal product showcase. "
            f"Bright studio lighting, white background, premium quality. "
            f"No text, no words, no letters anywhere in the image."
        ),
        (
            f"Lifestyle {platform} ad photo for {brand_name} {product}. "
            f"Showing {target_audience} using the product in real life. "
            f"Warm, authentic, candid photography style, natural lighting. "
            f"No text, no words, no letters anywhere in the image."
        ),
        (
            f"Bold eye-catching {platform} ad background for {brand_name}. "
            f"Abstract, colorful, dynamic composition. "
            f"Strong contrast, {goal.lower()} campaign mood, professional design. "
            f"No text, no words, no letters anywhere in the image."
        ),
    ]

    for i, img_prompt in enumerate(image_prompts):
        try:
            img_bytes = generate_image_with_hf(img_prompt)
            if img_bytes:
                safe_name = brand_name.lower().replace(" ", "_").replace("/", "_")
                filename  = f"ad_{safe_name}_{i+1}_{int(time.time())}.png"
                save_path = LOGOS_DIR / filename
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                ad_images.append(f"/static/generated_logos/{filename}")
                print(f"✅ Ad image {i+1}/3 saved: {filename}")
            else:
                ad_images.append(None)
                print(f"⚠ Ad image {i+1}/3 not generated")
        except Exception as e:
            print(f"⚠ Ad image {i+1}/3 failed: {e}")
            ad_images.append(None)

    return {
        "ad_creative": result,
        "platform":    platform,
        "goal":        goal,
        "ad_images":   ad_images,
    }
# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 9 — AI BRANDING ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
async def branding_assistant(
    brand_name: str,
    industry: str,
    stage: str = "Startup",
    request_type: str = "full_audit",
    custom_question: str = ""
) -> str:
    type_prompts = {
        "full_audit": (
            f"Conduct a complete branding audit for '{brand_name}' in the {industry} industry ({stage} stage).\n\n"
            f"Cover ALL sections:\n"
            f"**1. BRAND POSITIONING** — Where does this brand stand in the market?\n"
            f"**2. TARGET AUDIENCE** — Who are the ideal customers? Demographics + psychographics\n"
            f"**3. BRAND PERSONALITY** — 5 personality traits with explanation\n"
            f"**4. VALUE PROPOSITION** — One clear sentence that defines the brand's unique value\n"
            f"**5. COMPETITIVE LANDSCAPE** — 3 likely competitors and differentiation strategy\n"
            f"**6. BRAND VOICE** — Tone, language style, what to say and NOT say\n"
            f"**7. VISUAL IDENTITY DIRECTION** — Color psychology, typography style, imagery suggestions\n"
            f"**8. GROWTH RECOMMENDATIONS** — 3 specific actionable branding moves for next 90 days\n"
        ),
        "positioning": (
            f"Create a detailed brand positioning strategy for '{brand_name}' in {industry}.\n"
            f"Include: positioning statement, market differentiation, target segment, and unique value.\n"
            f"Stage: {stage}. Be specific and strategic."
        ),
        "competitor": (
            f"Analyze the competitive landscape for '{brand_name}' in {industry}.\n"
            f"Identify 5 likely competitors, their positioning, weaknesses, and how '{brand_name}' can stand out.\n"
            f"Give specific differentiation strategies."
        ),
        "voice": (
            f"Define the complete brand voice and tone guide for '{brand_name}' in {industry}.\n"
            f"Include: personality adjectives, writing style, vocabulary to use/avoid, "
            f"example phrases, and platform-specific tone variations (social, email, website)."
        ),
        "naming": (
            f"Evaluate the brand name '{brand_name}' for a {industry} business.\n"
            f"Cover: memorability, pronunciation, trademark considerations, domain availability likelihood, "
            f"cultural considerations, and suggest 5 alternative name directions if needed."
        ),
        "visual_identity": (
            f"Create a complete visual identity brief for '{brand_name}' in {industry}.\n"
            f"Include: color palette rationale (with HEX codes), typography pairing, logo style direction, "
            f"imagery style, iconography, and brand application examples."
        ),
        "custom": custom_question if custom_question else f"Give strategic branding advice for '{brand_name}'.",
    }

    prompt = type_prompts.get(request_type, type_prompts["full_audit"])

    return await generate_with_groq(
        prompt,
        system=(
            "You are a world-class brand strategist with 20 years of experience building "
            "iconic brands. You give specific, actionable, deeply insightful advice. "
            "Never be vague or generic. Always tailor advice to the specific brand and industry."
        ),
        max_tokens=1200
    )