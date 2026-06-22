"""Generate cover image for Building Neuromorphic AI using Imagen 4 Ultra (best text rendering)."""
import io
import json
import sys
from pathlib import Path
from PIL import Image

config = json.loads(Path.home().joinpath(".gemini-imagegen.json").read_text())

from google import genai
from google.genai import types

client = genai.Client(api_key=config["api_key"])

PROMPT = """Book cover for the technical textbook "Building Neuromorphic AI: From Spiking Neurons to Edge Intelligence" by Alexander Apartsin and Yehudit Aperstein.

Layout (portrait, top to bottom):
- TOP THIRD: Series label in small elegant caps: "HANDS-ON AI SCIENCE SERIES". Below it, the main title in large bold sans-serif: "BUILDING NEUROMORPHIC AI". Below the title, subtitle in smaller text: "From Spiking Neurons to Edge Intelligence".
- MIDDLE: Dramatic central illustration — a stylized biological neuron rendered as luminous geometric filaments against deep space. Axon branches extend outward like fiber optics, each ending in a glowing teal spike-pulse (#4ecdc4). A few orange-gold burst points (#f0a030) mark active firing events. The neuron transitions at the bottom into a silicon chip grid (dark circuit traces), bridging biology and hardware. A faint background grid of glowing teal nodes suggests a Loihi/SpiNNaker neuromorphic array.
- BOTTOM QUARTER: Author names in elegant serif: "Alexander Apartsin, Ph.D.  &  Yehudit Aperstein, Ph.D."

Color palette: deep navy-to-black background (#060d18 to #0a0f1c), teal pulses (#4ecdc4), violet-blue filaments (#7b9dff), rare amber spike bursts (#f0a030), white/light-grey text. Overall mood: precision, intelligence, scientific elegance. Publication-quality book cover art."""

output_path = Path(__file__).parent.parent / "images" / "book-cover.jpg"
TARGET_W, TARGET_H = 1000, 1600

print("Generating cover image with Imagen 4 Ultra...")
response = client.models.generate_images(
    model="imagen-4.0-ultra-generate-001",
    prompt=PROMPT,
    config=types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio="9:16",
        output_mime_type="image/jpeg",
    ),
)

if not response.generated_images:
    print("ERROR: no image in response", file=sys.stderr)
    sys.exit(1)

image_bytes = response.generated_images[0].image.image_bytes
img = Image.open(io.BytesIO(image_bytes))
print(f"Generated size: {img.size}")

# Resize to match VisionAI cover dimensions (1000x1600)
if img.size != (TARGET_W, TARGET_H):
    img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    print(f"Resized to {TARGET_W}x{TARGET_H}")

img.save(str(output_path), "JPEG", quality=90)
print(f"Saved: {output_path} ({output_path.stat().st_size // 1024} KB)")
