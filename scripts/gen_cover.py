"""Generate cover image for Building Neuromorphic AI using Gemini."""
import json
import sys
from pathlib import Path

config = json.loads(Path.home().joinpath(".gemini-imagegen.json").read_text())

from google import genai
from google.genai import types

client = genai.Client(api_key=config["api_key"])

PROMPT = """Book cover art for 'Building Neuromorphic AI: From Spiking Neurons to Edge Intelligence'.

Style: dark, dramatic, scientific. Deep navy-to-black background with subtle dark-teal and violet gradients.

Composition: A stylized spiking neuron in the center, rendered as luminous geometric filaments — axon branches extending outward like lightning or fiber optics, each branch ending in a glowing pulse (a spike event). Sparse bright dots of light travel along the axon paths, suggesting event-driven computation: only a few signals fire at once, most paths are dark. The neuron transitions at the bottom into a silicon chip grid or circuit board, bridging biology and hardware. In the far background, a faint grid of tiny glowing nodes suggests a large neuromorphic hardware array (Loihi/SpiNNaker scale). The palette is deep space: #060d18 background, teal pulses (#4ecdc4), violet-blue filaments (#7b9dff), rare orange-gold spike bursts (#f0a030). The overall mood: precision, sparsity, intelligence. Professional book cover, publication quality, no text.

Aspect ratio 2:3 (portrait, book-cover proportions). High detail, cinematic lighting."""

output_path = Path(__file__).parent.parent / "images" / "book-cover.jpg"

print("Generating cover image...")
response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="9:16", image_size="1K"),
    ),
)

saved = False
for part in response.candidates[0].content.parts:
    if hasattr(part, "inline_data") and part.inline_data:
        img = part.as_image()
        img.save(str(output_path))
        print(f"Saved: {output_path} ({output_path.stat().st_size // 1024} KB)")
        saved = True
        break

if not saved:
    print("ERROR: no image in response", file=sys.stderr)
    sys.exit(1)
