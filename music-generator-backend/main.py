import base64
import uuid
import modal
import os

from pydantic import BaseModel
import requests

app = modal.App("music-generator")

# Create a Modal container image with all dependencies
# This will run only once and will be cached for future runs (unless some configuration changes)
image = (
    modal.Image.debian_slim()  # Set up Operating System (Linux)
    .apt_install("git")  # Install git package for repository cloning
    .pip_install_from_requirements("requirements.txt")
    # Clone and install the ACE-Step music generation library
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", " cd /tmp/ACE-Step && pip install .",])
    # Set Hugging Face cache directory so we don't have to download the model again and again
    .env({"HF_HOME": "/.cache/huggingface"})
    # Add the local prompts module to the container
    .add_local_python_source("prompts")
)

# Create volumes to store files that can be shared between images
model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

# Load the secrets from Modal
music_gen_secrets = modal.Secret.from_name("music-gen-secret")

class GenerateMusicResponse(BaseModel):
  audio_data: str

@app.cls(
  image=image,
  gpu="L40S", # https://modal.com/pricing
  volumes={
    "/models": model_volume,
    "/.cache/huggingface": hf_volume,
  },
  secrets=[music_gen_secrets],
  scaledown_window=15, # keep the container warm for 15 seconds

)
class MusicGenServer:
  @modal.enter()
  def load_models(self):
    from acestep.pipeline_ace_step import ACEStepPipeline # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from diffusers import AutoPipelineForText2Image
    import torch

    # Music Generation Model
    self.music_model = ACEStepPipeline(
      checkpoint_dir="/models",
      dtype="bfloat16",
      torch_compile=False,
      cpu_offload=False,
      overlapped_decode=False
    )

    # Large Language Model (for generating lyrics)
    llm_model_id = "Qwen/Qwen2-7B-Instruct"
    self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    self.llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype="auto",
        device_map="auto",
        cache_dir="/.cache/huggingface"
    )

    # Stable Diffusion Model (for generating thumbnails)
    difussion_model_id = "stabilityai/sdxl-turbo"
    self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
      difussion_model_id,
      torch_dtype=torch.float16,
      variant="fp16",
      cache_dir="/.cache/huggingface"
    )
    self.image_pipeline.to("cuda")

  @modal.fastapi_endpoint(method="POST")
  def generate_music(self) -> GenerateMusicResponse:
    output_dir = "/tmp/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

    testing_lyrics = "[verse]\nWoke up in a city that's always alive\nNeon lights they shimmer they thrive\nElectric pulses beat they drive\nMy heart races just to survive\n\n[chorus]\nOh electric dreams they keep me high\nThrough the wires I soar and fly\nMidnight rhythms in the sky\nElectric dreams together we’ll defy\n\n[verse]\nLost in the labyrinth of screens\nVirtual love or so it seems\nIn the night the city gleams\nDigital faces haunted by memes\n\n[chorus]\nOh electric dreams they keep me high\nThrough the wires I soar and fly\nMidnight rhythms in the sky\nElectric dreams together we’ll defy\n\n[bridge]\nSilent whispers in my ear\nPixelated love serene and clear\nThrough the chaos find you near\nIn electric dreams no fear\n\n[verse]\nBound by circuits intertwined\nLove like ours is hard to find\nIn this world we’re truly blind\nBut electric dreams free the mind"

    self.music_model(
      prompt="synth-pop, electronic, pop, synthesizer, drums, bass, piano, 128 BPM, energetic, uplifting, modern",
      lyrics=testing_lyrics,
      audio_duration=221.27997916666666,
      infer_step=60,
      guidance_scale=15,
      save_path=output_path
    )

    with open(output_path, "rb") as f:
      audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    os.remove(output_path)

    return GenerateMusicResponse(audio_data=audio_b64)



@app.local_entrypoint()
def main():
  server = MusicGenServer()
  endpoint_url = server.generate_music.get_web_url()

  response = requests.post(endpoint_url)
  response.raise_for_status()

  result = GenerateMusicResponse(**response.json())

  audio_bytes = base64.b64decode(result.audio_data)
  output_filename = "generated.wav"
  with open(output_filename, "wb") as f:
    f.write(audio_bytes)

  print(f"Generated audio saved to {output_filename}")