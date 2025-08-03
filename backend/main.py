import base64
from typing import List
import uuid
import modal
import os

from pydantic import BaseModel
import requests
import boto3

from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT


# ------------------------------
# ------------------------------
# ------------------------------
# Container Setup
# ------------------------------
# ------------------------------
# ------------------------------

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


# ------------------------------
# ------------------------------
# ------------------------------
# Schema Models
# ------------------------------
# ------------------------------
# ------------------------------

class AudioGenerationBase(BaseModel):
  audio_duration: float = 180.0
  seed: int = -1
  guidance_scale: float = 15.0
  infer_step: int = 60
  instrumental: bool = False

class GenerateMusicFromDescriptionRequest(AudioGenerationBase): # Inherits from AudioGenerationBase so it has all the fields
  full_described_song: str

class GenerateMusicWithCustomLyricsRequest(AudioGenerationBase): # Inherits from AudioGenerationBase so it has all the fields
  prompt: str # style of the song
  lyrics: str # lyrics of the song

class GenerateMusicWithDescribedLyricsRequest(AudioGenerationBase): # Inherits from AudioGenerationBase so it has all the fields
  prompt: str # style of the song
  described_lyrics: str # description of the lyrics

class GenerateMusicResponseS3(BaseModel):
  s3_key: str
  cover_image_s3_key: str
  categories: List[str]

class GenerateMusicResponse(BaseModel):
  audio_data: str


# ------------------------------
# ------------------------------
# ------------------------------
# Main Server Class
# ------------------------------
# ------------------------------
# ------------------------------

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

  # ------------------------------
  # ------------------------------
  # ------------------------------
  # Utility Functions
  # ------------------------------
  # ------------------------------
  # ------------------------------

  def run_inference_on_qwen(self, question: str):
    # Use the qwen model (based on documentation found in huggingface)
    # https://huggingface.co/Qwen/Qwen2-7B-Instruct
    messages = [
      {"role": "user", "content": question}
    ]
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

    generated_ids = self.llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

  def generate_prompt(self, description: str):
    #  Insert description into template
    full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)

    #  Run LLM inference and return the result
    return self.run_inference_on_qwen(full_prompt)

  def generate_lyrics(self, description: str):
    #  Insert description into template
    full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)

    #  Run LLM inference and return the result
    return self.run_inference_on_qwen(full_prompt)

  def generate_categories(self, description: str) -> List[str]:
    prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-separated list. For example: Pop, Electronic, Sad, 80s. Description: '{description}'"

    response_text = self.run_inference_on_qwen(prompt)

    # Structure the response into a list of categories
    categories = [cat.strip() for cat in response_text.split(",") if cat.strip()]

    return categories

  def generate_and_upload_to_s3(
      self,
      prompt: str,
      lyrics: str,
      instrumental: bool,
      audio_duration: float,
      infer_step: int,
      guidance_scale: float,
      seed: int,
      description_for_categories: str,
  ) -> GenerateMusicResponseS3:

    final_lyrics = "[instrumental]" if instrumental else lyrics
    print(f"Generated lyrics: \n{final_lyrics}")
    print(f"Prompt: \n{prompt}")

    s3_client = boto3.client("s3")
    bucket_name = os.environ["S3_BUCKET_NAME"]

    # ------------------------------
    # ------------------------------
    #  Music Generation
    # ------------------------------
    # ------------------------------
    output_dir = "/tmp/outputs"
    os.makedirs(output_dir, exist_ok=True)
    song_output_path = os.path.join(output_dir, f"audio-{uuid.uuid4()}.wav")

    self.music_model(
      prompt=prompt,
      lyrics=final_lyrics,
      audio_duration=audio_duration,
      infer_step=infer_step,
      guidance_scale=guidance_scale,
      manual_seed=str(seed),
      save_path=song_output_path
    )

    audio_s3_key = f"audio-{uuid.uuid4()}.wav"
    s3_client.upload_file(song_output_path, bucket_name, audio_s3_key)
    os.remove(song_output_path)

    # ------------------------------
    # ------------------------------
    #  Thumbnail Generation
    # ------------------------------
    # ------------------------------
    thumbnail_prompt = f"{prompt}, album cover art"
    image = self.image_pipeline(prompt=thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]

    thumbnail_output_path = os.path.join(output_dir, f"thumbnail-{uuid.uuid4()}.png")
    image.save(thumbnail_output_path)

    thumbnail_s3_key = f"thumbnail-{uuid.uuid4()}.png"
    s3_client.upload_file(thumbnail_output_path, bucket_name, thumbnail_s3_key)
    os.remove(thumbnail_output_path)

    # ------------------------------
    # ------------------------------
    # Category Generation (ex. "hip-hop", "rock", jazz)
    # ------------------------------
    # ------------------------------
    categories = self.generate_categories(description_for_categories)

    return GenerateMusicResponseS3(
      s3_key=audio_s3_key,
      cover_image_s3_key=thumbnail_s3_key,
      categories=categories
    )

  # ------------------------------
  # ------------------------------
  # ------------------------------
  # Endpoints
  # ------------------------------
  # ------------------------------
  # ------------------------------

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

  @modal.fastapi_endpoint(method="POST")
  def generate_music_from_description(self, request: GenerateMusicFromDescriptionRequest) -> GenerateMusicResponseS3:
    # Generete prompt
    prompt = self.generate_prompt(request.full_described_song)

    # Generete lyrics
    lyrics = ""
    if not request.instrumental:
      lyrics = self.generate_lyrics(request.full_described_song)


  @modal.fastapi_endpoint(method="POST")
  def generate_music_with_lyrics(self, request: GenerateMusicWithCustomLyricsRequest) -> GenerateMusicResponseS3:
    pass

  @modal.fastapi_endpoint(method="POST")
  def generate_music_with_described_lyrics(self, request: GenerateMusicWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
    pass


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