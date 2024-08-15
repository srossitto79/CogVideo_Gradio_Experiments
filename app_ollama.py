import os
import tempfile
import threading
import time

import gradio as gr
import numpy as np
import torch
from diffusers import CogVideoXPipeline
from datetime import datetime, timedelta
from ollama import OllamaClient  # Replacing openai with ollama
import spaces
import imageio
import moviepy.editor as mp
from typing import List, Union
import PIL
import bitsandbytes as bnb

llm_model = "glm4:9b-chat-q8_0"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load the model with quantization using bitsandbytes
bnb_config = bnb.config.Config(
    load_in_8bit=True,
    # load_in_4bit=True,
)

# Load the model with quantization
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=dtype,
    device_map="auto",
    quantization_config=bnb_config,
).to(device)

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

def export_to_video_imageio(
        video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 8
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    client = OllamaClient()  # Initialize OllamaClient
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"'},
                {"role": "assistant",
                 "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance."},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"'},
                {"role": "assistant",
                 "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field."},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"'},
                {"role": "assistant",
                 "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background."},
                {"role": "user",
                 "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"'},
            ],
            model=llm_model,
            temperature=0.01,
            top_p=0.7,
            max_tokens=250,
        )
        
        if "message" in response:
            return response["message"]["content"]
        
    return prompt


@spaces.GPU(duration=240)
def infer(
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        progress=gr.Progress(track_tqdm=True)
):
    torch.cuda.empty_cache()

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,
        device=device,
        dtype=dtype,
    )

    video = pipe(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
    ).frames[0]

    return video


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video_imageio(tensor[1:], video_path)
    return video_path

def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace('.mp4', '.gif')
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        output_dir = './output'
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff:
                    os.remove(file_path)
        time.sleep(600)  # Sleep for 10 minutes


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-2B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> 
           </div>

           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            Users should strictly adhere to local laws and ethics.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use Ollama API to enrich your prompt automatically!")
                enhance_prompt_button = gr.Button("🚀 Enhance Prompt")
            enhanced_prompt = gr.Textbox(label="Enhanced Prompt", lines=5)
            enhance_prompt_button.click(fn=convert_prompt, inputs=prompt, outputs=enhanced_prompt)

            num_inference_steps = gr.Slider(0, 150, value=50, label="Inference Steps", step=1)
            guidance_scale = gr.Slider(0, 50, value=15, label="Guidance Scale", step=1)

            submit_button = gr.Button("Generate Video")
        with gr.Column():
            result = gr.Video(label="Generated Video", format="mp4")
            gif_result = gr.Image(label="GIF Preview")

    submit_button.click(fn=infer, inputs=[enhanced_prompt, num_inference_steps, guidance_scale], outputs=[result])
    submit_button.click(fn=save_video, inputs=[result], outputs=[])
    submit_button.click(fn=convert_to_gif, inputs=[result], outputs=[gif_result])

demo.queue()
demo.launch()
