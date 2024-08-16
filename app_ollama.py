import os
import tempfile
import threading
import time
import gc
import gradio as gr
import numpy as np
import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from datetime import datetime, timedelta
import ollama
import spaces
import imageio
import moviepy.editor as mp
from typing import List, Union
import PIL
from transformers import BitsAndBytesConfig

llm_model = "glm4:9b-chat-q8_0"
ollama_host = "http://192.168.1.123:11434"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

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


@spaces.GPU(duration=240)
def generate_video(
        prompt: str,
        output_path: str = "./output.mp4",
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype = torch.float16,
        progress=gr.Progress(track_tqdm=True)
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.float16).

    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (float16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=dtype
    ).to(device)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for better results.
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model and reset the memory, enable tiling.
    pipe.enable_model_cpu_offload()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    # Using with diffusers branch `main` to enable tiling. This will cost ONLY 12GB GPU memory.
    # pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps,so 48 frames and will plus 1 frame for the first frame.
    # for diffusers version `0.30.0`, this should be 48. and for `0.31.0` and after, this should be 49.
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        num_inference_steps=num_inference_steps,  # Number of inference steps
        num_frames=48, # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
        guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance
        generator=torch.Generator().manual_seed(42),  # Set the seed for reproducibility
    ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8
    export_to_video_imageio(video, output_path, fps=8)

def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    text = prompt.strip()

    for i in range(retry_times):
        response = ollama.Client(host=ollama_host).chat(
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
            model=llm_model
        )
        
        if "message" in response:
            return response["message"]["content"]
        
    return prompt

def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace('.mp4', '.gif')
    clip.write_gif(gif_path, fps=8)
    return gif_path


# def delete_old_files():
#     while True:
#         now = datetime.now()
#         cutoff = now - timedelta(minutes=10)
#         output_dir = './output'
#         for filename in os.listdir(output_dir):
#             file_path = os.path.join(output_dir, filename)
#             if os.path.isfile(file_path):
#                 file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
#                 if file_mtime < cutoff:
#                     os.remove(file_path)
#         time.sleep(600)  # Sleep for 10 minutes

#threading.Thread(target=delete_old_files, daemon=True).start()

def generate_and_process_video(enhanced_prompt, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
    # Run the inference to generate the video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    generate_video(prompt=enhanced_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,output_path=video_path, progress=progress)
    
    video_update = gr.update(visible=True, value=video_path)
    # Convert the video to GIF
    gif_path = convert_to_gif(video_path)
    
    gif_update = gr.update(visible=True, value=gif_path)
    
    return video_path, gif_path, video_update, gif_update



with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-2B with Ollama LLM, Gradio Demo
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 Original Version Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Original Version Github</a> |
               <a href="https://github.com/srossitto79/CogVideo_Gradio_Experiments">🌐 This Version Github</a> 
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
            guidance_scale = gr.Slider(0, 50, value=6, label="Guidance Scale", step=1)

            submit_button = gr.Button("Generate Video")
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
        with gr.Column():
            result = gr.Video(label="Generated Video", format="mp4")
            gif_result = gr.Image(label="GIF Preview")
    
    submit_button.click(fn=generate_and_process_video, inputs=[enhanced_prompt, num_inference_steps, guidance_scale], outputs=[result, gif_result, download_video_button, download_gif_button])

demo.queue()
demo.launch()
