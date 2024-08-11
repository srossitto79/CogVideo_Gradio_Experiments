---
title: CogVideoX-2B
emoji: 🎥
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.41.0
suggested_hardware: a10g-large
suggested_storage: large
app_port: 7860
app_file: app.py
models:
  - THUDM/CogVideoX-2b
tags:
  - cogvideox
  - video-generation
  - thudm
short_description: Text-to-Video
disable_embedding: false
---

# CogVideoX HF Space

## How to run this space

CogVideoX does not rely on any external API models.
However, during the training of CogVideoX, we used relatively long prompts. To enable users to achieve rendering with
shorter prompts, we integrated an LLM to refine the prompts for better results.
This step is not mandatory, but we recommend using an LLM to enhance the prompts.

### Using with GLM-4 Model

```shell
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/ OPENAI_API_KEY="ZHIPUAI_API_KEY" python gradio_demo.py
```

### Using with OpenAI GPT-4 Model

```shell
OPENAI_API_KEY="OPENAI_API_KEY" python gradio_demo.py
```

and change `app.py` here:

```
model="glm-4-0520"  # change to GPT-4o
```

### Not using LLM to refine prompts.

```shell
python gradio_demo.py
```