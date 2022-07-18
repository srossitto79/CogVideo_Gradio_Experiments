import gradio as gr
import os




os.environ['SAT_HOME'] = '/home/user/app/sharefs/cogview-new'

def inference(text):
    os.system("""bash ./scripts/inference_cogvideo_pipeline.sh""")
    return "output/out.mp4"

gr.Interface(inference,"text","video").launch()





