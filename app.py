import gradio as gr
import os

os.system("wget https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage1.zip")


os.environ['NLAYERS'] = 48
os.environ['NHIDDEN'] = 3072
os.environ['NATT'] = 48
os.environ['MAXSEQLEN'] = 1024
os.environ['MPSIZE'] = 1




os.environ['TEMP'] = 1.05
os.environ['TOPK'] = 12



os.environ['SAT_HOME'] = '/home/user/app/sharefs/cogview-new'

def inference(text):
    os.system("""python cogvideo_pipeline.py --input-source interactive --output-path ./output --parallel-size 1 --both-stages --use-guidance-stage1 --guidance-alpha 3.0 --generate-frame-num 5  --tokenizer-type fake --mode inference --distributed-backend nccl  --fp16 --model-parallel-size $MPSIZE --temperature $TEMP --coglm-temperature2 0.89 --top_k $TOPK --sandwich-ln --seed 1234 --num-workers 0 --batch-size 1 --max-inference-batch-size 1""")
    return "out/out.mp4"

gr.Interface(inference,"text","video").launch()





