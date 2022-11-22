#!/usr/bin/env python

from __future__ import annotations

import gradio as gr

# from model import AppModel

MAINTENANCE_NOTICE='Sorry, due to computing resources issues, this space is under maintenance, and will be restored as soon as possible. '

DESCRIPTION = '''# <a href="https://github.com/THUDM/CogVideo">CogVideo</a>

Currently, this Space only supports the first stage of the CogVideo pipeline due to hardware limitations.

The model accepts only Chinese as input.
By checking the "Translate to Chinese" checkbox, the results of English to Chinese translation with [this Space](https://huggingface.co/spaces/chinhon/translation_eng2ch) will be used as input.
Since the translation model may mistranslate, you may want to use the translation results from other translation services.
'''
NOTES = 'This app is adapted from <a href="https://github.com/hysts/CogVideo_demo">https://github.com/hysts/CogVideo_demo</a>. It would be recommended to use the repo if you want to run the app yourself.'
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=THUDM.CogVideo" />'


def main():
    only_first_stage = True
    # model = AppModel(only_first_stage)

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(MAINTENANCE_NOTICE)
        '''
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.Textbox(label='Input Text')
                    translate = gr.Checkbox(label='Translate to Chinese',
                                            value=False)
                    seed = gr.Slider(0,
                                     100000,
                                     step=1,
                                     value=1234,
                                     label='Seed')
                    only_first_stage = gr.Checkbox(
                        label='Only First Stage',
                        value=only_first_stage,
                        visible=not only_first_stage)
                    image_prompt = gr.Image(type="filepath",
                                            label="Image Prompt",
                                            value=None)
                    run_button = gr.Button('Run')

            with gr.Column():
                with gr.Group():
                    translated_text = gr.Textbox(label='Translated Text')
                    with gr.Tabs():
                        with gr.TabItem('Output (Video)'):
                            result_video = gr.Video(show_label=False)

        examples = gr.Examples(
            examples=[['骑滑板的皮卡丘', False, 1234, True,None],
                      ['a cat playing chess', True, 1253, True,None]],
            fn=model.run_with_translation,
            inputs=[text, translate, seed, only_first_stage,image_prompt],
            outputs=[translated_text, result_video],
            cache_examples=True)

        gr.Markdown(NOTES)
        gr.Markdown(FOOTER)
        print(gr.__version__)
        run_button.click(fn=model.run_with_translation,
                         inputs=[
                             text,
                             translate,
                             seed,
                             only_first_stage,
                             image_prompt
                         ],
                         outputs=[translated_text, result_video])
        print(gr.__version__)
        '''
    demo.launch()


if __name__ == '__main__':
    main()
