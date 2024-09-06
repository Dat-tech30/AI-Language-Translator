# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers ipywidgets gradio --upgrade

import gradio as gr # UI Library
from transformers import pipeline # Transformers Pipeline

# Loading up the pipeline
translation_pipeline = pipeline("translation_en_to_vi") 
results = translation_pipeline('I love learning about coding')
results[0]['translation_text']

def translate_transformers(from_text):
    results = translation_pipeline(from_text)
    return results[0]['translation_text']

translate_transformers('My name is Dat')

interface = gr.Interface(fn=translate_transformers, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Text to translate'),
                        outputs='text')

interface.launch()

