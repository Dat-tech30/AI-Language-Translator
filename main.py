# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers ipywidgets gradio --upgrade

import gradio as gr # UI Library
from transformers import pipeline # Transformers Pipeline

# Loading up the pipeline
translation_pipeline = pipeline("translation_en_to_vi") 