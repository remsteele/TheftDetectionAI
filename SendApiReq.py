import pathlib
import textwrap
import os
import PIL.Image
import google.generativeai as genai
import time
from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def get_api_req(prompt, file_name, API_KEY):
  genai.configure(api_key=API_KEY)
  model = genai.GenerativeModel('gemini-pro-vision')
  img = PIL.Image.open(file_name)
  response = model.generate_content([prompt, img], stream=True)
  response.resolve()
  return response.text