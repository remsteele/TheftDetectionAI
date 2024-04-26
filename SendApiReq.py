import PIL.Image
import google.generativeai as genai

def get_api_req(prompt, file_name, API_KEY):
  genai.configure(api_key=API_KEY)
  model = genai.GenerativeModel('gemini-pro-vision')
  img = PIL.Image.open(file_name)
  response = model.generate_content([prompt, img], stream=True)
  response.resolve()
  return response.text