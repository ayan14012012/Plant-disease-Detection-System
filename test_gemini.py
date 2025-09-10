import google.generativeai as genai

genai.configure(api_key="AIzaSyCwKHV15ClbD2-2BUAwtVoeVQqW62wcE0w")

model = genai.GenerativeModel(model_name="models/gemini-pro")

response = model.generate_content("What is organic farming?")
print(response.text)
