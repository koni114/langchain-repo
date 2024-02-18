import os
import openai

from openai import OpenAI
from base import OpenAIAssistantRunnable
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# Upload a file with an "assistants" purpose
file = client.files.create(
  file=open("bike.csv", "rb"),
  purpose='assistants'
)

assistant = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="You are a perfect data anaylst. When asked a question, tell me python code and run code to answer the question.",
    model="gpt-4-turbo-preview",
    tools=[{"type": "code_interpreter"}],
    client=client,
    file_ids=[file.id]
)

output = assistant.invoke({"content": "월별 평균 온도(temp) 의 추이는 어떻게 되나요?"})

# message = output[3]
for message in output:
  message_content = message.content[0]
  if isinstance(message_content, openai.types.beta.threads.message_content_text.MessageContentText):
    print(message_content.text.value)
  elif isinstance(message_content, openai.types.beta.threads.message_content_image_file.MessageContentImageFile):
    fild_id = message_content.image_file.file_id
    image_data = client.files.content(fild_id)
    image_data_bytes = image_data.read()
    with open("./my-image.png", "wb") as file:
        file.write(image_data_bytes)