import os

import pandas as pd
from openai import OpenAI


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


# 1. Assistant 생성
assistant = client.beta.assistants.create(
  instructions="You are a perfect data anaylst. When asked a question, tell me python code and run code to answer the question.",
  model="gpt-4-turbo-preview",
  tools=[{"type": "code_interpreter"}],
  file_ids=[file.id]
)

# 2. Thread 생성
thread = client.beta.threads.create()


# 3. 스레드에 메시지 추가하기
message = client.beta.threads.messages.create(
    thread_id=thread.id,

    role="user",
    content="`월별 평균 온도(temp) 의 추이는 어떻게 되나요?"
)

# Run the Assistant
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

# Check the Run status
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

# Display the Assitant's Response
messages = client.beta.threads.messages.list(
  thread_id=thread.id,
  
)
