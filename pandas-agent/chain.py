from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv

from common import get_python_code_in_inter_steps

load_dotenv()

df = pd.read_csv("bike.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    return_intermediate_steps=True,
    verbose=True
)

questions = [
    "온도(temp)가 자전거 대여 수(count)에 미치는 영향은 어떠한가요?",
    "습도(humidity)가 높을 때와 낮을 때의 자전거 대여 수 차이는 어떻게 되나요?",
    "풍속(windspeed)이 자전거 대여 수에 끼치는 영향은 어떤가요?",
    "시간대별(아침, 오후, 저녁, 밤) 자전거 대여 수는 어떻게 변화하나요?",
]

query_results = []
results_list = []

# question = questions[0]
for question in questions:
    results = agent.invoke({"input": question})
    results_list.append(results)
    query_results.append(get_python_code_in_inter_steps(results))

for idx, query in enumerate(query_results):
    print(f"### {idx}")
    print(query)