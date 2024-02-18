from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description

from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = OpenAI()

# Construct the ReAct agent

missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )


if missing_vars:
    raise ValueError(f"Prompt missing required variables: {missing_vars}")


prompt = prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

llm_with_stop = llm.bind(stop=["\nObservation"])

agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()  # AgentAction, AgentFinish 중 하나를 return. 
)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})


