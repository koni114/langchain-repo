import os
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a (int): first int
        b (int): second int
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [add, multiply]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-4-turbo-preview")

llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class add(BaseModel):
    """Add two integers together."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers together."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

chain = llm_with_tools | PydanticToolsParser(tools=[multiply, add])

first_func, second_func = chain.invoke(query)






















