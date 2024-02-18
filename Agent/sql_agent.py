from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

def _strip(text: str) -> str:
    return text.strip()

template="""Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most 10 results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}
"""

prompt = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=template,
)

llm = ChatOpenAI(temperature=0, streaming=True)

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

chain = (
        RunnablePassthrough()
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
)


response = chain.invoke({"input": "employee 들의 Email 에 대해서 알려줘 \nSQLQuery: ",
                         "table_info": db.get_table_info(table_names=["employees"]),
                         "dialect": db.dialect}
                        )
print(response)