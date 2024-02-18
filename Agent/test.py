import re
from langchain_core.agents import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"
text = """Question: What's the best-selling model name?
Thought: Now that I have the table names, I can see which table has the data I need
Action: sql_db_query_checker
Action Input: SELECT * FROM purchase_history WHERE model IS NOT NULL GROUP BY model HAVING SUM(price) > 1000"""

regex = (
    r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
)

action_match = re.search(regex, text, re.DOTALL)

print(action_match.group())

action = action_match.group(1).strip()
action_input = action_match.group(2)
tool_input = action_input.strip(" ")
tool_input = tool_input.strip('"')

AgentAction(action, tool_input, text)

