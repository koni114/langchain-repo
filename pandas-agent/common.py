def get_python_code_in_inter_steps(results: dict):
    import re
    if isinstance(results, dict):
        if not results['intermediate_steps']:
            output_results = results['output']
            pattern = r'```python\n(.*?)```'
            python_code_str = re.findall(pattern, output_results, re.DOTALL)
            if python_code_str:
                return python_code_str[-1]
            else:
                return "python code was not generated. Please answer the same question one more time."
        else:
            agent_action_message_log = results['intermediate_steps'][-1][0]
            execute_code = agent_action_message_log.__dict__['tool_input']['query']
            return execute_code
    else:
        raise TypeError("input value is not dict type.")