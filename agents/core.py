# Minimal stubs for Agent, Runner, function_tool
class Agent:
    def __init__(self, name, instructions, model, tools=None, mcp_servers=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.mcp_servers = mcp_servers or []

class Runner:
    @staticmethod
    async def run(starting_agent, input):
        class Result:
            final_output = f"[Stubbed response for: {input}]"
        return Result()

def function_tool(func):
    return func
