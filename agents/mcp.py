# Minimal stub for MCPServerSse
class MCPServerSse:
    def __init__(self, name, params):
        self.name = name
        self.params = params
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
