import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from tools.molecular_dynamics import (
    MDSimulationNVETool,
    MDSimulationNVTTool,
    MDSimulationNPTTool,
)
from langchain_core.messages import AIMessage, HumanMessage

OPENAI_API = os.getenv("OPENAI_API")

tools = [
    MDSimulationNVETool(),
    MDSimulationNVTTool(),
    MDSimulationNPTTool()
]

class MolecularDynamicsAgent:
    def __init__(self, api_key=OPENAI_API, tools=tools, verbose=True):
        self.chat_history = []
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        self.tools = tools
        self.verbose = verbose
        
        MEMORY_KEY = "chat_history"

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are tasked with performing molecular dynamics (MD) simulations using predefined tools.
You can perform MD simulations under 3 conditions: NVE (constant energy), NVT (constant temperature) and NPT (constant pressure and temperature).
For initializing atomic system without cif file, if lattice parameters are not given select appropiate values for the lattice parameters
and mention it explicitly to the user. For cubic crystal systems make sure that a=b=c when initializing lattice structures. All temperature
inputs to functions are in Kelvin so convert temperature if needed according before passing to the tools.
                    """
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm_with_tools = self.llm.bind_tools(tools)

        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    def _update_chat_history(self, user_input, agent_output):
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=agent_output),
        ])

    def get_tools(self):
        return self.tools

    def get_tools_info(self):
        return {tool.__class__.__name__: tool.__doc__ for tool in self.tools}

    def invoke(self, user_input):
        result = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        self._update_chat_history(user_input, result["output"])
        return result["output"]