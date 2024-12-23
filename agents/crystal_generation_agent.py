import os
from ase.io import read
from ase.visualize import view
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from tools.crystal_generation import CrystaLLMTool

OPENAI_API = os.getenv("OPENAI_API")

tools = [
    CrystaLLMTool()
]

class CrystalGenerationAgent:
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
                    You are powerful agent for generating crystal structures for a given composition and, optionally, space group.
            For chemical composition make sure to have number with element. For e.g. BaTiO3 should be passed as Ba1Ti1O3.
            From the description of cif file list the key points in a list.
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

    def invoke(self, user_input):
        result = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        self._update_chat_history(user_input, result["output"])
        return result["output"]