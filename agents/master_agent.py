import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.tools import tool

from agents.material_extraction_agent import MaterialExtractionAgent
from agents.continuum_simulation_agent import ContinuumSimulationAgent
from agents.crystal_generation_agent import CrystalGenerationAgent
from agents.molecular_dynamics_agent import MolecularDynamicsAgent

agents_dict = {
    "MaterialExtractionAgent" : MaterialExtractionAgent(),
    "ContinuumSimulationAgent" : ContinuumSimulationAgent(),
    "CrystalGenerationAgent" : CrystalGenerationAgent(),
    "MolecularDynamicsAgent" : MolecularDynamicsAgent()
}

OPENAI_API = os.getenv("OPENAI_API")

class MatSciAgent:
    def __init__(self, api_key=OPENAI_API, agents_dict=agents_dict, verbose=True):
        self.chat_history = []
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        self.agents = agents_dict
        self.verbose = verbose

        self.tools_info = self._gather_tools_info()

        # Initial selection prompt for determining which agent to use
        self.selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are an AI assistant that selects the correct specialized agent based on user input and available tools.
             In general for retrieval of specific materials or properties use MaterialExtractionAgent, for continuum
             simulations use ContinuumSimulationAgent for generating .cif files (crystal structures) use CrystalGenerationAgent
             and for molecular dynamic simulations use MolecularDynamicsAgent.
             If the appropriate agent exists, at end of your response, you MUST include the name of the specific AGENT to be used,
             not just the tools, along with reason for selection.
             If an appropriate agent is not available, suggest alternative methods or resources.
             """),
            ("system", "Here are the available tools:\n{tools_info}"),
            ("user", "{input}"),
        ])

        # Placeholder for the current agent and executor
        self.agent = None
        self.agent_executor = None

    def _gather_tools_info(self):
        """Dynamically gathers available tools and their docstrings from all agents."""
        tools_info = {}
        for agent_name, agent in self.agents.items():
            tools_info[agent_name] = agent.get_tools_info()
        return tools_info

    def _select_agent(self, user_input):
        """Determines the appropriate agent for the input."""
        formatted_prompt = self.selection_prompt.format(
            tools_info=self.tools_info,
            input=f"Which agent should handle this request? {user_input}"
        )
        response = self.llm.invoke(formatted_prompt)
        if self.verbose:
            print(f"LLM Agent Selector Response:\n{response.content.strip()}")

        for agent_name in self.agents:
            if agent_name in response.content:
                if self.verbose:
                    print(f"Selected agent: {agent_name}")
                return self.agents[agent_name]
        return None

    def _update_chat_history(self, user_input, agent_output):
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=agent_output),
        ])

    def invoke(self, user_input):
        """Main user interface — selects the agent and runs the appropriate toolchain."""
        selected_agent = self._select_agent(user_input)

        if selected_agent:
            inputs = {
                "input": user_input,
                "chat_history": self.chat_history,
            }
            response = selected_agent.invoke(inputs)

            # if isinstance(response, dict):
            #     output_text = response.get("output", "")
            # else:
            #     output_text = str(response)

            self._update_chat_history(user_input, response)
            return response
        else:
            fallback = "I couldn't determine the correct agent for this request."
            self._update_chat_history(user_input, fallback)
            return {"output": fallback}