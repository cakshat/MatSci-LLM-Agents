from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from tools.material_extraction import (
    get_material_ids,
    MPSurfacePropertiesTool,
    MPPhononTool,
    MPElasticityTool,
    MPThermoTool,
    MPDielectricTool,
    MPPiezoelectricTool,
    MPMagnetismTool,
    MPSynthesisTool,
    MPElectronicStructureTool,
    MPElectronicBandStructureTool,
    MPElectronicDensityOfStatesTool,
    MPOxidationStatesTool,
    MPBondsTool,
    MPAbsorptionTool,
    extract_materials_matweb
)
from tools.material_extraction import OPENAI_API
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)

tools = [
    get_material_ids,
    MPSurfacePropertiesTool(),
    MPPhononTool(),
    MPElasticityTool(),
    MPThermoTool(),
    MPDielectricTool(),
    MPPiezoelectricTool(),
    MPMagnetismTool(),
    MPSynthesisTool(),
    MPElectronicStructureTool(),
    MPElectronicBandStructureTool(),
    MPElectronicDensityOfStatesTool(),
    MPOxidationStatesTool(),
    MPBondsTool(),
    MPAbsorptionTool(),
    extract_materials_matweb
]

class MaterialExtractionAgent:
    def __init__(self, api_key=OPENAI_API, tools=tools, verbose=True):
        self.chat_history = []
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        self.tools = tools
        self.verbose = verbose
        
        MEMORY_KEY = "chat_history"

        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are a powerful agent for performing data extraction from the Materials Project and matweb.com.
            Your task is to determine the type of problem in the user's query before taking action. 
            First, classify the problem into one of the following types:
            1. Finding materials based on given physical properties (e.g., Young's modulus, tensile strength).
            2. Getting properties of a specific material (e.g., getting the properties of mp-149).
            
            Based on your classification, proceed with the appropriate action:
            - If it's Type 1 (Finding materials based on properties), explain the tool you are using (e.g., searching for materials based on physical properties), what parameters you are using, and why you chose those parameters. Then, execute the appropriate action to search for materials.
            - If it's Type 2 (Getting properties of a specific material), first find the material_id(s) for the material, then explain the tool you are using (e.g., fetching material properties), the parameters, and the reasoning behind those choices. Then, execute the appropriate action to retrieve the properties of the specific material.
            
            DO NOT MAKE YOUR OWN DATA UP. If Materials Project does not return any results, state that and ask for more information.
            After running the extraction from materials project, if the problem is type 1, extract from matweb as well if the property is
            relevant to the agent
            """
        )

        human_message = HumanMessagePromptTemplate.from_template(
            """
            Given the user's query: '{input}', first classify the problem as either '1' for finding materials
        based on physical properties or '2' for getting material properties. If it is '2', then state the chemical
        formula of the  specified materials and its contituent elements seperately. Use the chemical formula to get material_id(s)
        first then proceed. If it is type '1' use the appropiate tool.
            """
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                human_message,
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm_with_tools = self.llm.bind_tools(self.tools)

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
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=self.verbose)

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