# Multi-Agent LLM Framework for Materials Science  

This research project introduces a multi-agent framework that integrates Large Language Models with specialized computational tools to tackle the challenges of materials analysis, simulation, and design. By combining the natural language reasoning capabilities of LLMs with the precision of domain-specific tools, we aim to show the potential to develop a robust, automated, and adaptable system for materials science workflows.

The system leverages LLMs to interpret user queries, synthesize information, and coordinate workflows, while task-specific agents perform computationally intensive operations. The architecture ensures flexibility and extensibility, making it a versatile solution for researchers and engineers.

At its core, the framework utilizes OpenAI's GPT-3.5-turbo, enhanced with custom prompts for domain-specific reasoning and tool integration. The modular design supports diverse tasks, including:

- Data extraction from databases like the Materials Project and MatWeb.
- Continuum simulations like cellular automata and Monte Carlo Potts models.
- Crystal structure generation using the CrystaLLM model.
- Molecular dynamics simulations using the Atomic Simulation Environment (ASE).

![Framework](images/figure_3-transformed.png)  

## Getting Started 
Create a virtual environment  

```bash
conda create -n matllm python=3.10
conda activate matllm
```

Clone the repository  

```bash
git clone https://github.com/cakshat/Multi-Agent-LLM-Framework-MatSci.git
cd Multi-Agent-LLM-Framework-MatSci
```  

Install the dependencies  

```bash
pip install -r requirements.txt
```  

For using crystallm, create a folder named `crystallm_v1_small` in the `crystallm` folder and download the model from [here](https://zenodo.org/records/10642388) and place it in the folder.  

Create a `.env` file in the root directory and add the following environment variables:  

```python
OPENAI_API
MP_API
```  

For using the agent, import the necessary agent and invoke it with your prompt. For example:    

```bash
>>> from agents.material_extraction_agent import MaterialExtractionAgent
>>> mea = MaterialExtractionAgent()
>>> mea.invoke("Retrieve materials with Poisson's ratios ranging from 0.2 to 0.35, list their poissons ratios along with other elastic properties")  
```

Responses will be generated based on the prompt and the agent's capabilities.
 ```bash
> Entering new AgentExecutor chain...
Invoking: `MaterialsProjectElasticity` with `{'poisson_ratio': [0.2, 0.35], ...
...
The search for materials with Poisson's ratios ranging from 0.2 to 0.35 has provided several materials with their elastic properties. Here are some of the materials found along with their elastic properties:

1. Silicon (Si):
   - Poisson's Ratio: 0.318
   - Bulk Modulus: 67.85 GPa
   - Shear Modulus: 28.05 GPa
   - Anisotropy: 2.66

2. Oxygen (O₂):
   - Poisson's Ratio: 0.318
   - Bulk Modulus: 1.74 GPa
   - Shear Modulus: 1.04 GPa
   - Anisotropy: -17.18

...

5. Carbon (C):
   - Poisson's Ratio: 0.31
   - Bulk Modulus: 45.5 GPa
   - Shear Modulus: 30.7 GPa
   - Anisotropy: -3.046

These materials exhibit a range of elastic properties and can be further analyzed for specific applications requiring materials with these characteristics.

> Finished chain.
 ```  

 Note: Create these folders for the agent to store the data: `materials_project_data`, `matweb_data`, `crystallm_cifs`, `continuum_simulation_results`, `md_simulation_results`. You can also modify the paths in the agent code if you want to store the data in a different location.  


## Agents  
The framework consists of the following agents:  

### Material Extraction Agent  
Integrates computational resources like the Materials Project and MatWeb for material discovery and analysis, providing real-time, authoritative data on properties like elasticity, phonons, and thermodynamics. Bridges the gap between theoretical models and experimental datasets, addressing the domain-specific limitations of traditional LLMs.  

### Continuum Simulation Agent  
Executes advanced simulations using Cellular Automata for phase transformations and Monte Carlo Annealing for microstructural evolution. Provides insights into grain growth, nucleation, and phase kinetics, offering actionable results and visualization for materials design.  

### Crystal Generation Agent  
Uses the CrystaLLM model to generate plausible crystal structures based on compositions or alloy systems. Captures crystallographic patterns to predict lattice parameters, atomic positions, and symmetry, enhancing workflows for materials design and stability evaluations.  

### Molecular Dynamics Agent  
Simulates atomic-scale phenomena using tools like ASE, supporting various interatomic potentials and conditions (NVE, NVT, NPT). Automates workflows from initialization to visualization, enabling natural language-based queries for studying thermodynamic and mechanical properties of materials.  