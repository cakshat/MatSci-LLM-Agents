import os
from mp_api.client import MPRester
from langchain.agents import tool
from tqdm import tqdm
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from difflib import get_close_matches
from datetime import datetime
import tiktoken
from openai import OpenAI
from datetime import datetime
from typing import Type
from langchain.agents import tool
from langchain.tools import BaseTool
from pydantic import BaseModel

from emmet.core.surface_properties import SurfacePropDoc
from emmet.core.phonon import PhononBSDOSDoc
from emmet.core.elasticity import ElasticityDoc
from emmet.core.thermo import ThermoDoc
from emmet.core.synthesis import (
    SynthesisSearchResultModel,
)
from emmet.core.electronic_structure import (
    ElectronicStructureDoc,
)
from emmet.core.polar import DielectricDoc
from emmet.core.polar import PiezoelectricDoc
from emmet.core.magnetism import MagnetismDoc
from emmet.core.oxidation_states import OxidationStateDoc
from emmet.core.bonds import BondingDoc
from emmet.core.absorption import AbsorptionDoc
from mp_api.client import MPRester

from tools.schemas import (
    MPAbsorptionInput,
    MPBondsInput,
    MPDielectricInput,
    MPElasticityInput,
    MPElectronicBandStructureInput,
    MPElectronicDensityOfStatesInput,
    MPElectronicStructureInput,
    MPMagnetismInput,
    MPOxidationStatesInput,
    MPPhononInput,
    MPPiezoelectricInput,
    MPSurfacePropertiesInput,
    MPSynthesisInput,
    MPThermoInput
)

MAX_TOKENS = 16385
OPENAI_API = os.getenv("OPENAI_API")
MP_API = os.getenv("MP_API")
################

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.get_encoding("cl100k_base" if model == "gpt-3.5-turbo" else "p50k_base")
    tokens = encoding.encode(text)
    return len(tokens)

fields_to_remove = [
    "fields_not_requested",
    "builder_meta",
    "warnings",
    "fitting_data",
    "fitting_method",
    "state",
    "origins",
    "last_updated"
]

def remove_fields_from_docs(docs_list: list, fields_to_remove: list) -> list:
    """
    Removes specified fields from a list of documents.
    
    Args:
    docs_list (list): List of dictionaries (documents) to clean.
    fields_to_remove (list): List of field names to remove from each document.
    
    Returns:
    list: Cleaned list of documents without the specified fields.
    """
    cleaned_docs = []
    for doc in docs_list:
        if isinstance(doc, dict):
            cleaned_doc = {key: value for key, value in doc.items() if key not in fields_to_remove}
            cleaned_docs.append(cleaned_doc)
    return cleaned_docs

def generate_summary(docs_list) -> str:
    """
    Converts a list of various docs (e.g. ElasticityDoc, ElectronicStructureDoc, SurfacePropDoc, or matweb data)
    passed as a str extracted from Materials Project into simple text summary for the user.
    """
    MAX_DOC_LIMIT = 15
    summaries = []
    for doc_list in docs_list[:MAX_DOC_LIMIT]:
        results_str = json.dumps(doc_list)
        if count_tokens(results_str) > MAX_TOKENS:
            continue
        client = OpenAI(api_key=OPENAI_API)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that generates concise, informative summaries."
                },
                {
                    "role": "user", 
                    "content": f"""Please generate a concise, informative summary of these results.
                    Highlight key insights, patterns, and most significant findings.
                    Keep the summary under 300 words and focus on the most important information.
                    
                    Results:
                    {results_str}
                    
                    Summary:"""
                }
            ],
            max_tokens=350
        )
        summaries.append(response.choices[0].message.content)
    return " ".join(summaries)

matweb_property_fields = [
    "1% Secant Modulus (546 matls)",
    "1% Secant Modulus, MD (1953 matls)",
    "1% Secant Modulus, TD (1917 matls)",
    "2% Secant Modulus (546 matls)",
    "Arc Resistance (3268 matls)",
    "Charpy Impact (1055 matls)",
    "Charpy Impact Unnotched (11925 matls)",
    "Charpy Impact, Notched (16186 matls)",
    "Coefficient of Friction (3480 matls)",
    "Comparative Tracking Index (9096 matls)",
    "Compressive Modulus (1892 matls)",
    "Compressive Yield Strength (8620 matls)",
    "CTE, linear (29476 matls)",
    "CTE, linear, Transverse to Flow (7064 matls)",
    "Cure Time (4110 matls)",
    "Deflection Temperature at 0.46 MPa (66 psi) (24169 matls)",
    "Deflection Temperature at 1.8 MPa (264 psi) (35786 matls)",
    "Deflection Temperature at 8.0 MPa (895 matls)",
    "Density (108227 matls)",
    "Dielectric Constant (13463 matls)",
    "Dielectric Strength (15920 matls)",
    "Dissipation Factor (10601 matls)",
    "Drying Temperature (19391 matls)",
    "Electrical Resistivity (30440 matls)",
    "Elongation at Break (72346 matls)",
    "Elongation at Yield (14453 matls)",
    "Emissivity (0-1) (162 matls)",
    "Fatigue Strength (1166 matls)",
    "Film Elongation at Break, MD (3604 matls)",
    "Film Elongation at Break, TD (3361 matls)",
    "Film Elongation at Yield, MD (298 matls)",
    "Film Elongation at Yield, TD (255 matls)",
    "Film Tensile Strength at Break, MD (3775 matls)",
    "Film Tensile Strength at Break, TD (3522 matls)",
    "Film Tensile Strength at Yield, MD (1659 matls)",
    "Film Tensile Strength at Yield, TD (1631 matls)",
    "Flammability, UL94 (25332 matls)",
    "Flexural Modulus (45901 matls)",
    "Flexural Stiffness (45901 matls)",
    "Flexural Yield Strength (37318 matls)",
    "Fracture Toughness (608 matls)",
    "Gardner Impact (1578 matls)",
    "Glass Transition Temp, Tg (5864 matls)",
    "Gloss (3190 matls)",
    "Glow Wire Test (3959 matls)",
    "Hardness, Barcol (495 matls)",
    "Hardness, Brinell (4721 matls)",
    "Hardness, Knoop (3374 matls)",
    "Hardness, Rockwell A (667 matls)",
    "Hardness, Rockwell B (4155 matls)",
    "Hardness, Rockwell C (3210 matls)",
    "Hardness, Rockwell E (322 matls)",
    "Hardness, Rockwell M (3175 matls)",
    "Hardness, Rockwell R (9489 matls)",
    "Hardness, Shore A (17205 matls)",
    "Hardness, Shore D (11015 matls)",
    "Hardness, Vickers (4107 matls)",
    "Haze (5108 matls)",
    "Heat Distortion Temperature (377 matls)",
    "Heat of Fusion (1070 matls)",
    "Hot Ball Pressure Test (854 matls)",
    "Izod Impact (689 matls)",
    "Izod Impact, Notched (29459 matls)",
    "Izod Impact, Unnotched (10012 matls)",
    "K (wear) Factor (961 matls)",
    "Linear Mold Shrinkage (34696 matls)",
    "Liquidus (5121 matls)",
    "Magnetic Permeability (898 matls)",
    "Maximum Service Temperature, Air (20174 matls)",
    "Maximum Service Temperature, Inert (360 matls)",
    "Melt Flow (32725 matls)",
    "Melt Temperature (27542 matls)",
    "Melting Point (27230 matls)",
    "Minimum Service Temperature, Air (10058 matls)",
    "Modulus of Elasticity (41250 matls)",
    "Moisture Absorption at Equilibrium (9911 matls)",
    "Moisture Vapor Transmission (344 matls)",
    "Mold Temperature (25583 matls)",
    "Oxygen Index (5050 matls)",
    "Oxygen Transmission (585 matls)",
    "Poissons Ratio (7883 matls)",
    "Processing Temperature (11343 matls)",
    "Reflection Coefficient, Visible (0-1) (260 matls)",
    "Refractive Index (4704 matls)",
    "Ring & Ball Softening Point (430 matls)",
    "Rupture Strength (18 matls)",
    "Secant Modulus (546 matls)",
    "Shear Modulus (7784 matls)",
    "Shear Strength (4757 matls)",
    "Solidus (4854 matls)",
    "Specific Heat Capacity (9777 matls)",
    "Stiffness Modulus (45901 matls)",
    "Surface Resistance (12128 matls)",
    "Tack-Free Time (510 matls)",
    "Tear Strength (9151 matls)",
    "Tensile Modulus (41250 matls)",
    "Tensile Strength at Break (66430 matls)",
    "Tensile Strength, Ultimate (66430 matls)",
    "Tensile Strength, Yield (46238 matls)",
    "Thermal Conductivity (16755 matls)",
    "Transmission, Visible (6377 matls)",
    "UL RTI, Electrical (3287 matls)",
    "UL RTI, Mechanical with Impact (2989 matls)",
    "UL RTI, Mechanical without Impact (3002 matls)",
    "Vicat Softening Point (17839 matls)",
    "Water Absorption (18636 matls)",
    "Water Absorption at Saturation (4262 matls)"
]

### Materials Project Tools ###
@tool
def get_material_ids(compound: "str") -> list:
    """
    Returns a list of material_ids which can be used for extraction from materials project as needed.
    """
    mpr = MPRester(MP_API)
    with MPRester(MP_API, mute_progress_bars=True) as mpr:
        material_ids = mpr.get_material_ids(chemsys_formula=compound)
    return material_ids

class MPSurfacePropertiesTool(BaseTool):
    name: str = "MaterialsProjectSurfaceProperties"
    description: str = """
Query surface properties using a variety of criteria via API. Options include:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- has_reconstructed (bool): Filter by reconstructed surfaces.
- shape_factor (Tuple[float, float]): Range of shape factors.
- surface_energy_anisotropy (Tuple[float, float]): Range of surface energy anisotropy.
- weighted_surface_energy (Tuple[float, float]): Range of weighted surface energy (J/m²).
- weighted_work_function (Tuple[float, float]): Range of weighted work function (eV).
- num_chunks (int): Max chunks of data to yield; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id if all_fields=False).
"""
    args_schema: Type[BaseModel] = MPSurfacePropertiesInput

    def _run(self, **kwargs) -> list[SurfacePropDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/surface_properties_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.surface_properties.search(**kwargs)
        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPPhononTool(BaseTool):
    name: str = "MaterialsProjectPhonon"
    description: str = """
    Query phonon docs using a variety of search criteria.
    This tool uses a API call which uses any or all of the following properties:
    - material_ids : str, List[str]
    A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13]).

    - num_chunks : int
    Maximum number of chunks of data to yield. None will yield all possible.

    - chunk_size : int
    Number of data entries per chunk.

    - all_fields : bool
    Whether to return all fields in the document. Defaults to True.

    - fields : List[str]
    List of fields in PhononBSDOSDoc to return data for. Default is material_id, last_updated, and formula_pretty if all_fields is False.
    """
    args_schema: Type[BaseModel] = MPPhononInput

    def _run(self, **kwargs)-> list[PhononBSDOSDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/phonon{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.phonon.search(**kwargs)
        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPElasticityTool(BaseTool):
    name: str = "MaterialsProjectElasticity"
    description: str = """
Query elasticity properties using an API with the following options:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- elastic_anisotropy (Tuple[float, float]): Range for elastic anisotropy.
- g_voigt, g_reuss, g_vrh (Tuple[float, float]): Ranges for shear moduli (Voigt, Reuss, VRH) in GPa.
- k_voigt, k_reuss, k_vrh (Tuple[float, float]): Ranges for bulk moduli (Voigt, Reuss, VRH) in GPa.
- poisson_ratio (Tuple[float, float]): Range for Poisson's ratio.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, pretty_formula if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPElasticityInput

    def _run(self, **kwargs) -> list[ElasticityDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/elastic_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.elasticity.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPThermoTool(BaseTool):
    name: str = "MaterialsProjectThermo"
    description: str = """
Query core thermo data using an API with these options:
- is_stable (bool): Filter by material stability.
- material_ids (List[str]): List of Materials Project IDs.
- thermo_ids (List[str]): List of thermo IDs (e.g., mp-149_GGA_GGA+U).
- thermo_types (List[ThermoType]): Thermo types (e.g., ThermoType.GGA_GGA_U).
- num_elements (Tuple[int, int]): Range for the number of elements.
- total_energy, uncorrected_energy (Tuple[float, float]): Ranges for corrected/uncorrected total energy (eV/atom).
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPThermoInput

    def _run(self, **kwargs)-> list[ThermoDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/thermo_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.thermo.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPDielectricTool(BaseTool):
    name: str = "MaterialsProjectDielectric"
    description: str = """
Query dielectric data using an API with these options:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- e_total (Tuple[float, float]): Range for total dielectric constant.
- e_ionic (Tuple[float, float]): Range for ionic dielectric constant.
- e_electronic (Tuple[float, float]): Range for electronic dielectric constant.
- n (Tuple[float, float]): Range for refractive index.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPDielectricInput

    def _run(self, **kwargs)-> list[DielectricDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/dielectric_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.dielectric.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPPiezoelectricTool(BaseTool):
    name: str = "MaterialsProjectPiezoelectric"
    description: str = """
    Query piezoelectric data using a variety of search criteria.
    This tool uses a API call which uses any or all of the following properties:
    - material_ids : str, List[str]
    A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13]).

    - piezoelectric_modulus : Tuple[float,float]
    Minimum and maximum of the piezoelectric modulus in C/m² to consider.

    - num_chunks : int
    Maximum number of chunks of data to yield. None will yield all possible.

    - chunk_size : int
    Number of data entries per chunk.

    - all_fields : bool
    Whether to return all fields in the document. Defaults to True.

    - fields : List[str]
    List of fields in PiezoDoc to return data for. Default is material_id and last_updated if all_fields is False.
    """
    args_schema: Type[BaseModel] = MPPiezoelectricInput

    def _run(self, **kwargs) -> list[PiezoelectricDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/piezoelectric_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.piezoelectric.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPMagnetismTool(BaseTool):
    name: str = "MaterialsProjectMagnetism"
    description: str = """
Query magnetism data using an API with these options:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- num_magnetic_sites (Tuple[int, int]): Range for magnetic sites.
- num_unique_magnetic_sites (Tuple[int, int]): Range for unique magnetic sites.
- ordering (Ordering): Magnetic ordering of the material.
- total_magnetization (Tuple[float, float]): Range for total magnetization.
- total_magnetization_normalized_vol (Tuple[float, float]): Range for magnetization normalized by volume.
- total_magnetization_normalized_formula_units (Tuple[float, float]): Range for magnetization normalized by formula units.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""
    args_schema: Type[BaseModel] = MPMagnetismInput

    def _run(self, **kwargs) -> list[MagnetismDoc] | list[dict | str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/magnetism_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.magnetism.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPSynthesisTool(BaseTool):
    name: str = "MaterialsProjectSynthesis"
    description: str = """
Search synthesis recipe text using an API with these options:
- synthesis_type (List[SynthesisTypeEnum]): Types of synthesis to include.
- target_formula (str): Chemical formula of the target material.
- precursor_formula (str): Chemical formula of the precursor material.
- operations (List[OperationTypeEnum]): Required synthesis operations.
- condition_heating_temperature_min/max (float): Range for heating temperature.
- condition_heating_time_min/max (float): Range for heating time.
- condition_heating_atmosphere (List[str]): Required atmosphere (e.g., "air", "argon").
- condition_mixing_device (List[str]): Required mixing device (e.g., "zirconia").
- condition_mixing_media (List[str]): Required mixing media (e.g., "alcohol").
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
"""

    args_schema: Type[BaseModel] = MPSynthesisInput

    def _run(self, **kwargs) -> list[SynthesisSearchResultModel] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/synthesis_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.synthesis.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPElectronicStructureTool(BaseTool):
    name: str = "MaterialsProjectElectronicStructure"
    description: str = """
Query electronic structure data using an API with these options:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- band_gap (Tuple[float, float]): Range for band gap in eV.
- chemsys (str, List[str]): Chemical system or list of systems (e.g., Li-Fe-O).
- efermi (Tuple[float, float]): Range for fermi energy in eV.
- elements, exclude_elements (List[str]): List of included or excluded elements.
- formula (str, List[str]): Chemical formula or list (e.g., Fe2O3, ABO3).
- is_gap_direct (bool): Whether the material has a direct band gap.
- is_metal (bool): Whether the material is a metal.
- magnetic_ordering (Ordering): Magnetic ordering.
- num_elements (Tuple[int, int]): Range for number of elements.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPElectronicStructureInput

    def _run(self, **kwargs) -> list[ElectronicStructureDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/electronic_structure_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.electronic_structure.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPElectronicBandStructureTool(BaseTool):
    name: str = "MaterialsProjectElectronicbandStructure"
    description: str = """
Query band structure summary data using an API with these options:
- band_gap (Tuple[float, float]): Range for band gap in eV.
- efermi (Tuple[float, float]): Range for fermi energy in eV.
- is_gap_direct (bool): Whether the band gap is direct.
- is_metal (bool): Whether the material is a metal.
- magnetic_ordering (Ordering): Magnetic ordering.
- path_type (BSPathType): k-path selection convention for the band structure.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPElectronicBandStructureInput

    def _run(self, **kwargs) -> list[ElectronicStructureDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/electronic_band_structure_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.electronic_structure_bandstructure.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPElectronicDensityOfStatesTool(BaseTool):
    name: str = "MaterialsProjectElectronicDensityOfStates"
    description: str = """
Query density of states (DOS) summary data using an API with these options:
- band_gap (Tuple[float, float]): Range for band gap in eV.
- efermi (Tuple[float, float]): Range for fermi energy in eV.
- element (Element): Element for element-projected DOS.
- magnetic_ordering (Ordering): Magnetic ordering.
- orbital (OrbitalType): Orbital for orbital-projected DOS.
- projection_type (DOSProjectionType): Projection type (default: total DOS).
- spin (Spin): Spin channel (non-spin-polarized stored in Spin.up).
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPElectronicDensityOfStatesInput

    def _run(self, **kwargs) -> list[ElectronicStructureDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/electronic_dos_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.electronic_structure_dos.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPOxidationStatesTool(BaseTool):
    name: str = "MaterialsProjectOxidationStates"
    description: str = """
Query oxidation state docs using an API with these options:
- material_ids (str, List[str]): Single or multiple Material IDs (e.g., mp-149).
- chemsys (str, List[str]): Chemical system(s) (e.g., Li-Fe-O, Si-*).
- formula (str, List[str]): Chemical formula or list (e.g., Fe2O3, ABO3).
- possible_species (List[str]): List of elements with oxidation states (e.g. [Cr2+, O2-]).
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated, formula_pretty if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPOxidationStatesInput

    def _run(self, **kwargs) -> list[OxidationStateDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/oxidation_states_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.oxidation_states.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPBondsTool(BaseTool):
    name: str = "MaterialsProjectBonds"
    description: str = """
Query bonding docs using an API with these options:
- material_ids (str, List[str]): Search for bonding data for specified Material IDs.
- coordination_envs (List[str]): List of coordination environments (e.g. ['Mo-S(6)', 'S-Mo(3)']).
- coordination_envs_anonymous (List[str]): List of anonymous coordination environments (e.g. ['A-B(6)', 'A-B(3)']).
- max_bond_length (Tuple[float, float]): Range for maximum bond length in the structure.
- mean_bond_length (Tuple[float, float]): Range for mean bond length in the structure.
- min_bond_length (Tuple[float, float]): Range for minimum bond length in the structure.
- num_chunks (int): Max chunks of data; None for all.
- chunk_size (int): Entries per chunk.
- all_fields (bool): Return all fields (default: True).
- fields (List[str]): Specific fields to return (default: material_id, last_updated if all_fields=False).
"""

    args_schema: Type[BaseModel] = MPBondsInput

    def _run(self, **kwargs) -> list[BondingDoc] | list[dict] | str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/bonds_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.bonds.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
    
class MPAbsorptionTool(BaseTool):
    name: str = "MaterialsProjectAbsorption"
    description: str = """
Query for optical absorption spectra data.
This tool uses a API call which uses any or all of the following properties:
- material_ids : str, List[str]
Search for optical absorption data associated with the specified Material IDs
- chemsys : str, List[str]
A chemical system or list of chemical systems (e.g., Li-Fe-O, Si-*, [Si-O, Li-Fe-P]).
- elements : List[str]
A list of elements.
- exclude_elements : List[str]
A list of elements to exclude.
- formula : str, List[str]
A formula including anonymized formula or wild cards (e.g., Fe2O3, ABO3, Si*). A list of chemical formulas can also be passed (e.g., [Fe2O3, ABO3]).
- num_chunks : int
Maximum number of chunks of data to yield. None will yield all possible.
- chunk_size : int
Number of data entries per chunk.
- all_fields : bool
Whether to return all fields in the document. Defaults to True.
- fields : List[str]
List of fields in AbsorptionDoc to return data for.
    """
    args_schema: Type[BaseModel] = MPAbsorptionInput

    def _run(self, **kwargs) -> list[AbsorptionDoc] | list[dict | str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mpr = MPRester(MP_API)
        filename = f"materials_project_data/absorption_{timestamp}.json"
        with MPRester(MP_API, mute_progress_bars=True) as mpr:
            results = mpr.materials.absorption.search(**kwargs)

        docs_as_dicts = [doc.dict() for doc in results]
        docs_as_dicts = remove_fields_from_docs(docs_as_dicts, fields_to_remove)
        with open(filename, 'w') as f:
            json.dump(docs_as_dicts, f)
        summary = generate_summary(docs_as_dicts)
        return summary
###############################

### Matweb tool ###
@tool
def extract_materials_matweb(
    property="Modulus of Elasticity (41250 matls)",
    low=100,
    high=150,
    num_materials=5
) -> list:
    """
The agent supports the extraction of diverse properties, including but not limited to:
Mechanical Properties: Tensile strength, yield strength, elongation at break, hardness (Rockwell, Vickers, Shore), modulus of elasticity, shear modulus, fatigue strength, and fracture toughness.
Thermal Properties: Thermal conductivity, heat distortion temperature, glass transition temperature, specific heat capacity, and melting/solidus points.
Electrical Properties: Dielectric strength, resistivity, dissipation factor, and dielectric constant.
Optical and Surface Properties: Refractive index, haze, gloss, and surface resistance.
Chemical and Durability Characteristics: Water absorption, oxygen index, arc resistance, and maximum/minimum service temperatures.

Parameters:
property (str): Property to filter by
low (float): lower limit
high (float): higher limit
num_materials (int): number of materials to extract

Return:
materials_data (list) : List of the materials and data.
    """
    def get_valid_property(user_input, matweb_properties):
        matches = get_close_matches(user_input, matweb_properties)
        if matches:
            return matches[0]
        else:
            return None
    # Default incase of invalid value
    property = get_valid_property(property, matweb_property_fields)
    if property is None:
        print("Property cound not be detected, defaulting to Elastic modulus")
        property = "Modulus of Elasticity (41250 matls)"

    def extract_material_data(table_html, link, material):
        soup = BeautifulSoup(table_html, 'html.parser')
        material_data = {}
        material_data['Material'] = material
        material_data['link'] = link

        # Extract table data
        for row in soup.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) >= 2:
                property_name = cells[0].text.strip()
                property_value = cells[1].text.strip()
                if len(cells) >= 4:
                    property_value_comment = cells[3].text.strip()
                material_data[property_name] = property_value
                material_data[property_name + " Comment"] = property_value_comment

        return material_data
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    MATWEB_PATH = "https://www.matweb.com/search/PropertySearch.aspx"
    driver.get(MATWEB_PATH)

    ### Go to website ###
    link_text = "Property"
    property_link = driver.find_element(By.LINK_TEXT, link_text)
    property_link.click()
    #####################

    ### Enter property 1 ###
    material_property_1_name = "ctl00$ContentMain$ucPropertyDropdown1$drpPropertyList"
    material_property_1 = driver.find_element(By.NAME, material_property_1_name)
    select_material_property_1 = Select(material_property_1)
    select_material_property_1.select_by_visible_text(property)
    ########################

    ### Property 1 value ###
    min_property_1_name = "ctl00$ContentMain$ucPropertyEdit1$txtpMin"
    max_property_1_name = "ctl00$ContentMain$ucPropertyEdit1$txtpMax"

    time.sleep(3) # Else the filtering was not working

    min_property_1 = driver.find_element(By.NAME, min_property_1_name)
    min_property_1.clear()
    min_property_1.send_keys(low)

    max_property_1 = driver.find_element(By.NAME, max_property_1_name)
    max_property_1.clear()
    max_property_1.send_keys(high)
    ########################

    ### Select category as metal ###
    # metal_category_id = "ctl00_ContentMain_ucMatGroupTree_LODCS1_msTreeViewt3"
    # metal_link = driver.find_element(By.ID, metal_category_id)
    # metal_link.click()
    ################################

    ### Click submit ###
    submit_button_id = "ctl00_ContentMain_btnSubmit"
    submit_button = driver.find_element(By.ID, submit_button_id)
    submit_button.click()
    ####################

    ### Result counts ###
    result_counts_id = "ctl00_ContentMain_UcSearchResults1_lblResultCount2"
    result_counts = int(driver.find_element(By.ID, result_counts_id).text)
    print(f"Number of results: {result_counts}")
    #####################

    materials_data = []  # List to store material data

    while result_counts >= 0:

        table_id = "tblResults"
        table = driver.find_element(By.ID, table_id)
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Iterate through the table rows
        for row in tqdm(rows[::], desc="Collecting materials..."):
            try:
                # Find the link in each row
                link = row.find_element(By.TAG_NAME, "a").get_attribute("href")
                material = row.find_element(By.TAG_NAME, "a").text
                
                # Open the link in a new tab
                driver.execute_script(f"window.open('{link}', '_blank');")
                
                # Switch to the new tab
                driver.switch_to.window(driver.window_handles[-1])

                try:
                    # Locate the specific table
                    xpath = '//*[@id="ctl00_ContentMain_ucDataSheet1_pnlMaterialData"]/table[2]'
                    property_table = driver.find_element(By.XPATH, xpath)
                    property_table_html = property_table.get_attribute("outerHTML")

                    material_data = extract_material_data(property_table_html, link, material)
                    materials_data.append(material_data)
                    ############

                    num_materials -= 1

                    if num_materials == 0:
                        with open(f"matweb_data/materials_data_{timestamp}.json", "w") as json_file:
                            json.dump(materials_data, json_file, indent=4)
                        driver.quit()
                        return generate_summary(materials_data)
                    
                    # Close the new tab
                    driver.close()
                    
                    # Switch back to the original tab
                    driver.switch_to.window(driver.window_handles[0])

                except:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    continue
            
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        result_counts -= 50
        next_page_id = "ctl00_ContentMain_UcSearchResults1_lnkNextPage2"
        next_page_link = driver.find_element(By.ID, next_page_id)
        next_page_link.click()

    # Close the browser when done
    driver.quit()
    with open(f"matweb_data/materials_data_{timestamp}.json", "w") as json_file:
        json.dump(materials_data, json_file, indent=4)
    return generate_summary(materials_data)
###################