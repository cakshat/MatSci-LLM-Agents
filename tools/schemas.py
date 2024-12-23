from typing import List, Union, Optional
from pydantic import BaseModel, Field
from emmet.core.thermo import ThermoType
from pymatgen.analysis.magnetism import Ordering
from emmet.core.synthesis import (
    OperationTypeEnum,
    SynthesisTypeEnum,
)
from emmet.core.electronic_structure import (
    BSPathType,
    DOSProjectionType
)
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.core import OrbitalType, Spin

class MPSurfacePropertiesInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])"
    )
    has_reconstructed: Optional[bool] = Field(
        None, 
        description="Whether the entry has any reconstructed surfaces."
    )
    shape_factor: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum shape factor values to consider.",
        min_items=2,
        max_items=2
    )
    surface_energy_anisotropy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum surface energy anisotropy values to consider.",
        min_items=2,
        max_items=2
    )
    weighted_surface_energy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum weighted surface energy in J/m² to consider.",
        min_items=2,
        max_items=2
    )
    weighted_work_function: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum weighted work function in eV to consider.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in SurfacePropDoc to return data for. Default is material_id only if all_fields is False."
    )

class MPPhononInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="A single Material ID string or list of strings (e.g., 'mp-149', ['mp-149', 'mp-13'])."
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in PhononBSDOSDoc to return data for. "
            "Defaults to material_id, last_updated, and formula_pretty if all_fields is False."
        )
    )

class MPElasticityInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])."
    )
    elastic_anisotropy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value to consider for the elastic anisotropy.",
        min_items=2,
        max_items=2
    )
    g_voigt: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Voigt average of the shear modulus.",
        min_items=2,
        max_items=2
    )
    g_reuss: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Reuss average of the shear modulus.",
        min_items=2,
        max_items=2
    )
    g_vrh: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Voigt-Reuss-Hill average of the shear modulus.",
        min_items=2,
        max_items=2
    )
    k_voigt: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Voigt average of the bulk modulus.",
        min_items=2,
        max_items=2
    )
    k_reuss: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Reuss average of the bulk modulus.",
        min_items=2,
        max_items=2
    )
    k_vrh: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value in GPa to consider for the Voigt-Reuss-Hill average of the bulk modulus.",
        min_items=2,
        max_items=2
    )
    poisson_ratio: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value to consider for Poisson's ratio.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in ElasticityDoc to return data for. Default is material_id and prett-formula if all_fields is False."
    )
    
class MPThermoInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])."
    )
    chemsys: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A chemical system or list of chemical systems (e.g., Li-Fe-O, Si-*, [Si-O, Li-Fe-P])."
    )
    energy_above_hull: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum energy above the hull in eV/atom to consider.",
        min_items=2,
        max_items=2
    )
    equilibrium_reaction_energy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum equilibrium reaction energy in eV/atom to consider.",
        min_items=2,
        max_items=2
    )
    formation_energy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum formation energy in eV/atom to consider.",
        min_items=2,
        max_items=2
    )
    formula: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A formula including anonymized formula or wild cards (e.g., Fe2O3, ABO3, Si*). "
                    "A list of chemical formulas can also be passed (e.g., [Fe2O3, ABO3])."
    )
    is_stable: Optional[bool] = Field(
        None, 
        description="Whether the material is stable."
    )
    num_elements: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum number of elements in the material to consider.",
        min_items=2,
        max_items=2
    )
    thermo_ids: Optional[List[str]] = Field(
        None, 
        description="List of thermo IDs to return data for. This is a combination of the Materials Project ID "
                    "and thermo type (e.g., mp-149_GGA_GGA+U)."
    )
    thermo_types: Optional[List[Union[ThermoType, str]]] = Field(
        None, 
        description="List of thermo types to return data for (e.g., ThermoType.GGA_GGA_U)."
    )
    total_energy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum corrected total energy in eV/atom to consider.",
        min_items=2,
        max_items=2
    )
    uncorrected_energy: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum uncorrected total energy in eV/atom to consider.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in ThermoDoc to return data for. Default is material_id and last_updated "
                    "if all_fields is False."
    )

class MPDielectricInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])."
    )
    e_total: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum total dielectric constant to consider.",
        min_items=2,
        max_items=2
    )
    e_ionic: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum ionic dielectric constant to consider.",
        min_items=2,
        max_items=2
    )
    e_electronic: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum electronic dielectric constant to consider.",
        min_items=2,
        max_items=2
    )
    n: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum refractive index to consider.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in DielectricDoc to return data for. Default is material_id and last_updated if all_fields is False."
    )

class MPPiezoelectricInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])."
    )
    piezoelectric_modulus: Optional[List[Union[float, None]]] = Field(
        None, 
        description="Minimum and maximum value of the piezoelectric modulus in C/m² to consider.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in PiezoDoc to return data for. Default is material_id and last_updated if all_fields is False."
    )

class MPMagnetismInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None, 
        description="A single Material ID string or list of strings (e.g., mp-149, [mp-149, mp-13])."
    )
    num_magnetic_sites: Optional[List[Union[int, None]]] = Field(
        None, 
        description="Minimum and maximum number of magnetic sites to consider.",
        min_items=2,
        max_items=2
    )
    num_unique_magnetic_sites: Optional[List[Union[int, None]]] = Field(
        None, 
        description="Minimum and maximum number of unique magnetic sites to consider.",
        min_items=2,
        max_items=2
    )
    ordering: Optional[Ordering] = Field(
        None, 
        description="The magnetic ordering of the material."
    )
    total_magnetization: Optional[List[Union[int, None]]] = Field(
        None, 
        description="Minimum and maximum total magnetization values to consider.",
        min_items=2,
        max_items=2
    )
    total_magnetization_normalized_vol: Optional[List[Union[int, None]]] = Field(
        None, 
        description="Minimum and maximum total magnetization values normalized by volume to consider.",
        min_items=2,
        max_items=2
    )
    total_magnetization_normalized_formula_units: Optional[List[Union[int, None]]] = Field(
        None, 
        description="Minimum and maximum total magnetization values normalized by formula units to consider.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None, 
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000, 
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True, 
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None, 
        description="List of fields in MagnetismDoc to return data for. Default is material_id and last_updated if all_fields is False."
    )

class MPSynthesisInput(BaseModel):
    keywords: Optional[List[str]] = Field(
        None,
        description="List of string keywords to search synthesis paragraph text with."
    )
    synthesis_type: Optional[List[SynthesisTypeEnum]] = Field(
        None,
        description="Type of synthesis to include, defined by SynthesisTypeEnum."
    )
    target_formula: Optional[str] = Field(
        None,
        description="Chemical formula of the target material."
    )
    precursor_formula: Optional[str] = Field(
        None,
        description="Chemical formula of the precursor material."
    )
    operations: Optional[List[OperationTypeEnum]] = Field(
        None,
        description="List of operations that syntheses must have, defined by OperationTypeEnum."
    )
    condition_heating_temperature_min: Optional[float] = Field(
        None,
        description="Minimal heating temperature in the synthesis process."
    )
    condition_heating_temperature_max: Optional[float] = Field(
        None,
        description="Maximal heating temperature in the synthesis process."
    )
    condition_heating_time_min: Optional[float] = Field(
        None,
        description="Minimal heating time in the synthesis process."
    )
    condition_heating_time_max: Optional[float] = Field(
        None,
        description="Maximal heating time in the synthesis process."
    )
    condition_heating_atmosphere: Optional[List[str]] = Field(
        None,
        description='Required heating atmosphere, such as "air", "argon".'
    )
    condition_mixing_device: Optional[List[str]] = Field(
        None,
        description='Required mixing device, such as "zirconia", "Al2O3".'
    )
    condition_mixing_media: Optional[List[str]] = Field(
        None,
        description='Required mixing media, such as "alcohol", "water".'
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        10,
        description="Number of data entries per chunk."
    )

class MPElectronicStructureInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="A single Material ID string or list of strings (e.g., 'mp-149' or ['mp-149', 'mp-13'])."
    )
    band_gap: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum band gap in eV to consider.",
        min_items=2,
        max_items=2
    )
    chemsys: Optional[Union[str, List[str]]] = Field(
        None,
        description="A chemical system or list of systems (e.g., 'Li-Fe-O', 'Si-*', or ['Si-O', 'Li-Fe-P'])."
    )
    efermi: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum Fermi energy in eV to consider.",
        min_items=2,
        max_items=2
    )
    elements: Optional[List[str]] = Field(
        None,
        description="A list of elements to include."
    )
    exclude_elements: Optional[List[str]] = Field(
        None,
        description="A list of elements to exclude."
    )
    formula: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A formula including anonymized formula or wildcards (e.g., 'Fe2O3', 'ABO3', 'Si*'). "
            "A list of formulas can also be passed (e.g., ['Fe2O3', 'ABO3'])."
        )
    )
    is_gap_direct: Optional[bool] = Field(
        None,
        description="Whether the material has a direct band gap."
    )
    is_metal: Optional[bool] = Field(
        None,
        description="Whether the material is considered a metal."
    )
    magnetic_ordering: Optional[Ordering] = Field(
        None,
        description="Magnetic ordering of the material."
    )
    num_elements: Optional[List[Union[int, None]]] = Field(
        None,
        description="Minimum and maximum number of elements in the composition.",
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in the electronic structure document to return data for. "
            "Defaults to 'material_id' and 'last_updated' if 'all_fields' is False."
        )
    )

class MPElectronicBandStructureInput(BaseModel):
    band_gap: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum band gap in eV to consider.",
        min_items=2,
        max_items=2
    )
    efermi: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum Fermi energy in eV to consider.",
        min_items=2,
        max_items=2
    )
    is_gap_direct: Optional[bool] = Field(
        None,
        description="Whether the material has a direct band gap."
    )
    is_metal: Optional[bool] = Field(
        None,
        description="Whether the material is considered a metal."
    )
    magnetic_ordering: Optional[Ordering] = Field(
        None,
        description="Magnetic ordering of the material."
    )
    path_type: BSPathType = Field(
        BSPathType.setyawan_curtarolo,
        description="k-path selection convention for the band structure."
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in the electronic structure document to return data for. "
            "Defaults to 'material_id' and 'last_updated' if 'all_fields' is False."
        )
    )

class MPElectronicDensityOfStatesInput(BaseModel):
    band_gap: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum band gap in eV to consider.",
        min_items=2,
        max_items=2
    )
    efermi: Optional[List[Union[float, None]]] = Field(
        None,
        description="Minimum and maximum Fermi energy in eV to consider.",
        min_items=2,
        max_items=2
    )
    element: Optional[Element] = Field(
        None,
        description="Element for element-projected DOS data."
    )
    magnetic_ordering: Optional[Ordering] = Field(
        None,
        description="Magnetic ordering of the material."
    )
    orbital: Optional[OrbitalType] = Field(
        None,
        description="Orbital for orbital-projected DOS data."
    )
    projection_type: DOSProjectionType = Field(
        DOSProjectionType.total,
        description="Projection type of DOS data. Default is the total DOS."
    )
    spin: Spin = Field(
        Spin.up,
        description="Spin channel of DOS data. Non-spin-polarized data is stored in `Spin.up`."
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in the electronic density of states document to return data for. "
            "Defaults to 'material_id' and 'last_updated' if 'all_fields' is False."
        )
    )

class MPOxidationStatesInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A single Material ID string or list of strings "
            "(e.g., mp-149, [mp-149, mp-13])."
        )
    )
    chemsys: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A chemical system or list of chemical systems "
            "(e.g., Li-Fe-O, Si-*, [Si-O, Li-Fe-P])."
        )
    )
    formula: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A formula including anonymized formula or wild cards "
            "(e.g., Fe2O3, ABO3, Si*). A list of chemical formulas can also be passed "
            "(e.g., [Fe2O3, ABO3])."
        )
    )
    possible_species: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A list of element symbols appended with oxidation states "
            "(e.g., [Cr2+, O2-])."
        )
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in OxidationStateDoc to return data for. "
            "Defaults to material_id, last_updated, and formula_pretty if all_fields is False."
        )
    )

class MPBondsInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Search for bonding data for the specified Material IDs. "
            "Accepts a single Material ID or a list of IDs (e.g., 'mp-149', ['mp-149', 'mp-13'])."
        )
    )
    coordination_envs: Optional[List[str]] = Field(
        None,
        description=(
            "List of coordination environments to consider (e.g., ['Mo-S(6)', 'S-Mo(3)'])."
        )
    )
    coordination_envs_anonymous: Optional[List[str]] = Field(
        None,
        description=(
            "List of anonymous coordination environments to consider "
            "(e.g., ['A-B(6)', 'A-B(3)'])."
        )
    )
    max_bond_length: Optional[List[Union[float, None]]] = Field(
        None,
        description=(
            "Minimum and maximum value for the maximum bond length in the structure to consider."
        ),
        min_items=2,
        max_items=2
    )
    mean_bond_length: Optional[List[Union[float, None]]] = Field(
        None,
        description=(
            "Minimum and maximum value for the mean bond length in the structure to consider."
        ),
        min_items=2,
        max_items=2
    )
    min_bond_length: Optional[List[Union[float, None]]] = Field(
        None,
        description=(
            "Minimum and maximum value for the minimum bond length in the structure to consider."
        ),
        min_items=2,
        max_items=2
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in DielectricDoc to return data for. "
            "Defaults to material_id and last_updated if all_fields is False."
        )
    )

class MPAbsorptionInput(BaseModel):
    material_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Search for optical absorption data associated with the specified Material IDs. "
            "Accepts a single Material ID or a list of IDs (e.g., 'mp-149', ['mp-149', 'mp-13'])."
        )
    )
    chemsys: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A chemical system or list of chemical systems to search for (e.g., 'Li-Fe-O', 'Si-*', "
            "['Si-O', 'Li-Fe-P'])."
        )
    )
    elements: Optional[List[str]] = Field(
        None,
        description=(
            "A list of elements to include in the search (e.g., ['Li', 'Fe'])."
        )
    )
    exclude_elements: Optional[List[str]] = Field(
        None,
        description=(
            "A list of elements to exclude from the search (e.g., ['O', 'N'])."
        )
    )
    formula: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A chemical formula or list of formulas including anonymized formulas or wildcards "
            "(e.g., 'Fe2O3', 'ABO3', 'Si*'). Accepts single formula or a list (e.g., ['Fe2O3', 'ABO3'])."
        )
    )
    num_chunks: Optional[int] = Field(
        None,
        description="Maximum number of chunks of data to yield. None will yield all possible."
    )
    chunk_size: int = Field(
        1000,
        description="Number of data entries per chunk."
    )
    all_fields: bool = Field(
        True,
        description="Whether to return all fields in the document. Defaults to True."
    )
    fields: Optional[List[str]] = Field(
        None,
        description=(
            "List of fields in AbsorptionDoc to return data for. "
            "Defaults to material_id and last_updated if all_fields is False."
        )
    )