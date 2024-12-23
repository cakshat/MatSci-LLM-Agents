import os
import io
from ase.io import write, read
from ase.build import bulk
import numpy as np
from ase import Atoms
from ase.io import read
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution, 
    Stationary, 
    ZeroRotation
)
from ase.thermochemistry import IdealGasThermo
from ase.constraints import StrainFilter, UnitCellFilter
from ase.optimize import BFGS
from ase import units
from ase.io.trajectory import Trajectory
from ase.visualize import view
from ase.lattice.cubic import (
    SimpleCubic, 
    FaceCenteredCubic, 
    BodyCenteredCubic, 
    Diamond
)
from ase.lattice.tetragonal import (
    SimpleTetragonal, 
    CenteredTetragonal
)
from ase.lattice.orthorhombic import (
    SimpleOrthorhombic, 
    BaseCenteredOrthorhombic, 
    FaceCenteredOrthorhombic, 
    BodyCenteredOrthorhombic
)
from ase.lattice.hexagonal import Hexagonal
from langchain.agents import tool
from ase.calculators.eam import EAM
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet # Const. NVE
from ase.md.langevin import Langevin # Const. NVT
from ase.md.npt import NPT # Const. NPT
from ase.md.nvtberendsen import NVTBerendsen

from datetime import datetime
from ase.md import MDLogger
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from typing import List, Union, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

def initialize_atoms(
    cif_file: str=None, 
    name: str = "NaCl",
    crystalstructure: str = None,
    a: float = None,
    b: float = None,
    c: float = None,
    alpha: float = None,
    covera: float = None,
    u: float = None,
    orthorhombic: bool = False,
    cubic: bool = False, 
):
    """
Initialize an atomic structure using ASE based on parameters or a CIF file.

Parameters:
- cif_file (str): Path to a CIF file if provided.
- name (str): Chemical symbol(s) (e.g., 'MgO', 'NaCl').
- crystalstructure (str): Crystal type (sc, fcc, bcc, hcp, etc.).
- a, b, c (float): Lattice constants; `b` is interpreted as `c` if `c` is omitted.
- alpha (float): Angle for rhombohedral lattice (degrees).
- covera (float): c/a ratio for hcp (default: √8/3).
- u (float): Internal coordinate for Wurtzite.
- orthorhombic (bool): Build orthorhombic unit cell instead of primitive.
- cubic (bool): Build cubic unit cell if possible.

Returns:
- atoms (Atoms): ASE Atoms object.
    """
    
    if not cif_file:
        atoms = bulk(
            name=name,
            crystalstructure=crystalstructure,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            covera=covera,
            u=u,
            orthorhombic=orthorhombic,
            cubic=cubic
        )
    else:
        atoms = read(cif_file)
    
    atoms *= 3
    atoms.pbc = True
    return atoms

def define_interatomic_interactions(
    atoms: Atoms = None,
    potential_type: str="emt", 
    potential_file: str=None, 
    lj_epsilon: float=0.0103, 
    lj_sigma: float=3.4
):
    """
Define interatomic interactions for MD simulations, with fallback options.

Parameters:
- atoms (Atoms): ASE Atoms object
- potential_type (str): Type of potential to use ("eam", "lj", "sw", "emt").
    "eam": Embedded Atom Model, often used for metallic systems and alloys.
    "lj": Lennard-Jones potential, a basic model for interactions in simple systems like noble gases.
    "emt": Effective Medium Theory, designed for metals, especially transition metals.
- potential_file (str): Path to the potential file, required for EAM potential types.
- lj_epsilon (float): epsilon value if potential type is Leonard-Jones potential.
- lj_sigma (float) sigma value if potential type is Leonard-Jones potential.

Returns:
- atoms (Atoms): ASE Atoms object attached with the calculator based on the interatomic interactions.
    """
    try:
        if potential_type == "eam":
            # EAM potential, requires a potential file
            if not potential_file:
                print("Warning: EAM potential selected but no potential file provided. Falling back to EMT.")
                calculator = EMT()
            else:
                calculator = EAM(potential=potential_file)
                
        elif potential_type == "lj":
            # Lennard-Jones potential, simple systems
            epsilon = lj_epsilon 
            sigma = lj_sigma      
            calculator = LennardJones(epsilon=epsilon, sigma=sigma)

        elif potential_type == "emt":
            # EMT potential for metals
            calculator = EMT()
            
        else:
            print(f"Warning: Unsupported potential type '{potential_type}'. Falling back to EMT.")
            calculator = EMT()
    
    except Exception as e:
        print(f"Error initializing '{potential_type}' potential: {e}. Falling back to EMT.")
        calculator = EMT()

    atoms.calc = calculator
    return atoms

def run_NVE_dynamics(
    atoms: Atoms = None,
    time_step: int=1,
    temperature: float=300,
    num_md_steps: int = 10000,
    num_interval: int = 1000
):
    """
Run an NVE (constant energy) molecular dynamics simulation using the Velocity Verlet integrator.
Parameters:
- atoms (Atoms): ASE Atoms object
- time_step (int): Time step in femtoseconds (default=1).
- temperature (float): Initial temperature in Kelvin (default=300).
- num_md_steps (int): Total MD steps to run (default=10000).
- num_interval (int): Logging interval (default=1000).

Behavior:
- Initializes atomic velocities at specified temperature using Maxwell-Boltzmann distribution.
- Removes net momentum and angular momentum.
- Logs energy (total, kinetic, potential) and temperature, saving trajectory and plots.

Outputs:
- Log file, trajectory file, and energy/temperature plots stored in a timestamped folder.

Notes:
- Results are saved to disk; no return value.
- Example use: `run_NVE_dynamics(atoms, time_step=2, temperature=500, num_md_steps=5000)`.
"""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove net momentum
    ZeroRotation(atoms)  # Remove net angular momentum

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    folder_path = 'md_simulation_results/MD_' + datetime_str
    os.makedirs(folder_path)

    output_filename = folder_path + "/nve"
    log_filename = output_filename + ".log"
    print("log_filename = ",log_filename)
    traj_filename = output_filename + ".traj"
    print("traj_filename = ",traj_filename)

    dyn = VelocityVerlet(
        atoms,
        time_step * units.fs,
        trajectory = traj_filename,
        loginterval=num_interval
    )
    def print_dyn():
        imd = dyn.get_number_of_steps()
        time_md = time_step*imd
        etot  = atoms.get_total_energy()
        ekin  = atoms.get_kinetic_energy()
        epot  = atoms.get_potential_energy()
        temp_K = atoms.get_temperature()
        print(f"   {imd: >3}     {etot:.9f}     {ekin:.9f}    {epot:.9f}   {temp_K:.2f}")

    dyn.attach(print_dyn, interval=num_interval)
    dyn.attach(MDLogger(dyn, atoms, log_filename, header=True, stress=False,peratom=False, mode="w"), interval=num_interval)
    print(f"\n    imd     Etot(eV)    Ekin(eV)    Epot(eV)    T(K)")
    dyn.run(num_md_steps)

    df = pd.read_csv(log_filename, delim_whitespace=True)
    df.head()
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Tot E (eV)')
    ax1.set_ylim([0, 2*df["Etot[eV]"][0]])
    ax1.plot(df["Time[ps]"], df["Etot[eV]"], color="blue")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_xticklabels([])
    ax2.set_ylabel('P.E. (eV)')
    ax2.plot(df["Time[ps]"], df["Epot[eV]"], color="green")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_xticklabels([])
    ax3.set_ylabel('K.E. (eV)')
    ax3.plot(df["Time[ps]"], df["Ekin[eV]"], color="orange")

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_xlabel('time (ps)')
    ax4.set_ylabel('Temp. (K)')
    ax4.plot(df["Time[ps]"], df["T[K]"], color="red")

    fig.suptitle("Time evolution of total, potential, and kinetic energies, and temperature.")

    plt.savefig(f"{folder_path}/energy_plots.png")
    plt.show()

def run_NVT_dynamics(
    atoms: Atoms = None,
    time_step: int=1,
    temperature: float=300,
    taut: float = 1.0,
    num_md_steps: int = 10000,
    num_interval: int = 1000
):
    """
Run an NVT (constant temperature and volume) MD simulation using the Berendsen thermostat.
Parameters:
- atoms (Atoms): ASE Atoms object
- time_step (int): Time step in femtoseconds (default=1).
- temperature (float): Target temperature in Kelvin (default=300).
- taut (float): Berendsen thermostat time constant in fs (default=1.0).
- num_md_steps (int): Total MD steps to run (default=10000).
- num_interval (int): Logging interval (default=1000).

Behavior:
- Initializes atomic velocities at specified temperature using Maxwell-Boltzmann distribution.
- Removes net momentum and angular momentum.
- Logs energy (total, kinetic, potential), temperature, and stress tensor.
- Saves trajectory and energy/temperature plots.

Outputs:
- Log file, trajectory file, and energy/temperature plots in a timestamped folder.

Example:
run_NVT_dynamics(atoms, time_step=2, temperature=500, taut=0.5, num_md_steps=5000)
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove net momentum
    ZeroRotation(atoms)  # Remove net angular momentum

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    folder_path = 'md_simulation_results/MD_' + datetime_str
    os.makedirs(folder_path)

    output_filename = folder_path + "/nvt"
    log_filename = output_filename + ".log"
    print("log_filename = ",log_filename)
    traj_filename = output_filename + ".traj"
    print("traj_filename = ",traj_filename)

    dyn = NVTBerendsen(
        atoms, 
        time_step*units.fs, 
        temperature_K = temperature, 
        taut=taut*units.fs, 
        loginterval=num_interval, 
        trajectory=traj_filename
    )
    def print_dyn():
        imd = dyn.get_number_of_steps()
        etot  = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True)/units.GPa
        stress_ave = (stress[0]+stress[1]+stress[2])/3.0
        elapsed_time = perf_counter() - start_time
        print(f"  {imd: >3}   {etot:.3f}    {temp_K:.2f}    {stress_ave:.2f}  {stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  {stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")

    dyn.attach(print_dyn, interval=num_interval)
    dyn.attach(MDLogger(dyn, atoms, output_filename+".log", header=True, stress=True, peratom=True, mode="a"), interval=num_interval)
    start_time = perf_counter()
    print(f"    imd     Etot(eV)    T(K)    stress(mean,xx,yy,zz,yz,xz,xy)(GPa)  elapsed_time(sec)")
    dyn.run(num_md_steps)

    df = pd.read_csv(
        log_filename,
        delim_whitespace=True,
        names=["Time[ps]", "Etot/N[eV]", "Epot/N[eV]", "Ekin/N[eV]", "T[K]",
            "stressxx", "stressyy", "stresszz", "stressyz", "stressxz", "stressxy"],
        skiprows=1,
        header=None,
    )
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Tot E (eV)')
    ax1.plot(df["Time[ps]"], df["Etot/N[eV]"], color="blue",alpha=0.5)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_xticklabels([])
    ax2.set_ylabel('P.E. (eV)')
    ax2.plot(df["Time[ps]"], df["Epot/N[eV]"], color="green",alpha=0.5)

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_xticklabels([])
    ax3.set_ylabel('K.E. (eV)')
    ax3.set_ylim([0.0, 0.2])
    ax3.plot(df["Time[ps]"], df["Ekin/N[eV]"], color="orange",alpha=0.5)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_xlabel('time (ps)')
    ax4.set_ylabel('Temp. (K)')
    ax4.plot(df["Time[ps]"], df["T[K]"], color="red",alpha=0.5)
    ax4.set_ylim([temperature*0.5, temperature*2])

    fig.suptitle("Time evolution of total, potential, and kinetic energies, and temperature in NVT.", y=0)

    plt.savefig(f"{folder_path}/energy_plots.png")  
    plt.show()

def run_NPT_dynamics(
    atoms: Atoms = None,
    time_step: int=1,
    temperature: float=300,
    num_md_steps: int = 10000,
    num_interval: int = 1000,
    sigma   = 1.0,     # External pressure in bar
    ttime   = 20.0,    # Time constant in fs
    pfactor = 2e6     # Barostat parameter in GPa
):
    """
Run an NPT (constant pressure, temperature, and volume) MD simulation using the Berendsen barostat.
Parameters:
- atoms (Atoms): ASE Atoms object
- time_step (int): Time step in femtoseconds (default=1).
- temperature (float): Target temperature in Kelvin (default=300).
- num_md_steps (int): Total MD steps (default=10000).
- num_interval (int): Logging interval (default=1000).
- sigma (float): External pressure in bar (default=1.0).
- ttime (float): Barostat time constant in fs (default=20.0).
- pfactor (float): Barostat parameter in GPa (default=2e6).

Behavior:
- Initializes velocities at the target temperature using Maxwell-Boltzmann distribution.
- Removes net momentum and angular momentum.
- Logs energy, temperature, and stress tensor components.
- Saves trajectory and energy/temperature plots.

Example:
run_NPT_dynamics(atoms, time_step=2, temperature=500, sigma=1.0, ttime=20.0, pfactor=2e6)
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)  # Remove net momentum
    ZeroRotation(atoms)  # Remove net angular momentum

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    folder_path = 'md_simulation_results/MD_' + datetime_str
    os.makedirs(folder_path)

    output_filename = folder_path + "/npt"
    log_filename = output_filename + ".log"
    print("log_filename = ",log_filename)
    traj_filename = output_filename + ".traj"
    print("traj_filename = ",traj_filename)

    dyn = NPT(atoms,
        time_step*units.fs,
        temperature_K = temperature,
        externalstress = sigma*units.bar,
        ttime = ttime*units.fs,
        pfactor = pfactor*units.GPa*(units.fs**2),
        logfile = log_filename,
        trajectory = traj_filename,
        loginterval=num_interval
    )

    def print_dyn():
        imd = dyn.get_number_of_steps()
        etot  = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True)/units.GPa
        stress_ave = (stress[0]+stress[1]+stress[2])/3.0
        elapsed_time = perf_counter() - start_time
        print(f"  {imd: >3}   {etot:.3f}    {temp_K:.2f}    {stress_ave:.2f}  {stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  {stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")

    print_interval = num_interval
    dyn.attach(print_dyn, interval=print_interval)
    dyn.attach(MDLogger(dyn, atoms, output_filename+".log", header=True, stress=True, peratom=True, mode="a"), interval=num_interval)

    start_time = perf_counter()
    print(f"    imd     Etot(eV)    T(K)    stress(mean,xx,yy,zz,yz,xz,xy)(GPa)  elapsed_time(sec)")
    dyn.run(num_md_steps)

    df = pd.read_csv(
        log_filename,
        delim_whitespace=True,
        names=["Time[ps]", "Etot/N[eV]", "Epot/N[eV]", "Ekin/N[eV]", "T[K]",
            "stressxx", "stressyy", "stresszz", "stressyz", "stressxz", "stressxy"],
        skiprows=3,
        header=None,
    )
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Tot E (eV)')
    ax1.plot(df["Time[ps]"], df["Etot/N[eV]"], color="blue",alpha=0.5)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.set_xticklabels([])
    ax2.set_ylabel('P.E. (eV)')
    ax2.plot(df["Time[ps]"], df["Epot/N[eV]"], color="green",alpha=0.5)

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_xticklabels([])
    ax3.set_ylabel('K.E. (eV)')
    ax3.plot(df["Time[ps]"], df["Ekin/N[eV]"], color="orange",alpha=0.5)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_xlabel('time (ps)')
    ax4.set_ylabel('Temp. (K)')
    ax4.plot(df["Time[ps]"], df["T[K]"], color="red",alpha=0.5)
    ax4.set_ylim([temperature*0.5, temperature*2])

    fig.suptitle("Time evolution of total, potential, and kinetic energies, and temperature in NPT.", y=0)

    plt.savefig(f"{folder_path}/energy_plots.png")  
    plt.show()


class MDSimulationNVEInput(BaseModel):
    ### Initializing atoms ###
    cif_file: Optional[str] = Field(None, description="Path to a CIF file if provided.")
    name: str = Field("NaCl", description="Chemical symbol(s) (e.g., 'MgO', 'NaCl').")
    crystalstructure: Optional[str] = Field(None, description="Crystal type (sc, fcc, bcc, hcp, etc.).")
    a: Optional[float] = Field(None, description="Lattice constant a.")
    b: Optional[float] = Field(None, description="Lattice constant b (interpreted as c if c is omitted).")
    c: Optional[float] = Field(None, description="Lattice constant c.")
    alpha: Optional[float] = Field(None, description="Angle for rhombohedral lattice (degrees).")
    covera: Optional[float] = Field(None, description="c/a ratio for hcp (default: √8/3).")
    u: Optional[float] = Field(None, description="Internal coordinate for Wurtzite.")
    orthorhombic: bool = Field(False, description="Build orthorhombic unit cell instead of primitive.")
    cubic: bool = Field(True, description="Build cubic unit cell if possible. Set as True if crystal structure is of cubic type")
    ##########################

    ### Interatomic potentials ###
    potential_type: str = Field("emt", description="Type of potential to use ('eam', 'lj', 'sw', 'emt').")
    potential_file: Optional[str] = Field(None, description="Path to the potential file (required for EAM potential types).")
    lj_epsilon: float = Field(0.0103, description="Epsilon value if potential type is Lennard-Jones.")
    lj_sigma: float = Field(3.4, description="Sigma value if potential type is Lennard-Jones.")
    ##############################

    ### NVE Inputs ###
    time_step: int = Field(1, description="Time step for the simulation in femtoseconds (fs).")
    temperature: float = Field(300, description="Initial temperature for the simulation in Kelvin (K).")
    num_md_steps: int = Field(10000, description="Total number of molecular dynamics steps to run.")
    num_interval: int = Field(1000, description="Interval at which data is logged and printed during the simulation.")
    ##################

class MDSimulationNVTInput(BaseModel):
    ### Initializing atoms ###
    cif_file: Optional[str] = Field(None, description="Path to a CIF file if provided.")
    name: str = Field("NaCl", description="Chemical symbol(s) (e.g., 'MgO', 'NaCl').")
    crystalstructure: Optional[str] = Field(None, description="Crystal type (sc, fcc, bcc, hcp, etc.).")
    a: Optional[float] = Field(None, description="Lattice constant a.")
    b: Optional[float] = Field(None, description="Lattice constant b (interpreted as c if c is omitted).")
    c: Optional[float] = Field(None, description="Lattice constant c.")
    alpha: Optional[float] = Field(None, description="Angle for rhombohedral lattice (degrees).")
    covera: Optional[float] = Field(None, description="c/a ratio for hcp (default: √8/3).")
    u: Optional[float] = Field(None, description="Internal coordinate for Wurtzite.")
    orthorhombic: bool = Field(False, description="Build orthorhombic unit cell instead of primitive.")
    cubic: bool = Field(True, description="Build cubic unit cell if possible. Set as True if crystal structure is of cubic type")
    ##########################

    ### Interatomic potentials ###
    potential_type: str = Field("emt", description="Type of potential to use ('eam', 'lj', 'sw', 'emt').")
    potential_file: Optional[str] = Field(None, description="Path to the potential file (required for EAM potential types).")
    lj_epsilon: float = Field(0.0103, description="Epsilon value if potential type is Lennard-Jones.")
    lj_sigma: float = Field(3.4, description="Sigma value if potential type is Lennard-Jones.")
    ##############################

    ### NVT Inputs ###
    time_step: int = Field(1, description="Time step for the simulation in femtoseconds (fs).")
    temperature: float = Field(300, description="Target temperature for the simulation in Kelvin (K).")
    taut: float = Field(1.0, description="Time constant for the Berendsen thermostat in femtoseconds (fs).")
    num_md_steps: int = Field(10000, description="Total number of molecular dynamics steps to run.")
    num_interval: int = Field(1000, description="Interval at which data is logged and printed during the simulation.")
    ##################

class MDSimulationNPTInput(BaseModel):
    ### Initializing atoms ###
    cif_file: Optional[str] = Field(None, description="Path to a CIF file if provided.")
    name: str = Field("NaCl", description="Chemical symbol(s) (e.g., 'MgO', 'NaCl').")
    crystalstructure: Optional[str] = Field(None, description="Crystal type (sc, fcc, bcc, hcp, etc.).")
    a: Optional[float] = Field(None, description="Lattice constant a.")
    b: Optional[float] = Field(None, description="Lattice constant b (interpreted as c if c is omitted).")
    c: Optional[float] = Field(None, description="Lattice constant c.")
    alpha: Optional[float] = Field(None, description="Angle for rhombohedral lattice (degrees).")
    covera: Optional[float] = Field(None, description="c/a ratio for hcp (default: √8/3).")
    u: Optional[float] = Field(None, description="Internal coordinate for Wurtzite.")
    orthorhombic: bool = Field(False, description="Build orthorhombic unit cell instead of primitive.")
    cubic: bool = Field(True, description="Build cubic unit cell if possible. Set as True if crystal structure is of cubic type")
    ##########################

    ### Interatomic potentials ###
    potential_type: str = Field("emt", description="Type of potential to use ('eam', 'lj', 'sw', 'emt').")
    potential_file: Optional[str] = Field(None, description="Path to the potential file (required for EAM potential types).")
    lj_epsilon: float = Field(0.0103, description="Epsilon value if potential type is Lennard-Jones.")
    lj_sigma: float = Field(3.4, description="Sigma value if potential type is Lennard-Jones.")
    ##############################

    ### NPT Inputs ###
    time_step: int = Field(1, description="Time step for the simulation in femtoseconds (fs).")
    temperature: float = Field(300, description="Target temperature for the simulation in Kelvin (K).")
    num_md_steps: int = Field(10000, description="Total number of molecular dynamics steps to run.")
    num_interval: int = Field(1000, description="Interval at which data is logged and printed during the simulation.")
    sigma: float = Field(1.0, description="External pressure in bar applied during the simulation.")
    ttime: float = Field(20.0, description="Time constant for the Berendsen barostat in femtoseconds (fs).")
    pfactor: float = Field(2e6, description="Barostat parameter in GPa used to control pressure.")
    ##################

class MDSimulationNVETool(BaseTool):
    name: str = "MDSimulationNVETool"
    description: str = """
    Tool to run an NVE molecular dynamics simulation with various input options.
    """
    args_schema: Type[BaseModel] = MDSimulationNVEInput

    def _run(self, **kwargs):
        """
        This method runs the NVE simulation using the unpacked parameters passed through kwargs.
        """
        cif_file = kwargs.get("cif_file", None)
        name = kwargs.get("name", "NaCl")
        crystalstructure = kwargs.get("crystalstructure", None)
        a = kwargs.get("a", None)
        b = kwargs.get("b", None)
        c = kwargs.get("c", None)
        alpha = kwargs.get("alpha", None)
        covera = kwargs.get("covera", None)
        u = kwargs.get("u", None)
        orthorhombic = kwargs.get("orthorhombic", False)
        cubic = kwargs.get("cubic", False)

        potential_type = kwargs.get("potential_type", "emt")
        potential_file = kwargs.get("potential_file", None)
        lj_epsilon = kwargs.get("lj_epsilon", 0.0103)
        lj_sigma = kwargs.get("lj_sigma", 3.4)

        time_step = kwargs.get("time_step", 1)
        temperature = kwargs.get("temperature", 300)
        num_md_steps = kwargs.get("num_md_steps", 10000)
        num_interval = kwargs.get("num_interval", 1000)

        atoms = initialize_atoms(
            cif_file,
            name,
            crystalstructure,
            a,
            b,
            c,
            alpha,
            covera,
            u,
            orthorhombic,
            cubic
        )
        
        atoms = define_interatomic_interactions(
            atoms,
            potential_type,
            potential_file,
            lj_epsilon,
            lj_sigma
        )
        
        run_NVE_dynamics(
            atoms,
            time_step,
            temperature,
            num_md_steps,
            num_interval
        )

class MDSimulationNVTTool(BaseTool):
    name: str = "MDSImulationNVTTool"
    description: str = """
    Tool to run an NVT molecular dynamics simulation with various input options.
"""
    args_schema: Type[BaseModel] = MDSimulationNVTInput

    def _run(self, **kwargs):
        """
        This method runs the NVT simulation using the unpacked parameters passed through kwargs.
        """
        cif_file = kwargs.get("cif_file", None)
        name = kwargs.get("name", "NaCl")
        crystalstructure = kwargs.get("crystalstructure", None)
        a = kwargs.get("a", None)
        b = kwargs.get("b", None)
        c = kwargs.get("c", None)
        alpha = kwargs.get("alpha", None)
        covera = kwargs.get("covera", None)
        u = kwargs.get("u", None)
        orthorhombic = kwargs.get("orthorhombic", False)
        cubic = kwargs.get("cubic", False)

        potential_type = kwargs.get("potential_type", "emt")
        potential_file = kwargs.get("potential_file", None)
        lj_epsilon = kwargs.get("lj_epsilon", 0.0103)
        lj_sigma = kwargs.get("lj_sigma", 3.4)

        time_step = kwargs.get("time_step", 1)
        temperature = kwargs.get("temperature", 300)
        taut = kwargs.get("taut", 1)
        num_md_steps = kwargs.get("num_md_steps", 10000)
        num_interval = kwargs.get("num_interval", 1000)

        atoms = initialize_atoms(
            cif_file,
            name,
            crystalstructure,
            a,
            b,
            c,
            alpha,
            covera,
            u,
            orthorhombic,
            cubic
        )
        
        atoms = define_interatomic_interactions(
            atoms,
            potential_type,
            potential_file,
            lj_epsilon,
            lj_sigma
        )

        run_NVT_dynamics(
            atoms,
            time_step,
            temperature, taut,
            num_md_steps,
            num_interval
        )

class MDSimulationNPTTool(BaseTool):
    name: str = "MDSimulationNPTTool"
    description: str = """
    Tool to run an NPT molecular dynamics simulation with various input options.
"""
    args_schema: Type[BaseModel] = MDSimulationNPTInput

    def _run(self, **kwargs):
        """
        This method runs the NPT simulation using the unpacked parameters passed through kwargs.
        """
        cif_file = kwargs.get("cif_file", None)
        name = kwargs.get("name", "NaCl")
        crystalstructure = kwargs.get("crystalstructure", None)
        a = kwargs.get("a", None)
        b = kwargs.get("b", None)
        c = kwargs.get("c", None)
        alpha = kwargs.get("alpha", None)
        covera = kwargs.get("covera", None)
        u = kwargs.get("u", None)
        orthorhombic = kwargs.get("orthorhombic", False)
        cubic = kwargs.get("cubic", False)

        potential_type = kwargs.get("potential_type", "emt")
        potential_file = kwargs.get("potential_file", None)
        lj_epsilon = kwargs.get("lj_epsilon", 0.0103)
        lj_sigma = kwargs.get("lj_sigma", 3.4)

        time_step = kwargs.get("time_step", 1)
        temperature = kwargs.get("temperature", 300)
        num_md_steps = kwargs.get("num_md_steps", 10000)
        num_interval = kwargs.get("num_interval", 1000)
        sigma = kwargs.get("sigma", 1.0)
        ttime = kwargs.get("ttime", 20.0)
        pfactor = kwargs.get("pfactor", 2e6)

        atoms = initialize_atoms(
            cif_file,
            name,
            crystalstructure,
            a,
            b,
            c,
            alpha,
            covera,
            u,
            orthorhombic,
            cubic
        )

        atoms = define_interatomic_interactions(
            atoms,
            potential_type,
            potential_file,
            lj_epsilon,
            lj_sigma
        )

        run_NPT_dynamics(
            atoms,
            time_step,
            temperature,
            num_md_steps,
            num_interval,
            sigma,
            ttime,
            pfactor
        )