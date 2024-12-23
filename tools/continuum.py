### Imports ###
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import tool
from typing import Optional, Type
from datetime import datetime
from tqdm import tqdm
###############

### Cellular Automata Solidification ###
class CASolid():

    def __init__(self, width=101, attach_prob=1, num_solid_states = 1):
      """
      This class is for Cellular Automata Solidification Model

      Attributes:
        width (int): The system will be square with side lengths of width
        attach_prob (float): Attachment probability of pCA
        num_solid_states (int): 1 for single phase solidification, >1 for recrystalization

      Methods:
        evolve(Nsteps): Evolves CA state Nsteps times
        update_neighbours(x, y): Updates the neighbouring states of neighbours of x,y
        transformed_area(): Total solid area
        plot(): Plots current system state 
      """

      self.width = width # width of the system; assume the system is square
      self.area_tot=(width-2)**2 # total system area: edge cells are not counted
      self.attach_prob = attach_prob # attachment probability for probabilistic CA
      self.ttot = 0 #a variable to record total time passed

      self.Nnuc = 1 #the number of nuclei in the initial state
      self.transformed_area_arr = [ ] #an array to store the total transformed area over time
      self.transformed_area_arr.append(self.Nnuc) # initial transformed area = number of nuclei
      self.num_solid_states = num_solid_states

      # Create an initial state that is all liquid = 0
      self.A0 = np.zeros((self.width, self.width))
      # Add nuclei to your initial state
      x_init = int(width/2)
      y_init = int(width/2)

      # Now we set the nucleus value depending whether we are simulating solidification or recrystalization
      if self.num_solid_states == 1: self.A0[y_init, x_init] = 1
      else: self.A0[y_init, x_init] = np.random.randint(1, self.num_solid_states+1)

      #Create a second array for storing the planned updated state
      #useful for avoiding overwriting current states
      self.A = self.A0.copy()

      #Create Neighborhood Array (width x width x Num Neighbors):
      # 0 : up; 1 : right; 2 : down; 3 : left
      # To store values of neighbors of each site
      self.B = np.zeros((self.width, self.width, 4))
      # Setting neighbour values as 1 for neighbouring cells of initial nucleus
      self.update_neighbours(x_init, y_init)
      self.num_nuclei = None

      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      self.output_dir = f"continuum_simulation_results/CA_solidification_{timestamp}"
      os.makedirs(self.output_dir, exist_ok=True)

    # a class that evolves the CA state for Nsteps
    # It should evolve the interior cells
    def place_nuclei(self, num_nuclei):
      """
      Places num_nuclei number of nuclei; each belonging to self.num_solid_states in the self.width x self.width neighbourhood.
      Position of placement and the phase of the nuclei are selected randomly.
      """
      self.num_nuclei = num_nuclei
      self.A = np.zeros((self.width, self.width))
      for _ in range(num_nuclei):
        x = random.randint(1, self.width-2)
        y = random.randint(1, self.width-2)
        phase = random.randint(1, self.num_solid_states)
        self.A[y, x] = phase
        self.update_neighbours(x, y)
        self.A0 = self.A.copy()
        self.A = self.A0.copy()
        self.transformed_area_arr = [num_nuclei]

    def evolve(self, Nsteps):
      
      """
      Simulates a cellular automata phase transformation process over a 2D grid for a given number of steps.
      The function iteratively updates the grid `A` based on the neighborhood conditions of grid `B`. 
      At each step, the function compares the current state of the grid `A` with its previous state, and updates the grid points based on the presence of neighboring phases in grid `B`.
      If a grid point in `A` is empty (0), it probabilistically adopts a phase based on the most common neighboring phase in `B`. 
      The function also keeps track of the total transformed area (points in `A` that are not zero) across all iterations.
  
      Parameters:
      -----------
      Nsteps : int
          The number of steps to run the simulation.
  
      Process:
      --------
      - For each iteration (step):
          - The function compares the current state of grid `A` to its previous state (`A_original`).
          - For each non-border point in grid `A`, if the value is 0 and the corresponding neighborhood in grid `B` contains phases (values greater than 0), the grid point in `A` may adopt the most frequent neighboring phase from `B` based on a given probability (`attach_prob`).
          - In case of ties (multiple phases having the same frequency), a random phase is selected.
          - After updating the grid point, the neighborhood is also updated to reflect the change.
      - The function tracks the total number of transformed points (non-zero values in `A`) after each step and appends it to `self.transformed_area_arr`.
  
      Returns:
      --------
      None
  
      Additional Notes for LLM Agent:
      -------------------------------
      - This function should be invoked in scenarios where a user asks for cellular automata phase transformation simulations, particularly when they request iterative updates based on neighborhood conditions.
      - Example prompts: "Simulate phase changes over time on a 2D grid." or "Run a phase transformation process for N steps."
      - Avoid running this function if the user query does not mention iterative phase transformations or simulations involving neighboring grid updates.
      """
      self.plot_phase_diagram(show_plot=False)
      for _ in range(Nsteps):
        A_original = self.A.copy() # At each iteration need to compare to previous state
        B_original = self.B.copy()
        for y in range(1, self.width-1):
          for x in range(1, self.width-1):
            if A_original[y, x] == 0 and np.sum(B_original[y,x]) > 0:
            # Updating with given probability
              if random.random() <= self.attach_prob:
                # Selecting the most common phase in neighbourhood
                counter = collections.Counter([phase for phase in B_original[y,x] if phase != 0])
                max_occurance = max(counter.values())
                phases_max_occurance = [phase for phase, count in counter.items() if count == max_occurance]
                # Random choice to break ties
                new_phase = random.choice(phases_max_occurance)
                self.A[y, x] = new_phase
                # We also need to update neighbouring points' neighbourhoods
                self.update_neighbours(x, y)                     

        transformed_area = np.sum(self.A > 0)
        self.transformed_area_arr.append(transformed_area)   
        self.plot_phase_diagram(show_plot=False)
   
    
    def update_neighbours(self, x, y):
      """
      Stores the neighbour at 4 neighbouring positions.
      0 : up
      1 : right
      2 : down
      3 : left
      """
      if x-1 > 0: self.B[y, x-1, 1] = self.A[y, x]
      if x+1 < self.width-1: self.B[y, x+1, 3] = self.A[y, x]
      if y+1 < self.width-1: self.B[y+1, x, 0] = self.A[y, x]
      if y-1 > 0: self.B[y-1, x, 2] = self.A[y, x]

    # define a method for computing total solid area
    def transformed_area(self):
      """
      Total transformned area at current time step, that is all positive positions. (0 -> liquid; >0 -> solid)
      """
      return np.sum(self.A > 0)

    def plot_phase_diagram(self, title='Current State', save_plot = True, show_plot = True):
        """
        Plots the phase diagram showing the solid and liquid phases at the current time step.
        """
        plt.imshow(self.A,cmap='nipy_spectral')
        plt.title(title)
        if save_plot:
          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          plt.axis("off")
          plt.savefig(os.path.join(self.output_dir, f"{timestamp}.png"))
        if show_plot:
          plt.show()

    def plot_transformed_area(self, plot_analytical = False, title='Transformed area', save_plot = False):
      """
      Plots liquid to solid transformed area against the time.
      """
      plt.figure()
      plt.xlabel('Time')
      plt.ylabel('Fraction of Transformed Area')
      plt.title(title)
      plt.plot(np.array(self.transformed_area_arr) / self.area_tot, 'o', markeredgecolor='b', markerfacecolor='none', linestyle='None', label='CA Simulation')
      if plot_analytical:
        times = np.arange(0, len(self.transformed_area_arr))
        pA_nuc = self.attach_prob*(1+2*times+2*(times**2))
        ca_area_transformed_fraction_JMAK_prob = 1 - np.exp(-self.num_nuclei*(pA_nuc) / self.area_tot)
        plt.plot(ca_area_transformed_fraction_JMAK_prob, 'k-', label='Exact Solution')
      plt.legend()
      if save_plot:
          plt.axis("off")
          plt.savefig(title + ".png")
      plt.show()

class CASolidInput(BaseModel):
  width: int = Field(description="width of the NxN grid on which simulation takes place"),
  attach_prob: float = Field(1, description="likelihood of a cell in the grid transitioning from a liquid or untransformed state to a solidified or transformed state.")
  num_solid_states: int = Field(description="Maximum number of possible solid phases.")
  num_nuclei: int = Field(description="Initial number of nuclei to initialize in the grid.")

class CASolidTool(BaseTool):

  name : str = "CellularAutomataPhaseTransform"
  description : str = """
CellularAutomataPhaseTransform is designed to model phase transformations in a solidifying material using cellular automata on a 2D grid. It simulates iterative transformations based on neighboring conditions and attachment probability, ideal for modeling processes like single-phase solidification and recrystallization.
Key Features:
Evolve Simulation: Iteratively runs phase transformations across the grid.
Neighborhood Update: Continuously updates neighboring cells to reflect transformations.
Track Transformed Area: Calculates and logs the total solidified area across iterations.
Visualization: Plots the current grid state to illustrate transformation progress.
"""
  args_schema: Type[BaseModel] = CASolidInput

  def _run(
      self,
      width : float = 81,
      attach_prob : float = 1,
      num_solid_states : int = 5,
      num_nuclei : int = 3
  )->None:
    ca = CASolid(width=width, attach_prob=attach_prob, num_solid_states=num_solid_states)
    ca.place_nuclei(num_nuclei)
    ca.plot_phase_diagram("Initial")
    while(ca.transformed_area() < ca.area_tot):
        ca.evolve(1)
    ca.plot_phase_diagram("Final")
    ca.plot_transformed_area(True)
    return None

########################################

### Ising Model Monte Carlo ###
###############################

### Biased Potts Annealing ###
class MCAnnealing:
    def __init__(self, Q=16, L=64, MC_steps=120, biasing_strategy=1):
        """
        Class for Monte Carlo Annealing at T ~ 0.
        Neighbourhood: Moore sq(1,2)
        
        Variables:
            Q: Number of flavours
            L: Grid size
            MC_steps: Number of Monte Carlo steps
            biasing_strategy:
                1 => Unbiased
                2 => Unbiased site, neighbouring flavour
                3 => Interface site, neighbouring flavour
            mask: Positions of interfaces

        Functions:
            initialize_random_lattice(): 
                Randomly initializes each site with 1 to Q flavour

            find_interface_sites():
                1. Find sites which are at interface and update mask. 
                2. Comaprision is done with 8 neighbours.
                3. If all are equal to site the not interface site else interface site.

            update_interface_sites():
                1. Updates mask of site and 8 neighbours after switch in flavour
        
        """
        self.Q = Q
        self.L = L
        self.MC_steps = MC_steps
        self.biasing_strategy = biasing_strategy

        self.s = np.zeros((L, L), dtype=int)
        self.energy = 0

        self.mask = None
        self.coordinates = None
        self.s_init = None

        self.twoDconfig_history = []
        self.energy_history = []
        self.acceptance_history = []
        self.config_history = []

    def initialize_random_lattice(self):
        self.s = np.random.randint(1, self.Q+1, (self.L, self.L))
        self.s_init = np.copy(self.s)
        self.find_interface_sites()
        self.energy = self.calc_energy()
        self.energy_history.append(np.copy(self.energy))
        self.config_history.append(np.copy(self.s))

    def neighbour_flavours(self, i, j):
        L = self.L
        neighbour_flavours = np.array([
            self.s[(i-1)%L, j],
            self.s[(i-1)%L, (j+1)%L],
            self.s[i, (j+1)%L],
            self.s[(i+1)%L, (j+1)%L],
            self.s[(i+1)%L, j],
            self.s[(i+1)%L, (j-1)%L],
            self.s[i, (j-1)%L],
            self.s[(i-1)%L, (j-1)%L]
        ])
        return neighbour_flavours 
    
    def calc_energy(self):
        energy = 0
        L = self.L
        for i in range(self.L):
            for j in range(self.L):
                energy += (1 - (self.s[i,j] == self.s[(i+1)%L, j]))+\
                (1 - (self.s[i,j] == self.s[i, (j+1)%L]))+\
                (1 - (self.s[i,j] == self.s[(i-1)%L, (j+1)%L]))+\
                (1 - (self.s[i,j] == self.s[(i-1)%L, ((j-1)%L)]))
        return energy
    
    def find_interface_sites(self):
        """
        Sets up mask (initally None) for interfacial sites
        """
        shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)] # 8 neighbouring directions
        shifted_configs = []
        for shift in shifts:
            shifted_configs.append(np.roll(self.s, shift=shift, axis=(0,1))) # 8 rolled configurations
        self.mask = np.ones_like(self.s)
        for config in shifted_configs:
            self.mask *= (self.s == config) # Setting 1 for all sites in bulk (sites equivalent to neighbours)
        self.mask = 1 - self.mask # Inverting ... Setting 1 for all sites in interface

    def update_interface_sites(self, i, j):
        L = self.L
        # Checking for the site (i,j) and its 8 neighbours
        directions = [(0,0), (0,1), (1,0), (1,1), (0,-1), (-1,0), (-1,-1), (-1,1), (1,-1)] 

        for dir in directions:
            i_n = (i+dir[0])%L; j_n = (j+dir[1])%L
            neighbour_flavours = self.neighbour_flavours(i_n, j_n)

            if np.sum(self.s[i_n, j_n] == neighbour_flavours) == 8:
                self.mask[i_n, j_n] = 0
            else:
                self.mask[i_n, j_n] = 1

    def metropolis_step(self):
        L = self.L
        i, j = -1, -1 # site to select

        if self.biasing_strategy != 3:
            i, j = np.random.randint(self.L, size=2) # for bias 1 & 2 randomly select (i,j)
        else:
            coordinates = np.argwhere(self.mask)            
            idx = np.random.randint(coordinates.shape[0])
            i, j = coordinates[idx] # for bias 3, select (i,j) from interface
            
        neighbour_flavours = self.neighbour_flavours(i, j)

        q_old = self.s[i, j]

        if self.biasing_strategy == 1:
            q_new = np.random.randint(1, self.Q+1) # For bias 1, randomly select potential flavour
        else:
            q_unique = np.unique(neighbour_flavours) 
            q_new = np.random.choice(q_unique) # For bias 2 & 3, select potential flavour from neighbours

        n_diff_old = np.sum(q_old != neighbour_flavours)
        n_diff_new = np.sum(q_new != neighbour_flavours)
        delta_E = n_diff_new - n_diff_old
        
        if delta_E > 0:
            return 0
        
        if delta_E <= 0:
            self.s[i, j] = q_new
            self.energy += delta_E

            if self.biasing_strategy == 3:
                self.update_interface_sites(i, j) # Mask (interfacial sites) only utilized in bias 3
        
        return q_old != q_new # Return to keep count of number of acceptances

    def simulate(self):
        for i in tqdm(range(self.MC_steps)):
            num_acceptances = 0 # To count number of accepted flips per single MC step
            for _ in range(self.L**2):
                num_acceptances += self.metropolis_step()
            
            self.energy_history.append(np.copy(self.energy))
            self.acceptance_history.append(num_acceptances)
            self.config_history.append(np.copy(self.s))
        
        return self.energy_history, self.acceptance_history, self.config_history
    
    def plot_output(self, save_config_history=True):

        if save_config_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"continuum_simulation_results/MonteCarlo_Annealing_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            for step, config in enumerate(self.config_history):
                plt.imshow(config, cmap='tab20b')
                plt.title(f"Configuration at Step {step}")
                plt.colorbar()
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, f"Config_Step_{step}.png"))
                plt.close()

        plt.figure(figsize=(12,12))
        plt.suptitle(f"Q={self.Q} --- L={self.L} --- steps={self.MC_steps} --- Biasing:{self.biasing_strategy}")

        plt.subplot(2,2,1)
        plt.grid(1)
        plt.plot(self.energy_history)
        plt.xlabel('Total Steps')
        plt.ylabel('Energy')
        
        plt.subplot(2,2,2)
        plt.grid(1)
        plt.plot(100*np.array(self.acceptance_history)/self.L**2)
        plt.xlabel('Monte Carlo Steps')
        plt.ylabel('Percent Number of Accepted Flips')

        plt.subplot(2,2,3)
        plt.imshow(self.s_init, cmap='tab20b')
        plt.title('Initial Configuration')

        plt.subplot(2,2,4)
        plt.imshow(self.s, cmap='tab20b')
        plt.title('Final Configuration')

        plt.show()

class MCAnnealingInput(BaseModel):
   Q: int = Field(16, description="Number of phases")
   L: int = Field(64, description="Grid size (NxN)")
   MC_steps : int = Field(120, description="Number of Monte-Carlo steps")
   biasing_strategy : int = Field(1, description="""
Biasing strategy:
1 => Unbiased
2 => Unbiased site, neighbouring flavour
3 => Interface site, neighbouring flavour                     
""")
   
class MCAnnealingTool(BaseTool):
   name : str = "BiasedPottsMonteCarloAnnealing"
   description : str = """
The MCAnnealing class implements a Monte Carlo Potts Model for annealing a 2D grain structure at T≈0. 
It simulates the process over a 2D grid using the Moore neighborhood (eight surrounding sites) to 
identify and update the interface and bulk sites in each Monte Carlo step. This setup uses three biasing 
strategies to examine their effects on the acceptance rate, microstructure, and energy reduction over time.
Key Parameters and Variables:
Q: Number of different "flavors" or states that a site can take.
L: Grid size, representing the width and height of the 2D lattice.
MC_steps: Number of Monte Carlo steps for the simulation.
biasing_strategy: Sampling strategy for selecting and flipping states. Options include:
Unbiased random selection of site and flavor.
Unbiased site with neighboring flavor.
Interface site with neighboring flavor.
mask: Binary mask indicating interface sites, where sites are assigned 1 if they are on an interface and 0 if they are in the bulk.
"""
   args_schema : Type[BaseModel] = MCAnnealingInput

   def _run(
         self,
         Q : int = 16,
         L :int = 64,
         MC_steps : int = 120,
         biasing_strategy : int = 1
   )->None:
    mca = MCAnnealing(Q=Q, L=L, MC_steps=MC_steps, biasing_strategy=biasing_strategy)
    mca.initialize_random_lattice()
    mca.simulate()
    mca.plot_output()
    return None

##############################

### Kinetic Monte Carlo ###
###########################

### Atomic Monte Carlo ###
##########################