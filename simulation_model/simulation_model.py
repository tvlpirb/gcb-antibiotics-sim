import mesa
from mesa.space import PropertyLayer
import numpy as np

class BacteriaAgent(mesa.Agent):
    def __init__(self, unique_id, model, params):
        super().__init__(unique_id, model)
        self.params = params
        self.uptake_rate = params["uptake_rate"]
        self.size = params["initial_size"]
        self.biomass = params["initial_biomass"]
        self.split_threshold = 2 * params["initial_size"]  # Bacteria splits when its size has doubled
        self.biomass_threshold = params["biomass_threshold"]
        self.alive = True
    
    def step(self):
        # self.uptake_nutrient()
        self.grow()
        if self.ready_to_split():
            self.split()
            
    def total_biomass(self):
        total_biomass = 0
        for agent in self.model.schedule.agents:
            total_biomass += agent.biomass
        return total_biomass
    
    def grow(self):
        # Get the nutrient uptake
        nutrient_uptake = self.uptake_nutrient()
        # Increase the size based on the nutrient uptake
        self.size += nutrient_uptake
        # Update biomass
        self.biomass = self.size * 1.5 # Assume a constant conversion factor of 1.5 (this is an arbitary number for now)

    def ready_to_split(self):
        # Bacteria is ready to split if its size is greater than the split 
        # threshold and the environment can support more biomass
        return self.size >= self.split_threshold and self.total_biomass() < self.params["biomass_threshold"]
    
    def split(self):
        # Create a new bacterium with half the size and biomass of the current one
        paramsC = self.params.copy()
        paramsC["initial_size"] = self.size / 2
        paramsC["initial_biomass"] = self.biomass / 2
        new_bacteria = BacteriaAgent(self.model.next_id(), self.model, paramsC)
        self.model.schedule.add(new_bacteria)
        self.model.grid.place_agent(new_bacteria, self.pos)
        # Halve the size and weight of the current bacterium
        self.size /= 2
        self.biomass /= 2

    def uptake_nutrient(self):
        # Get the current cell of the bacterium
        x, y = self.pos
        # Get the current nutrient level in this cell
        nutrient = self.model.grid.properties["nutrient"].data[x][y]
        # Uptake only the available nutrients
        # Subtract the uptaken nutrients from the nutrient level in the cell
        self.model.grid.properties["nutrient"].data[x][y] -= min(self.uptake_rate * nutrient, nutrient)
        return nutrient
        # return min(self.uptake_rate * nutrient, nutrient)

    # need to implement living status of bacteria
    def is_alive(self):
        if (self.biomass > 0):
            self.alive = True 
        else:
            self.alive = False
        # NEED MORE LOGIC HERE

    # def interact_with_antibiotic(self, antibiotic):

    # def interact_with_enzyme(self, enzyme):

class SimModel(mesa.Model):
    def __init__(self, params):
        super().__init__()
        self.width = params["width"]
        self.height = params["height"]
        self.diffusion_coefficient = params["diffusion_coefficient"]
        self.num_agents = params["num_agents"]

        # Initialize Grid Properties
        self.grid = mesa.space.MultiGrid(self.width,self.height,True)
        nutrient_layer = PropertyLayer("nutrient",self.width,self.height,default_value=0)
        nutrient_layer.modify_cells(lambda x: np.random.random())
        self.grid.add_property_layer(nutrient_layer)
        
        # Initialize Scheduler
        self.schedule = mesa.time.RandomActivation(self)
       
        # Initialize Agents
        for i in range(self.num_agents):
            a = BacteriaAgent(i,self, params)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()
        # Vectorized version should have a major speedup
        # Runtime went from 75 seconds to 5 seconds for 1000 steps
        self.diffuse_nutrients_vectorized()
        #self.total_biomass()

    
    
    def diffuse_nutrients_vectorized(self):
        # Extract the current nutrient grid for convenience
        nutrient_grid = self.grid.properties["nutrient"].data
        # Create padded grid to handle edge wrapping more easily
        padded = np.pad(nutrient_grid, pad_width=1, mode='wrap')
        # Calculate the diffusion from each cell to its neighbors
        # For direct neighbors
        direct_diffusion = padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1] - 4 * nutrient_grid
        # For diagonal neighbors, adjusted by 1/sqrt(2)
        diagonal_diffusion = (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:] - 4 * nutrient_grid) / np.sqrt(2)
        # Sum of both diffusion effects
        total_diffusion = direct_diffusion + diagonal_diffusion
        # Apply the diffusion coefficient
        nutrient_grid += total_diffusion * self.diffusion_coefficient
        # Update the nutrient grid
        self.grid.properties["nutrient"].data = nutrient_grid
