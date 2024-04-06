import mesa
from mesa.space import PropertyLayer
import numpy as np

class BacteriaAgent(mesa.Agent):
    def __init__(self, unique_id, model, params):
        super().__init__(unique_id, model)
        self.params = params
        self.uptake_rate_antibiotic = params["uptake_rate_antibiotic"]
        self.uptake_rate = params["uptake_rate"]
        self.biomass = params["initial_biomass"]
        self.biomass_threshold = params["biomass_threshold"]
        self.alive = True
    
    def step(self):
        self.grow()
        if self.ready_to_split():
            self.split()
            
    def total_biomass(self):
        total_biomass = 0
        for agent in self.model.schedule.agents: # type: ignore
            total_biomass += agent.biomass
        return total_biomass
    
    def grow(self):
        # Get the nutrient uptake
        nutrient_uptake = self.uptake_nutrient()
        # Increase the size based on the nutrient uptake
        self.biomass += nutrient_uptake

    def ready_to_split(self):
        # Bacteria is ready to split if its size is greater than the split 
        # threshold 
        return self.biomass >= self.biomass_threshold
    
    def split(self):
        # Create a new bacterium with half the size and biomass of the current one
        paramsC = self.params.copy()
        paramsC["initial_biomass"] = self.biomass / 2
        new_bacteria = BacteriaAgent(self.model.next_id(), self.model, paramsC)
        self.model.schedule.add(new_bacteria) # type: ignore
        self.model.grid.place_agent(new_bacteria, self.pos) # type: ignore
        # Halve the size and weight of the current bacterium
        self.biomass /= 2

    def uptake_nutrient(self):
        x, y = self.pos # type: ignore
        # Get the current nutrient level in this cell
        nutrient = self.model.grid.properties["nutrient"].data[x][y] # type: ignore
        # Uptake only the available nutrients
        self.model.grid.properties["nutrient"].data[x][y] -= min(self.uptake_rate * nutrient, nutrient) # type: ignore
        return nutrient

    def is_alive(self):
        minimum_nutrient_level = 0.1 # arbitrary value
        if self.biomass > 0:
            x, y = self.pos # type: ignore
            nutrient = self.model.grid.properties["nutrient"].data[x][y] # type: ignore
            if nutrient >= minimum_nutrient_level:
                self.alive = True
            else:
                self.alive = False
        else:
            self.alive = False

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
        #nutrient_layer.modify_cells(lambda x: np.random.random())
        nutrient_layer.set_cell((0,0),100)
        self.grid.add_property_layer(nutrient_layer)
        
        # Initialize Scheduler
        self.schedule = mesa.time.RandomActivation(self)
       
        # Initialize Agents
        for i in range(self.num_agents):
            a = BacteriaAgent(i,self, params)
            #self.schedule.add(a)
            x = self.random.randrange(self.grid.width) # type: ignore
            y = self.random.randrange(self.grid.height) # type: ignore
            #self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()
        # Vectorized version should have a major speedup
        # Runtime went from 75 seconds to 5 seconds for 1000 steps
        self.diffuse_nutrients_vectorized()
    
    def diffuse_nutrients_vectorized(self):
        nutrient_grid = self.grid.properties["nutrient"].data
        
        # Use 'edge' mode for reflective boundaries
        padded = np.pad(nutrient_grid, pad_width=1, mode='edge')
        
        # Calculate direct neighbors diffusion with reflective boundary adjustments
        direct_diffusion = (
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[:-2, 1:-1] + padded[2:, 1:-1] - 4 * nutrient_grid
        )
        
        # Compute diffusion for diagonal neighbors with reflective boundary adjustments and adjusted by 1/sqrt(2)
        diagonal_diffusion = (
            padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:] - 4 * nutrient_grid
        ) / np.sqrt(2)
        
        # Combine both effects
        total_diffusion = (direct_diffusion + diagonal_diffusion) * self.diffusion_coefficient
        
        # Apply diffusion symmetrically
        nutrient_grid += total_diffusion
        
        # Normalize or ensure nutrient conservation if needed here.
        nutrient_grid += -(nutrient_grid.sum() - self.grid.properties["nutrient"].data.sum()) / np.prod(nutrient_grid.shape)
        self.grid.properties["nutrient"].data = nutrient_grid
