import mesa
from mesa.space import PropertyLayer
import numpy as np 

class BacteriaAgent(mesa.Agent):
    def __init__(self, unique_id, model, uptake_rate):
        super().__init__(unique_id, model)
        self.uptake_rate = uptake_rate # added
    
    def step(self):
        self.uptake_nutrient()
        #pass

    def uptake_nutrient(self):
        # Get the current cell of the bacterium
        x, y = self.pos
        # Get the current nutrient level in this cell
        nutrient = self.model.grid.properties["nutrient"].data[x][y]
        # Uptake only the available nutrients
        # Subtract the uptaken nutrients from the nutrient level in the cell
        self.model.grid.properties["nutrient"].data[x][y] -= min(self.uptake_rate * nutrient, nutrient)
        return nutrient

class SimModel(mesa.Model):
    def __init__(self, params):
        super().__init__()
        self.num_agents = params["num_agents"]
        self.width = params["width"]
        self.height = params["height"]
        uptake_rate = params["uptake_rate"]
        self.diffusion_coefficient = params["diffusion_coefficient"]
        
        # Initialize Grid Properties
        self.grid = mesa.space.MultiGrid(self.width,self.height,True)
        nutrient_layer = PropertyLayer("nutrient",self.width,self.height,default_value=0)
        nutrient_layer.modify_cells(lambda x: np.random.random())
        self.grid.add_property_layer(nutrient_layer)
        
        # Initialize Scheduler
        self.schedule = mesa.time.RandomActivation(self)
       
        # Initialize Agents
        for i in range(self.num_agents):
            a = BacteriaAgent(i,self, uptake_rate) 
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()
        # Vectorized version should have a major speedup
        # Runtime went from 75 seconds to 5 seconds for 1000 steps
        #self.diffuse_nutrients_vectorized()
        self.diffuse_nutrients()
        
    def diffuse_nutrients(self):
        new_nutrient_distribution = np.copy(self.grid.properties["nutrient"].data)
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue  # Skip self
                        x2 = (x + dx) % self.grid.width
                        y2 = (y + dy) % self.grid.height
                        diff = (self.grid.properties["nutrient"].data[x2][y2] -
                                self.grid.properties["nutrient"].data[x][y])
                        if dx == 0 or dy == 0:  # Adjacent
                            transfer = diff * self.diffusion_coefficient
                        else:  # Diagonal
                            transfer = (diff * self.diffusion_coefficient) / np.sqrt(2)
                        new_nutrient_distribution[x][y] += transfer
                        new_nutrient_distribution[x2][y2] -= transfer
        
        self.grid.properties["nutrient"].data = new_nutrient_distribution

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
