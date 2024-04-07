import mesa
import random
from mesa.space import PropertyLayer
import numpy as np

class BacteriaAgent(mesa.Agent):
    def __init__(self, unique_id, model, params):
        super().__init__(unique_id, model)
        self.params = params
        self.antibiotic_intake = params["antibiotic_intake"]
        self.nutrient_intake = params["nutrient_intake"]
        self.biomass = params["initial_biomass"]
        self.biomass_threshold = params["biomass_threshold"]
        self.alive = True
        # new parameters that still need to be tested
        self.lag_phase_length = params["lag_phase_length"]
        self.survival_cost = params["survival_cost"]
        #self.stationary_phase_metabolic_rate = params["stationary_phase_metabolic_rate"]
        #self.beta_lactamase_production_rate = params["beta_lactamase_production_rate"]
        #self.beta_lactamase_production_cost = params["beta_lactamase_production_cost"]
        self.lag_phase = params["lag_phase_true"]
        self.stationary_phase = False
    
    def step(self):
        if self.lag_phase:
            self.lag_phase_length -= 1
            #self.nutrient_intake += (self.params["nutrient_intake"] - self.nutrient_intake) / self.lag_phase_length
            if self.lag_phase_length <= 0:
                self.lag_phase = False
        self.grow()
        self.is_alive()
        self.move()
        if self.ready_to_split() and not self.lag_phase:
            self.split()
            
    def total_biomass(self):
        total_biomass = 0
        for agent in self.model.schedule.agents: # type: ignore
            total_biomass += agent.biomass
        return total_biomass
    
    def grow(self):
        # Get the nutrient uptake
        nutrient_intake = self.intake_nutrient()
        self.biomass += nutrient_intake

        # Subtract survival cost
        if self.stationary_phase:
            self.biomass -= self.survival_cost * self.stationary_phase_metabolic_rate
        else:
            self.biomass -= self.survival_cost

    def ready_to_split(self):
        # Bacteria is ready to split if its size is greater than the split 
        # threshold 
        return self.biomass >= self.biomass_threshold
    
    def split(self):
        # Create a new bacterium with half the size and biomass of the current one
        paramsC = self.params.copy()
        paramsC["initial_biomass"] = self.biomass / 2
        paramsC["lag_phase_true"] = False,
        new_bacteria = BacteriaAgent(self.model.next_id(), self.model, paramsC)
        self.model.schedule.add(new_bacteria) # type: ignore
        self.model.grid.place_agent(new_bacteria, self.pos) # type: ignore
        # Halve the size and weight of the current bacterium
        self.biomass /= 2

    def intake_nutrient(self):
        x, y = self.pos # type: ignore
        # Get the current nutrient level in this cell
        nutrient = self.model.grid.properties["nutrient"].data[x][y] # type: ignore
        # If nutrients are less than intake rate we take the remaining amount
        intake = min(self.nutrient_intake,nutrient)
        # Uptake only the available nutrients
        self.model.grid.properties["nutrient"].data[x][y] -= intake # type: ignore
        return intake 

    def is_alive(self):
        if self.biomass < 0:
            self.alive = False
            return
        x, y = self.pos # type: ignore
        if self.biomass >= self.params["minimum_biomass"]:
            self.alive = True
        else:
            self.alive = False

    # TODO Consider checking that the cell isn't full before moving to it
    def move(self):
        x, y = self.pos # type: ignore
        neighbors = self.model.grid.get_neighborhood((x, y), moore=True, include_center=True) # type: ignore
        # Get neighborhood nutrient levels 
        nutrient_levels = [(nx, ny, self.model.grid.properties["nutrient"].data[nx][ny]) for nx, ny in neighbors] # type: ignore
        max_nutrient = max([level for _, _, level in nutrient_levels])
        # Filter locations that have the maximum nutrient level
        best_locations = [(nx, ny) for nx, ny, level in nutrient_levels if level == max_nutrient]
        new_x, new_y = random.choice(best_locations)
        # Move the agent to the chosen location with more or equal
        self.model.grid.move_agent(self, (new_x, new_y)) # type: ignore

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
        self.grid = mesa.space.MultiGrid(self.width,self.height,False)
        nutrient_layer = PropertyLayer("nutrient",self.width,self.height,default_value=0)
        # TODO We need a valid distribution of nutrients
        #nutrient_layer.modify_cells(lambda x: np.random.random())
        nutrient_layer.set_cell((5,5),25)
        nutrient_layer.set_cell((8,3),100)
        self.grid.add_property_layer(nutrient_layer)
        
        # Initialize Scheduler
        self.schedule = mesa.time.RandomActivation(self)
       
        # Initialize Agents
        for i in range(self.num_agents):
            a = BacteriaAgent(i,self, params)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width) # type: ignore
            y = self.random.randrange(self.grid.height) # type: ignore
            x = 5
            y = 5
            self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()
        # Vectorized version should have a major speedup
        # Runtime went from 75 seconds to 5 seconds for 1000 steps
        #self.diffuse_nutrients_vectorized()
    
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
