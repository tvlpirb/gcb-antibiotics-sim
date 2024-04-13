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
        self.resistant = params["resistant"]
        self.alive = True
        self.growth_inhibited = False
        # new parameters that still need to be tested
        self.lag_phase_length = params["lag_phase_length"]
        self.survival_cost = params["survival_cost"]
        #self.stationary_phase_metabolic_rate = params["stationary_phase_metabolic_rate"] # 0.2
        self.enzyme_production_rate = params["enzyme_production_rate"]
        #self.beta_lactamase_production_cost = params["beta_lactamase_production_cost"]
        self.lag_phase = params["lag_phase_true"]
        self.stationary_phase = False
    
    def step(self):
        self.interact_with_antibiotic()
        if self.lag_phase:
            self.lag_phase_length -= 1
            #self.nutrient_intake += ((self.params["nutrient_intake"] - self.nutrient_intake)
            #     / self.lag_phase_length) # revisit might have a bug
            # End the lag phase if its length is 0
            if self.lag_phase_length <= 0:
                self.lag_phase = False

        self.grow()
        self.is_alive()
        if not self.alive:
            return
        self.move()
        if self.ready_to_split():
            self.split()
    
    def grow(self):
        if not self.alive:
            return
        nutrient_intake = self.intake_nutrient()
        # NOTE Nutrient intake is constantly updated if this is true for each step. Do 
        # we really want it to constantly increase/decrease? 
        # If the model is in the stationary phase, adjust the nutrient intake
        if self.stationary_phase:
            nutrient_intake *= self.params["stationary_phase_metabolic_rate"]

        self.biomass += nutrient_intake

        # Subtract survival cost
        if self.stationary_phase:
            # If the model is in the stationary phase, adjust the survival cost
            self.biomass -= self.params["survival_cost"] * self.params["stationary_phase_metabolic_rate"]
        else:
            # If the model is not in the stationary phase, subtract the survival cost directly
            self.biomass -= self.params["survival_cost"]

    def ready_to_split(self):
        # Bacteria is ready to split if its size is greater than the split 
        # threshold and it's past the lag phase 
        return self.biomass >= self.biomass_threshold and \
            not self.growth_inhibited and \
            not self.lag_phase
    
    # Once a bacteria is ready to split we create a new agent and pass the same
    # parameters with some alterations to it.
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

    # This function implements nutrient intake which is experimentally
    # determined
    def intake_nutrient(self):
        x, y = self.pos # type: ignore
        # Get the current nutrient level in this cell
        nutrient = self.model.grid.properties["nutrient"].data[x][y] # type: ignore
        # If nutrients are less than intake rate we take the remaining amount
        intake = min(self.nutrient_intake,nutrient)
        # Uptake only the available nutrients
        self.model.grid.properties["nutrient"].data[x][y] -= intake # type: ignore
        return intake 

    # The cell is dead once it goes below it's minimum biomass
    def is_alive(self):
        x, y = self.pos # type: ignore
        if self.biomass < self.params["minimum_biomass"]:
            self.alive = False

    # Movement logic for bacteria, the bacterium either move towards the 
    # patch with more nutrients if it is at most one away or it moves in
    # a random fashion in search of nutrients. We take into account overcrowding
    # which is experimentally determined. The bacteria don't move into a cell if
    # it will cause overcrowding and if there is overcrowding then we call 
    # move_overcrowded to deal with this condition
    def move(self):
        x, y = self.pos # type: ignore
        current_patch_agents = self.model.grid.get_cell_list_contents([(x, y)]) # type: ignore

        # Overcrowding condition
        if len(current_patch_agents) > self.params["max_in_patch"]:
            self.move_overcrowded(x,y)
        else:
            neighbors = self.model.grid.get_neighborhood((x, y), moore=True, include_center=True) # type: ignore
            # Get neighborhood nutrient levels 
            nutrient_levels = [(nx, ny, self.model.grid.properties["nutrient"].data[nx][ny]) \
                               for nx, ny in neighbors] 
            max_nutrient = max([level for _, _, level in nutrient_levels])
            # Filter locations that have the maximum nutrient level
            best_locations = [(nx, ny) for nx, ny, level in nutrient_levels \
                              if level == max_nutrient]

            # Further filter locations based on agent count to avoid overcrowding  
            suitable_locations = []
            for loc in best_locations:
                # Check how many agents are in the considered cell
                cell_agents_count = len(self.model.grid.get_cell_list_contents([loc])) # type: ignore
                # If moving to this cell does not cause overcrowding, add to suitable locations
                if cell_agents_count <= self.params["max_in_patch"]:
                    suitable_locations.append(loc)

            if suitable_locations:
                new_x, new_y = random.choice(suitable_locations)
                self.model.grid.move_agent(self, (new_x, new_y)) # type: ignore

    # Given that the current patch is overcrowded we move bacteria to a patch
    # relative to its biomass. We calculate probabilities with the surrounding
    # patches and forcefully move the bacterium to that patch.
    # NOTE There is a current flaw which is if the neighboring cells are also
    # overcrowded. We could possibly kill the bacterium and release the nutrients
    # to the environment but this case needs further consideration.
    def move_overcrowded(self,x,y):
        neighbors = self.model.grid.get_neighborhood( # type: ignore
            (x, y), moore=True, include_center=True) 
        neighbor_biomass = dict()
        for nx, ny in neighbors:
            agents = self.model.grid.get_cell_list_contents([(nx, ny)])  # type: ignore
            total_biomass = sum(agent.biomass for agent in agents)
            neighbor_biomass[(nx, ny)] = total_biomass

        total_neighbor_biomass = sum(neighbor_biomass.values()) + 0.001  # Avoid division by zero
        probabilities = {loc: (1 - biomass / total_neighbor_biomass) \
                         for loc, biomass in neighbor_biomass.items()}
        
        chosen_location = random.choices(
            list(probabilities.keys()), weights=probabilities.values())[0] # type: ignore
        
        # Move agent to the new location
        self.model.grid.move_agent(self, chosen_location)  # type: ignore

    # When the bacteria interact with antibiotics they intake antibiotics however,
    # in our model we won't take this detail into account and factor activation only
    # when the minimum inhibitory concentration (MIC) is present. Due to the cell wall
    # being damaged by the antibiotic we will assume that some biomass is lost and that
    # our cell cannot grow. Our cell dies when it can't grow and it reaches a specific
    # minimum threshold biomass where it can no longer survive. 
    def interact_with_antibiotic(self):
        x, y = self.pos # type: ignore
        antibiotic_concentration = self.model.grid.properties["antibiotic"].data[x][y] # type: ignore

        # Antibiotic concentration is at MIC, response mechanism is activated
        if antibiotic_concentration >= self.params["MIC"]:
            self.growth_inhibited = True
            self.biomass -= 1 # NOTE Arbitrary value cost
            # Start producing enzymes
            # NOTE Consider checking if we have a resistant strain or not
            if self.resistant:
                self.model.grid.properties["time_enzyme"].data[x][y] += 1 # type: ignore
                self.model.grid.properties["enzyme"].data[x][y] += self.enzyme_production_rate # type: ignore
        else:
            self.growth_inhibited = False


class SimModel(mesa.Model):
    def __init__(self, params):
        super().__init__()
        self.width = params["width"]
        self.height = params["height"]
        self.params = params
        # NOTE Diffusion happens a little too fast so consider altering this 
        # constant for now to account for it
        self.diffusion_coefficient = params["diffusion_coefficient"]
        self.num_agents = params["num_agents"]

        # Initialize Grid Properties
        self.grid = mesa.space.MultiGrid(self.width,self.height,False)
        nutrient_layer = PropertyLayer("nutrient",self.width,self.height,default_value=0)
        antibiotic_layer = PropertyLayer("antibiotic",self.width,self.height,default_value=0)
        enzyme_layer = PropertyLayer("enzyme",self.width, self.height, default_value=0)
        time_layer = PropertyLayer("time_enzyme",self.width, self.height,default_value=-1)
        #nutrient_layer.modify_cells(lambda x: np.random.random())
        # NOTE This is hardcoded for testing purposes, REMOVE
        nutrient_layer.set_cell((5,5),100)
        nutrient_layer.set_cell((8,3),100)
        self.grid.add_property_layer(nutrient_layer)
        self.grid.add_property_layer(antibiotic_layer)
        self.grid.add_property_layer(enzyme_layer)
        self.grid.add_property_layer(time_layer)
        
        # Initialize Scheduler
        self.schedule = mesa.time.RandomActivation(self)
       
        # Initialize Agents
        for i in range(self.num_agents):
            a = BacteriaAgent(i,self, params)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width) # type: ignore
            y = self.random.randrange(self.grid.height) # type: ignore
            # NOTE JUST FOR TESTING PURPOSES
            x = 5
            y = 5
            self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()
        # Vectorized version should have a major speedup
        # Runtime went from 75 seconds to 5 seconds for 1000 steps
        # NOTE DISABLED TEMPORATILY
        #self.diffuse_vectorized("nutrient",self.diffusion_coefficient)
        self.degrade_antibiotic()

    # Go through every cell and take into account the antibiotic concentration
    # and the enzyme level to degrade the antibiotics
    # We make use of the formula V0 = V_max * [S] /K_m + [S] to determine the 
    # rate at which the antibiotics should degrade, this equation comes from 
    # the standard Michealis Menten kinetic equations and the inspiration for 
    # this specific choice of modelling comes from DOI: 10.1089/cmb.2018.0064. 
    def degrade_antibiotic(self):
        for x in range(self.width):
            for y in range (self.height):
                antibiotic_conc = self.grid.properties["antibiotic"].data[x][y] 
                V_0 = ((self.params["v_max"] * antibiotic_conc) / 
                       (self.params["k_m"] + antibiotic_conc))
                new_conc = max(0, antibiotic_conc-V_0)
                self.grid.properties["antibiotic"].data[x][y] = new_conc

    # Degrade the enzymes by halving it each time, we have a time layer to track
    # how long an enzyme has been in a patch and we aim to halve it only after the
    # set time. There is however an issue as this doesn't yet account for diffusion 
    # of enzymes 
    def degrade_enzyme(self):
        cells_halve = np.where(self.grid.properties["time_enzyme"].data >= self.params["enzyme_half_life"])
        for x,y in zip(*cells_halve):
            self.grid.properties["enzyme"].data[x][y] /= 2
            # Reset the timer
            self.grid.properties["time_enzyme"].data[x][y] = 0

    # Given a key and a coefficient we will modify the appropriate grid layer
    # for diffusion, we make use of a discretized version of ficks law where
    # for adjacent patches we use D(delt concentration) and for diagonal patches
    # we use D(delta concentration)/sqrt(2) where D is the diffusion constant.
    # The following implementation treats the grid as non-torus and therefore
    # we pad the edges with zeros and have reflective boundaries for diffusion.
    # Following benchmarking this new method is magnitudes faster due to vectorization
    def diffuse_vectorized(self,key,coefficient):
        nutrient_grid = self.grid.properties[key].data
        
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
        total_diffusion = (direct_diffusion + diagonal_diffusion) * coefficient
        nutrient_grid += total_diffusion
        # Normalize or ensure nutrient conservation due to it being lost
        nutrient_grid += (-(nutrient_grid.sum() - self.grid.properties[key].data.sum()) / 
                          np.prod(nutrient_grid.shape))
        self.grid.properties[key].data = nutrient_grid
