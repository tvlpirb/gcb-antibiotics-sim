from mesa import Agent, Model, datacollection
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import random

class Bacteria(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
    
    # Defines how the agent should be drawn in a visualization
    def get_portrayal(self):
        return {"Shape": "circle", "Color": "red", "Filled": "true", "r": 0.5}

class BacteriaModel(Model):
    def __init__(self, width, height, num_bacteria):
        super().__init__()
        self._steps = 0
        self._time = 0 
        self.grid = MultiGrid(width, height, True)
        # Activates each agent once per step in a random order
        self.schedule = RandomActivation(self)
        # Collecting the count of bacteria agents at each step
        self.datacollector = datacollection.DataCollector(
            {"Bacteria": lambda m: m.schedule.get_agent_count()})

        # Create bacteria
        for i in range(num_bacteria):
            bacteria = Bacteria(i, self)
            self.schedule.add(bacteria)

            # Place the bacteria in a random cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(bacteria, (x, y))
            
    # What happens at each step on model execution
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self._steps += 1
        self._time += 1

# Create the model
model = BacteriaModel(10, 10, 100)

# Run the model
for i in range(100):
    model.step()

# Is this even doing anything? - did it for the error
def portrayal(agent):
    return {"Shape": "circle", "Color": "red", "Filled": "true", "r": 0.5, "Layer": 0}

grid = CanvasGrid(portrayal, 10, 10, 500, 500)
chart = ChartModule([{"Label": "Bacteria", "Color": "Black"}])

server = ModularServer(BacteriaModel, [grid, chart], "Bacteria Model", {"width": 10, "height": 10, "num_bacteria": 100})
server.launch(port=8522)
