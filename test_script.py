from MacroColumn import MacroColumn
from data_generation import generate_objects

objectlist = generate_objects(20, 10, 30, 30)

agent = MacroColumn(3)
for object in objectlist:
    agent.sense_object(object, True)
    agent.sense_object(object, True)