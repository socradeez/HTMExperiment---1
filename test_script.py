from MacroColumn import MacroColumn
from data_generation import generate_objects

objectlist = generate_objects(20, 10, 30, 30)

agent = MacroColumn(3)
for input_object in objectlist:

    agent.train_on_object(input_object)