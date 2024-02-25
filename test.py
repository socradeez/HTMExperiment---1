import Column
from data_generation import generate_objects

data = generate_objects(1, 10, 30, 30)

testobject = data[0]

column = Column.Column()

for _ in range(20):
    for (feature, location) in testobject:
        column.update_activity(feature, location)
        column.learn()
        L2count, L4count = column.get_active_counts()

for (feature, location) in testobject:
    column.update_activity(feature, location)
    column.learn()
    L2count, L4count = column.get_active_counts()
    print("L2 active neuron count is", L2count)
    print("L4 active neuron count is", L4count)
