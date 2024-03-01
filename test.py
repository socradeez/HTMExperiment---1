import Column
from data_generation import generate_objects

data = generate_objects(1, 4, 30, 30)

testobject = data[0]

column = Column.Column()
column2 = Column.Column()

column.add_lateral_connection(column2)
column2.add_lateral_connection(column)

for _ in range(5):
    print('new run on object')
    for (feature, location) in testobject:
        print('starting new feature')
        column.update_activity(feature, location)
        column2.update_activity(feature, location)
        column.learn()
        column2.learn()
        L2count, L4count = column.get_active_counts()
        print("L4 active neuron count is", L4count)
        print("L2 active neuron count is", L2count)



