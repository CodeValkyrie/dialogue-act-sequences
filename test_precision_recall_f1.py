import pandas as pd
from data import Statistics

""" This is a script that tests the correctness of the precision_recall_f1() function in the Statistics class."""

# Constructs a test DataFrame with 3 classes of which the precision, recall and f1 score must be computed.
labels = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'b']
predictions = ['a', 'b', 'c', 'a', 'a', 'b', 'c', 'c', 'a', 'b', 'a', 'b', 'a', 'e']
classes = ['a', 'b', 'c', 'd', 'e']

data = pd.DataFrame(list(zip(labels, predictions)), columns =['labels', 'predictions'])
print(data)

""" Data: 

      labels   predictions
0       a           a
1       a           b
2       a           c
3       a           a
4       b           a
5       b           b
6       b           c
7       b           c
8       c           a
9       c           b
10      c           a
11      c           b
12      d           a
13      b           e


"""

statistics = Statistics(data)
for class_name in classes:
    precision, recall, f1 = statistics.precision_recall_f1(data, ['labels', 'predictions'], class_name)
    print(class_name + ' precision is: ' + str(round(precision, 4)))
    print(class_name + ' recall is: ' + str(round(recall, 4)))
    print(class_name + ' f1 is: ' + str(round(f1, 4)))

""" 
Desired output:

a precision = 2/6 = 0.33
a recall = 2/4 = 0.5
a f1 = 2 * (0.33 * 0.5) / 0.83 = 0.4

b precision = 1/4 = 0.25
b recall = 1/5 = 0.20
b f1 = 2 * (0.25 * 0.20) / 0.45 = 0.2222

c precision = 0/3 = 0
c recall = 0/4 = 0
c f1 = 0

d precision = 0/0 = NaN
d recall = 0/1 = 0
d f1 = NaN

e precision = 0/1 = 0
e recall = 0/0 = NaN
e F1 = NaN


Output of script:

a precision is: 0.3333
a recall is: 0.5
a f1 is: 0.4
b precision is: 0.25
b recall is: 0.2
b f1 is: 0.2222
c precision is: 0.0
c recall is: 0.0
c f1 is: 0
d precision is: nan
d recall is: 0.0
d f1 is: nan
e precision is: 0.0
e recall is: nan
e f1 is: nan



"""