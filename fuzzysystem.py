import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

#initialise inputs and outputs
temperature = ctrl.Antecedent(np.arange(16, 32, 0.1), 'temperature')
number_of_people = ctrl.Antecedent(np.arange(0, 13, 0.1), 'number of people')
cooling_power = ctrl.Consequent(np.arange(0, 100, 0.1), 'cooling power')
energy_consumption = ctrl.Consequent(np.arange(1, 10, 0.1), 'energy consumption')

#Membership fit vector
temperature['cold'] = mf.trimf(temperature.universe, [16, 18, 20])
temperature['slightly cold'] = mf.trimf(temperature.universe, [19, 21, 23])
temperature['warm'] = mf.trimf(temperature.universe, [22, 24, 26])
temperature['slightly warm'] = mf.trimf(temperature.universe, [25, 27, 29])
temperature['hot'] = mf.trimf(temperature.universe, [29, 32, 35])

number_of_people['empty'] = mf.trimf(number_of_people.universe, [0, 0, 0])
number_of_people['few'] = mf.trimf(number_of_people.universe, [1, 2, 4])
number_of_people['moderate'] = mf.trimf(number_of_people.universe, [4, 5, 7])
number_of_people['many'] = mf.trimf(number_of_people.universe, [6, 8, 11])
number_of_people['crowded'] = mf.trimf(number_of_people.universe, [10, 11, 13])

cooling_power['low'] = mf.trimf(cooling_power.universe, [0, 12.5, 25])
cooling_power['slightly low'] = mf.trimf(cooling_power.universe, [20, 32.5, 45])
cooling_power['medium'] = mf.trimf(cooling_power.universe, [40, 52.5, 65])
cooling_power['slightly high'] = mf.trimf(cooling_power.universe, [60, 70, 80])
cooling_power['high'] = mf.trimf(cooling_power.universe, [75, 87.5, 100])

energy_consumption['low'] = mf.trimf(energy_consumption.universe, [1, 2, 3])
energy_consumption['slightly low'] = mf.trimf(energy_consumption.universe, [2, 3, 4])
energy_consumption['medium'] = mf.trimf(energy_consumption.universe, [4, 5, 6])
energy_consumption['slightly high'] = mf.trimf(energy_consumption.universe, [5, 6, 7])
energy_consumption['high'] = mf.trimf(energy_consumption.universe, [7, 8.5, 10])


#Rules
rule1 = ctrl.Rule(temperature['cold'] & number_of_people['empty'], (cooling_power['low'], energy_consumption['low']))
rule2 = ctrl.Rule(temperature['slightly cold'] & number_of_people['empty'], (cooling_power['low'], energy_consumption['low']))
rule3 = ctrl.Rule(temperature['warm'] & number_of_people['empty'], (cooling_power['slightly low'], energy_consumption['slightly low']))
rule4 = ctrl.Rule(temperature['slightly warm'] & number_of_people['empty'], (cooling_power['medium'], energy_consumption['medium']))
rule5 = ctrl.Rule(temperature['hot'] & number_of_people['empty'], (cooling_power['slightly high'], energy_consumption['slightly high']))

rule6 = ctrl.Rule(temperature['cold'] & number_of_people['few'], (cooling_power['low'], energy_consumption['low']))
rule7 = ctrl.Rule(temperature['slightly cold'] & number_of_people['few'], (cooling_power['slightly low'], energy_consumption['slightly low']))
rule8 = ctrl.Rule(temperature['warm'] & number_of_people['few'], (cooling_power['medium'], energy_consumption['medium']))
rule9 = ctrl.Rule(temperature['slightly warm'] & number_of_people['few'], (cooling_power['slightly high'], energy_consumption['slightly high']))
rule10 = ctrl.Rule(temperature['hot'] & number_of_people['few'], (cooling_power['high'], energy_consumption['high']))

rule11 = ctrl.Rule(temperature['cold'] & number_of_people['moderate'], (cooling_power['slightly low'], energy_consumption['slightly low']))
rule12 = ctrl.Rule(temperature['slightly cold'] & number_of_people['moderate'], (cooling_power['medium'], energy_consumption['medium']))
rule13 = ctrl.Rule(temperature['warm'] & number_of_people['moderate'], (cooling_power['medium'], energy_consumption['medium']))
rule14 = ctrl.Rule(temperature['slightly warm'] & number_of_people['moderate'], (cooling_power['slightly high'], energy_consumption['slightly high']))
rule15 = ctrl.Rule(temperature['hot'] & number_of_people['moderate'], (cooling_power['high'], energy_consumption['high']))

rule16 = ctrl.Rule(temperature['cold'] & number_of_people['many'], (cooling_power['slightly low'], energy_consumption['slightly low']))
rule17 = ctrl.Rule(temperature['slightly cold'] & number_of_people['many'], (cooling_power['medium'], energy_consumption['medium']))
rule18 = ctrl.Rule(temperature['warm'] & number_of_people['many'], (cooling_power['slightly high'], energy_consumption['slightly high']))
rule19 = ctrl.Rule(temperature['slightly warm'] & number_of_people['many'], (cooling_power['high'], energy_consumption['high']))
rule20 = ctrl.Rule(temperature['hot'] & number_of_people['many'], (cooling_power['high'], energy_consumption['high']))

rule21 = ctrl.Rule(temperature['cold'] & number_of_people['crowded'], (cooling_power['medium'], energy_consumption['medium']))
rule22 = ctrl.Rule(temperature['slightly cold'] & number_of_people['crowded'], (cooling_power['slightly high'], energy_consumption['slightly high']))
rule23 = ctrl.Rule(temperature['warm'] & number_of_people['crowded'], (cooling_power['high'], energy_consumption['high']))
rule24 = ctrl.Rule(temperature['slightly warm'] & number_of_people['crowded'], (cooling_power['high'], energy_consumption['high']))
rule25 = ctrl.Rule(temperature['hot'] & number_of_people['crowded'], (cooling_power['high'], energy_consumption['high']))

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25]

#
#Construct fuzzy system
#
train_ctrl = ctrl.ControlSystem(rules=rules)
train = ctrl.ControlSystemSimulation(control_system=train_ctrl)

# define the values for the inputs
train.input['temperature'] = 30
train.input['number of people'] = 8

# compute the outputs
train.compute()

# print the output values
print(train.output)

# to extract one of the outputs
print(train.output['cooling power'])


cooling_power.view(sim=train)
energy_consumption.view(sim=train)

#
#View the control/output space
#
x, y = np.meshgrid(np.linspace(temperature.universe.min(), temperature.universe.max(), 100),
                   np.linspace(number_of_people.universe.min(), number_of_people.universe.max(), 100))
z_cooling_power = np.zeros_like(x, dtype=float)
z_energy_consumption = np.zeros_like(x, dtype=float)

for i,r in enumerate(x):
  for j,c in enumerate(r):
    train.input['temperature'] = x[i,j]
    train.input['number of people'] = y[i,j]
    try:
      train.compute()
      z_cooling_power[i,j] = train.output['cooling power']
      z_energy_consumption[i,j] = train.output['energy consumption']
    except:
      z_cooling_power[i,j] = float('inf')
      z_energy_consumption[i,j] = float('inf')
    
  

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(x,y,z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

  ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

  ax.view_init(30, 200)

plot3d(x, y, z_cooling_power)
plot3d(x, y, z_energy_consumption)
