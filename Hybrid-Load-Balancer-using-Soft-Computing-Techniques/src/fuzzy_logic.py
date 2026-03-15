import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyController:
    def __init__(self):
        # Antecedents (Inputs)
        self.load = ctrl.Antecedent(np.arange(0, 101, 1), 'load')
        self.trend = ctrl.Antecedent(np.arange(0, 101, 1), 'trend')
        # Consequent (Output)
        self.priority = ctrl.Consequent(np.arange(0, 11, 1), 'priority')

        # Membership Functions
        self.load.automf(3, names=['low', 'medium', 'high'])
        self.trend.automf(3, names=['stable', 'increasing', 'spiking'])
        self.priority['low'] = fuzz.trimf(self.priority.universe, [0, 0, 5])
        self.priority['medium'] = fuzz.trimf(self.priority.universe, [2, 5, 8])
        self.priority['high'] = fuzz.trimf(self.priority.universe, [5, 10, 10])

        # Rules
        rule1 = ctrl.Rule(self.load['high'] | self.trend['spiking'], self.priority['low'])
        rule2 = ctrl.Rule(self.load['medium'], self.priority['medium'])
        rule3 = ctrl.Rule(self.load['low'] & self.trend['stable'], self.priority['high'])

        self.lb_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.sim = ctrl.ControlSystemSimulation(self.lb_ctrl)

    def compute_priority(self, current_load, predicted_trend):
        self.sim.input['load'] = current_load
        self.sim.input['trend'] = predicted_trend
        self.sim.compute()
        return self.sim.output['priority']