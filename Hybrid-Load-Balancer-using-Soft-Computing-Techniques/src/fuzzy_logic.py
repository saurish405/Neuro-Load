import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyController:
    """
    Fuzzy controller with speed-proportional overflow routing.

    compute_priority(load, trend, server_speed) returns a final priority
    score in [0, 100].

    Speed-proportional overflow
    ---------------------------
    When S3 is full and both S1 (speed=0.5) and S2 (speed=1.0) are candidates,
    we want a ~2:1 split in favour of S2.  We achieve this by multiplying the
    raw fuzzy output by a speed multiplier BEFORE returning it, so the scoring
    in main2.py naturally prefers faster servers in proportion to their speed:

        S2 score ≈ fuzzy_out * (1.0 / max_speed)  →  higher
        S1 score ≈ fuzzy_out * (0.5 / max_speed)  →  lower

    The multiplier is normalised against the cluster's fastest server (1.5 t/s)
    so scores stay in a comparable range.
    """

    # Fastest server speed in the cluster — used to normalise the multiplier.
    # Update this if you change the cluster config in main2.py.
    MAX_SPEED = 1.5

    def __init__(self):
        # ── Antecedents (inputs) ─────────────────────────────────────────────
        self.load  = ctrl.Antecedent(np.arange(0, 101, 1), 'load')
        self.trend = ctrl.Antecedent(np.arange(0, 101, 1), 'trend')

        # ── Consequent (output) ──────────────────────────────────────────────
        self.priority = ctrl.Consequent(np.arange(0, 101, 1), 'priority')

        # ── Membership functions ─────────────────────────────────────────────
        # Load bands
        self.load['low']    = fuzz.trimf(self.load.universe, [0,   0,  40])
        self.load['medium'] = fuzz.trimf(self.load.universe, [20, 50,  80])
        self.load['high']   = fuzz.trimf(self.load.universe, [60, 100, 100])

        # Trend bands — tuned so NASA's 5-25% steady traffic sits in 'stable',
        # a 30-60% score hits 'rising', and 65%+ hits 'critical'.
        self.trend['stable']   = fuzz.trimf(self.trend.universe, [0,   0,  35])
        self.trend['rising']   = fuzz.trimf(self.trend.universe, [25,  50, 75])
        self.trend['critical'] = fuzz.trimf(self.trend.universe, [60, 100, 100])

        # Priority output bands
        self.priority['low']    = fuzz.trimf(self.priority.universe, [0,   0,  40])
        self.priority['medium'] = fuzz.trimf(self.priority.universe, [20,  50, 80])
        self.priority['high']   = fuzz.trimf(self.priority.universe, [60, 100, 100])

        # ── Rules ────────────────────────────────────────────────────────────
        # Low load: always welcome new tasks
        r1 = ctrl.Rule(self.load['low'] & self.trend['stable'],   self.priority['high'])
        r2 = ctrl.Rule(self.load['low'] & self.trend['rising'],   self.priority['high'])
        r3 = ctrl.Rule(self.load['low'] & self.trend['critical'], self.priority['high'])

        # Medium load: back off only when a critical spike is incoming
        r4 = ctrl.Rule(self.load['medium'] & self.trend['stable'],   self.priority['medium'])
        r5 = ctrl.Rule(self.load['medium'] & self.trend['rising'],   self.priority['medium'])
        r6 = ctrl.Rule(self.load['medium'] & self.trend['critical'], self.priority['low'])

        # High load: never take more work regardless of trend
        r7 = ctrl.Rule(self.load['high'] & self.trend['stable'],   self.priority['low'])
        r8 = ctrl.Rule(self.load['high'] & self.trend['rising'],   self.priority['low'])
        r9 = ctrl.Rule(self.load['high'] & self.trend['critical'], self.priority['low'])

        self._ctrl = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9])
        self._sim  = ctrl.ControlSystemSimulation(self._ctrl)

    def compute_priority(self, 
                         current_load:    float, 
                         predicted_trend: float, 
                         server_speed:    float = 1.0) -> float:
        
        current_load    = float(np.clip(current_load,    0, 100))
        predicted_trend = float(np.clip(predicted_trend, 0, 100))

        try:
            self._sim.input['load']  = current_load
            self._sim.input['trend'] = predicted_trend
            self._sim.compute()
            fuzzy_out = float(self._sim.output['priority'])
        except Exception:
            # Fallback: priority is inverse of load
            fuzzy_out = 100.0 - current_load

        # --- IMPROVED SCORING LOGIC ---
        # 1. Base Score: This is the logic from the Fuzzy rules (0-100)
        # 2. Speed Bonus: Give faster servers a small nudge, but don't let it 
        #    overpower the fact that a server is full.
        
        # We change the multiplier to be less aggressive:
        # Old: 0.33 for S1. New: 0.7 for S1, 1.0 for S3.
        speed_bonus = (server_speed / self.MAX_SPEED) * 0.3 + 0.7 
        
        # 3. Final Score
        # If fuzzy_out is high (server is empty), speed_bonus helps S3 win.
        # If fuzzy_out is low (server is full), S3's score drops, allowing S1 to win.
        final_score = fuzzy_out * speed_bonus

        return float(np.clip(final_score, 0.0, 100.0))
