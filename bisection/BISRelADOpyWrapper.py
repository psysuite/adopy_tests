import numpy as np

from bisection.BISADOpyWrapper import BISADOpyWrapper


# manage relative latencies (0-300 ms)
# can use model-derived and externally-defined latencies

class BISRelADOpyWrapper(BISADOpyWrapper):

    """
    get stimulus value [in milliseconds], add random, clip to min/max,  
    optional: X attempts to avoid exclusion windows,  of stimuli around offset latency.
    """
    def get(self, addNoise=True):

        model = self.engine.get_design("optimal")
        stim_q = model["stimulus"]

        if addNoise:
            stim_q += int((np.random.rand() - 0.5) * self.noise_perc * self.range)

        stim_q = np.clip(stim_q, self.min, self.max)  # clip stimulus

        return stim_q

    def set(self, success, response, q_value, stim_value):

        model = {"stimulus": q_value}

        self.model_stim.append(model)
        self.stimuli_ms.append(stim_value)
        self.responses.append(response)
        self.successes.append(success)

        self.engine.update(model, success)
        
