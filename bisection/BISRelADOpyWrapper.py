import numpy as np

from bisection.BISADOpyWrapper import BISADOpyWrapper


# manage relative latencies (0-300 ms)
# can use model-derived and externally-defined latencies

class BISRelADOpyWrapper(BISADOpyWrapper):

    """
    get stimulus value [in milliseconds], add random, clip to min/max,  
    optional: X attempts to avoid exclusion windows,  of stimuli around offset latency.
    """
    def get(self, is_pre = None, addNoise=True):

        model = self.engine.get_design("optimal")
        stim_q = model["stimulus"]

        if addNoise:
            stim_q += int((np.random.rand() - 0.5) * self.noise_perc * self.range)

        stim_q = np.clip(stim_q, self.min, self.max)  # clip stimulus

        stim_ms = self.offset - stim_q if is_pre is True else self.offset + stim_q


        model["stimulus"] = stim_q

        self.stimuli_ms.append(stim_ms)
        self.model_stim.append(model)

        return stim_q


    def set(self, success, response, q_value=None, index=-1):

        # if q_value is given, crete a new model_stim
        if q_value is not None:
            model = {"stimulus": q_value}
            self.model_stim.append(model)
            index = -1  # Use the last element just added
        else:
            # if q_value is not given, model_stim cannot be empty
            if len(self.model_stim) == 0:
                print("ERROR: No stimulus model available. Call get_rel() first or provide q_value.")
                raise Exception("ERROR: No stimulus model available. Call get_rel() first or provide q_value.")
        
        model = self.model_stim[index]
        self.engine.update(model, success)
        
        self.successes.append(success)
        self.responses.append(response)
