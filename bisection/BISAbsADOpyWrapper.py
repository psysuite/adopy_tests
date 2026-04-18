# crea BISRelAdopyWrapper,BISAbsAdopyWrapper


import numpy as np

from bisection.BISADOpyWrapper import BISADOpyWrapper


# manage absolute latencies (e.g. 200-800 ms)
# can use model-derived and externally-defined latencies
# optionally: try (10 times) to avoid presenting stimuli within an exclusion_width large window
class BISAbsADOpyWrapper(BISADOpyWrapper):

    def __init__(self, adoparams=None, taskparams=None):

        super().__init__(adoparams, taskparams)

        self.exclusion_width    = taskparams.get("exclusion_width", 20)  # Exclude ms around offset
        self.pre_offset_nstim   = 0
        self.post_offset_nstim  = 0

    """
    get stimulus value [in milliseconds], add random, clip to min/max,  
    optional: X attempts to avoid exclusion windows,  of stimuli around offset latency.
    """
    def get(self, addNoise=True, exclude_zone=False):

        model   = self.engine.get_design("optimal")
        stim_ms = model["stimulus"]

        if addNoise:
            stim_ms += int((np.random.rand() - 0.5) * self.noise_perc * self.range)

        if exclude_zone:
            stim_ms = self.apply_exclusion_window(stim_ms)

        stim_ms = np.clip(stim_ms, self.min, self.max)  # clip stimulus

        # update counter of pre/post stimuli
        if stim_ms > self.offset:
            self.post_offset_nstim += 1
        else:
            self.pre_offset_nstim += 1

        return stim_ms

    # accept 0/1 response
    # The 2AFC model will learn P(response=1) as a function of stimulus
    # This should be monotonically increasing, with threshold at offset
    def set(self, response, stim_ms, index=-1):

        if response not in (0, 1):
            print("WARNING, response value not valid")
            return

        self.stimuli_ms.append(stim_ms)
        model = {"stimulus": stim_ms}
        self.model_stim.append(model)

        self.engine.update(model, response)

        # Calculate success based on stimulus position relative to offset
        if (self.stimuli_ms[-1] > self.offset and response == 1) or (self.stimuli_ms[-1] < self.offset and response == 0):
            success = 1
        else:
            success = 0

        self.responses.append(response)
        self.successes.append(success)

    # manage exclusion zone
    def apply_exclusion_window(self, stim, max_attempts = 10, multiplier=2):
        for attempt in range(max_attempts):
            # Check if result falls in exclusion zone
            if abs(stim - self.offset) > self.exclusion_width/2:
                break
            # Otherwise, try again with different random noise with width * multiplier
            stim += int((np.random.rand() - 0.5) * multiplier * self.noise_perc * self.range)
        else:
            # If all attempts failed, send it to the excluded limit of the least represented side
            stim = self.offset - self.exclusion_width/2 if self.post_offset_nstim > self.pre_offset_nstim else self.offset + self.exclusion_width / 2
            print(f"Warning: Could not avoid exclusion zone after {max_attempts} attempts, set {stim:.1f}")

        return stim
