import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from adopy.tasks.psi import Task2AFC, ModelLogistic
from adopy.tasks.psi import EnginePsi
from scipy.stats import bernoulli


# base class for temporal bisection task
class BISADOpyWrapper:

    def __init__(self, adoparams=None, taskparams=None):

        self.min            = taskparams.get("min", 150)
        self.max            = taskparams.get("max", 850)
        self.ntrials        = taskparams.get("ntrials", 50)
        self.offset         = taskparams.get("offset", 500)

        self.range          = self.max - self.min

        self.guess_rate = adoparams["guess_rate"]
        self.lapse_rate = adoparams["lapse_rate"]
        self.noise_perc = adoparams["noise_perc"]

        self.model_stim = []
        self.stimuli_q  = []
        self.responses  = []
        self.stimuli_ms = []
        self.successes  = []

        self.task   = Task2AFC()
        self.model  = ModelLogistic()

        # Create continuous stimulus design space (no exclusion zone)
        self.designs = {
            'stimulus': np.linspace(
                self.min,
                self.max,
                200
            )
        }
        self.params = {
            'guess_rate': [self.guess_rate],
            'lapse_rate': [self.lapse_rate],
            'threshold' : np.linspace(
                                    self.min,
                                    self.max,
                                    200),
            'slope'     : np.logspace(-2, 1, 200)  # 0.01 to 1.0, emphasizes shallow slopes
        }

        self.engine = EnginePsi(self.model, self.designs, self.params)

    def gausFit(self, binSize):
        """ Fit cumulative Gaussian to response data """

        # Use stimuli_ms and successes for absolute mode
        if len(self.stimuli_ms) == 0:
            raise ValueError("No data collected yet. Run experiment first.")

        SR          = sorted(list(zip(self.stimuli_ms, self.responses)))
        bins        = [i * binSize for i in range(math.floor(SR[0][0] / binSize), math.ceil(SR[-1][0] / binSize) + 1)]
        x           = [s[0] for s in SR]
        i_binned    = np.digitize(x, bins)
        x_binned    = np.asarray(bins)[i_binned - 1]
        r           = np.asarray([sr[1] for sr in SR])
        f           = [sum(r[x_binned == b]) / len(r[x_binned == b]) if len(r[x_binned == b]) > 0 else math.nan for b in bins]
        goodx       = [not math.isnan(y) for y in f]
        f           = np.asarray(f)[goodx]
        bins        = np.asarray(bins)[goodx]

        # Fit cumulative Gaussian
        try:
            # Better initial guess: threshold at offset, sigma as reasonable fraction of range
            p0 = [self.offset, self.range / 10]
            # Add bounds to prevent unreasonable fits
            bounds = ([self.offset - self.range, 1], [self.offset + self.range, self.range])
            mu1, sigma1 = curve_fit(norm.cdf, bins, f, p0=p0, bounds=bounds, maxfev=10000)[0]
        except Exception as e:
            print(f"Fit failed: {e}")
            # If fit fails, return mean and std of stimuli
            mu1 = np.mean(self.stimuli_ms)
            sigma1 = np.std(self.stimuli_ms)

        return mu1, abs(sigma1)  # Ensure sigma is positive

    def plot_psychometric(self, outfile_name, binSize=10):
        """ Plot psychometric function with fitted curve """

        if len(self.stimuli_ms) == 0:
            print("No data to plot")
            return None

        # Compute binned data
        SR = sorted(list(zip(self.stimuli_ms, self.responses)))
        bins = [i * binSize for i in range(math.floor(SR[0][0] / binSize), math.ceil(SR[-1][0] / binSize) + 1)]
        x = [s[0] for s in SR]
        i_binned = np.digitize(x, bins)
        x_binned = np.asarray(bins)[i_binned - 1]
        r = np.asarray([sr[1] for sr in SR])
        f = [sum(r[x_binned == b]) / len(r[x_binned == b]) if len(r[x_binned == b]) > 0 else math.nan for b in bins]
        goodx = [not math.isnan(y) for y in f]
        f = np.asarray(f)[goodx]
        bins = np.asarray(bins)[goodx]

        # Get fitted parameters
        mu1, sigma1 = self.gausFit(binSize)

        # Plot
        plt.figure(figsize=(10, 6))
        x_fit = np.linspace(min(self.stimuli_ms) - 50, max(self.stimuli_ms) + 50, 200)
        plt.plot(x_fit, norm.cdf(x_fit, mu1, sigma1), 'b-', alpha=0.7, label=f'Fit: μ={mu1:.1f}, σ={sigma1:.1f}')
        plt.plot(bins, f, 'ro', markersize=8, alpha=0.7, label='Data')
        plt.axvline(self.offset, color='g', linestyle='--', label=f'True threshold ({self.offset}ms)')
        plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Stimulus time (ms)')
        plt.ylabel('P(response = 2)')
        plt.title(outfile_name)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(outfile_name, bbox_inches='tight')
        plt.close()

        return mu1, sigma1

    def print_statistics(self):
        """
        Calculate and return statistics about collected data as a dictionary.
        
        Returns:
            Dictionary with statistics or empty dict if no data
        """
        stats = {}
        
        if len(self.stimuli_ms) == 0:
            print("No data collected yet")
            return stats
        
        stimuli = np.array(self.stimuli_ms)
        below_offset = stimuli < self.offset
        above_offset = stimuli > self.offset
        
        stats['total_trials'] = len(stimuli)
        stats['offset'] = self.offset
        stats['trials_below_offset'] = int(np.sum(below_offset))
        stats['trials_above_offset'] = int(np.sum(above_offset))
        stats['pct_below_offset'] = float(100 * np.sum(below_offset) / len(stimuli))
        stats['pct_above_offset'] = float(100 * np.sum(above_offset) / len(stimuli))
        stats['stim_min'] = float(stimuli.min())
        stats['stim_max'] = float(stimuli.max())
        stats['stim_mean'] = float(stimuli.mean())
        stats['stim_median'] = float(np.median(stimuli))
        stats['stim_std'] = float(stimuli.std())
        
        if len(self.successes) > 0:
            successes = np.array(self.successes)
            stats['response_accuracy'] = float(100 * successes.mean())
        
        # Print for console visibility
        print("\n" + "="*50)
        print("EXPERIMENT STATISTICS")
        print("="*50)
        print(f"Total trials: {stats['total_trials']}")
        print(f"Offset (true threshold): {stats['offset']} ms")
        print(f"\nStimulus distribution:")
        print(f"  Below {stats['offset']}ms: {stats['trials_below_offset']} trials ({stats['pct_below_offset']:.1f}%)")
        print(f"  Above {stats['offset']}ms: {stats['trials_above_offset']} trials ({stats['pct_above_offset']:.1f}%)")
        print(f"\nStimulus range: {stats['stim_min']:.1f} - {stats['stim_max']:.1f} ms")
        print(f"Mean stimulus: {stats['stim_mean']:.1f} ms")
        print(f"Median stimulus: {stats['stim_median']:.1f} ms")
        print(f"Std deviation: {stats['stim_std']:.1f} ms")
        if 'response_accuracy' in stats:
            print(f"\nResponse accuracy: {stats['response_accuracy']:.1f}%")
        print("="*50 + "\n")
        
        return stats

    def get_simulated_response(self, model, design):
        # Compute a probability to respond positively.
        p_obs = model.compute(choice=[[[0, 1]]], stimulus=design["stimulus"],guess_rate=0.5, lapse_rate=0.04,threshold=20, slope=1.5)
        # Compute a random binary choice
        return bernoulli.rvs(p_obs)