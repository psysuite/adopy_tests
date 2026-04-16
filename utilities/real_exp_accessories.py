import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os


# Global cache for tone WAV file
_tone_wav_cache = {}


def generate_tone(duration_ms, frequency, sample_rate):
    """Generate a pure sine wave tone with specified duration in milliseconds and smooth envelope"""
    samples = int(duration_ms * sample_rate / 1000)
    t = np.arange(samples) / sample_rate
    
    # Generate sine wave with higher amplitude
    tone = 1.0 * np.sin(2 * np.pi * frequency * t)
    
    # Apply smooth envelope (cosine fade in/out) to avoid clicks
    # Fade duration: 10ms at start and end (increased from 5ms)
    fade_samples = int(10 * sample_rate / 1000)
    fade_samples = min(fade_samples, samples // 3)  # Ensure fade doesn't exceed 33% of tone
    
    if fade_samples > 0:
        # Fade in (first fade_samples)
        fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        tone[:fade_samples] *= fade_in
        
        # Fade out (last fade_samples)
        fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        tone[-fade_samples:] *= fade_out
    
    return tone.astype(np.float32)


def get_or_create_tone_wav(duration_ms, frequency, sample_rate):
    """
    Get or create a WAV file for a tone with specified parameters.
    Uses caching to avoid regenerating the same tone multiple times.
    """
    cache_key = (duration_ms, frequency, sample_rate)
    
    if cache_key in _tone_wav_cache:
        return _tone_wav_cache[cache_key]
    
    # Generate tone
    tone = generate_tone(duration_ms, frequency, sample_rate)
    
    # Create temporary WAV file
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, f"tone_{duration_ms}ms_{frequency}hz_{sample_rate}sr.wav")
    
    # Write WAV file (uncompressed PCM)
    sf.write(wav_path, tone, sample_rate, subtype='PCM_16')
    
    # Cache the path
    _tone_wav_cache[cache_key] = wav_path
    
    return wav_path


def create_trial_sequence(onset_ms, tone_duration, total_duration, sample_rate, frequency):
    """
    Create a trial sequence with three 50ms tones:
    - Tone 1 at 0 ms
    - Tone 2 at onset_ms (calculated by algorithm)
    - Tone 3 at 1000 ms
    Returns audio array with precise timing
    """
    # Add padding at start and end to avoid clicks
    padding_ms = 50
    
    # Ensure total duration is long enough to contain the last tone completely plus padding
    min_duration = padding_ms + 1000 + tone_duration + padding_ms
    actual_duration = max(total_duration + 2 * padding_ms, min_duration)

    total_samples = int(actual_duration * sample_rate / 1000)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Load tone from WAV file (cached)
    tone_wav_path = get_or_create_tone_wav(tone_duration, frequency, sample_rate)
    tone, _ = sf.read(tone_wav_path, dtype='float32')
    tone_samples = len(tone)
    
    # Calculate padding offset
    padding_samples = int(padding_ms * sample_rate / 1000)

    # Tone 1 at padding_ms (not at 0 to avoid click)
    start_pos = padding_samples
    audio[start_pos:start_pos + tone_samples] = tone

    # Tone 2 at onset_ms + padding
    onset_samples = int(onset_ms * sample_rate / 1000) + padding_samples
    end_sample = min(onset_samples + tone_samples, total_samples)
    audio[onset_samples:end_sample] = tone[:end_sample - onset_samples]

    # Tone 3 at 1000 ms + padding
    final_onset_samples = int(1000 * sample_rate / 1000) + padding_samples
    end_sample = min(final_onset_samples + tone_samples, total_samples)
    audio[final_onset_samples:end_sample] = tone[:end_sample - final_onset_samples]

    return audio

# ask for 1,2 and returns 0,1
def run_bisection_trial(onset_ms, tone_duration, total_duration, sample_rate, frequency):
    """
    Run a single bisection trial with three 50ms tones:
    - Tone 1 at 0 ms
    - Tone 2 at onset_ms (variable, calculated by algorithm)
    - Tone 3 at 1000 ms
    User indicates if tone 2 is closer to tone 1 (1) or tone 3 (2)
    """
    print(f"Trial onset: {onset_ms:.1f} ms")
    # Create and play trial sequence
    audio = create_trial_sequence(onset_ms, tone_duration, total_duration, sample_rate, frequency)
    sd.play(audio, samplerate=sample_rate, blocking=True)
    sd.wait()  # Extra wait to ensure playback completes

    # Get user response
    while True:
        print("Is the second tone closer to the first (1) or the third (2)? ", end="", flush=True)
        try:
            user_ans = input().strip()
            if user_ans in ("1", "2"):
                return int(user_ans) - 1
            else:
                print("Invalid input. Please enter 1 or2.")
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")
