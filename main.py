import numpy as np
from scipy.io import wavfile


class SoundWaveFactory:
    SAMPLING_RATE = 44100   # Standard audio sampling rate
    MAX_AMPLITUDE = 2 ** 13
    NOTES = {
        '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654, 'a0': 27.50000,
        'a#0': 29.13524, 'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783, 'd0': 36.70810, 'd#0': 38.89087,
        'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930, 'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000,
        'a#1': 58.27047, 'b1': 61.73541, 'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175,
        'e2': 82.40689, 'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000,
        'a#2': 116.5409, 'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324, 'd#2': 155.5635,
        'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977, 'g#3': 207.6523, 'a3': 220.0000,
        'a#3': 233.0819, 'b3': 246.9417, 'c3': 261.6256, 'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270,
        'e4': 329.6276, 'f4': 349.2282, 'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000,
        'a#4': 466.1638, 'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
        'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094, 'a5': 880.0000,
        'a#5': 932.3275, 'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731, 'd5': 1174.659, 'd#5': 1244.508,
        'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978, 'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000,
        'a#6': 1864.655, 'b6': 1975.533, 'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016,
        'e7': 2637.020, 'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000,
        'a#7': 3729.310, 'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636, 'd#7': 4978.032,
    }

    def __init__(self, duration_seconds=5):
        self.duration_seconds = duration_seconds
        self.sound_array_len = self.SAMPLING_RATE * duration_seconds
        self.common_timeline = np.linspace(0, duration_seconds, num=self.sound_array_len)

    def get_normed_sin(self, frequency):
        return self.MAX_AMPLITUDE * np.sin(2 * np.pi * frequency * self.common_timeline)

    def get_soundwave(self, note):
        return self.get_normed_sin(self.NOTES[note])

    def create_note(self, note="a4", name=None, wave_type="sine"):
        sound_wave = self.get_wave(note, wave_type).astype(np.int16)
        if name is None:
            file_name = f"{note}_sin.wav".replace("#", "s")
        else:
            file_name = f"{name}.wav"
        wavfile.write(file_name, self.SAMPLING_RATE, sound_wave)
        return sound_wave

    def get_wave(self, note, wave_type="sine"):
        if wave_type == "sine":
            return self.get_soundwave(note)
        elif wave_type == "square":
            return self.get_square_wave(note)
        elif wave_type == "triangle":
            return self.get_triangle_wave(note)
        else:
            raise ValueError(f"Unknown wave_type: {wave_type}")

    def get_square_wave(self, note):
        frequency = self.NOTES[note]
        return self.MAX_AMPLITUDE * np.sign(np.sin(2 * np.pi * frequency * self.common_timeline))

    def get_triangle_wave(self, note):
        frequency = self.NOTES[note]
        return self.MAX_AMPLITUDE * (2 * np.arcsin(np.sin(2 * np.pi * frequency * self.common_timeline)))

    def read_wave_from_txt(self, file_name):
        """ This method reads a wave from .txt file """
        return np.loadtxt(file_name)

    def print_wave_details(self, wave_data):
        """ Prints the important details of the wave """
        print(f"Length of wave: {len(wave_data)}; Max amplitude: {np.max(wave_data)}; Min amplitude: {np.min(wave_data)}")

    def normalize_sound_waves(self, *waves):
        """ normalizes an arbitrary number of sound waves in both length and amplitude."""
        # Normalize length
        min_len = min(map(len, waves))
        normalized_waves = [wave[:min_len] for wave in waves]

        # Normalize amplitude
        max_amp = max([np.max(np.abs(wave)) for wave in normalized_waves])
        normalized_waves = [wave * (self.MAX_AMPLITUDE / max_amp) for wave in normalized_waves]

        return normalized_waves

    def save_wave(self, wave_data, file_name, file_type="txt"):
        if file_type == "txt":
            np.savetxt(file_name, wave_data)
        elif file_type == "wav":
            wavfile.write(file_name, self.SAMPLING_RATE, wave_data.astype(np.int16))
        else:
            raise ValueError("Unknown file_type. Use 'txt' or 'wav'.")

    def combine_waves(self, *waves):
        combined_wave = np.concatenate(waves)
        return combined_wave

    def generate_melody(self, melody_str):
        melody = []
        parts = melody_str.split()
        i = 0
        print(parts)
        while i < len(parts):
            print(parts[i], parts[i+1])
            if parts[i] in self.NOTES:
                # Handle individual notes
                duration = float(parts[i + 1].replace('s', ''))
                wave = self.get_soundwave(parts[i])
                length = int(self.SAMPLING_RATE * duration)
                melody.append(wave[:length])
                i += 2  # Move to the next note and duration
            elif parts[i].startswith("("):  # Chord handling
                chord = []
                i += 1  # Skip the '(' part
                while not parts[i].endswith(")"):
                    chord.append(parts[i])
                    i += 1
                chord.append(parts[i].rstrip(")"))  # Add the last note of the chord

                # Get the duration after the chord
                duration = float(parts[i + 1].replace('s', ''))
                chord_waves = [self.get_soundwave(note)[:int(self.SAMPLING_RATE * duration)] for note in chord]
                combined_chord_wave = np.mean(chord_waves, axis=0)  # Combine the waves by averaging
                melody.append(combined_chord_wave)
                i += 2  # Skip past the chord and duration
            else:
                raise ValueError(f"Invalid note or chord: {parts[i]}")

        return self.combine_waves(*melody)


if __name__ == "__main__":
    factory = SoundWaveFactory()

    a4_wave = factory.create_note()  # Create a note
    factory.print_wave_details(a4_wave)  # Print details

    c4_wave = factory.create_note("c4")  # Normalize two waves
    normalized_waves = factory.normalize_sound_waves(a4_wave, c4_wave)

    # Save to file
    factory.save_wave(normalized_waves[0], "a4_normalized.txt")
    factory.save_wave(normalized_waves[0], "a4_normalized.wav", file_type="wav")

    # Generate melody with a chord
    melody_wave = factory.generate_melody("g4 0.2s b4 0.2s (g3 d5 g5) 0.5s")
    factory.save_wave(melody_wave, "melody.wav", file_type="wav")
