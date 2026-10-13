import numpy as np
from scipy.io import wavfile
import os

SR = 44100
BPM = 111
Q = 60.0 / BPM
E = Q / 2.0

NOTES = {
    'B3': 246.94, 'C#4': 277.18, 'D4': 293.66, 'E4': 329.63,
    'F#4': 369.99, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    'C#5': 554.37, 'D5': 587.33, 'E5': 659.25
}

def generate_note(freq, duration):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    fundamental = np.sin(2 * np.pi * freq * t) * 0.60
    harmonic_2 = np.cos(2 * np.pi * freq * 2 * t) * 0.25
    harmonic_3 = np.sin(2 * np.pi * freq * 3 * t) * 0.10
    signal = fundamental + harmonic_2 + harmonic_3
    
    n = len(signal)
    attack = min(int(0.02 * SR), n // 4)
    release = min(int(0.05 * SR), n // 4)
    
    env = np.ones(n)
    if attack: env[:attack] = np.linspace(0, 1, attack)
    if release: env[-release:] = np.linspace(1, 0, release)
    
    return signal * env

def silence(duration):
    return np.zeros(int(SR * duration))

melody_data = [
    ('D4', E), ('E4', E), ('F#4', E), ('G4', E), ('F#4', E), ('E4', E), ('D4', Q),
    ('D4', E), ('E4', E), ('F#4', E), ('G4', E), ('F#4', E), ('E4', E), ('B3', Q),
    ('D4', E), ('E4', E), ('F#4', E), ('G4', E), ('A4', E), ('G4', E), ('F#4', Q),
    ('F#4', E), ('G4', E), ('A4', E), ('B4', E), ('A4', E), ('G4', E), ('F#4', Q),
    ('G4', E), ('A4', E), ('B4', E), ('C#5', E), ('B4', E), ('A4', E), ('G4', Q),
    ('G4', E), ('A4', E), ('B4', E), ('C#5', E), ('D5', E), ('C#5', E), ('B4', Q),
    ('A4', E), ('B4', E), ('C#5', E), ('D5', E), ('C#5', E), ('B4', E), ('A4', Q),
    ('B4', E), ('C#5', E), ('D5', E), ('E5', E), ('D5', E), ('C#5', E), ('B4', Q),
    ('D5', E), ('C#5', E), ('B4', E), ('A4', E), ('G4', Q), ('F#4', Q),
]

print("Начинаем синтез мелодии...")

audio_segments = []
for note_name, duration in melody_data:
    if note_name is None:
        audio_segments.append(silence(duration))
    else:
        freq = NOTES[note_name]
        audio_segments.append(generate_note(freq, duration))

final_audio = np.concatenate(audio_segments)

max_amplitude = np.max(np.abs(final_audio))
if max_amplitude > 0:
    final_audio = final_audio / max_amplitude * 0.85

audio_int16 = np.int16(final_audio * 32767)

subdirectory = "audiofiles"
os.makedirs(subdirectory, exist_ok=True)
output_filename = os.path.join(subdirectory, "melody.wav")

wavfile.write(output_filename, SR, audio_int16)

print("=" * 50)
print("Синтез завершен успешно")
print("=" * 50)
print(f"Файл сохранен: {os.path.abspath(output_filename)}")
print(f"Длительность: {len(final_audio) / SR:.2f} секунд")
print(f"Тональность: Bm | Темп: {BPM} BPM")
print("Метод синтеза: Аддитивный (сумма sin и cos)")
print("=" * 50)