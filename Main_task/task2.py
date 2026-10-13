import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.io import wavfile
import sounddevice as sd
import qrcode
import os

f = 5

f_max = 3 * f
fs_min = 2 * f_max

print("=" * 60)
print(" ПРАКТИЧЕСКАЯ РАБОТА №1. Все задания (1-14) + доп.")
print("=" * 60)

t_cont = np.linspace(0, 1, 1000)
y_cont = np.sin(np.pi * f * t_cont + np.pi / 6) + np.sin(6 * np.pi * f * t_cont)

plt.figure(figsize=(10, 4))
plt.plot(t_cont, y_cont, 'b-', linewidth=1.5, label='y(t) = sin(5*pi*t+pi/6) + sin(30*pi*t)')
plt.title('Задание 1: Непрерывный сигнал')
plt.xlabel('Время t (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('task1_continuous_signal.png', dpi=150)
print("График Задания 1 сохранен: task1_continuous_signal.png")
plt.close()

print(f"\n[Задание 2] Максимальная частота в спектре: {f_max} Гц")
print(f"[Задание 3] Минимальная частота дискретизации (Котельников): {fs_min} Гц")

def process_signal(fs, label):
    N = int(fs * 1)
    t_disc = np.linspace(0, 1, N, endpoint=False)
    y_disc = np.sin(np.pi * f * t_disc + np.pi / 6) + np.sin(6 * np.pi * f * t_disc)

    Y = fft(y_disc)
    freqs = fftfreq(N, 1 / fs)
    mags = np.abs(Y)
    threshold = np.max(mags) * 0.01
    sig_freqs = np.abs(freqs[mags > threshold])
    spec_width = (np.max(sig_freqs) - np.min(sig_freqs)) if len(sig_freqs) > 0 else 0

    mem_bytes = y_disc.nbytes
    mem_bits = mem_bytes * 8

    interp_func = interp1d(t_disc, y_disc, kind='linear', fill_value='extrapolate')
    y_rec = interp_func(t_cont)
    mse = np.mean((y_cont - y_rec) ** 2)

    print(f"\n--- {label} (fs = {fs} Гц) ---")
    print(f"  Отсчетов за 1 сек: {N}")
    print(f"  Ширина спектра: {spec_width:.2f} Гц")
    print(f"  Память: {mem_bytes} байт ({mem_bits} бит), тип: {y_disc.dtype}")
    print(f"  MSE восстановления: {mse:.6f}")

    return (t_disc, y_disc, freqs[:N // 2], mags[:N // 2] * 2 / N, y_rec)

t30, y30, f30, m30, y_rec30 = process_signal(fs_min, "Базовая дискретизация")
t120, y120, f120, m120, y_rec120 = process_signal(fs_min * 4, "Увеличенная дискретизация (x4)")

print("\n" + "=" * 60)
print(" ЗАДАНИЯ 8-12: АНАЛИЗ ГОЛОСОВОГО СИГНАЛА")
print("=" * 60)

subdirectory = "audiofiles" 
voice_file = os.path.join(subdirectory, "voice.wav")

if not os.path.exists(voice_file):
    print(f"Файл '{voice_file}' не найден. Генерирую тестовый сигнал.")
    print("Замените его на реальную запись голоса для защиты работы!")
    t_voice = np.linspace(0, 2, 88200, endpoint=False)
    y_voice_float = (0.5 * np.sin(2 * np.pi * 150 * t_voice) +
                     0.3 * np.sin(2 * np.pi * 300 * t_voice) +
                     0.2 * np.random.randn(len(t_voice)))
    y_voice = np.int16(y_voice_float * 32767)
    wavfile.write(voice_file, 44100, y_voice)

Fs, y_voice_raw = wavfile.read(voice_file)
if len(y_voice_raw.shape) > 1:
    y_voice_raw = y_voice_raw[:, 0]

y_voice = y_voice_raw.astype(np.float64)
if y_voice_raw.dtype == np.int16:
    y_voice = y_voice / 32768.0

duration = len(y_voice) / Fs
print(f"[Задание 9] Fs из метаданных: {Fs} Гц, длительность: {duration:.2f} с")

Fs_calc = len(y_voice) / duration
print(f"[Задание 10] Вычисленная Fs = {len(y_voice)} / {duration:.4f} = {Fs_calc:.1f} Гц. Совпадает: {np.isclose(Fs, Fs_calc)}")

factor = 10
y_pruned = y_voice[::factor]
Fs_pruned = Fs / factor
print(f"[Задание 11] Прореживание в {factor} раз. Новая Fs: {Fs_pruned} Гц")

print("Воспроизведение прореженного сигнала (3 с)...")
try:
    sd.play(y_pruned[:int(Fs_pruned * 3)], int(Fs_pruned))
    sd.wait()
    print("Воспроизведение завершено.")
except Exception as e:
    print(f"Ошибка воспроизведения: {e}")

def get_spectrum(sig, fs):
    N = len(sig)
    window = np.hanning(N)
    Y = fft(sig * window)
    freqs = fftfreq(N, 1 / fs)
    return freqs[:N // 2], 2.0 / N * np.abs(Y[:N // 2])

f_orig, m_orig = get_spectrum(y_voice, Fs)
f_pruned, m_pruned = get_spectrum(y_pruned, Fs_pruned)

threshold_voice = 0.02 * np.max(m_orig)
f_max_voice = np.max(f_orig[m_orig > threshold_voice])
print(f"[Задание 12] Максимальная значимая частота голоса: ~{f_max_voice:.0f} Гц")
print(f"Рекомендуемая Fs по Котельникову: ~{2 * f_max_voice:.0f} Гц")

print("\n" + "=" * 60)
print(" ЗАДАНИЕ 13: ВЛИЯНИЕ РАЗРЯДНОСТИ АЦП (квантование по уровню)")
print("=" * 60)

def quantize_signal(signal, bits):
    levels = 2 ** bits
    y_min, y_max = np.min(signal), np.max(signal)
    y_norm = (signal - y_min) / (y_max - y_min)
    y_q_norm = np.round(y_norm * (levels - 1)) / (levels - 1)
    return y_q_norm * (y_max - y_min) + y_min

y_test = y120
print("Разрядность | Уровней | Средняя ошибка квантования (MAE)")
print("-" * 55)

plt.figure(figsize=(12, 5))
for bits in [3, 4, 5, 6]:
    y_q = quantize_signal(y_test, bits)
    error = np.mean(np.abs(y_test - y_q))
    print(f"   {bits} бит   |   {2 ** bits:<7} |   {error:.6f}")

    if bits in [3, 6]:
        Y_q = fft(y_q)
        freqs_q = fftfreq(len(y_q), 1 / 120)
        mags_q = np.abs(Y_q) * 2 / len(y_q)
        plt.plot(freqs_q[:len(y_q) // 2], mags_q[:len(y_q) // 2],
                 label=f'Квантование {bits} бит', alpha=0.8)

Y_orig = fft(y_test)
mags_orig = (np.abs(Y_orig) * 2 / len(y_test))[:len(y_test) // 2]
plt.plot(f120, mags_orig, 'k--', linewidth=2, label='Исходный (без квантования)')

plt.title('Задание 13: Влияние квантования на спектр сигнала (fs = 120 Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, 25)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('task13_quantization_spectrum.png', dpi=150)
print("График сохранен: task13_quantization_spectrum.png")
plt.close()

print("\n[Задание 14] Генерация сводных графиков для отчета...")

fig = plt.figure(figsize=(15, 12))

ax1 = plt.subplot(3, 3, 1)
ax1.plot(t_cont, y_cont, 'b-', alpha=0.5, label='Оригинал')
ax1.stem(t30, y30, linefmt='r-', markerfmt='ro', basefmt=' ', label='Отсчеты (30 Гц)')
ax1.set_title('Зад. 4: Дискретизация по Котельникову (30 Гц)')
ax1.legend()
ax1.grid(True)

ax2 = plt.subplot(3, 3, 2)
ax2.stem(f30, m30, linefmt='b-', markerfmt='bo', basefmt=' ')
ax2.set_title('Зад. 5: Спектр (30 Гц)')
ax2.set_xlim(0, 25)
ax2.grid(True)

ax3 = plt.subplot(3, 3, 3)
ax3.plot(t_cont, y_cont, 'b-', alpha=0.5, label='Оригинал')
ax3.stem(t120[::4], y120[::4], linefmt='g-', markerfmt='go', basefmt=' ',
         label='Отсчеты (120 Гц, каждый 4-й)')
ax3.set_title('Зад. 7: Увеличенная дискретизация (120 Гц)')
ax3.legend()
ax3.grid(True)

ax4 = plt.subplot(3, 3, 4)
ax4.stem(f120, m120, linefmt='g-', markerfmt='go', basefmt=' ')
ax4.set_title('Зад. 7: Спектр (120 Гц)')
ax4.set_xlim(0, 25)
ax4.grid(True)

ax5 = plt.subplot(3, 3, 5)
ax5.plot(t_cont, y_cont, 'b-', linewidth=2, label='Оригинал')
ax5.plot(t_cont, y_rec30, 'r--', linewidth=1, label='Восстановленный')
ax5.set_title(f'Зад. 6: Восстановление (MSE={np.mean((y_cont - y_rec30) ** 2):.4f})')
ax5.legend()
ax5.grid(True)

ax6 = plt.subplot(3, 3, 6)
ax6.plot(t_cont, y_cont, 'b-', linewidth=2, label='Оригинал')
ax6.plot(t_cont, y_rec120, 'g--', linewidth=1, label='Восстановленный')
ax6.set_title(f'Зад. 7: Восстановление (MSE={np.mean((y_cont - y_rec120) ** 2):.4f})')
ax6.legend()
ax6.grid(True)

ax7 = plt.subplot(3, 3, 7)
ax7.plot(f_orig, m_orig, 'b-', linewidth=0.5)
ax7.set_title(f'Зад. 12: Спектр голоса (Fs = {Fs} Гц)')
ax7.set_xlim(0, 4000)
ax7.grid(True)

ax8 = plt.subplot(3, 3, 8)
ax8.plot(f_pruned, m_pruned, 'r-', linewidth=0.5)
ax8.set_title(f'Зад. 12: Спектр прореженного (Fs = {Fs_pruned} Гц)')
ax8.set_xlim(0, Fs_pruned / 2)
ax8.grid(True)

plt.tight_layout()
plt.savefig('lab1_full_report_plots.png', dpi=150, bbox_inches='tight')
print("Сводные графики сохранены: lab1_full_report_plots.png")
plt.close()

print("\n" + "=" * 60)
print(" ВСЕ ЗАДАНИЯ (1-14) ВЫПОЛНЕНЫ УСПЕШНО!")
print(" Проверьте папку со скриптом на наличие сохраненных изображений.")
print("=" * 60)