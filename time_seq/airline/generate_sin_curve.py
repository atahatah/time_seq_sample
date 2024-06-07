import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# パラメータ設定
sample_rate = 1000  # サンプルレート（Hz）
duration = 2  # 信号の長さ（秒）
frequency = 5  # サイン波の周波数（Hz）
noise_amplitude = 0.5  # ノイズの振幅

# 時間ベクトルの生成
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# サイン波の生成
signal = np.sin(2 * np.pi * frequency * t)

# ノイズの生成
noise = noise_amplitude * np.random.normal(size=t.shape)

# ノイズが乗ったサイン波の生成
noisy_signal = signal + noise

# データフレームの作成
df = pd.DataFrame({"Time": t, "Noisy_Sine_Wave": noisy_signal})

# CSVファイルに出力
csv_file_path = "/work/time_seq/airline/inputs/noisy_sine_wave.csv"
df.to_csv(csv_file_path, index=False)

# 結果のプロット
plt.plot(t, noisy_signal, label="Noisy Sine Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Noisy Sine Wave")
plt.legend()
plt.savefig("/work/time_seq/airline/outputs/noisy_sine_wave.png")
plt.close()
