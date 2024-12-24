import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from respiration_signal import extract_respiratory_signal
from rppg_signal import extract_rppg_signal, filter_rppg_signal, analyze_rppg_signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
import csv

def analyze_signal(signal, fs):
    """
    Menganalisis sinyal yang diberikan untuk mengekstrak metrik yang relevan seperti frekuensi.

    Parameter:
        signal (numpy.ndarray): Sinyal input yang akan dianalisis.
        fs (float): Sampling rate dari sinyal.

    Mengembalikan:
        dict: Sebuah kamus yang berisi metrik yang dianalisis seperti frekuensi.
    """
    N = len(signal)
    if N < 256:  # Pastikan panjang sinyal cukup untuk FFT
        return {'peak_frequency': 0.0, 'bpm': 0.0}

    # Normalisasi sinyal
    signal = signal - np.mean(signal)

    frekuensi = np.fft.fftfreq(N, 1/fs)
    nilai_fft = np.abs(fft(signal))

    # Temukan frekuensi puncak
    frekuensi_puncak = frekuensi[np.argmax(nilai_fft)]
    
    # Pastikan frekuensi puncak positif
    if frekuensi_puncak < 0:
        frekuensi_puncak = -frekuensi_puncak
    
    # Konversi frekuensi puncak ke detak per menit (BPM)
    bpm = frekuensi_puncak * 60.0

    metrik = {
        'peak_frequency': frekuensi_puncak,
        'bpm': bpm
    }
    return metrik

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def save_analysis_results(respiratory_metrics, rppg_metrics, respiratory_signals, rppg_signals):
    with open('analysis_results.txt', 'w') as f:
        f.write(f"Respiratory Metrics: {respiratory_metrics}\n")
        f.write(f"rPPG Metrics: {rppg_metrics}\n")
        f.write(f"Heart Rate: {rppg_metrics['heart_rate']} bpm\n")
        f.write(f"Respiratory Signal Values: {respiratory_signals[:10]}\n")
        f.write(f"rPPG Signal Values: {rppg_signals[:10]}\n")

    with open('analysis_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Respiratory Metrics', 'rPPG Metrics', 'Heart Rate', 'Respiratory Signal Values', 'rPPG Signal Values']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'Respiratory Metrics': respiratory_metrics,
            'rPPG Metrics': rppg_metrics,
            'Heart Rate': rppg_metrics['heart_rate'],
            'Respiratory Signal Values': respiratory_signals[:10],
            'rPPG Signal Values': rppg_signals[:10]
        })

def main():
    # Inisialisasi feed webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    respiratory_signals = []
    rppg_signals = []
    frames = []

    plt.ion()  # Aktifkan mode interaktif
    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    respiratory_line, = ax1.plot([], [], label='Respiratory Signal')
    rppg_line, = ax2.plot([], [], label='rPPG Signal', color='r')
    ax1.set_title('Respiratory Signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax2.set_title('Remote Photoplethysmography Signal')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    fig.tight_layout()

    try:
        while True:
            # Tangkap frame demi frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Tambahkan frame ke daftar
            frames.append(frame)

            # Proses frame untuk mengekstrak sinyal pernapasan dan rPPG
            if len(frames) >= 30:  # Proses setiap 30 frame
                respiratory_signal = extract_respiratory_signal(frames)
                respiratory_signals.extend(respiratory_signal)  # Rata-rata daftar
                frames = []  # Hapus frame setelah diproses

            rppg_signal = extract_rppg_signal(frame)
            rppg_signals.append(rppg_signal)

            # Perbarui plot
            respiratory_line.set_data(range(len(respiratory_signals)), respiratory_signals)
            rppg_line.set_data(range(len(rppg_signals)), rppg_signals)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            canvas.draw()

            # Konversi plot ke gambar
            plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (4,))
            plot_img = plot_img[:, :, [1, 2, 3]]  # Konversi ARGB ke RGB

            # Ubah ukuran gambar plot agar sesuai dengan tinggi frame webcam
            plot_img_resized = cv2.resize(plot_img, (int(plot_img.shape[1] * frame.shape[0] / plot_img.shape[0]), frame.shape[0]))

            # Gabungkan feed webcam dan plot
            combined_img = np.hstack((frame, plot_img_resized))

            # Tampilkan gambar gabungan
            cv2.imshow('Project', combined_img)

            # Hentikan loop dengan menekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Lepaskan webcam dan tutup jendela
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()  # Matikan mode interaktif

        # Plot sinyal akhir
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(respiratory_signals, label='Respiratory Signal')
        plt.title('Respiratory Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(rppg_signals, label='rPPG Signal', color='r')
        plt.title('Remote Photoplethysmography Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        
        # Simpan figure sebagai file gambar
        plt.savefig('output_figure.png')

        plt.show()

        # Analisis sinyal
        fs = 30.0  # Sampling rate
        respiratory_metrics = analyze_signal(respiratory_signals, fs)
        filtered_rppg_signals = filter_rppg_signal(np.array(rppg_signals), lowcut=0.7, highcut=2.5, fs=fs)
        rppg_metrics = analyze_rppg_signal(filtered_rppg_signals, fs)

        print("Respiratory Metrics:", respiratory_metrics)
        print("rPPG Metrics:", rppg_metrics)

        # Cetak detak jantung
        print(f"Heart Rate: {rppg_metrics['heart_rate']} bpm")

        # Cetak beberapa nilai sinyal untuk debugging
        print("Respiratory Signal Values:", respiratory_signals[:10])
        print("rPPG Signal Values:", rppg_signals[:10])

        # Simpan hasil analisis ke file
        save_analysis_results(respiratory_metrics, rppg_metrics, respiratory_signals, rppg_signals)

if __name__ == "__main__":
    main()