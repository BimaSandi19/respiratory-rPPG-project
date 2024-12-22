import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from respiration_signal import extract_respiratory_signal
from rppg_signal import extract_rppg_signal, filter_rppg_signal, analyze_rppg_signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

def analyze_signal(signal, fs):
    """
    Analyzes the given signal to extract relevant metrics such as frequency.

    Parameters:
        signal (numpy.ndarray): The input signal to analyze.
        fs (float): The sampling rate of the signal.

    Returns:
        dict: A dictionary containing analyzed metrics such as frequency.
    """
    N = len(signal)
    if N < 256:  # Ensure the signal length is sufficient for FFT
        return {'peak_frequency': 0.0, 'bpm': 0.0}

    # Normalize the signal
    signal = signal - np.mean(signal)

    freqs = np.fft.fftfreq(N, 1/fs)
    fft_values = np.abs(fft(signal))

    # Find the peak frequency
    peak_freq = freqs[np.argmax(fft_values)]
    
    # Ensure the peak frequency is positive
    if peak_freq < 0:
        peak_freq = -peak_freq
    
    # Convert peak frequency to beats per minute (BPM)
    bpm = peak_freq * 60.0

    metrics = {
        'peak_frequency': peak_freq,
        'bpm': bpm
    }
    return metrics

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def main():
    # Initialize webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    respiratory_signals = []
    rppg_signals = []
    frames = []

    plt.ion()  # Turn on interactive mode
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
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Append frame to the list
            frames.append(frame)

            # Process the frames to extract respiratory and rPPG signals
            if len(frames) >= 30:  # Process every 30 frames
                respiratory_signal = extract_respiratory_signal(frames)
                respiratory_signals.extend(respiratory_signal)  # Flatten the list
                frames = []  # Clear frames after processing

            rppg_signal = extract_rppg_signal(frame)
            rppg_signals.append(rppg_signal)

            # Update the plots
            respiratory_line.set_data(range(len(respiratory_signals)), respiratory_signals)
            rppg_line.set_data(range(len(rppg_signals)), rppg_signals)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            canvas.draw()

            # Convert plot to image
            plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (4,))
            plot_img = plot_img[:, :, [1, 2, 3]]  # Convert ARGB to RGB

            # Resize plot image to match webcam frame height
            plot_img_resized = cv2.resize(plot_img, (int(plot_img.shape[1] * frame.shape[0] / plot_img.shape[0]), frame.shape[0]))

            # Combine webcam feed and plot
            combined_img = np.hstack((frame, plot_img_resized))

            # Display the combined image
            cv2.imshow('Project', combined_img)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()  # Turn off interactive mode

        # Plot the final signals
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
        plt.show()

        # Analyze the signals
        fs = 30.0  # Sampling rate
        respiratory_metrics = analyze_signal(respiratory_signals, fs)
        filtered_rppg_signals = filter_rppg_signal(np.array(rppg_signals), lowcut=0.7, highcut=2.5, fs=fs)
        rppg_metrics = analyze_rppg_signal(filtered_rppg_signals, fs)

        print("Respiratory Metrics:", respiratory_metrics)
        print("rPPG Metrics:", rppg_metrics)

        # Print heart rate
        print(f"Heart Rate: {rppg_metrics['heart_rate']} bpm")

        # Print some values of the signals for debugging
        print("Respiratory Signal Values:", respiratory_signals[:10])
        print("rPPG Signal Values:", rppg_signals[:10])

if __name__ == "__main__":
    main()