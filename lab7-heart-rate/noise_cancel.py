"""
> python3 lab7_main.py -f NC
"""

import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy import fftpack

import warnings
warnings.filterwarnings("ignore") # Hide the warning at execution

def noise_cancel():
	#wf = wave.open("data/audio/dual.wav", "rb")
	wf = wave.open("data/audio/noise.wav", "rb")

	# information of the wave
	ch = wf.getnchannels() # Returns number of audio channels (1:mono, 2:stereo).
	wd = wf.getsampwidth() # Returns sample width in bytes.
	fr = wf.getframerate() # Returns sampling frequency.
	fn = wf.getnframes()   # Returns number of audio frames
	time = 1.0 * fn / fr

	print("Original waveform")
	print("Number of channel: ", ch)
	print("Sample width: ", wd)
	print("Sample frequency: ", fr)
	print("Number of audio frames: ", fn)
	print("Play time: ", time)
	print("\n")

	fig, axs = plt.subplots(2)

	buf = wf.readframes(1000) # read n frames out of 132301 frames
	#buf = wf.readframes(-1)	# read all points
	sig = np.frombuffer(buf, dtype="int16") # 16bit
	axs[0].plot(sig, label="Original Signal")

	# Plot FFT of the original signal
	sig_fft = fftpack.fft(sig)
	power = np.abs(sig_fft) ** 2
	sample_freq = fftpack.fftfreq(sig.size)
	axs[1].plot(sample_freq, power)
	axs[1].set_ylabel("Power")
	axs[1].set_xlabel("Frequency [Hz]")

	# Find the peak frequency
	pos_mask = np.where(sample_freq > 0)
	freqs = sample_freq[pos_mask]
	peak_freq = freqs[power[pos_mask].argmax()]

	# Remove all the high frequency
	high_freq_fft = sig_fft.copy()
	high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
	filtered_sig = fftpack.ifft(high_freq_fft)
	axs[0].plot(filtered_sig, label="Filtered Signal")

	# Plot FFT
	sig_fft1 = fftpack.fft(filtered_sig)
	power1 = np.abs(sig_fft1) ** 2
	sample_freq1 = fftpack.fftfreq(filtered_sig.size)
	axs[1].plot(sample_freq1, power1)
	axs[1].set_ylabel("Power")
	axs[1].set_xlabel("Frequency [Hz]")

	# Generate a filtered signal as wav
	ww = wave.open("data/audio/filtered.wav", "wb")
	ww.setnchannels(ch) # Set the number of channels
	ww.setsampwidth(wd) # Set the sample width to n bytes
	ww.setframerate(fr) # Set the frame rate to n
	ww.setnframes(fn)   # Set the number of frames to n
	ww.writeframes(filtered_sig) # Write audio frames

	# information of the generated wave (filtered)
	ch_fil = ww.getnchannels() # Returns number of audio channels (1:mono, 2:stereo).
	wd_fil = ww.getsampwidth() # Returns sample width in bytes.
	fr_fil = ww.getframerate() # Returns sampling frequency.
	fn_fil = ww.getnframes()   # Returns number of audio frames
	time1 = 1.0 * fn / fr

	print("Filtered waveform")
	print("Number of channel: ", ch_fil)
	print("Sample width: ", wd_fil)
	print("Sample frequency: ", fr_fil)
	print("Number of audio frames: ", fn_fil)
	print("Play time: ", time1)

	plt.show()
