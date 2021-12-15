
import wave
import heartpy as hp
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore") # Hide the warning at execution


###########################################################################
# Heart Rate Analysis – Time Domain Measurements – Biotechnology
###########################################################################
def heart_rate():
	wf = wave.open("data/heart/beat.wav", "rb")
	#wf = wave.open("data/heart/murmur.wav", "rb")
	#wf = wave.open("data/heart/hb1.wav", "rb")

	# information of the wave
	ch = wf.getnchannels() # Returns number of audio channels (1:mono, 2:stereo).
	wh = wf.getsampwidth() # Returns sample width in bytes.
	fr = wf.getframerate() # Returns sampling frequency.
	fn = wf.getnframes()   # Returns number of audio frames
	time = 1.0 * fn / fr

	print("*** Information of the wave file ***")
	print("Number of channel: ", ch)
	print("Sample width: ", wh)
	print("Sample frequency: ", fr)
	print("Number of audio frames: ", fn)
	print("Play time: ", time)
	print("\n")

	buf = wf.readframes(-1) # read all data
	sig = np.frombuffer(buf, dtype="int16")

	#norm = sig/np.linalg.norm(sig) # normalization of 1d array
	#plt.plot(sig)

	wd, m = hp.process(sig, fr) # wd:"dict" type 
	#print(wd.values())

	hp.plotter(wd, m)

	print("*** Heart Rate Anaysis ***")
	for measure in m.keys():
		print("%s: %f" %(measure, m[measure])) # print the time domain of each parameter

	plt.grid()
	plt.show()



###########################################################################
# Heart Rate Diagnostic Analysis – Biotechnology
# DATA SOURCE: http://www.peterjbentley.com/heartchallenge/
###########################################################################
def heart_rate2():
	# sig:audio time series, sr:sampling rate of sig
	#sig, sr = librosa.load("data/heart/murmur.wav")
	sig, sr = librosa.load("data/heart/murmur2.wav")

	dur = librosa.get_duration(sig)
	print(dur) # 7.935555555555555

	# Plot the original signal
	fig, axs = plt.subplots(3)
	librosa.display.waveplot(sig, sr, ax=axs[0], label="Original Heart Beats")
	axs[0].set_title("Murmur Heart Beats")
	axs[0].set_xlim(0,8)
	axs[0].grid()
	axs[0].legend()

	# Locate note onset events.
	on_env = librosa.onset.onset_strength(y=sig, sr=sr)
	times = librosa.times_like(on_env, sr=sr)
	onset_frames = librosa.onset.onset_detect(onset_envelope=on_env, sr=sr)

	# Display onset
	axs[1].plot(times, on_env, label="Onset Strength")
	axs[1].vlines(times[onset_frames], 0, on_env.max(), color="r", linestyle="--", label="Onsets")
	axs[1].set_title("Onset Strength")
	axs[1].set_xlim(0,8)
	axs[1].grid()
	axs[1].legend()

	# Estimate tempo from onset correlation
	tempo, beats = librosa.beat.beat_track(y=sig, sr=sr)
	onset_env = librosa.onset.onset_strength(y=sig, sr=sr, aggregate=np.median)
	tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,sr=sr)
	axs[2].plot(times, librosa.util.normalize(onset_env), label="Onset strength")
	axs[2].vlines(times[beats], 0, 1, alpha=0.5, color="r", linestyle="--", label="Beats")
	axs[2].set_xlim(0,8)
	axs[2].grid()
	axs[2].legend()

	plt.show()
