
import argparse

import noise_cancel
import heart_rate

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-f",
		type=str,
		metavar="file",
		dest="file",
		help="The executed file",
	)
	args = parser.parse_args()

	if (args.file == "NC"):
		noise_cancel.noise_cancel()

	elif (args.file == "HR"):
		heart_rate.heart_rate()

	elif (args.file == "HR2"):
		heart_rate.heart_rate2()