import dataclasses

import numpy as np
import pandas as pd
import scipy.stats

from .dataset_manager import DatasetManager

@dataclasses.dataclass
class BitMix(DatasetManager):
	project_root: str = None

	# --- Constants ---
	dataset_name: str = dataclasses.field(init=False, default="BitMix")
	n_training_samples: int = dataclasses.field(init=False, default=100000, repr=False)
	n_testing_samples1: int = dataclasses.field(init=False, default=1000, repr=False)
	n_testing_samples2: int = dataclasses.field(init=False, default=1000, repr=False)
	training_random_state: int = dataclasses.field(init=False, default=0, repr=False)
	testing_samples1: int = dataclasses.field(init=False, default=1, repr=False)
	testing_samples2: int = dataclasses.field(init=False, default=2, repr=False)

	# --- Filenames ---
	TRAINING_FILE: str = dataclasses.field(init=False, default="training.csv", repr=False)
	TESTING_FILE1: str = dataclasses.field(init=False, default="is_testing.csv", repr=False)
	TESTING_FILE2: str = dataclasses.field(init=False, default="oos_testing.csv", repr=False)

	def __post_init__(self, project_root=None):
		super().__post_init__()

	# --- Generate Download ---
	def generate_dataset(self, *args, **kwargs):
		self.log("Generating synthetic data...")

		df = BitMix.generate_training_data(self.n_training_samples, random_state=self.training_random_state)
		path = self.get_raw_file_path(self.TRAINING_FILE)
		self._save_csv_dataframe(
			df, path, append=False, index=False, suppress_logs=False,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)

		df = BitMix.generate_testing_data1(self.n_testing_samples1, random_state=self.testing_samples1)
		path = self.get_raw_file_path(self.TESTING_FILE1)
		self._save_csv_dataframe(
			df, path, append=False, index=False, suppress_logs=False,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)

		df = BitMix.generate_testing_data2(self.n_testing_samples2, random_state=self.testing_samples2)
		path = self.get_raw_file_path(self.TESTING_FILE2)
		self._save_csv_dataframe(
			df, path, append=False, index=False, suppress_logs=False,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)

	# --- Raw Data Loaders ---
	def load_data(self, verbose=True):
		return (
			self.load_training_data(verbose=verbose),
			self.load_testing_data1(verbose=verbose),
			self.load_testing_data2(verbose=verbose)
		)

	def load_training_data(self, verbose=True):
		return self.load_raw_dataframe(
			self.TRAINING_FILE, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_testing_data1(self, verbose=True):
		return self.load_raw_dataframe(
			self.TESTING_FILE1, header=0, index_col=None, nrows=None, verbose=verbose
		)

	def load_testing_data2(self, verbose=True):
		return self.load_raw_dataframe(
			self.TESTING_FILE2, header=0, index_col=None, nrows=None, verbose=verbose
		)

	# --- Data Generation --- 
	@staticmethod
	def generate_features(n_samples, dist_params, random_state=None):
		"""
		dist_params : dict
			A mapping of feature names to distribution settings.
			Each entry should include a `"type"` key and any parameters
			required for that distribution. Supported types:
			  - "normal": {"type": "normal", "mean": mu, "std": sigma}
			  - "uniform": {"type": "uniform", "low": a, "high": b}
			  - "skew": {"type": "skew", "a": shape, "mean": mu, "std": sigma}
			Example:
				{
					"A": {"type": "normal", "mean": 0, "std": 1},
					"B": {"type": "uniform", "low": -2, "high": 2},
					"C": {"type": "skew", "a": 5, "mean": 0, "std": 1}
				}
		"""
		random = np.random.default_rng(random_state)
		features = {}

		for feature, spec in dist_params.items():
			dist_type = spec.get("type", "normal").lower()
			if dist_type == "normal":
				mean = spec.get("mean", 0)
				std = spec.get('std', 1)
				features[feature] = random.normal(mean, std, size=n_samples)
			elif dist_type == "uniform":
				low = spec.get("low", 0)
				high = spec.get("high", 1)
				features[feature] = random.uniform(low, high, size=n_samples)
			elif dist_type == "skew":
				a = spec.get("a", 0)
				mean = spec.get("mean", 0)
				std = spec.get("std", 1)
				samples = scipy.stats.skewnorm.rvs(a, size=n_samples, random_state=random)
				samples = mean + std * (samples - np.mean(samples)) / np.std(samples)
				features[feature] = samples
			else:
				raise ValueError(f"Unsupported distribution type: {feature}")

		return features

	@staticmethod
	def generate_mixer(n_samples, n_bits=4, dist_params=None, random_state=None):
		"""
		dist_params : dict or None
			Distribution specification. Supported types:
			  - {"type": "uniform"} → uniform over [0, 2**n_bits - 1]
			  - {"type": "choice", "values": [...], "p": [...]} - choose from a list of integers with optional probabilities
			  - {"type": "skewed", "bias": float} → skewed distribution favoring small or large numbers
		"""
		random = np.random.default_rng(random_state)
		max_val = 2 ** n_bits

		if dist_params is None:
			dist_params = {"type": "uniform"}

		dist_type = dist_params.get("type", "uniform").lower()

		if dist_type == "uniform":
			values = random.integers(0, max_val, size=n_samples)
		elif dist_type == "choice":
			values = np.asarray(dist_params.get("values", np.arange(max_val)))
			probs = dist_params.get("p", None)
			values = random.choice(values, size=n_samples, p=probs)
		elif dist_type == "skewed":
			bias = dist_params.get("bias", 2.0)
			xx = np.arange(max_val)
			if bias >= 0:
				probs = (max_val - xx) ** bias
			else:
				probs = (xx + 1) ** (-bias)
			probs = probs / probs.sum()
			values = random.choice(xx, size=n_samples, p=probs)
		else:
			raise ValueError(f"Unsupported distribution type: {dist_type}")

		return values.astype(int)

	@staticmethod
	def get_bit_matrix(F, n_bits=4):
		F = np.asarray(F, dtype=int)
		bit_positions = np.arange(n_bits)
		bit_matrix = (F[:, None] >> bit_positions) & 1
		return bit_matrix.astype(int)
	
	@staticmethod
	def generate_data(n_samples, dist_params, random_state=None):
		features = BitMix.generate_features(n_samples, dist_params, random_state=random_state)
		features["x5"] = BitMix.generate_mixer(
			n_samples, n_bits=4, dist_params={"type": "uniform"}, random_state=random_state
		)

		values = np.stack([features["x1"], features["x2"], features["x3"], features["x4"]], axis=1)
		bit_matrix = BitMix.get_bit_matrix(features["x5"], len(dist_params))
		features["y"] = (bit_matrix * values).sum(axis=1)

		return pd.DataFrame(features)

	@staticmethod
	def generate_in_sample_data(n_samples, random_state=None):
		dist_params = {
			"x1": {"type": "uniform", "low": -1, "high": 1},
			"x2": {"type": "normal", "mean": 0, "std": 1},
			"x3": {"type": "skew", "a": 5, "mean": 0, "std": 1},
			"x4": {"type": "skew", "a": -5, "mean": 2, "std": 0.5}
		}
		return BitMix.generate_data(n_samples, dist_params, random_state)

	@staticmethod
	def generate_out_of_sample_data(n_samples, random_state=None):
		dist_params = {
			"x1": {"type": "uniform", "low": 1, "high": 2},
			"x2": {"type": "normal", "mean": 1, "std": 1},
			"x3": {"type": "skew", "a": -5, "mean": 2, "std": 1},
			"x4": {"type": "skew", "a": 5, "mean": 0, "std": 0.5}
		}
		return BitMix.generate_data(n_samples, dist_params, random_state)

	@staticmethod
	def generate_training_data(n_samples, random_state=None):
		return BitMix.generate_in_sample_data(n_samples, random_state)

	@staticmethod
	def generate_testing_data1(n_samples, random_state=None):
		return BitMix.generate_in_sample_data(n_samples, random_state)

	@staticmethod
	def generate_testing_data2(n_samples, random_state=None):
		return BitMix.generate_out_of_sample_data(n_samples, random_state)

