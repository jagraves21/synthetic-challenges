import dataclasses
from abc import ABC, abstractmethod
import datetime
import glob
import logging
import os
import re
import requests

import pandas as pd

from .utils import path_utils
from .utils import dataframe_utils

@dataclasses.dataclass
class DatasetManager(ABC):
	dataset_name: str
	project_root: str = None

	# initialized in __post_init__
	paths: path_utils.ProjectPaths = dataclasses.field(init=False, repr=False)
	dataset_dir: str = dataclasses.field(init=False)
	raw_dir: str = dataclasses.field(init=False)
	processed_dir: str = dataclasses.field(init=False)
	artifacts_dir: str = dataclasses.field(init=False)
	results_dir: str = dataclasses.field(init=False)
	figures_dir: str = dataclasses.field(init=False)
	logger: logging.Logger = dataclasses.field(init=False, repr=False)

	def __post_init__(self):
		self.paths = path_utils.ProjectPaths(project_root=self.project_root)
		self.dataset_dir = os.path.join(self.paths.get_data_dir(), self.dataset_name)
		self.raw_dir = os.path.join(self.dataset_dir, "raw")
		self.processed_dir = os.path.join(self.dataset_dir, "processed")
		self.artifacts_dir = os.path.join(self.dataset_dir, "artifacts")
		self.results_dir = os.path.join(self.dataset_dir, "results")
		self.figures_dir = os.path.join(self.dataset_dir, "figures")

		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.INFO)
		if not self.logger.handlers:
			ch = logging.StreamHandler()
			ch.setLevel(logging.INFO)
			formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
			ch.setFormatter(formatter)
			self.logger.addHandler(ch)

	# --- Optional Methods for Subclass Implementation ---
	def generate_dataset(self, *args, **kwargs):
		raise NotImplementedError(f"{self.dataset_name} does not implement generate_dataset.")

	# --- Directory Helpers ---
	def get_dataset_dir(self):
		return self.dataset_dir

	def get_raw_dir(self):
		return self.raw_dir

	def get_processed_dir(self):
		return self.processed_dir

	def get_artifacts_dir(self):
		return self.artifacts_dir

	def get_results_dir(self):
		return self.results_dir

	def get_figures_dir(self):
		return self.figures_dir

	# --- Path Helpers ---
	def get_raw_file_path(self, filename=None):
		return self.raw_dir if filename is None else os.path.join(self.raw_dir, filename)

	def get_processed_file_path(self, filename=None):
		return self.processed_dir if filename is None else os.path.join(self.processed_dir, filename)

	def get_artifacts_file_path(self, filename=None):
		return self.artifacts_dir if filename is None else os.path.join(self.artifacts_dir, filename)

	def get_results_file_path(self, filename=None):
		return self.results_dir if filename is None else os.path.join(self.results_dir, filename)

	def get_figures_file_path(self, filename=None):
		return self.figures_dir if filename is None else os.path.join(self.figures_dir, filename)

	# --- Private Save/Load Methods ---
	def _save_csv_dataframe(self, df, path, append=False, index=True, suppress_logs=False, log_message=None):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		if append and os.path.exists(path):
			df.to_csv(path, mode="a", header=False, index=index)
			if not suppress_logs:
				self.log(log_message or f"Appended {len(df)} rows to existing CSV: {path}")
		else:
			df.to_csv(path, index=index)
			if not suppress_logs:
				self.log(log_message or f"Saved DataFrame to: {path}")

	def _load_csv_dataframe(self, path, header=None, index_col=0, nrows=None, verbose=False):
		if not os.path.exists(path):
			raise FileNotFoundError(f"File not found: {path}")
		df = pd.read_csv(path, header=header, index_col=index_col, nrows=nrows)
		if verbose:
			dataframe_utils.print_dataframe_info(df, name=os.path.basename(path))
		return df

	def _load_xls_dataframe(self, path, verbose=False, **read_excel_kwargs):
		if verbose:
			try:
				from IPython.display import display, HTML
			except ImportError:
				raise ImportError("IPython is required for verbose display but is not installed.")

		sheet_map = pd.read_excel(path, sheet_name=None, **read_excel_kwargs)
		result = {}
		for sheet_name, df in sheet_map.items():
			if verbose:
				display(HTML(f"<h1>{sheet_name}</h1>"))
				dataframe_utils.print_dataframe_info(df, name=sheet_name)
			result[sheet_name] = df
		return result

	@staticmethod
	def _get_latest_file(directory, base_filename):
		base_name, file_extension = os.path.splitext(base_filename)
		base_path = os.path.join(directory, base_filename)
		pattern = os.path.join(directory, f"{base_name}_*{file_extension}")
		matching_files = glob.glob(pattern)

		timestamp_regex = re.compile(
			rf"^{re.escape(base_name)}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}{re.escape(file_extension)}$"
		)

		valid_files = [
			filename
			for filename in matching_files
			if timestamp_regex.match(os.path.basename(filename))
		]

		if valid_files:
			return sorted(valid_files)[-1]

		if os.path.exists(base_path):
			return base_path

		return None

	# --- Public Save Methods ---
	def save_processed_dataframe(self, df, filename, append=False, index=True, suppress_logs=False):
		path = self.get_processed_file_path(filename)
		self._save_csv_dataframe(
			df, path, append=append, index=index, suppress_logs=suppress_logs,
			log_message=f"Saved DataFrame to processed directory: {path}"
		)
		return path
	
	def save_model(self, model, base_filename):
		import torch
		timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		base_name, file_extension = os.path.splitext(base_filename)
		timestamped_filename = f"{base_name}_{timestamp_str}{file_extension}"
		path = self.get_artifacts_file_path(timestamped_filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torch.save(model, path)
		self.log(f"Saved model to artifacts directory: {path}")
		return path

	def save_results_dataframe(self, df, filename, append=False, index=True, suppress_logs=False):
		path = self.get_results_file_path(filename)
		self._save_csv_dataframe(
			df, path, append=append, index=index, suppress_logs=suppress_logs,
			log_message=f"Saved DataFrame to results directory: {path}"
		)
		return path
	
	def save_results_figure(self, fig, filename, suppress_logs=False):
		path = self.get_figures_file_path(filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path, bbox_inches="tight")
		if not suppress_logs:
			self.log(f"Saved figure to figures directory: {path}")
		return path

	# --- Public Load Methods ---
	def load_raw_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_raw_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)

	def load_processed_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_processed_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)
	
	def load_model(self, base_filename):
		import torch
		path = DatasetManager._get_latest_file(self.get_artifacts_dir(), base_filename)
		if path is None:
			raise FileNotFoundError(f"No file found for '{base_filename}' in {self.get_artifacts_dir()}")
		return torch.load(path, map_location="cpu")

	def load_results_dataframe(self, filename, header=None, index_col=0, nrows=None, verbose=True):
		path = self.get_results_file_path(filename)
		return self._load_csv_dataframe(
			path, header=header, index_col=index_col, nrows=nrows, verbose=verbose
		)

	# --- Download File ---
	@staticmethod
	def download_file(url, path):
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
						  "AppleWebKit/537.36 (KHTML, like Gecko) "
						  "Chrome/120.0.0.0 Safari/537.36",
			"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
			"Referer": "https://robjhyndman.com/"
		}

		with requests.Session() as session:
			response = session.get(url, headers=headers, stream=True)
			response.raise_for_status()  # Will raise HTTPError if blocked

			with open(path, "wb") as fp:
				for chunk in response.iter_content(chunk_size=8192):
					if chunk:
						fp.write(chunk)

	# --- Logger Helper ---
	def log(self, message, level="info"):
		level = level.lower()
		if level == "debug":
			self.logger.debug(message)
		elif level == "info":
			self.logger.info(message)
		elif level == "warning":
			self.logger.warning(message)
		elif level == "error":
			self.logger.error(message)
		elif level == "critical":
			self.logger.critical(message)
		else:
			raise ValueError(f"Invalid log level: {level}")

