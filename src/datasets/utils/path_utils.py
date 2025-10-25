import dataclasses
import os
import pathlib

def print_tree(startpath, show_hidden=False):
    startpath = os.path.abspath(startpath)
    root_name = os.path.basename(startpath.rstrip(os.sep))
    print(root_name)

    def _walk(current_path, prefix=""):
        items = sorted(os.listdir(current_path))
        if not show_hidden:
            items = [i for i in items if not i.startswith(".")]

        for ii, item in enumerate(items):
            path = os.path.join(current_path, item)
            is_last = ii == len(items) - 1
            branch = "└── " if is_last else "├── "
            print(prefix + branch + item)

            if os.path.isdir(path):
                extension = "    " if is_last else "│   "
                _walk(path, prefix + extension)

    _walk(startpath)

@dataclasses.dataclass
class ProjectPaths:
    project_root: pathlib.Path = None
    data_folder_name: str = "data"

	# initialized in __post_init__
    drive_mounted: bool = dataclasses.field(init=False, default=False)
    project_name: str = dataclasses.field(init=False)
    data_folder: pathlib.Path = dataclasses.field(init=False)
    colab_base_path: pathlib.Path = dataclasses.field(init=False)

    def __post_init__(self):
        try:
            current_file = pathlib.Path(__file__).resolve()
        except NameError:
            current_file = pathlib.Path.cwd()

        # Find 'src' directory
        path = current_file
        while path != path.parent:
            if path.name == "src":
                break
            path = path.parent
        else:
            raise FileNotFoundError(
                "'src' directory not found in any parent directories of the current file."
            )

        # Project root = parent of src
        self.project_root = pathlib.Path(self.project_root) if self.project_root else path.parent
        self.project_name = self.project_root.name

        # Paths
        self.data_folder = self.project_root.joinpath(self.data_folder_name)
        self.colab_base_path = pathlib.Path("/content/drive/MyDrive").joinpath(self.project_name)

    @staticmethod
    def in_colab():
        try:
            return "google.colab" in str(get_ipython())
        except NameError:
            return False

    def mount_drive_once(self):
        if not self.drive_mounted:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            self.drive_mounted = True

    def get_data_dir(self):
        if self.in_colab():
            self.mount_drive_once()
            data_dir = self.colab_base_path.joinpath(self.data_folder.name)
        else:
            data_dir = self.data_folder

        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

