"""
@Description: Downloads the data sets to data directory if they are not already there
@Notes:
- UAVDT source link: https://datasetninja.com/uavdt#download

"""

import os


def downloadUAVDT(data_path: str, replace: bool = False):
    import dataset_tools as dtools

    assert os.path.isdir(data_path), f"EE: data path {data_path} does not exist"

    if (not os.path.exists(os.path.join(data_path, "uavdt")) or replace):
        print(f"Downloading data to: {data_path}")
        dtools.download(dataset="UAVDT", dst_dir=data_path)
    else:
        print(f"II: uavdt dataset already exists at {data_path}")


# The archive dataset_tools downloads, taken from its released_datasets.json.
UAVDT_URL = (
    "https://www.dropbox.com/scl/fi/vaqpimf247amiu0ijeyen/uavdt-DatasetNinja.tar"
    "?rlkey=gwydhrv0hef7rssvrs3wxvh9a&dl=1"
)


def downloadUAVDTDirect(data_path: str, replace: bool = False, url: str = UAVDT_URL) -> str:
    """Download UAVDT without dataset_tools, unpacking the tar while it streams in.

    Gives the same data/uavdt folder as downloadUAVDT. dataset_tools pins exact
    numpy, pandas, and pillow versions, which breaks Colab's preinstalled torch,
    so Colab uses this. The 14 GB tar is never written to disk.
    """
    import shutil
    import tarfile
    import time
    import urllib.request

    assert os.path.isdir(data_path), f"EE: data path {data_path} does not exist"
    dataset_dir = os.path.join(data_path, "uavdt")
    if os.path.isfile(os.path.join(dataset_dir, "meta.json")) and not replace:
        print(f"II: uavdt dataset already exists at {data_path}")
        return dataset_dir

    # Clear a half-finished download so no stale files are left behind.
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    print(f"Downloading and unpacking UAVDT (about 14 GB) to: {dataset_dir}")
    started = time.time()
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        reader = _ProgressReader(response, total)
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            archive.extractall(dataset_dir, filter="data")
    print(f"\nDone in {(time.time() - started) / 60:.1f} min")

    assert os.path.isfile(os.path.join(dataset_dir, "meta.json")), f"EE: no meta.json after unpacking to {dataset_dir}"
    return dataset_dir


class _ProgressReader:
    """Wraps the HTTP response and prints progress every 500 MB."""

    def __init__(self, stream, total: int, step: int = 500 * 1024**2):
        self.stream, self.total, self.step = stream, total, step
        self.read_bytes, self.next_report = 0, step

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.read_bytes += len(chunk)
        if self.read_bytes >= self.next_report:
            done = f"{self.read_bytes / 1024**3:.1f} GB"
            if self.total:
                done += f" / {self.total / 1024**3:.1f} GB"
            print(f"  {done}", flush=True)
            self.next_report += self.step
        return chunk

