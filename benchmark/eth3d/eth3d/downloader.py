import os
import requests
import shutil
import pathlib
import functools

import py7zr
from py7zr import callbacks
from tqdm.auto import tqdm


class ProgressCallback(callbacks.ExtractCallback):
    def __init__(self, uncompressed_size):
        super().__init__()
        self.uncompressed_size = uncompressed_size
        self.pbar = tqdm(
            total=self.uncompressed_size, unit="B", unit_scale=True, desc="Extracting"
        )

    def report_start_preparation(self):
        pass

    def report_start(self, processing_file_path, processing_bytes):
        pass

    def report_update(self, decompressed_bytes):
        if self.pbar:
            self.pbar.update(int(decompressed_bytes))
        pass

    def report_end(self, processing_file_path, wrote_bytes):
        pass

    def report_warning(self, message):
        pass

    def report_postprocess(self):
        pass


def extract_7z(archive_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(
            out_path, callback=ProgressCallback(archive.archiveinfo().uncompressed)
        )


def download_file(url, local_filename):
    """based on Mike's answer to:
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    """
    try:
        with requests.get(url, stream=True, allow_redirects=True) as r:
            if 300 < r.status_code < 200:
                r.raise_for_status()
                raise RuntimeError(
                    f"Downloading file from {url} returned {r.status_code}"
                )

            file_size = int(r.headers.get("Content-Length", 0))

            path = pathlib.Path(local_filename).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)

            desc = "(Unknown total file size)" if file_size == 0 else ""
            r.raw.read = functools.partial(
                r.raw.read, decode_content=True
            )  # Decompress if needed
            with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
                with path.open("wb") as f:
                    shutil.copyfileobj(r_raw, f)
    except Exception as e:
        # if an exception happens, remove the
        # temporary file
        if os.path.exists(local_filename):
            os.remove(local_filename)
        raise e


def download_extract_archive(url, local_dir):
    """
    Downloads and extracts a 7z compressed archive

    Args:
        url: url of the file to download
        local_dir: local directory to place the downloaded file
    """
    local_path_7z = os.path.join(local_dir, os.path.basename(url))
    local_path, ext = os.path.splitext(local_path_7z)
    if ext != ".7z":
        raise TypeError("URL does not have a `.7z` extension.")
    download_file(url, local_path_7z)
    print(f"Extracting content from `{local_path_7z}` archive to `{local_path}`...")
    extract_7z(local_path_7z, local_path)
    os.remove(local_path_7z)
    return os.path.join(local_dir, local_path)


def download_dataset(
    dataset_name: str,
    datasets_dir: str,
    with_original_jpg: bool,
    with_eval: bool,
    with_depth: bool,
):
    scan_eval_url = f"https://www.eth3d.net/data/{dataset_name}_dslr_scan_eval.7z"
    undistorted_url = f"https://www.eth3d.net/data/{dataset_name}_dslr_undistorted.7z"
    depth_url = f"https://www.eth3d.net/data/{dataset_name}_dslr_depth.7z"
    jpg_url = f"https://www.eth3d.net/data/{dataset_name}_dslr_jpg.7z"

    local_dataset_path = os.path.join(datasets_dir, dataset_name)
    if os.path.exists(local_dataset_path):
        shutil.rmtree(local_dataset_path)
    os.makedirs(local_dataset_path, exist_ok=True)

    print("Downloading undistorted archive...")
    download_extract_archive(undistorted_url, local_dataset_path)
    if with_eval:
        print("Downloading evaluation archive...")
        download_extract_archive(scan_eval_url, local_dataset_path)
    if with_depth:
        print("Downloading depth archive...")
        download_extract_archive(depth_url, local_dataset_path)
    if with_original_jpg:
        download_extract_archive(jpg_url, local_dataset_path)
