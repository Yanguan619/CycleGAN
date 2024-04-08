# from __future__ import print_function

import tarfile
from pathlib import Path
from warnings import warn
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup


class DownloadData(object):
    """A Python script for downloading CycleGAN datasets.

    Examples:
        >>> from util.get_data import DownloadData
        >>> gd = DownloadData(technique='cyclegan', save_path='./datasets')
        # options will be displayed.
    """

    def __init__(self, technique="CycleGAN", save_path="./data", verbose=True):
        url_dict = {
            "cyclegan": "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/",
        }
        self.url = url_dict.get(technique.lower())
        self._verbose = verbose
        self.get(save_path=save_path)

    def _print(self, text: str):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text)
        options = [
            h.text
            for h in soup.find_all("a", href=True)
            if h.text.endswith((".zip", "tar.gz"))
        ]
        return options

    def _present_options(self):
        print(self.url)
        r = requests.get(self.url)
        options = self._get_options(r)
        print("Options:\n")
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input(
            "\nPlease enter the number of the " "dataset above you wish to download: "
        )
        return options[int(choice)]

    def _download_data(self, dataset_url: str, dataset_path: Path):
        dataset_path.mkdir(exist_ok=True)

        save_path = Path(dataset_path).joinpath(Path(dataset_url).name)

        import urllib.request

        urllib.request.urlretrieve(dataset_url, save_path)
        print("--> 下载完成 ")
        if save_path.suffix == ".tar.gz":
            obj = tarfile.open(save_path)
        elif save_path.suffix == ".zip":
            obj = ZipFile(save_path, "r")
        else:
            raise ValueError("Unknown File Type: {0}.".format(save_path))
        self._print("Unpacking Data...")
        obj.extractall(dataset_path)
        obj.close()

    def get(self, save_path: str, dataset=None):
        save_path = Path(save_path)
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset
        save_path_full = save_path.joinpath(selected_dataset.split(".")[0])
        print(f"Downloading Data to {save_path_full}")

        if save_path_full.is_dir():
            warn(f"\n'{save_path_full}' already exists.")
        else:
            url = "{0}/{1}".format(self.url, selected_dataset)
            self._print(f"Downloading Data from {url}")
            self._download_data(url, save_path)
        return Path(save_path_full)


if __name__ == "__main__":
    gd = DownloadData()
