import tarfile
from pathlib import Path
from warnings import warn
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup


class GetData(object):
    """A Python script for downloading CycleGAN datasets.

    Examples:
        >>> from util.get_data import GetData
        >>> gd = GetData(technique='cyclegan', save_path='./datasets')
        # options will be displayed.
    """

    def __init__(
        self, technique="cycle_gan", save_dir="./weights/cycle_gan", verbose=True
    ):
        url_dict = {
            "cycle_gan": "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/",
        }
        self.url = url_dict.get(technique.lower())
        self._verbose = verbose
        self.get(save_dir=save_dir)

    def _print(self, text: str):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, "lxml")
        options = [
            h.text for h in soup.find_all("a", href=True) if h.text.endswith((".pth"))
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

    def _download_data(self, dataset_url: str, save_path: Path):
        import urllib.request

        urllib.request.urlretrieve(dataset_url, save_path)

    def get(self, save_dir: Path, dataset=None):
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset
        save_path = Path(save_dir, selected_dataset)
        print(f"Downloading Data to {save_path}")

        if save_path.is_file():
            warn(f"\n'{save_path}' already exists.")
        else:
            url = "{0}/{1}".format(self.url, selected_dataset)
            self._print(f"Downloading Data from {url}")
            self._download_data(url, save_path)
            self._print("--> 下载完成 ")
        return Path(save_dir)


if __name__ == "__main__":
    gd = GetData()
