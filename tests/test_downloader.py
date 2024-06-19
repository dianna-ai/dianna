import os
from shutil import rmtree
import pooch
import pytest
from dianna.utils.downloader import data
from dianna.utils.downloader import download
from dianna.utils.downloader import labels
from dianna.utils.downloader import models

# create dict containing all possible file downloads
# note: not creating one combined dict in the unlikely case that
# the same filename appears in multiple dicts
# keys in this dict must match the available file types in the
# download function
all_files = {'model': models, 'data': data, 'label': labels}


@pytest.mark.parametrize('file_type', ['model', 'data', 'label'])
def test_downloader(file_type):
    """All files are available and have correct checksum."""
    files = all_files[file_type]

    # ensure we start with an empty cache folder
    cache_dir = pooch.os_cache("dianna")
    if os.path.isdir(cache_dir):
        rmtree(cache_dir)

    for fname, (_, checksum) in files.items():
        print(fname)
        assert checksum is not None
        path = download(fname, file_type, cache_dir)
        assert path is not None
