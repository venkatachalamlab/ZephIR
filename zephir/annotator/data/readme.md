# Read and write data

The `vlab.data.io` module allows access to time-series data in a somewhat
flexible way. Every dataset is a folder on disk. To handle different operating
systems, use the `pathlib` module to create these folders:

    dataset = pathlib.Path(r"N:\data\my_great_recording")

The folder can hold any data and files that you want. It should be **read only**
and it should have a file called `getters.py` containing functions with
the following signatures:

    def get_slice(t: int) -> np.ndarray:
        ...

    def get_metadata() -> dict:
        ...

    def get_times() -> np.ndarray:
        ...

There are some defaults for these in `vlab.data.io`, but it is better to make
your own.

You can define functions to access any other data you want. It is good practice
to also include a jupyter notebook in that data directory showing examples of
how to access the data and what the data looks like.

Generally, the contents of a data folder should be frozen after the folder is
created and filled. If you wish to create derived data, the best practice is to
make a new folder with the derived data.

## How should data and code be separated?

Any code required to access data should generally live with the data
(e.g. `getters.py`). Code used to generate the data should also live with the
data (e.g. `make.py`). If either of those use very general functions, those
functions can be moved to github, but copying and pasting code between different
data folders is OK, because it ensures that the data can be accessed in the
future after we change some convention (e.g. switching from hdf5 to a binary
disk format).