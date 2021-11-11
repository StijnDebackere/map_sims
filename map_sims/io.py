from typing import Any, Tuple
import logging
import sys

import astropy.units as u
import h5py


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)


def to_schema(
    value: Any,
) -> Tuple[Any, dict]:
    """Convert value to a type that can be saved to hdf5, with necessary
    attributes to reconstruct the default type."""
    if isinstance(value, u.Quantity):
        val = value.value
        attrs = {
            "instance": str(u.Quantity),
            "unit": str(value.unit),
        }
    elif value is None:
        val = "NULL"
        attrs = {
            "instance": str(None),
        }
    else:
        # try simply saving
        val = value
        attrs = {
            "instance": str(type(value)),
        }

    return val, attrs


def from_schema(
    value: Any,
    attrs: dict
) -> Any:
    """Convert value saved to hdf5 according to schema, to the known
    type."""
    if attrs["instance"] == str(u.Quantity):
        val = value * u.Unit(str(attrs["unit"]))

    elif attrs["instance"] == str(None):
        val = None

    else:
        val = value

    return val


def write_to_hdf5(
    h5file: h5py.File,
    path: str,
    value: Any,
    overwrite: bool,
) -> None:
    """Save value to h5file[path] following our hdf5 schema."""
    val, attrs = to_schema(value=value)

    # check whether path exists and check whether to overwrite
    if path in h5file:
        if isinstance(h5file[path], h5py.Dataset):
            if not overwrite:
                logging.warn(f"{path} is already a Dataset in {h5file.filename}, skipping")
            else:
                logging.warn(f"{path} is already a Dataset in {h5file.filename}, overwriting")
                del h5file[path]
                h5file[path] = val
                for k, v in attrs.items():
                    h5file[path].attrs[k] = v

        elif isinstance(h5file[path], h5py.Group):
            if not overwrite:
                logging.warn(f"{path} is already a Group in {h5file.filename}, skipping")
            else:
                logging.warn(f"{path} is already a Group in {h5file.filename}, overwriting")
                del h5file[path]
                h5file[path] = val
                for k, v in attrs.items():
                    h5file[path].attrs[k] = v

        else:
            # should not get here...
            breakpoint()

    # add path to h5file
    else:
        # initial path could already be in
        base_path = "/".join(path.split("/")[:-1])
        if base_path in h5file and isinstance(h5file[base_path], h5py.Dataset):
            if not overwrite:
                logging.warn(f"{base_path} is already a Dataset in {h5file.filename}, skipping")
            else:
                logging.warn(f"{base_path} is already a Dataset in {h5file.filename}, overwriting")
                del h5file[base_path]
                h5file[path] = val
        else:
            try:
                h5file[path] = val
                for k, v in attrs.items():
                    h5file[path].attrs[k] = v
            except (TypeError, OSError) as e:
                breakpoint()

    return


def read_from_hdf5(
    h5file: h5py.File,
    path: str,
) -> Any:
    """Read value from h5file[path] following our hdf5 schema."""
    if not isinstance(h5file[path], h5py.Dataset):
        raise ValueError(f"{path} should be a h5py.Dataset, not {type(h5file[path])}")

    value = h5file[path][()]
    attrs = {k: v for k, v in h5file[path].attrs.items()}
    val = from_schema(value=value, attrs=attrs)

    return val


def recursively_save_dict_hdf5(
    data: dict,
    h5file: h5py.File,
    path: str,
    overwrite: bool,
) -> None:
    """Extract data recursively into h5file.

    Parameters
    ----------
    data : dict
        data at path
    h5file : h5py.File
        hdf5 file to save data to
    path : str
        current path position in h5file
    overwrite : bool
        overwrite existing Groups and Datasets

    """
    for key, val in data.items():
        new_path = f"{path}/{key}"
        if type(val) is dict:
            recursively_save_dict_hdf5(h5file=h5file, path=new_path, data=val, overwrite=overwrite)
        else:
            write_to_hdf5(h5file=h5file, path=new_path, value=val, overwrite=overwrite)


def recursively_fill_dict_hdf5(
    h5file: h5py.File,
    path: str = "/",
):
    """Extract h5file recursively.

    Parameters
    ----------
    h5file : h5py.File
        hdf5 file to load data from
    path : str
        current path position in h5file

    Returns
    -------
    data : dict
        extracted dict at path

    """
    data = {}

    for key, val in h5file[path].items():
        new_path = f"{path.rstrip('/')}/{key}"
        if isinstance(val, h5py.Group):
            data[key] = recursively_fill_dict_hdf5(
                h5file=h5file,
                path=new_path,
            )
        elif isinstance(val, h5py.Dataset):
            data[key] = read_from_hdf5(h5file=h5file, path=new_path)
        else:
            # don't know how we would end up here, best to debug...
            breakpoint()

    return data


def dict_to_hdf5(
    fname: str,
    data: dict,
    attrs: dict = None,
    overwrite: bool = False,
) -> None:
    """Convert a possibly nested dict to a hdf5 file.

    Parameters
    ----------
    fname : str
        filename to save data to
    data : dict
        possibly nested dictionary, also possibly containing astropy.units.Quantity objects
    attrs : dict, optional
        non-nested dict with metadata attributes to save to fname
        WARNING: if included as key in data, it will be popped!
    overwrite : bool
        overwrite existing Datasets/Groups if fname already exists

    """
    # attributes are either passed as kwarg or in data
    if attrs is None:
        if "attrs" in data.keys():
            logging.warn(f"popping attrs from data keys")
            attrs = data.pop("attrs")
        else:
            attrs = {}
    else:
        if "attrs" in data.keys():
            raise ValueError(f"attrs both included in data and as kwarg, choose one")

    with h5py.File(fname, mode="a") as f:
        for k, v in attrs.items():
            f.attrs[k] = v

        recursively_save_dict_hdf5(h5file=f, path="", data=data, overwrite=overwrite)


def hdf5_to_dict(fname: str) -> dict:
    """Load an hdf5 file into a nested dict, possibly containing
    astropy.units.Quantity objects.

    Parameters
    ----------
    fname : str
        filename to load data from

    Returns
    -------
    data : dict
        groups and datasets from fname

    """
    with h5py.File(fname, "r") as f:
        data = recursively_fill_dict_hdf5(h5file=f, path="/")
        if "attrs" in data.keys():
            raise ValueError("attrs is a reserved key for loading hdf5 attributes")

        data["attrs"] = {k: v for k, v in f.attrs.items()}

    return data


def merge_dicts(a, b):
    """Merge possibly nested dictionaries a and b. For matching keys, a
    takes precedence."""
    for key, val in a.items():
        if type(val) == dict:
            if key in b and type(b[key] == dict):
                merge_dicts(a[key], b[key])
        else:
            if key in b:
                logging.warn(f"{key=} in b already in a, skipping")

    for key, val in b.items():
        if not key in a:
            a[key] = val

    return a
