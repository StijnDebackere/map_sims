from typing import Tuple

import h5py
import numpy as np
from copy import deepcopy
from multiprocessing import Process, Queue
import os
from pathlib import Path
import time

import astropy.units as u
import gadget
import h5py
import numpy as np
from tqdm import tqdm


def on_queue(queue, func, *args, **kwargs):
    res = func(*args, **kwargs)
    queue.put([os.getpid(), res])


def get_group_pids(
    sim_dir,
    snapnum,
    cut_var=None,
    cut_vals=None,
    N=None,
    hydro=False,
    verbose=False,
):
    """Get ParticleIDs for all DM particles, sorted by binding energy, for
    each group with cut_var between cut_vals in sim_dir with snapnum.

    Parameters
    ----------
    sim_dir : str
        base location for gadget data
    snapnum : int
        snapshot
    cut_var : dict or None
        name and HDF5 dataset to apply cut_vals to
    cut_vals : (2,) array-like
        values for cut_var selection
    N : int
        N from sim_dir, cube root of number of DM particles to correct hydro sim PIDs offset
    hydro : bool
        is sim_dir hydrodynamical simulation, if true, offset to PIDs starting from 1 instead of N**3 + 1
    verbose : bool
        verbose output

    Returns
    -------
    pids_groups : dict
        dictionary with group_ids as keys and all PIDs belonging to group as values

    """
    group_info = gadget.Gadget(sim_dir, "subh", snapnum, verbose=verbose)

    group_offset = group_info.read_var("FOF/GroupOffset")
    group_length = group_info.read_var("FOF/GroupLength")
    group_number = np.arange(0, len(group_offset))

    pids = group_info.read_var("IDs/ParticleID")
    try:
        pids_be = group_info.read_var("IDs/Particle_Binding_Energy")
        bound = pids_be < 0
    except KeyError:
        # no binding energy available
        pids_be = None
        bound = None


    # if hydro simulation, DM PIDs start at N**3 + 1 => only include these PIDs
    if hydro and N is not None:
        N3 = N ** 3
        dm = pids > N3
    elif hydro and N is None:
        raise ValueError("cannot correct dark matter ParticleIDs if N is not given.")

    # if selection cut is provided, determine which groups to include
    if cut_var is not None and cut_vals is not None:
        var = group_info.read_var(list(cut_var.values())[0])
        if type(cut_vals) is not u.Quantity:
            cut_vals = cut_vals * var.unit

        # select only groups matching cut_vals
        selection = np.where((var >= cut_vals[0]) & (var < cut_vals[1]))[0]
    else:
        # select all
        selection = ()

    iterator = zip(
        group_number[selection], group_offset[selection], group_length[selection]
    )
    if verbose:
        iterator = tqdm(
            iterator, total=len(group_offset[selection]), desc="Loading pids for groups"
        )

    pids_groups = {}
    for g_n, g_o, g_l in iterator:
        if hydro and bound is not None:
            selection = dm[g_o : g_o + g_l] & bound[g_o : g_o + g_l]
        elif hydro and bound is None:
            selection = dm[g_o : g_o + g_l]
        elif not hydro and bound is None:
            selection = ()
        else:
            selection = bound[g_o : g_o + g_l]

        if pids_be is not None:
            pids_be_group = pids_be[g_o : g_o + g_l][selection]
            be_sort = np.argsort(pids_be_group)
        else:
            be_sort = ()

        if hydro:
            pids_group = pids[g_o : g_o + g_l][selection][be_sort] - N3
        else:
            pids_group = pids[g_o : g_o + g_l][selection][be_sort]

        pids_groups[g_n] = pids_group

    return pids_groups


def match_group_pids(
    pids_groups_ref,
    pids_groups_other,
    n_mb,
    skip_fraction=0.1,
    verbose=False,
    logger=None,
):
    """Link group_ids between group pids for ref and other if more than
    n_mb of most-bound particles are matched.

    Parameters
    ----------
    pids_groups_ref : dict
        dictionary with group_ids for ref as keys and all PIDs sorted by binding energy
        belonging to group as values
    pids_groups_other : dict
        dictionary with group_ids for other as keys and all PIDs sorted by binding energy
        belonging to group as values
    n_mb : int
        number of most-bound particles required
    skip_fraction : float
        skip iteration of match is not found before other haloes with only skip_fraction
        of ref particles

    Returns
    -------
    linked_gids : (n, 2) array-like
        gids from

    """
    iterator = enumerate(pids_groups_ref.items())
    if verbose:
        iterator = tqdm(iterator, desc="Linking ref to other")

    linked_gids = []
    n_tot = len(pids_groups_ref.keys())

    non_matched_keys = set(pids_groups_other.keys())
    for idx, (gid_ref, pids_ref) in iterator:
        for gid_other in non_matched_keys:
            n_ref = len(pids_ref)
            n_other = len(pids_groups_other[gid_other])

            # should not have haloes that have such a large difference in particles...
            if n_other / n_ref < skip_fraction:
                if logger:
                    logger.info(
                        f"skipped {gid_ref=}, no matches with more than {skip_fraction=}"
                    )
                break

            matches_other = np.isin(
                pids_groups_other[gid_other], pids_ref[:n_mb], assume_unique=True
            )
            n_match = matches_other.sum()

            # group matched
            if n_match >= 0.5 * n_mb:
                linked_gids.append([gid_ref, gid_other])
                non_matched_keys = non_matched_keys - {gid_other}
                break

        if idx % 100 == 0:
            if logger:
                logger.info(f"{os.getpid()} - matched {idx} haloes")

    return linked_gids


def get_group_pids_mp(ref_sim_dir, other_sim_dirs, snapnum, cut_var, cut_vals, verbose, logger):
    # set up queue for passing different sim process results
    queue = Queue()

    procs = []
    for sim_dir in [ref_sim_dir, *other_sim_dirs]:
        if "DMONLY" in Path(sim_dir).name:
            hydro = False
        else:
            hydro = True

        proc = Process(
            target=on_queue,
            kwargs={
                "queue": queue,
                "func": get_group_pids,
                "sim_dir": sim_dir,
                "snapnum": snapnum,
                "cut_var": cut_var,
                "cut_vals": cut_vals,
                "hydro": hydro,
                "N": N,
                "verbose": verbose,
            },
        )
        procs.append(proc)
        proc.start()
        if logger:
            logger.info(f"{sim_dir=} passed to pid={proc.pid}")

    results = []
    for _ in range(len(procs)):
        results.append(queue.get())

    for proc in procs:
        proc.join()

    # results contains lists of [pid, pids_groups]
    # sort to correct pid order => have ref and others
    results.sort()
    pids_groups_ref = results[0][1]
    pids_groups_others = [res[1] for res in results[1:]]
    return pids_groups_ref, pids_groups_others


def link_sims(
    ref_sim_dir="/hpcdata0/simulations/BAHAMAS/DMONLY_nu0_L400N1024_WMAP9",
    other_sim_dirs=[
        "/hpcdata0/simulations/BAHAMAS/AGN_TUNED_nu0_L400N1024_WMAP9",
        "/hpcdata0/simulations/BAHAMAS/AGN_7p6_nu0_L400N1024_WMAP9",
        "/hpcdata0/simulations/BAHAMAS/AGN_8p0_nu0_L400N1024_WMAP9",
    ],
    snapnum=28,
    N=1024,
    n_mb=50,
    cut_var={"m200m": "FOF/Group_M_Mean200"},
    cut_vals=(10 ** 12.5, 10 ** 16) * u.Msun / u.littleh,
    save_dir="/hpcdata0/simulations/BAHAMAS/extsdeba/matches/",
    fname_append="",
    overwrite=True,
    verbose=False,
    log_dir=None,
    log_fname=None,
):
    """Link hydro_sim and dmo_sim groupnumbers with n_mb matching DM particles."""
    if log_dir is not None and log_fname is not None:
        logger = get_logger(log_dir=log_dir, fname=log_fname)
        verbose = False

    else:
        logger = None

    # make save_dir if given
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if overwrite:
            mode = "w"
        else:
            mode = "a"

    # extract sim names
    ref_sim = Path(ref_sim_dir).name
    other_sims = [Path(other_sim_dir).name for other_sim_dir in other_sim_dirs]

    pids_groups_ref, pids_groups_other = get_group_pids_mp(
        ref_sim_dir=ref_sim_dir,
        other_sim_dirs=other_sim_dirs,
        snapnum=snapnum,
        cut_var=cut_var,
        cut_vals=cut_vals,
        verbose=verbose,
    )

    if save_dir is not None:
        if verbose:
            print("Saving match files")
        if fname_append != "":
            fname_append = f"_{fname_append}"

        with h5py.File(
            f"{save_dir}/{ref_sim}_{snapnum:03d}_groups_pids{fname_append}.hdf5",
            mode=mode,
        ) as f:
            f.attrs["snapnum"] = snapnum
            for gid, pids in pids_groups_ref.items():
                # unique key for given cut_var and cut_vals
                key = (
                    f"{gid:d}_{list(cut_var.keys())[0]}"
                    f"_{np.log10(cut_vals[0]):.2f}-{np.log10(cut_vals[1]):.2f}"
                )
                if key in f:
                    if np.allclose(f[key][()], pids):
                        pass
                    else:
                        raise ValueError(f"not all pids for {key} match for existing {ref_sim}")
                else:
                    f[key] = pids
                    f[key].attrs[
                        "description"
                    ] = "dark matter particle ids for each group in order of boundedness"
                    f[key].attrs["cut_var"] = list(cut_var.values())[0]
                    f[key].attrs["cut_vals"] = cut_vals

        if logger:
            logger.info(f"saved groups_pids for {ref_sim=}")
        for idx, other_sim in enumerate(other_sims):
            with h5py.File(
                f"{save_dir}/{other_sim}_{snapnum:03d}_groups_pids{fname_append}.hdf5",
                mode=mode,
            ) as f:
                f.attrs["snapnum"] = snapnum
                for gid, pids in pids_groups_others[idx].items():
                    key = (
                        f"{gid:d}_{list(cut_var.keys())[0]}"
                        f"_{np.log10(cut_vals[0]):.2f}-{np.log10(cut_vals[1]):.2f}"
                    )
                    if key in f:
                        if np.allclose(f[key][()], pids):
                            pass
                        else:
                            raise ValueError(f"not all pids for {key} match for existing {other_sim}")
                    else:
                        f[key] = pids
                        f[key].attrs[
                            "description"
                        ] = "dark matter particle ids for each group in order of boundedness"
                        f[key].attrs["cut_var"] = list(cut_var.values())[0]
                        f[key].attrs["cut_vals"] = cut_vals

            if logger:
                logger.info(f"saved groups_pids for {other_sim=}")

    # bijectively link haloes between ref_sim and other_sims
    procs = []
    for pids_groups_other, other_sim in zip(pids_groups_others, other_sims):
        for i in range(2):
            if i == 0:
                ref = pids_groups_ref
                other = pids_groups_other
            if i == 1:
                ref = pids_groups_other
                other = pids_groups_ref

            proc = Process(
                target=on_queue,
                kwargs={
                    "queue": queue,
                    "func": match_group_pids,
                    "pids_groups_ref": ref,
                    "pids_groups_other": other,
                    "n_mb": n_mb,
                    "verbose": verbose,
                    "logger": logger,
                },
            )
            procs.append(proc)
            proc.start()
            logger.info(f"{other_sim=} {i=} passed to pid={proc.pid}")

    results = []
    for _ in range(len(procs)):
        results.append(queue.get())

    for proc in procs:
        proc.join()

    # results contains lists of [pid, [ids_hydro, ids_dmo]]
    # sort to correct pid order => have ref2other on even, other2ref on odd
    results.sort()

    linked_ref2others = [np.array(res[1]) for res in results[0::2]]
    linked_others2ref = [np.array(res[1]) for res in results[1::2]]

    # extract double matches
    linked_gids = []
    iterator = enumerate(zip(linked_ref2others, linked_others2ref))
    if verbose:
        iterator = tqdm(
            iterator, desc="Comparing matches", total=len(linked_ref2others)
        )

    if save_dir is not None:
        h5file = h5py.File(
            f"{save_dir}/{ref_sim}_{snapnum:03d}_matches{fname_append}.hdf5", mode=mode
        )
    else:
        h5file = None

    linked_gids = {other_sim: [] for other_sim in other_sims}
    for idx, (linked_ref2other, linked_other2ref) in iterator:
        for match in linked_ref2other:
            # matches reversed in other_sim
            if match[::-1] in linked_other2ref:
                linked_gids[other_sims[idx]].append(match)

        if h5file:
            key = (
                f"{other_sims[idx]}_{list(cut_var.keys())[0]}"
                f"_{np.log10(cut_vals[0]):.2f}-{np.log10(cut_vals[1]):.2f}"
                f"_n_mb_{n_mb:d}"
            )
            if key in f:
                if np.allclose(f[key][()], pids):
                    pass
                else:
                    raise ValueError(f"not all pids for {key} match for existing {other_sim}")
            else:
                h5file[key] = linked_gids[other_sims[idx]]
                h5file[key].attrs[
                    "description"
                ] = "column 0: group_ids in reference, column 1: matched group_ids in sim"
                h5file[key].attrs["cut_var"] = cut_var
                h5file[key].attrs["cut_vals"] = cut_vals
                h5file[key].attrs["n_mb"] = n_mb

    if logger:
        logger.info(f"saved matches in {h5file.filename}")

    if h5file:
        h5file.close()

    return linked_gids


def match_ids(
    group_ids_ref: np.ndarray,
    group_ids_other: np.ndarray,
    group_ids_matched: np.ndarray,
    return_slices: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the group_ids of ref and other that have been successfully matched.

    Parameters
    ----------
    group_ids_ref : array-like
        list of group_ids from reference simulation
    group_ids_other : array-like
        list of group_ids from other simulation
    group_ids_matched : (n, 2) array-like
        list of matched group_ids between ref (column 0) and other (column 1)
    return_slices : bool [Default: False]
        also return slices for group_ids_ref and group_ids_other

    Returns
    -------
    matched_ids_ref : array-like
        list of matched group_ids in group_ids_ref

        if return_slices: (n, 2) array-like
            corresponding ids in group_ids_ref (column 1)

    matched_ids_other : array-like
        list of corresponding group_ids in group_ids_other

        if return_slices: (n, 2) array-like
            corresponding ids in group_ids_other (column 1)
    """
    # will contain [gid_ref, idx_ref] and [gid_other, idx_other] for all matches
    matches_ref = []
    matches_other = []

    iterator = enumerate(group_ids_ref)
    if verbose:
        iterator = tqdm(iterator, total=len(group_ids_ref), desc="Matching ids")

    for idx_ref, gid_ref in iterator:
        # which matched group_id corresponds to gid_ref in group_ids_ref?
        idx_matched_ref = np.where(group_ids_matched[:, 0] == gid_ref)[0]

        # gid_ref was not matched
        if idx_matched_ref.size == 1:
            # match found, extract index
            idx_matched_ref = idx_matched_ref[0]
            matched_ids = group_ids_matched[idx_matched_ref]

            # which gid_other matches gid_ref?
            idx_other = np.where(group_ids_other == matched_ids[1])[0]
            if idx_other.size == 1:
                # match found, extract index
                idx_other = idx_other[0]
                ids_ref = [matched_ids[0], idx_ref]
                ids_other = [matched_ids[1], idx_other]
            else:
                ids_ref = [-1, idx_ref]
                ids_other = [-1, -1]
        else:
            ids_ref = [-1, idx_ref]
            ids_other = [-1, -1]

        matches_ref.append(ids_ref)
        matches_other.append(ids_other)

    matches_ref = np.asarray(matches_ref, dtype=int)
    matches_other = np.asarray(matches_other, dtype=int)

    matches_ref = matches_ref[matches_ref[:, 0] != -1]
    matches_other = matches_other[matches_other[:, 0] != -1]

    if return_slices:
        return matches_ref, matches_other

    else:
        return matches_ref[:, 0], matches_other[:, 0]


def link_ref_other(
    ref_sim: str,
    other_sim: str,
    ref_results: dict,
    other_results: dict,
    matches_file: str,
    verbose: bool = False,
) -> Tuple[dict, dict]:
    """Link ref_results and other_results using matches_file between
    ref_sim and other_sim.

    Parameters
    ----------
    ref_sim : str
        reference simulation in matches_file
    other_sim : str
        other simulation name in matches_file
    ref_results : dict
        properties for ref groups
    other_results : dict
        properties for other groups
    matches_file : str
        filename for matches between ref and other

    Returns
    -------
    ref_results : dict
        results for all matched haloes in ref
    other_results : dict
        results for all matched haloes in other

    """
    # now load the matching ids
    with h5py.File(matches_file, "r") as h5_matches:
        matched_ids = h5_matches[other_sim][()]

    matches_ref, matches_other = match_ids(
        group_ids_ref=ref_results["group_ids"],
        group_ids_other=other_results["group_ids"],
        group_ids_matched=matched_ids,
        return_slices=True,
        verbose=verbose,
    )

    ref_results = {k: v[matches_ref[:, 1]] for k, v in ref_results.items()}
    other_results = {k: v[matches_other[:, 1]] for k, v in other_results.items()}

    return ref_results, other_results
