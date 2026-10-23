"""
Fermi-surface pockets areas and QO frequencies.

Functions
---------
get_orbit_areas
merge_orbits
"""

import numpy as np
from scipy import ndimage

__all__ = ["get_orbit_areas", "merge_orbits"]


def get_orbit_areas(bands, mu, dk, unit_cell_to_cm2, min_pixels=30, connectivity=1):
    """
    One entry per closed Fermi contour of a flavor (T -> 0).
    Regions: E0 < mu (conduction sea) and E1 > mu (valence holes).
    Outer contour -> area of the region with holes filled;
    each hole -> its own (inner) orbit. Nested areas overlap by construction.
    Returns list of dict: kind ('cond'/'val'), inner, f_cm2 (A/(2pi)^2 in cm^-2), trunc.
    """
    E0, E1 = bands
    # Define neighbors; must be dual: (2, 1) <-> (2, 2)
    fg = ndimage.generate_binary_structure(2, connectivity)         # foreground = pocket
    bg = ndimage.generate_binary_structure(2, 3 - connectivity)     # background

    # Loop over the two possible types of pockets (masks)
    orbits = []
    for kind, mask in (("cond", E0 < mu), ("val", E1 > mu)):
        if not mask.any():
            continue

        # Look for connected regions and label them
        lab, _ = ndimage.label(mask, structure=fg)      # returns (lab, n) with lab=0 for bg and 1,..., n for pockets

        edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            # Separate true exterior from inner pockets
            comp = lab[sl] == i
            filled = ndimage.binary_fill_holes(comp, structure=bg)      # fload from the borders
            if filled.sum() < min_pixels:
                continue
            orbits.append(dict(kind=kind, inner=False,
                               A=filled.sum() * dk**2, trunc=(i in edge)))

            # Look for inner pockets
            holes, nh = ndimage.label(filled & ~comp, structure=bg)
            if nh:
                sizes = np.bincount(holes.ravel())[1:]      # count the pixels per pocket
                orbits += [dict(kind=kind, inner=True, A=s * dk**2, trunc=False)
                           for s in sizes if s >= min_pixels]
    for o in orbits:
        o["f_cm2"] = o.pop("A") / (2 * np.pi)**2 * unit_cell_to_cm2
    return orbits


def merge_orbits(orbits, rtol=2e-2):
    """
    Merge orbits with the same (kind, inner) and areas within rtol (C3 copies, degenerate flavors).
    Returns list of dict with f_cm2 and mult.
    """
    merged = []
    for o in sorted(orbits, key=lambda o: o["f_cm2"]):
        for m in merged:
            if (m["kind"], m["inner"]) == (o["kind"], o["inner"]) and \
               abs(m["f_cm2"] - o["f_cm2"]) <= rtol * m["f_cm2"]:
                m["f_cm2"] = (m["f_cm2"] * m["mult"] + o["f_cm2"]) / (m["mult"] + 1)
                m["mult"] += 1
                m["trunc"] |= o["trunc"]
                break
        else:
            merged.append(dict(o, mult=1))
    return merged