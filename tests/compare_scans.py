"""
compare_scans.py
================
Load two phase-diagram scans (.npz) and print every saved parameter side by side,
flagging the ones that differ. Also summarizes how the results differ.

Usage:
    python compare_scans.py scan_A.npz scan_B.npz
or edit FILE_A / FILE_B below and run with no arguments.
"""

import sys
import numpy as np
import model.config as config

FILE_A = config.DATA_DIR / "phase_diagram_nD_PIP_170pts.npz"
FILE_B = config.DATA_DIR / "phase_diagram_nD_150pts.npz"

if len(sys.argv) == 3:
    FILE_A, FILE_B = sys.argv[1], sys.argv[2]

# allow_pickle is needed because phase_labels is an object array
A = np.load(FILE_A, allow_pickle=True)
B = np.load(FILE_B, allow_pickle=True)

# Keys that hold the scan axes / scalar parameters vs. the result maps
RESULT_KEYS = {"phase_map", "phase_labels", "energy_map",
               "density_map", "mu_map", "conv_map"}


def fmt(x):
    """Compact one-line description of a saved entry."""
    x = np.asarray(x)
    if x.ndim == 0:
        return f"{x.item():.6g}" if x.dtype.kind in "fc" else str(x.item())
    if x.dtype.kind in "fiu":
        return f"shape={x.shape}  [{np.nanmin(x):.6g} .. {np.nanmax(x):.6g}]"
    return f"shape={x.shape}  dtype={x.dtype}"


def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc" and y.dtype.kind in "fc":
        return np.allclose(x, y, equal_nan=True)
    return np.array_equal(x, y)


print(f"A: {FILE_A}\nB: {FILE_B}\n")

# ---------- Parameters and axes ----------
keys = sorted(set(A.files) | set(B.files))
param_keys = [k for k in keys if k not in RESULT_KEYS]

print("=" * 78)
print("PARAMETERS & AXES")
print("=" * 78)
for k in param_keys:
    if k not in A.files or k not in B.files:
        print(f"  [!] {k:12s} only in {'A' if k in A.files else 'B'}")
        continue
    flag = "   " if same(A[k], B[k]) else "[!]"
    print(f"  {flag} {k}")
    print(f"        A: {fmt(A[k])}")
    print(f"        B: {fmt(B[k])}")

# ---------- Results ----------
print("\n" + "=" * 78)
print("RESULTS")
print("=" * 78)
for k in sorted(RESULT_KEYS & set(A.files) & set(B.files)):
    print(f"  {k:12s} A: shape={A[k].shape}  B: shape={B[k].shape}")

if "conv_map" in A.files and "conv_map" in B.files:
    ca, cb = A["conv_map"], B["conv_map"]
    print(f"\n  Converged points:  A: {ca.sum()}/{ca.size}   B: {cb.sum()}/{cb.size}")

if "phase_map" in A.files and "phase_map" in B.files:
    pa, pb = A["phase_map"], B["phase_map"]
    print(f"  Phase indices found:  A: {sorted(np.unique(pa[pa >= 0]).tolist())}"
          f"   B: {sorted(np.unique(pb[pb >= 0]).tolist())}")

# ---------- Point-by-point comparison (only if grids are identical) ----------
grids_match = all(
    k in A.files and k in B.files and same(A[k], B[k])
    for k in ("n_vals", "D_vals")
)

print("\n" + "-" * 78)
if not grids_match:
    print("  (n, D) grids differ -> skipping point-by-point comparison.")
else:
    print("  Same (n, D) grid -> point-by-point comparison:")
    pa, pb = A["phase_map"], B["phase_map"]
    diff = pa != pb
    print(f"    Points with different phase: {diff.sum()}/{diff.size} "
          f"({100 * diff.mean():.2f}%)")

    ea, eb = A["energy_map"], B["energy_map"]
    both = np.isfinite(ea) & np.isfinite(eb)
    if both.any():
        dE = eb[both] - ea[both]
        print(f"    Energy B - A [eV cm^-2]: mean={dE.mean():.3e}  "
              f"min={dE.min():.3e}  max={dE.max():.3e}")
        print(f"    B lower in energy at {(dE < 0).sum()}/{both.sum()} points")

    if diff.any():
        iD, iN = np.where(diff)
        print("    First differing points (iD, iN, D[meV], n[cm^-2], phaseA -> phaseB):")
        for d, n in list(zip(iD, iN))[:10]:
            print(f"      ({d:3d}, {n:3d})  D={A['D_vals'][d]*1e3:6.2f}  "
                  f"n={A['n_vals'][n]:.3e}  {pa[d, n]} -> {pb[d, n]}")