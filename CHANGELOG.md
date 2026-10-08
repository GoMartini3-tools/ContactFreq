# Changelog

## v1.0.0 (2026-10-08)

Fixes and improvements applied on top of `contact_freq.py` (including the `--ff`, `--ff-dir`, `--map-dir` and `--sigma` options already on `main`).

### Fixed
- Sequence separation filter (`|i2 - i1| >= 4`) is now applied only to contacts within the same chain. Inter-chain contacts between residues with close numbering were previously discarded.
- High-frequency threshold is applied to the exact fraction instead of the value rounded to two decimals (a frequency of 0.696 previously passed a 0.7 threshold).
- Pair frequencies use an orientation independent key and no longer include residue names, so the same contact is no longer split across entries when `contact_map` reports (A,B) vs (B,A) or when residue names vary (HIS/HID/HIE).
- Missing-contact distances are measured on the deduplicated frame list; `*_CG.pdb` files and duplicate PDB/CIF versions of the same frame are no longer included.
- `contact_map` handles are closed, stderr is reported, and a missing executable or all-empty maps now raise a clear error.
- Only files produced by the pipeline are moved to `output_files/` (previously every `*.txt` in the directory).
- `-cys` is no longer passed twice to martinize2; `next(hf)` on an empty file no longer raises; `run.log` is no longer written for `-h`.

### Changed
- Command-line options of `contact_freq.py`, `traj_to_pdb.py` and `traj_to_cif.py` now use a single dash, as in martinize2 (`-dssp`, `-merge`, `-go-eps`, `-trajectory`, ...). The double-dash spelling is still accepted and translated with a deprecation note. Abbreviated option names are no longer accepted.
- Distances for missing contacts are computed by reading CA records directly and in parallel, replacing one MDAnalysis selection per pair and frame.
- Duplicate bead pairs are removed from `missing_high_freq.itp`.
- The Go site prefix is a single constant (`MOLNAME`).
- `run.log` also records the martinize2 version.

### Added
- martinize2 options: `--ignore`, `--model`, `--posres-fc`, `--martinize-extra`; `--merge` is repeatable; `--dssp` accepts no value (mdtraj) and is mutually exclusive with `--ss`.
- `--min-seq-sep` for the intra-chain separation filter.
