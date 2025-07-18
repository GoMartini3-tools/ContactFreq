# High-frequency-Contacts
We provide a set of scripts to calculate high frequency native contacts from a MD trayectories.

To better capture the nanomechanical behavior of this complex, we can use the AA-MD simulation used in Part I. This simulation will allow to determine the most persistent protein contacts, as described in [8]. The AA-MD trajectory is located in the All-atom directory. We will use `traj_to_pdb.py` and `contact_calculation.py` (located in the template folder) to convert the trajectory to pdb format and to calculate high-frequency contacts as follows:

```bash
python traj_to_pdb.py  --trajectory 6ZH9_WT_dry_100.nc --topology 6ZH9_WT_dry.parm7  --ranges 1-195,195-323 --outdir . --stride 1
```

* --trajectory specifies the trajectory file (e.g., .xtc, .dcd). In this case, the file 6ZH9_WT_dry.nc is located in the 6ZH9_AA folder, with each frame representing 1 ns.

* --topology refers to the coordinate file (e.g., .pdb, .gro)

* --ranges defines the residue blocks corresponding to each chain (e.g., A: 2-196, B: 197-325)

* --outdir sets the output directory for the generated PDB files

* --stride allows skipping n frames


The `contact_calculation.py` script automates the analysis of PDB trajectory frames to generate a CG topology with filtered high-frequency contacts. It performs the following steps:

* Contact detection: Identifies all frame_*.pdb files and runs contact_map in parallel to compute atom contacts per frame.

* Annotation: Labels contacts as intra- or inter-chain, removes symmetric duplicates, and annotates files accordingly.

* Frequency calculation: Aggregates all contacts to compute their framewise frequency. Writes normalized_type.txt and selects high-frequency contacts (--threshold) into high_type.txt.

* Reference selection: Identifies the frame with the most high-frequency contacts to serve as input for coarse-graining.

* Coarse-graining and Gō potential generation: Runs martinize2 on the selected reference. Constructs a virtual-site Gō model, filters native contacts, and builds go_nbparams.itp.

* Optional optimization: Contacts missing from the reference but exceeding the threshold are appended to go_nbparams.itp under a clearly marked section. This can be removed manually if undesired.

Output organization: Moves all .txt, .map, and intermediate .pdb files (except the final CG PDB) into the output_files/ folder.




```bash

python contact_calculation.py --short 3.0 --long 11.0 --cm /path/to/contact_map --type inter --cpus 15 --threshold 0.7 --merge all --dssp /usr/bin/mkdssp --go-eps 15.0
```

* --short and --long define the minimum and maximum distance cutoffs for contact detection (in Å).
* --cm points to the directory containing the contact_map executable. Available here [here](https://zenodo.org/records/3817447) or [here](https://github.com/Martini-Force-Field-Initiative/GoMartini/tree/main/ContactMapGenerator)
* --type sets the analysis scope (`both`, `intra`, or `inter`).
* --cpus defines the number of parallel workers for contact computation.
* --threshold is the minimum frequency to classify a contact as high-frequency (0.7 is recommended).
* --merge lists the chains to merge prior to running martinize2.
* --dssp provides the path to the DSSP executable (e.g., /usr/bin/mkdssp).
* --go-eps sets the epsilon value (in kJ/mol) for the Gō-model potential (a default value of 12.0 or 15.0 is suggested).
* --from indicates the origin of the topology (amber or charmm).
