#!/usr/bin/env python3

"""
prepare_docking_trio_benchmark.py

Create benchmark PDBs containing:

    receptor
      +
    connected ligand heavy-atom triplet A
      +
    connected ligand heavy-atom triplet B

The two retained triplets must be separated in the ligand graph by a
user-defined number of intervening atoms.

Supported input modes
---------------------

1. Complex PDB containing receptor + ligand:

    python prepare_docking_trio_benchmark.py \
        --complex 5TUE.pdb \
        --output-dir 5TUE_cases

2. Separate receptor and ligand:

    python prepare_docking_trio_benchmark.py \
        --receptor receptor.pdb \
        --ligand ligand.sdf \
        --output-dir test_cases

3. Override automatic ligand identification in a complex:

    python prepare_docking_trio_benchmark.py \
        --complex complex.pdb \
        --ligand-resname LIG \
        --ligand-chain A \
        --ligand-resid 501 \
        --output-dir cases

4. Exclude all-carbon atom trios:

    python prepare_docking_trio_benchmark.py \
        --complex complex.pdb \
        --output-dir cases \
        --exclude-all-carbon-trios

Ligand formats supported through RDKit:
    .pdb
    .sdf
    .mol
    .mol2

Notes
-----
* Hydrogens are excluded from trio enumeration.
* Only the largest covalently connected ligand fragment is used.
* PDB receptor cleanup removes HETATM records, including waters and ions.
* Receptor chains can optionally be removed if they are distant from ligand.
* --exclude-all-carbon-trios removes C-C-C trios while retaining other
  homogeneous trios such as N-N-N.
* A manifest.csv is written describing every output benchmark case.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors


# ---------------------------------------------------------------------------
# Common PDB non-drug components
# ---------------------------------------------------------------------------

WATER_NAMES = {
    "HOH", "WAT", "H2O", "DOD",
}

ION_NAMES = {
    "NA", "K", "CL", "BR", "I", "F",
    "CA", "MG", "ZN", "MN", "FE", "CO",
    "CU", "NI", "CD", "HG", "SR", "CS",
    "LI", "RB",
}

COMMON_ADDITIVES = {
    "SO4", "PO4", "NO3",
    "GOL", "EDO", "PEG", "PG4",
    "DMS", "MPD",
    "ACT", "FMT", "ACE",
    "TRS", "MES", "HEP",
    "BME",
}

COMMON_COFACTORS = {
    "FAD", "FMN",
    "NAD", "NAP", "NDP", "NAI",
    "ATP", "ADP", "AMP",
    "GTP", "GDP", "GMP",
    "CTP", "CDP",
    "UTP", "UDP",
    "HEM", "HEC", "HEA",
    "PLP",
    "SAM", "SAH",
    "COA",
}

COMMON_LIPIDS = {
    "CLR",
    "CHL",
    "OLA",
    "PLM",
    "STE",
    "POP",
    "POPC",
    "POPE",
    "DOPC",
    "DOPS",
}

AUTO_EXCLUDE = (
    WATER_NAMES
    | ION_NAMES
    | COMMON_ADDITIVES
    | COMMON_COFACTORS
    | COMMON_LIPIDS
)


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class PDBLigandCandidate:
    resname: str
    chain: str
    resid: str
    lines: list[str]
    mol: Chem.Mol | None = None
    heavy_atoms: int = 0
    molecular_weight: float | None = None
    clogp: float | None = None
    hetero_atoms: int = 0
    score: float = float("-inf")


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def first_model_lines(path: Path) -> list[str]:
    """
    Read only the first MODEL of a PDB.

    If no MODEL records exist, return the complete file.
    """

    lines = path.read_text().splitlines()

    has_model = any(line.startswith("MODEL") for line in lines)

    if not has_model:
        return lines

    output = []
    in_first_model = False

    for line in lines:
        if line.startswith("MODEL"):
            if not in_first_model:
                in_first_model = True
                continue

        if line.startswith("ENDMDL") and in_first_model:
            break

        if in_first_model:
            output.append(line)

    return output


def pdb_xyz(line: str) -> np.ndarray:
    return np.array(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        dtype=float,
    )


def pdb_element(line: str) -> str:
    """
    Get an element from a PDB atom line.

    Prefer the official element field, then fall back to the atom name.
    """

    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem:
            return elem

    atom_name = line[12:16].strip()

    atom_name = atom_name.lstrip("0123456789")

    if not atom_name:
        return ""

    if len(atom_name) >= 2 and atom_name[:2].title() in {
        "Cl", "Br", "Si", "Se", "Na", "Ca", "Mg", "Zn", "Fe",
        "Mn", "Co", "Cu", "Ni",
    }:
        return atom_name[:2].title()

    return atom_name[0].upper()


def group_hetero_residues(lines: list[str]):
    groups = defaultdict(list)

    for line in lines:
        if not line.startswith("HETATM"):
            continue

        resname = line[17:20].strip()
        chain = line[21].strip()
        resid = line[22:26].strip()
        icode = line[26].strip()

        key = (
            resname,
            chain,
            resid + icode,
        )

        groups[key].append(line)

    return groups


# ---------------------------------------------------------------------------
# RDKit loading
# ---------------------------------------------------------------------------

def rdkit_from_pdb_lines(lines: list[str]) -> Chem.Mol | None:

    block = "\n".join(lines) + "\nEND\n"

    mol = Chem.MolFromPDBBlock(
        block,
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol


def load_external_ligand(path: Path) -> Chem.Mol:

    suffix = path.suffix.lower()

    mol = None

    if suffix == ".pdb":
        mol = Chem.MolFromPDBFile(
            str(path),
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )

    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(
            str(path),
            sanitize=False,
            removeHs=False,
        )

    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(
            str(path),
            sanitize=False,
            removeHs=False,
        )

    elif suffix == ".sdf":
        supplier = Chem.SDMolSupplier(
            str(path),
            sanitize=False,
            removeHs=False,
        )

        molecules = [m for m in supplier if m is not None]

        if not molecules:
            raise RuntimeError(f"No molecules could be read from {path}")

        mol = max(
            molecules,
            key=lambda m: sum(
                a.GetAtomicNum() > 1
                for a in m.GetAtoms()
            ),
        )

    else:
        raise ValueError(
            f"Unsupported ligand extension: {suffix}\n"
            "Supported: PDB, SDF, MOL, MOL2"
        )

    if mol is None:
        raise RuntimeError(f"RDKit could not read ligand: {path}")

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        print(
            f"WARNING: RDKit sanitization failed for {path}: {exc}",
            file=sys.stderr,
        )

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            f"{path} does not contain ligand coordinates. "
            "A benchmark ligand must already have its experimental/"
            "reference pose."
        )

    return mol


# ---------------------------------------------------------------------------
# Ligand utilities
# ---------------------------------------------------------------------------

def largest_fragment_atom_indices(mol: Chem.Mol) -> set[int]:
    """
    Return atom indices belonging to the largest connected component,
    measured by heavy-atom count.
    """

    fragments = Chem.GetMolFrags(
        mol,
        asMols=False,
        sanitizeFrags=False,
    )

    if not fragments:
        return set()

    def fragment_score(fragment):
        return sum(
            mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            for i in fragment
        )

    largest = max(fragments, key=fragment_score)

    return set(largest)


def ligand_heavy_atom_indices(mol: Chem.Mol) -> list[int]:

    main_fragment = largest_fragment_atom_indices(mol)

    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if (
            atom.GetIdx() in main_fragment
            and atom.GetAtomicNum() > 1
        )
    ]


def ligand_coordinates(
    mol: Chem.Mol,
    indices: list[int] | None = None,
) -> np.ndarray:

    conf = mol.GetConformer()

    if indices is None:
        indices = ligand_heavy_atom_indices(mol)

    xyz = []

    for idx in indices:
        p = conf.GetAtomPosition(idx)
        xyz.append([p.x, p.y, p.z])

    return np.asarray(xyz, dtype=float)


# ---------------------------------------------------------------------------
# Automatic ligand identification
# ---------------------------------------------------------------------------

def evaluate_candidate(candidate: PDBLigandCandidate):

    resname = candidate.resname.upper()

    if resname in AUTO_EXCLUDE:
        return

    heavy_from_pdb = sum(
        pdb_element(line).upper() not in {"H", "D"}
        for line in candidate.lines
    )

    if heavy_from_pdb < 6:
        return

    mol = rdkit_from_pdb_lines(candidate.lines)

    if mol is None:
        return

    candidate.mol = mol

    heavy_indices = ligand_heavy_atom_indices(mol)

    heavy_atoms = len(heavy_indices)

    if heavy_atoms < 6:
        return

    candidate.heavy_atoms = heavy_atoms

    candidate.hetero_atoms = sum(
        mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6}
        for i in heavy_indices
    )

    if candidate.hetero_atoms == 0:
        return

    mw = None
    clogp = None

    try:
        mw = Descriptors.MolWt(mol)
    except Exception:
        pass

    try:
        clogp = Crippen.MolLogP(mol)
    except Exception:
        pass

    candidate.molecular_weight = mw
    candidate.clogp = clogp

    if mw is not None and mw < 120:
        return

    if mw is not None and mw > 1200:
        return

    if clogp is not None and clogp > 7.0:
        return

    candidate.score = (
        min(heavy_atoms, 50)
        + 2.0 * min(candidate.hetero_atoms, 10)
    )

    if mw is not None and 180 <= mw <= 700:
        candidate.score += 10

    if clogp is not None and -2 <= clogp <= 6:
        candidate.score += 5


def select_ligand_from_complex(
    pdb_lines: list[str],
    resname_override: str | None = None,
    chain_override: str | None = None,
    resid_override: str | None = None,
) -> PDBLigandCandidate:

    groups = group_hetero_residues(pdb_lines)

    candidates = []

    for (resname, chain, resid), lines in groups.items():

        if (
            resname_override is not None
            and resname != resname_override
        ):
            continue

        if (
            chain_override is not None
            and chain != chain_override
        ):
            continue

        if (
            resid_override is not None
            and resid != resid_override
        ):
            continue

        candidate = PDBLigandCandidate(
            resname=resname,
            chain=chain,
            resid=resid,
            lines=lines,
        )

        evaluate_candidate(candidate)

        if candidate.mol is not None and math.isfinite(candidate.score):
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "No suitable ligand was identified.\n\n"
            "For a known complex, specify it explicitly with e.g.:\n"
            "  --ligand-resname ABC --ligand-chain A "
            "--ligand-resid 501"
        )

    candidates.sort(
        key=lambda c: c.score,
        reverse=True,
    )

    print("\nLigand candidates:")
    for c in candidates[:10]:
        print(
            f"  {c.resname:>4s} "
            f"chain={c.chain or '-':1s} "
            f"resid={c.resid:>5s} "
            f"heavy={c.heavy_atoms:3d} "
            f"hetero={c.hetero_atoms:2d} "
            f"MW={c.molecular_weight!s:>8} "
            f"cLogP={c.clogp!s:>8} "
            f"score={c.score:.1f}"
        )

    winner = candidates[0]

    print(
        "\nSelected ligand:",
        winner.resname,
        f"chain={winner.chain or '-'}",
        f"resid={winner.resid}",
    )

    return winner


# ---------------------------------------------------------------------------
# Connected atom triplets
# ---------------------------------------------------------------------------

def enumerate_connected_triplets(
    mol: Chem.Mol,
    allowed_indices: set[int],
) -> list[tuple[int, int, int]]:
    """
    Enumerate all unique connected 3-heavy-atom sets.

    Every connected three-node graph must contain an atom connected to
    the other two, so enumeration can be done by taking pairs of
    neighbors around every possible center atom.
    """

    triplets = set()

    for center in allowed_indices:

        atom = mol.GetAtomWithIdx(center)

        neighbors = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if (
                nbr.GetIdx() in allowed_indices
                and nbr.GetAtomicNum() > 1
            )
        ]

        for left, right in itertools.combinations(neighbors, 2):

            triplet = tuple(sorted((left, center, right)))

            if len(set(triplet)) == 3:
                triplets.add(triplet)

    return sorted(triplets)


def triplet_is_all_carbon(
    mol: Chem.Mol,
    triplet: tuple[int, int, int],
) -> bool:
    """
    Return True if all three atoms in the triplet are carbon.

    Examples:
        C-C-C -> True
        C-C-N -> False
        C-N-O -> False
        N-N-N -> False

    This filter is intentionally specific to carbon-only trios.
    """

    return all(
        mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
        for atom_idx in triplet
    )


def graph_distance_matrix(mol: Chem.Mol):
    """
    RDKit topological distance matrix:
        1 = directly bonded
        2 = two bonds
        etc.
    """

    return Chem.GetDistanceMatrix(mol)


def triplet_pair_is_valid(
    triplet_a,
    triplet_b,
    graph_distances,
    minimum_intervening_atoms: int,
) -> bool:

    if set(triplet_a) & set(triplet_b):
        return False

    minimum_bond_distance = minimum_intervening_atoms + 1

    for a in triplet_a:
        for b in triplet_b:

            distance = graph_distances[a, b]

            if distance < minimum_bond_distance:
                return False

    return True


# ---------------------------------------------------------------------------
# Receptor cleanup
# ---------------------------------------------------------------------------

def receptor_atom_lines(pdb_lines: list[str]) -> list[str]:
    """
    Keep standard ATOM records only.

    This removes:
        waters
        ions
        ligand
        crystallization additives
        most cofactors

    Note that HETATM-formatted modified protein residues such as MSE
    are also omitted by this simple implementation.
    """

    return [
        line
        for line in pdb_lines
        if line.startswith("ATOM")
    ]


def select_nearby_chains(
    receptor_lines: list[str],
    ligand_xyz: np.ndarray,
    cutoff: float | None,
) -> list[str]:

    if cutoff is None:
        return receptor_lines

    chain_coords = defaultdict(list)

    for line in receptor_lines:
        chain = line[21].strip()
        chain_coords[chain].append(pdb_xyz(line))

    keep_chains = set()

    cutoff_squared = cutoff * cutoff

    for chain, coords in chain_coords.items():

        coords = np.asarray(coords)

        delta = (
            coords[:, None, :]
            - ligand_xyz[None, :, :]
        )

        d2 = np.sum(delta * delta, axis=2)

        if np.min(d2) <= cutoff_squared:
            keep_chains.add(chain)

    if not keep_chains:
        raise RuntimeError(
            "No receptor chains fall within "
            f"{cutoff:.1f} Å of the ligand."
        )

    print(
        "Retaining receptor chain(s):",
        ", ".join(sorted(c or "<blank>" for c in keep_chains)),
    )

    return [
        line
        for line in receptor_lines
        if line[21].strip() in keep_chains
    ]


# ---------------------------------------------------------------------------
# PDB output
# ---------------------------------------------------------------------------

def atom_name_for_rdkit_atom(atom: Chem.Atom, idx: int) -> str:

    info = atom.GetPDBResidueInfo()

    if info is not None:
        name = info.GetName().strip()
        if name:
            return name[:4]

    return f"{atom.GetSymbol()}{idx + 1}"[:4]


def ligand_atom_pdb_line(
    mol: Chem.Mol,
    atom_idx: int,
    serial: int,
    resname: str,
    chain: str,
    resid: int,
) -> str:

    atom = mol.GetAtomWithIdx(atom_idx)
    conf = mol.GetConformer()

    p = conf.GetAtomPosition(atom_idx)

    atom_name = atom_name_for_rdkit_atom(atom, atom_idx)

    element = atom.GetSymbol().upper()

    return (
        f"HETATM{serial:5d} "
        f"{atom_name:>4s} "
        f"{resname:>3s} "
        f"{chain:1s}"
        f"{resid:4d}    "
        f"{p.x:8.3f}"
        f"{p.y:8.3f}"
        f"{p.z:8.3f}"
        f"{1.00:6.2f}"
        f"{0.00:6.2f}"
        f"          "
        f"{element:>2s}"
    )


def max_pdb_serial(lines: list[str]) -> int:

    maximum = 0

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                maximum = max(
                    maximum,
                    int(line[6:11]),
                )
            except ValueError:
                pass

    return maximum


def trio_bonds(mol, atom_indices):

    atom_set = set(atom_indices)

    bonds = []

    for bond in mol.GetBonds():

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in atom_set and b in atom_set:
            bonds.append((a, b))

    return bonds


def write_case(
    output_path: Path,
    receptor_lines: list[str],
    ligand: Chem.Mol,
    triplet_a,
    triplet_b,
):

    serial = max_pdb_serial(receptor_lines) + 1

    atom_serials = {}

    ligand_lines = []

    for resname, chain, resid, triplet in [
        ("T01", "X", 1, triplet_a),
        ("T02", "Y", 2, triplet_b),
    ]:

        for atom_idx in triplet:

            atom_serials[atom_idx] = serial

            ligand_lines.append(
                ligand_atom_pdb_line(
                    ligand,
                    atom_idx,
                    serial,
                    resname,
                    chain,
                    resid,
                )
            )

            serial += 1

    conect = []

    for triplet in (triplet_a, triplet_b):

        for a, b in trio_bonds(ligand, triplet):

            sa = atom_serials[a]
            sb = atom_serials[b]

            conect.append(f"CONECT{sa:5d}{sb:5d}")
            conect.append(f"CONECT{sb:5d}{sa:5d}")

    with output_path.open("w") as fh:

        fh.write(
            "REMARK Generated ligand-triplet docking benchmark\n"
        )

        for line in receptor_lines:
            fh.write(line.rstrip() + "\n")

        fh.write("TER\n")

        for line in ligand_lines:
            fh.write(line + "\n")

        for line in sorted(set(conect)):
            fh.write(line + "\n")

        fh.write("END\n")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def atom_description(mol: Chem.Mol, idx: int) -> str:

    atom = mol.GetAtomWithIdx(idx)

    name = atom_name_for_rdkit_atom(atom, idx)

    return f"{idx}:{name}:{atom.GetSymbol()}"


def triplet_element_string(
    mol: Chem.Mol,
    triplet: tuple[int, int, int],
) -> str:
    """
    Return a compact element description such as:
        C-C-N
        N-N-N
        C-O-S
    """

    return "-".join(
        mol.GetAtomWithIdx(i).GetSymbol()
        for i in triplet
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--complex",
        type=Path,
        help="PDB containing receptor and ligand",
    )

    mode.add_argument(
        "--receptor",
        type=Path,
        help="Receptor PDB when ligand is provided separately",
    )

    parser.add_argument(
        "--ligand",
        type=Path,
        help="Separate ligand file: PDB/SDF/MOL/MOL2",
    )

    parser.add_argument(
        "--ligand-resname",
        help="Explicit ligand residue name for complex PDB",
    )

    parser.add_argument(
        "--ligand-chain",
        help="Explicit ligand chain ID for complex PDB",
    )

    parser.add_argument(
        "--ligand-resid",
        help="Explicit ligand residue number for complex PDB",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--min-intervening-atoms",
        type=int,
        default=2,
        help=(
            "Minimum number of ligand atoms lying between every atom "
            "of one retained trio and every atom of the other trio"
        ),
    )

    parser.add_argument(
        "--chain-distance-cutoff",
        type=float,
        default=12.0,
        help=(
            "Remove receptor chains having no atom within this many "
            "Angstroms of the ligand. Use 0 to keep every chain."
        ),
    )

    parser.add_argument(
        "--exclude-all-carbon-trios",
        action="store_true",
        help=(
            "Exclude connected ligand atom triplets composed entirely "
            "of carbon atoms (C-C-C). Other homogeneous triplets such "
            "as N-N-N are retained."
        ),
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of outputs, useful for testing",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Load receptor + ligand
    # ---------------------------------------------------------------

    if args.complex:

        pdb_lines = first_model_lines(args.complex)

        ligand_candidate = select_ligand_from_complex(
            pdb_lines,
            args.ligand_resname,
            args.ligand_chain,
            args.ligand_resid,
        )

        ligand = ligand_candidate.mol

        receptor_lines = receptor_atom_lines(pdb_lines)

    else:

        if args.ligand is None:
            parser.error(
                "--receptor requires --ligand"
            )

        pdb_lines = first_model_lines(args.receptor)

        receptor_lines = receptor_atom_lines(pdb_lines)

        ligand = load_external_ligand(args.ligand)

    if ligand is None:
        raise RuntimeError("No ligand loaded.")

    # ---------------------------------------------------------------
    # Heavy atoms / primary fragment
    # ---------------------------------------------------------------

    heavy_indices = ligand_heavy_atom_indices(ligand)

    print(
        f"\nLigand heavy atoms in largest connected fragment: "
        f"{len(heavy_indices)}"
    )

    if len(heavy_indices) < 6:
        raise RuntimeError(
            "Ligand contains fewer than six connected heavy atoms; "
            "two independent three-atom motifs cannot be generated."
        )

    allowed = set(heavy_indices)

    # ---------------------------------------------------------------
    # Clean receptor
    # ---------------------------------------------------------------

    ligand_xyz = ligand_coordinates(
        ligand,
        heavy_indices,
    )

    cutoff = args.chain_distance_cutoff

    if cutoff == 0:
        cutoff = None

    receptor_lines = select_nearby_chains(
        receptor_lines,
        ligand_xyz,
        cutoff,
    )

    print(
        f"Receptor atoms retained: {len(receptor_lines)}"
    )

    # ---------------------------------------------------------------
    # Enumerate triplets
    # ---------------------------------------------------------------

    triplets = enumerate_connected_triplets(
        ligand,
        allowed,
    )

    print(
        f"Connected heavy-atom triplets: {len(triplets)}"
    )

    if args.exclude_all_carbon_trios:

        original_count = len(triplets)

        triplets = [
            triplet
            for triplet in triplets
            if not triplet_is_all_carbon(
                ligand,
                triplet,
            )
        ]

        removed_count = original_count - len(triplets)

        print(
            "All-carbon trio filtering enabled: "
            f"{len(triplets)}/{original_count} triplets retained "
            f"({removed_count} C-C-C triplets removed)"
        )

    if len(triplets) < 2:
        raise RuntimeError(
            "Fewer than two eligible triplets remain after filtering."
        )

    graph_distances = graph_distance_matrix(ligand)

    valid_pairs = []

    for triplet_a, triplet_b in itertools.combinations(
        triplets,
        2,
    ):

        if triplet_pair_is_valid(
            triplet_a,
            triplet_b,
            graph_distances,
            args.min_intervening_atoms,
        ):
            valid_pairs.append(
                (triplet_a, triplet_b)
            )

    print(
        f"Valid separated triplet pairs: {len(valid_pairs)}"
    )

    if args.max_cases is not None:
        valid_pairs = valid_pairs[:args.max_cases]

        print(
            f"Writing first {len(valid_pairs)} cases "
            f"because --max-cases was specified."
        )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    manifest_path = (
        args.output_dir / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
    ) as manifest_file:

        writer = csv.writer(manifest_file)

        writer.writerow(
            [
                "case",
                "filename",
                "triplet_A_indices",
                "triplet_A_atoms",
                "triplet_A_elements",
                "triplet_B_indices",
                "triplet_B_atoms",
                "triplet_B_elements",
                "minimum_graph_distance",
            ]
        )

        for case_number, (ta, tb) in enumerate(
            valid_pairs,
            start=1,
        ):

            filename = (
                f"case_{case_number:06d}.pdb"
            )

            output_path = (
                args.output_dir / filename
            )

            write_case(
                output_path,
                receptor_lines,
                ligand,
                ta,
                tb,
            )

            min_distance = min(
                int(graph_distances[a, b])
                for a in ta
                for b in tb
            )

            writer.writerow(
                [
                    case_number,
                    filename,
                    ";".join(map(str, ta)),
                    ";".join(
                        atom_description(ligand, i)
                        for i in ta
                    ),
                    triplet_element_string(
                        ligand,
                        ta,
                    ),
                    ";".join(map(str, tb)),
                    ";".join(
                        atom_description(ligand, i)
                        for i in tb
                    ),
                    triplet_element_string(
                        ligand,
                        tb,
                    ),
                    min_distance,
                ]
            )

    print(
        f"\nFinished."
        f"\nPDB cases: {args.output_dir}"
        f"\nManifest:  {manifest_path}"
    )


if __name__ == "__main__":
    main()
