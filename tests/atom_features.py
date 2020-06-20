from pathlib import Path
import molnet
import numpy as np
from rdkit import Chem
from collections import defaultdict
from tqdm import tqdm

count = defaultdict(int)
for name in molnet.molnet_list:
    print(f'On dataset {name}')
    datasets = molnet.load(name, '../datasets', split=(1,0,0))
    for mol in tqdm(datasets[0].mols):
        for atom in mol.GetAtoms():
            s = atom.GetSymbol()
            count[s] += 1

print(count)
"""
All MoleculeNet datasets except QM7b, 762841 molecules in total.
[('C', 12158671),
 ('O', 2059952),
 ('N', 1870105),
 ('S', 315260),
 ('F', 138033),
 ('Cl', 112903),
 ('Br', 27867),
 ('P', 5120),
 ('I', 1651),
 ('Na', 1003),
 ('Si', 795), <-- 0.104%
 ('B', 279), <-- 0.036%
 ('Sn', 228),
 ('Se', 201),
 ('Cu', 148),
 ('As', 147),
 ('H', 143),
 ('Co', 140),
 ('Fe', 102),
 ('K', 94),
 ('Ni', 88),
 ('Pt', 85),
 ('Al', 83),
 ('Zn', 80),
 ('Hg', 68),
 ('Pd', 49),
 ('Ca', 45),
 ('Sb', 44),
 ('Mn', 42),
 ('Cr', 42),
 ('Ge', 37),
 ('Au', 31),
 ('Bi', 31),
 ('Mo', 29),
 ('Li', 23),
 ('Rh', 22),
 ('Ru', 20),
 ('Ir', 20),
 ('Gd', 17),
 ('Ti', 13),
 ('Ba', 13),
 ('Mg', 13),
 ('Ag', 12),
 ('W', 12),
 ('Pb', 11),
 ('Cd', 10),
 ('Ga', 10),
 ('Tl', 10),
 ('Sr', 9),
 ('In', 9),
 ('Zr', 7),
 ('Re', 6),
 ('Te', 6),
 ('V', 4),
 ('Nd', 3),
 ('U', 3),
 ('Dy', 3),
 ('La', 3),
 ('Tc', 3),
 ('Be', 2),
 ('Yb', 2),
 ('Cf', 2),
 ('Ho', 1),
 ('Tb', 1),
 ('Cs', 1),
 ('Ac', 1),
 ('Eu', 1),
 ('Sc', 1),
 ('Ra', 1),
 ('Y', 1),
 ('Sm', 1),
 ('*', 1)]
"""