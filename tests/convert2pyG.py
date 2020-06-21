import molnet.convert2graph as cvt
from rdkit import Chem


smilst = open('test.smi').readlines()
mollst = [Chem.MolFromSmiles(s) for s in smilst]
mol = mollst[0]
print(Chem.MolToSmiles(mol))

for idx, atom in enumerate(mol.GetAtoms()):
    f = cvt.atom_features(atom)
    print(idx, f[:13], f[13:19], f[19:21], f[21:27], f[27], f[27:32], f[32:36], f[36:])

data = cvt.mol_to_pyG_data(mol)
print(data)
