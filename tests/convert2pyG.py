import molnet.convert2graph as cvt
import molnet
from rdkit import Chem


train_mol, valid_mol, test_mol = molnet.load('Tox21', '../datasets')

train_set = cvt.CustomMoleculeDataset('../datasets/tox21', mols=train_mol.mols, y=train_mol.y, w=train_mol.w, name='Tox21_train')
valid_set = cvt.CustomMoleculeDataset('../datasets/tox21', mols=valid_mol.mols, y=valid_mol.y, w=valid_mol.w, name='Tox21_valid')
test_set =  cvt.CustomMoleculeDataset('../datasets/tox21', mols=test_mol.mols, y=test_mol.y, w=test_mol.w, name='Tox21_test')

print(train_set.data.x.min(), train_set.data.x.max())
std, mean = train_set.normalize()
print(mean, std)
print(train_set.data.x.min(), train_set.data.x.max())

print(train_set)
