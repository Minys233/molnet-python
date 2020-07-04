import molnet.convert2graph as cvt
import molnet
from rdkit import Chem


train_mol, valid_mol, test_mol = molnet.load('BBBP', '../datasets')

train_set = cvt.CustomMoleculeDataset('../datasets/bbbp', mols=train_mol.mols, y=train_mol.y, w=train_mol.w, name='BBBP_train')
valid_set = cvt.CustomMoleculeDataset('../datasets/bbbp', mols=valid_mol.mols, y=valid_mol.y, w=valid_mol.w, name='BBBP_valid')
test_set =  cvt.CustomMoleculeDataset('../datasets/bbbp', mols=test_mol.mols, y=test_mol.y, w=test_mol.w, name='BBBP_test')

print(train_set)
