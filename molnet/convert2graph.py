import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch

import numpy as np
from rdkit import Chem
"""
This script is modified from https://github.com/deepchem/deepchem/blob/master/deepchem/feat/graph_features.py
"""


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def safe_index(lst, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return lst.index(e)
    except:
        return len(lst)


def calc_gasteiger_charges(mol_or_atom, iter=12):
    if isinstance(mol_or_atom, Chem.Atom):
        atom = mol_or_atom
    else:
        assert isinstance(mol_or_atom, Chem.Mol)
        atom = mol_or_atom.GetAtomWithIdx(0)
    try:
        mol_or_atom.GetProp('_GasteigerCharge')
    except KeyError:
        mol = atom.GetOwningMol()
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=iter, throwOnParamFailure=True)


def get_feature_list(atom):
    possible_atom_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'Te', 'I']
    possible_numH_list = [0, 1, 2, 3, 4]
    possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
    possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
    possible_number_radical_e_list = [0, 1, 2]
    possible_hybridization_list = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features = 8 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list, atom.GetNumRadicalElectrons())
    features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
    calc_gasteiger_charges(atom)
    features[6] = float(atom.GetProp('_GasteigerCharge'))
    features[7] = float(atom.GetProp('_GasteigerHCharge'))
    return features


def atom_features(atom, explicit_H=False, use_chirality=True, gasteiger_charges_iter=12):
    results = one_of_k_encoding_unk(atom.GetSymbol(),
                                    ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'Te', 'I', 'other']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [
                atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]
    calc_gasteiger_charges(atom)
    results += [float(atom.GetProp('_GasteigerCharge')), float(atom.GetProp('_GasteigerHCharge'))]
    return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def mol_to_pyG_data(mol, explicit_H=False, use_chirality=True, gasteiger_charges_iter=12):
    calc_gasteiger_charges(mol)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_features(atom, explicit_H=explicit_H, use_chirality=use_chirality)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    # bonds, we will force molecules have at least 2 atoms
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feature = bond_features(bond, use_chirality=use_chirality)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
