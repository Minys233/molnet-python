import molnet
from molnet.convert2graph import CustomMoleculeDataset

for name in molnet.molnet_list:
    print(name)
    train_dataset, valid_dataset, test_dataset = molnet.load(name, '../datasets', save_whole_dataset=True)
    graph_dataset_train = CustomMoleculeDataset('../datasets/custom', train_dataset.mols, train_dataset.y,
                                                train_dataset.w, name='train')
    graph_dataset_valid = CustomMoleculeDataset('../datasets/custom', valid_dataset.mols, valid_dataset.y,
                                                valid_dataset.w, name='valid')
    graph_dataset_test  = CustomMoleculeDataset('../datasets/custom', test_dataset.mols, test_dataset.y,
                                                test_dataset.w, name='test')
    print(train_dataset, valid_dataset, test_dataset)
    print(graph_dataset_train, graph_dataset_valid, graph_dataset_test)

