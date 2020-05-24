from pathlib import Path
import molnet
import numpy as np

datasets = molnet.load('ESOL', './', False, False, 1.0, 1,)
print(datasets)
print(datasets[0].mols[:5])
print()
print(datasets[0].y[:5])
print()
print(datasets[0].w[:5])

