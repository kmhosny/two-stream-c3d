import h5py
from lazy5.inspect import get_datasets
from lazy5.inspect import get_attrs_dset
import sys
from PyQt5.QtWidgets import QApplication
from lazy5.ui.QtHdfLoad import HdfLoad

app = QApplication(sys.argv)

result = HdfLoad.getFileDataSets(pth='.')
print('Result: {}'.format(result))

sys.exit()

filename = 'models/sports1M_weights_tf.h5'
fid = h5py.File(filename, 'r')

dset_list = get_datasets(fid)

print('Datasets:')
for dset in dset_list:
    print(dset)

attr_dict = get_attrs_dset(filename, '/')
print('Dataset Attributes:')
for k in attr_dict:
    print attr_dict
    print('{} : {}'.format(k, attr_dict[k]))
fid.close()
