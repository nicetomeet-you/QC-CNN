import os
from torch.utils.data import Dataset
import sys
import numpy as np
import torch
from sklearn.preprocessing import normalize


# visualize
# fig = plt.figure()
# examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

class MyCSVDatasetReader(Dataset):

    def __init__(self, csv_path):
        print(csv_path)
        csv_path = "D:/Pycharm/RangeNet/QC-CNN" + csv_path.strip(".")
        assert os.path.isfile(csv_path)
        self.DATA = np.genfromtxt(csv_path, delimiter = ',')
        #self.DATA = self.DATA[:200]
        self.X = self.DATA[:, 0:-1]
        self.Y = self.DATA[:, -1]
        #self.X = np.pi*normalize(self.X, axis = 0, norm = 'max')
        self.X = np.pi*self.X/255
        
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = torch.FloatTensor(self.X[idx, :])
        Y = self.Y[idx]
        sample = {'feature': X, 'label': Y}
        return sample

    def _get_labels(self):
        return self.Y

if __name__ == '__main__':
    dataset = MyCSVDatasetReader('./app/digits.csv')
    sample = dataset.__getitem__(int(sys.argv[1]))
    print(sample)
