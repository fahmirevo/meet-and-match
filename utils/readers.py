from PIL import Image
import os
import random
import itertools
from torchvision import transforms


class ImageReader:

    def __init__(self, n_labels, n_samples, root='dataset'):
        self.n_labels = n_labels
        self.n_samples = n_samples
        self.root = root
        self.data = []

        self.load()

    def get_dirs(self, path):
        """
        os.listdir but append the root
        """

        dirs = os.listdir(path)
        return [os.path.join(path, dir) for dir in dirs]

    def prepare(self):
        self.labels = self.get_dirs(self.root)
        self.paths = list(map(self.get_dirs, self.labels))

    def load(self):
        self.prepare()
        combined = list(zip(self.labels, self.paths))
        random.shuffle(combined)
        self.labels, self.paths = zip(*combined)
        data = self.paths[:self.n_labels]
        data = [datum[:self.n_samples] for datum in data]
        self.data = list(itertools.chain.from_iterable(data))
        random.shuffle(self.data)


    def read(self, key=None):
        path = None

        if not self.data:
            self.load()
            return False

        if key is not None:
            assert key < self.n_labels
            label = self.labels[key]
            for i in range(len(self.data)):
                if label in self.data[i]:
                    path =  self.data.pop(i)
                    break
        else:
            path = self.data.pop()

        if path is None:
            return

        for i, label in enumerate(self.labels):
            if label in path:
                break

        try:
            im = Image.open(path).convert('RGB')
            im = transforms.ToTensor()(im)
            return i, im
        except OSError:
            return self.read(key=key)
