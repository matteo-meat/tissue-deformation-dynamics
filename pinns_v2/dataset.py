from torch.utils.data import Dataset
import numpy as np
import math

class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n, batchsize = None, shuffle = True, period = 1, seed = 1234):
        self.name = "DomainDataset"
        self.xmin = np.array(xmin, dtype="f")
        self.xmax = np.array(xmax, dtype="f")
        self.dim = len(xmin)
        self.n = n
        self.shuffle = shuffle
        self.period = period
        if batchsize != None:
            self.n_points_per_axis = batchsize
        else:
            self.n_points_per_axis = n
        self.rng = np.random.default_rng(seed)
        self.seed = self.rng.integers(0, 10000)
        self.rng2 = np.random.default_rng(self.seed)
        if shuffle:
            self.period_counter = 0 
        self.counter = 0

    def __len__(self):
        return 1 if self.n_points_per_axis == self.n else int(math.ceil(self.n/self.n_points_per_axis))
    
    def _sample_items(self, length):
        x = self.rng2.uniform(low=self.xmin[0], high=np.nextafter(self.xmax[0], self.xmax[0]+1), size=(length, 1))
        for i in range(1, self.dim):
            s = self.rng2.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(length, 1))
            if i == self.dim - 1:
                #sort for ascending time
                s = s.reshape(length, )
                s = np.sort(s)
                s = s.reshape(length, 1)
            x = np.hstack((x, s))
        return x
    
    def __getitem__(self, idx):
        if self.counter >= self.n:
            #time to reset the dataset generator
            self.rng2 = np.random.default_rng(self.seed)
            self.counter = 0
            if self.shuffle:
                self.period_counter += 1
        if self.shuffle and self.period_counter >= self.period:
            #time to change the dataset generator
            self.seed = self.rng.integers(0, 10000)
            self.rng2 = np.random.default_rng(self.seed)
            self.counter = 0
            self.period_counter = 0
        length = self.n - self.counter if self.n - self.counter < self.n_points_per_axis else self.n_points_per_axis
        x = self._sample_items(length)
        self.counter += length 
        return x
    
    def get_params(self):
        return {"x_min": self.xmin, "x_max": self.xmax, "n": self.n, "batchsize": self.n_points_per_axis, "shuffle": self.shuffle, "period":self.period}
    
    def __str__(self):
        return f"{self.name}: {self.get_params()}"
    

class ICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, batchsize = None, shuffle = True, period = 1):
        super().__init__(xmin, xmax, n, batchsize = batchsize, shuffle = shuffle, period = period)
        self.name = "ICDataset"
        
    def _sample_items(self, length):

        x = self.rng2.uniform(low=self.xmin[0], high=np.nextafter(self.xmax[0], self.xmax[0]+1), size=(length, ))
        for i in range(1, self.dim):
            s = self.rng2.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(length, ))
            x = np.vstack((x, s))
        x = np.vstack((x, np.zeros(length, )))
        return x.T
    
    
    def __str__(self):
        s = f"ICDataset({self.xmin}, {self.xmax}, n={self.n}, shuffle={self.shuffle}, period={self.period})"
        return s