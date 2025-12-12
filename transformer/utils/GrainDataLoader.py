import grain


class Source(grain.Dataloader):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return index


data_loader = grain.DataLoader(Source(10))
