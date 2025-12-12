import grain
from grain.samplers import IndexSampler
from grain.transforms import Batch


class Source(grain.Dataloader):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return index


source = Source(10)
index_sampler = IndexSampler(
    num_records=len(source),
    shard_options=grain.sharding.NoSharding,
    shuffle=True,
    num_epochs=1,
    seed=42,
)

grain.sample

data_loader = grain.DataLoader(
    data_source=source,
    sampler=index_sampler,
    operations=[Batch(batch_size=2, drop_remainder=True)],
    worker_count=2,
)
