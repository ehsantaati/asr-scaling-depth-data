from typing import Iterator, TypeVar

from torch.utils import data

from . import datasets

T_co = TypeVar("T_co", covariant=True)


class PartitionedDataset(datasets.SizedIterableDataset):
    """
    Wraps a SizedIterableDataset and yields only samples belonging to a specific partition.
    Partitioning is done using modulo arithmetic on the sample index.
    """

    def __init__(
        self,
        dataset: datasets.SizedIterableDataset,
        partition_index: int,
        total_partitions: int,
    ) -> None:
        super().__init__()
        if partition_index < 0 or partition_index >= total_partitions:
            raise ValueError(
                f"partition_index must be in [0, {total_partitions - 1}], got {partition_index}"
            )
        
        self._dataset = dataset
        self._partition_index = partition_index
        self._total_partitions = total_partitions
        
        # Calculate length: floor(len / total)
        # Note: This is an approximation. Some partitions might have +1 sample.
        # For simplicity in training progress bars, we use floor.
        self._length = len(dataset) // total_partitions
        self._name = f"{dataset.name}.part_{partition_index}_of_{total_partitions}"

    def __iter__(self) -> Iterator[T_co]:
        for i, sample in enumerate(self._dataset):
            if i % self._total_partitions == self._partition_index:
                yield sample

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return f"PartitionedDataset({self._dataset}, part={self._partition_index}/{self._total_partitions})"

    @property
    def name(self) -> str:
        return self._name
