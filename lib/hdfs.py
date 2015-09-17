from math import ceil


class HDFS:
    def __init__(self, data_size, block_size):
        """
        :param numeric block_size: HDFS block size in bytes
        :param numeric data_size: Input data size in bytes
        """
        self.data_size = data_size
        self.block_size = block_size

    @property
    def n_blocks(self):
        """Number of blocks"""
        return ceil(self.data_size / self.block_size)
