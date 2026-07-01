from nnssl.ssl_data.dataloading.data_loader_3d import nnsslIndexableCenterCropDataLoader3D


class IndexableSingleThreadedAugmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """

    def __init__(self, data_loader: nnsslIndexableCenterCropDataLoader3D, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __getitem__(self, index):
        item = self.data_loader.generate_train_batch(index)
        if self.transform is not None:
            item = self.transform(**item)
        return item

    def __len__(self):
        return len(self.data_loader)
