import torch
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

splits1 = "/dataset_LUNA16.json"
list_dir = "/vol/research/mscmiproj/people/okafor/ConvMAE_UNEt_3D/Pretrain/jsons"
jsonlist1 = list_dir + splits1
datadir1 = "/vol/research/mscmiproj/people/okafor/ConvMAE_UNEt_3D/Pretrain/dataset"
num_workers = 1

train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[
                        128,128,128]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[
                             128,128,128]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[ 128,128,128],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

def get_data_loader():
    datalist1 = load_decathlon_datalist(
            jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))

    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    datalist = new_datalist1
    print("Dataset all training: number of data: {}".format(len(datalist)))

    train_ds = Dataset(data=datalist, transform=train_transforms)
    sampler_train = torch.utils.data.RandomSampler(train_ds)
    train_loader = DataLoader(
            train_ds, batch_size=2, num_workers=num_workers, sampler=sampler_train, drop_last=True
        )
    return train_loader
