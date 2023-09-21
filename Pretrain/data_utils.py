import torch
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist, DistributedSampler
from train_utils import get_world_size, get_rank
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
splits2 = "/dataset_Covid_19.json"
splits3 = "/dataset_HNSCC.json"
splits4 = "/dataset_CT_Colonography.json"
splits5 = "/dataset_LIDC_IDRI.json"
list_dir = "/vol/research/mscmiproj/people/okafor/ConvMAE_UNEt_3D/Pretrain/jsons"
jsonlist1 = list_dir + splits1
jsonlist2 = list_dir + splits2
jsonlist3 = list_dir + splits3
jsonlist4 = list_dir + splits4
jsonlist5 = list_dir + splits5
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
                        96,96,96]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[
                             96,96,96]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[ 96,96,96],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

def get_data_loader(args):
    datalist1 = load_decathlon_datalist(
            jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))

    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    
    """
    -------------------------------------------------------------------------
    """
    datalist2 = load_decathlon_datalist(
            jsonlist2, False, "training", base_dir=datadir1)
    print("Dataset 2 Covid-19: number of data: {}".format(len(datalist2)))

    new_datalist2 = []
    for item in datalist2:
        item_dict = {"image": item["image"]}
        new_datalist2.append(item_dict)

    """
    -------------------------------------------------------------------------
    """

    datalist3 = load_decathlon_datalist(
            jsonlist3, False, "training", base_dir=datadir1)
    print("Dataset 3 HNSCC: number of data: {}".format(len(datalist3)))

    new_datalist3 = []
    for item in datalist3:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)
   
    """
    -------------------------------------------------------------------------
    """

    datalist4 = load_decathlon_datalist(
            jsonlist4, False, "training", base_dir=datadir1)
    print("Dataset 4 CT Colonography: number of data: {}".format(len(datalist4)))

    new_datalist4 = []
    for item in datalist4:
        item_dict = {"image": item["image"]}
        new_datalist4.append(item_dict)


    """
    -------------------------------------------------------------------------
    """

    datalist5 = load_decathlon_datalist(
            jsonlist5, False, "training", base_dir=datadir1)
    print("Dataset 5 LIDC_IDRI: number of data: {}".format(len(datalist5)))

    new_datalist5 = []
    for item in datalist5:
        item_dict = {"image": item["image"]}
        new_datalist5.append(item_dict)



    datalist = new_datalist1 + datalist2 + datalist3 + datalist4 + datalist5
    print("Dataset all training: number of data: {}".format(len(datalist)))

    train_ds = Dataset(data=datalist, transform=train_transforms)
    
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank,  even_divisible=True, shuffle=True)
    else:
        sampler_train = None

    train_loader = DataLoader(
            train_ds, batch_size=4, num_workers=num_workers, sampler=sampler_train, drop_last=True
        )
    return train_loader
