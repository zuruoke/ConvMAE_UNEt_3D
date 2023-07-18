from monai.data import load_decathlon_datalist
from monai.data import  DataLoader, Dataset, load_decathlon_datalist
from utils.data_utils import train_transforms, val_transforms


def get_data_loader(args):
    data_dir = f"{args.root_dir}data/"
    split_json = "dataset_0.json"

    datasets = data_dir + split_json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")


    train_ds = Dataset(data=datalist, transform=train_transforms)
    train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=1, num_workers=0, sampler=train_sampler, drop_last=True,
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1,
                            num_workers=0, shuffle=False, drop_last=True,)
    
    return train_loader, val_loader