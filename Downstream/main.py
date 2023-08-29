import os
from model.convmae.model import ConvMAE
from model.core.model import Model
from utils.args_parser import get_args_parser
from utils.save_file import save_to_json
from einops import rearrange
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from utils.data_loader import get_data_loader

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:120'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    batch_size = 1
    epochs = 400
    accum_iter = 1
    model = 'convmae_convvit_base_patch16'
    input_size = 96
    mask_ratio = 0.75
    norm_pix_loss = False
    weight_decay = 0.05
    lr = None
    blr = 1e-3
    min_lr = 0.0
    warmup_epochs = 40
    data_path = '/datasets01/imagenet_full_size/061417/'
    output_dir = '/content/drive/MyDrive/convmae/model_real'
    log_dir = '/content/drive/MyDrive/convmae/logs'
    device = 'cuda'
    seed = 0
    resume = ''
    start_epoch = 0
    num_workers = 10
    pin_mem = True
    world_size = 1
    local_rank = -1
    dist_on_itp = False
    dist_url = 'env://'

# args = Args()

args = get_args_parser()
args = args.parse_args()

max_iterations = 10000
eval_num = 100
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# get data loaders

train_data_loader, val_data_loader = get_data_loader(args)

convmae = ConvMAE().to(device)

# checkpoint = torch.load('/content/drive/MyDrive/convmae/model_pretrain/checkpoint.pth')
checkpoint = torch.load(f"{args.root_dir}/data/checkpoint.pth")
convmae.load_state_dict(checkpoint['model'], strict=False)


model = Model(convmae).to(device)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
    


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
        del x, y, loss  # free up GPU memory
        torch.cuda.empty_cache()
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_data_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            save_to_json(epoch_loss_values, f"{args.root_dir}/results", 'epoch_loss_values')
            save_to_json(metric_values, f"{args.root_dir}/results", 'metric_values')
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(f"{args.root_dir}/results", "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

if __name__ == '__main__':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_data_loader, dice_val_best, global_step_best)
    
