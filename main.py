import argparse
import numpy as np
import torch.utils.data
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker
from utils.scheduler import load_scheduler
from models.get_model import get_model
from loss import make_loss
from train_pipeline import train, test

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Bi-directional Domain Adaptation for Cross-domain HSI classification")
    parser.add_argument("--model", type=str, default='BiDA')
    parser.add_argument('--source_name', type=str, default='Houston13',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='Houston18',
                        help='the name of the test dir')
    parser.add_argument("--dataset_dir", type=str, default='./Houston/')
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str,
                        default='softmax')
    parser.add_argument("--labelsmooth", type=str,
                        default='off') 
    parser.add_argument("--lambda1", type=float, default=1e-1)
    parser.add_argument("--lambda2", type=float, default=1e+0)
    parser.add_argument("--log_interval", type=float, default=10)
    parser.add_argument('--seed', type=int, default=2100,
                        help='random seed ')
    parser.add_argument('--re_ratio', type=int, default=1,
                        help='random seed ')
    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))    
    print("source dataset = {}".format(opts.source_name))
    print("target dataset = {}".format(opts.target_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("depth = {}".format(opts.depth))
    print("{} for training, {} for validation and {} testing".format(opts.ratio, 1-opts.ratio, 1))

    seed_worker(opts.seed) 
    print("running an experiment with the {} model".format(opts.model))

    img_src, gt_src, labels = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='normband')
    img_tar, gt_tar, labels = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')

    num_classes = len(labels)
    num_bands = img_src.shape[-1]
    train_gt_src, val_gt_src = sample_gt(gt_src, opts.ratio, opts.seed, mode='random')
    test_gt_tar, _ = sample_gt(gt_tar, 1, opts.seed, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    
    for i in range(opts.re_ratio-1):
        img_src_con = np.concatenate((img_src_con,img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
        val_gt_src_con = np.concatenate((val_gt_src_con,val_gt_src))

    r = opts.patch_size // 2
    img_src_con = np.pad(img_src_con, ((r, r), (r, r), (0, 0)), mode='reflect')
    train_gt_src_con = np.pad(train_gt_src_con, ((r, r), (r, r)), mode='reflect')
    val_gt_src_con = np.pad(val_gt_src_con, ((r, r), (r, r)), mode='reflect')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), mode='reflect')
    test_gt_tar = np.pad(test_gt_tar, ((r, r), (r, r)), mode='reflect')

    train_set = HSIDataset(img_src_con, train_gt_src_con, patch_size=opts.patch_size, data_aug=True,
                            flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False)
    val_set = HSIDataset(img_src_con, val_gt_src_con, patch_size=opts.patch_size, data_aug=False)
    test_dataset_noise= HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=True,
                            flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False)
    test_dataset = HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=False)

    g = torch.Generator()
    g.manual_seed(opts.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_set, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)
    test_loader_noise = torch.utils.data.DataLoader(
        test_dataset_noise, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)

    # load model and loss
    model = get_model(opts.model, opts.source_name, opts.patch_size, opts)
    model_ema = get_model(opts.model, opts.source_name, opts.patch_size, opts, ema=True)
    
    model = model.to(device)
    model_ema = model_ema.to(device)
    
    optimizer, scheduler = load_scheduler(opts.model, model, opts)

    # criterion = nn.CrossEntropyLoss()
    criterion, center_criterion = make_loss(opts, num_classes=num_classes)
    
    # where to save checkpoint model
    model_dir = "./checkpoints/" + opts.model + '/' + opts.source_name + 'to' + opts.target_name

    try:
        train(model, model_ema, optimizer, criterion, num_classes, train_loader, val_loader, test_loader_noise, test_loader, opts, model_dir, device, scheduler)
    except KeyboardInterrupt:
        print('"ctrl+c" is pused, the training is over')

