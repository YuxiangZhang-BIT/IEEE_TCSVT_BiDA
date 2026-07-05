import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import scipy.io as io
from utils.utils_HSI import grouper, sliding_window, count_sliding_window
from utils.utils_HSI import metrics
from loss.mmd_loss import MMD_loss

def train(network, network_ema, optimizer, criterion, num_classes, train_loader, val_loader, test_loader_noise, test_loader, opts, saving_path, device, scheduler):

    global_step = 0
    best_test_acc = 0
    losses = []
    Align_losses = []
    Distill_losses = []
    MMD_criterion = MMD_loss()
    
    for e in tqdm(range(1, opts.epoch+1), desc="training the network"):
        network.train()
        for batch_idx, (data_src, data_tar) in enumerate(zip(train_loader, test_loader_noise)):

            images, targets = data_src
            images_tar, targets_tar = data_tar
            images, targets = images.to(device), targets.to(device)
            images_tar, targets_tar = images_tar.to(
                device), targets_tar.to(device)
            if images.shape[0] == images_tar.shape[0]:
                optimizer.zero_grad()
                out_x, out_x_tar, out_x_fusion, out_x_fusion_src = network(images, images_tar)
                with torch.no_grad():
                    out_x_ema, out_x_tar_ema, _, _ = network_ema(images, images_tar)
                loss_cls = criterion(out_x, targets)
                loss_dis = distill_loss(out_x_fusion, out_x_tar)
                loss_dis_src = distill_loss(out_x_fusion_src, out_x)
                loss_con_tar = softmax_mse_loss(F.softmax(out_x_tar, dim=1), F.softmax(out_x_tar_ema, dim=1)) / data_tar[0].shape[0]
                loss_con_src = softmax_mse_loss(F.softmax(out_x, dim=1), F.softmax(out_x_ema, dim=1)) / data_src[0].shape[0]
                
                if e > 100 and opts.bs == images.shape[0] and opts.bs == images_tar.shape[0]:
                    Align_loss = (MMD_criterion(
                        out_x, out_x_tar) + MMD_criterion(
                        out_x_fusion_src, out_x_fusion_src))/2
                    loss = loss_cls + opts.lambda1*Align_loss + opts.lambda1*(loss_dis + loss_dis_src) + opts.lambda2*(loss_con_tar + loss_con_src)

                    Align_losses.append(Align_loss.item())
                else:
                    loss = loss_cls + opts.lambda1*(loss_dis + loss_dis_src) + opts.lambda2*(loss_con_tar + loss_con_src)

                Distill_losses.append((loss_dis + loss_dis_src).item())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                global_step += 1
                update_ema_variables(network, network_ema, opts.ema_decay, global_step)
                
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            if Align_losses:
                A_mean_losses = np.mean(Align_losses)
            else:
                A_mean_losses = 0
            A_mean_Distill_losses = np.mean(Distill_losses)
            train_info = "train at epoch {}/{}, loss={:.6f}, Align_loss={:.6f}, Distill_loss={:.6f}"
            train_info = train_info.format(
                e, opts.epoch,  mean_losses, A_mean_losses, A_mean_Distill_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        if scheduler is not None:
            scheduler.step()

        if e % opts.log_interval == 0:
            ts_acc, results = validation(network, test_loader, device, num_classes, show=True)
            is_best = ts_acc >= best_test_acc
            print('best_test_acc: {:.4f}'.format(best_test_acc))
            if ts_acc > best_test_acc:
                best_test_acc = max(ts_acc, best_test_acc)
                save_ts_checkpoint(network, is_best, saving_path,
                                epoch=e, acc=best_test_acc, tmp_acc=ts_acc, seed=opts.seed)

                io.savemat(os.path.join(saving_path,
                                        'results' + f'_{best_test_acc:.4f}_{opts.seed}' +'.mat'),
                            {'lr': opts.lr, 'lambda1': opts.lambda1, 'depth': opts.depth, 're_ratio':opts.re_ratio, 'results': results, 
                             'seed': opts.seed, 'lambda2':opts.lambda2})


def validation(network, val_loader, device, num_classes, show=False, ema=False):
    num_correct = 0.
    total_num = 0.
    if ema:
        pass
    else:
        network.eval()

    ps = []
    ys = []
    for batch_idx, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)
        if ema:
            _, outputs, _, _ = network(images, images)
        else:
            _, outputs, _ = network(images, images)
        _, outputs = torch.max(outputs, dim=1)
        ps.append(outputs.detach().cpu().numpy())
        ys.append(targets.detach().cpu().numpy())
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            total_num = total_num + 1
    overall_acc = num_correct / total_num
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    
    if show:
        results = metrics(ps, ys, n_classes=num_classes)
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(
            results['TPR']*100, 2), '\n', 'OA:', results['Accuracy'])
        return overall_acc, results
    return overall_acc


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size),
                   (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(
        image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_ts_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)
    if is_best:
        tqdm.write(
            "epoch = {epoch}: best test OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(
            saving_path, 'model_ts_best{acc:.4f}_{seed}.pth'.format(**kwargs)))
    else:
        tqdm.write(
            "epoch = {epoch}: best test OA = {acc:.4f}".format(**kwargs))
        

def update_ema_variables(model, ema_model, alpha, global_step):
    """EMA for model parameters"""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        
        
def distill_loss(teacher_output, student_out):
    """Takes softmax on both sides and returns KL divergence loss"""
    teacher_out = F.softmax(teacher_output, dim=-1)
    loss = torch.sum(-teacher_out *
                     F.log_softmax(student_out, dim=-1), dim=-1)
    return loss.mean()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss"""
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes

