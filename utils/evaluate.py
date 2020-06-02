import numpy as np
import torch

def torch2np(pred, mask):
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=1)
    mask = mask.cpu().detach().numpy()

    return pred, mask

n_class = 21
ignore_index = 255

def pixel_accuracy(preds, masks):
    preds, masks = torch2np(preds, masks)
    batch_size = masks.shape[0]
    acc = 0
    for i in range(batch_size):
        pred = preds[i, :, :]
        mask = masks[i, :, :]
        n_ignore = (mask == 255).sum()
        n_correct = (pred == mask).sum()
        n_pixels = pred.shape[0] * pred.shape[1] - n_ignore
        acc += n_correct / n_pixels
    
    return acc / batch_size

def mean_pixel_accuracy(preds, masks):
    preds, masks = torch2np(preds, masks)
    batch_size = masks.shape[0]
    acc = 0
    for i in range(batch_size):
        pred = preds[i, :, :]
        mask = masks[i, :, :]
        exist_classes = 0
        batch_acc = 0
        for k in range(n_class):
            pred_class = (pred == k)
            mask_class = (mask == k)
            n_correct = (pred_class & mask_class).sum()
            n_mask = mask_class.sum()
            if n_mask == 0:
                continue
            exist_classes += 1
            batch_acc += n_correct / n_mask
        acc += batch_acc / exist_classes
    
    return acc / batch_size

def mean_IOU(preds, masks):
    preds, masks = torch2np(preds, masks)
    batch_size = masks.shape[0]
    IoU = 0
    for i in range(batch_size):
        pred = preds[i, :, :]
        mask = masks[i, :, :]
        exist_classes = 0
        batch_IoU = 0
        for k in range(n_class):
            pred_class = (pred == k)
            mask_class = (mask == k)
            n_mask = mask_class.sum()
            if n_mask == 0:
                continue
            n_pred = pred_class.sum()
            n_correct = (pred_class & mask_class).sum()
            exist_classes += 1
            batch_IoU += n_correct / (n_mask + n_pred - n_correct)
        IoU += batch_IoU / exist_classes
    
    return IoU / batch_size
    

if __name__ == "__main__":
    output = torch.randn((1, 21, 500, 500), requires_grad=True)
    target = torch.empty((1, 500, 500), dtype=torch.long).random_(21)
    pa = pixel_accuracy(output, target)
    mpa = mean_pixel_accuracy(output, target)
    miou = mean_IOU(output, target)
    print(pa, mpa, miou)