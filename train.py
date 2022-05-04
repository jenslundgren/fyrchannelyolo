import config
import torch
import torch.optim as optim
from dataset import DistanceDataset
from torch.utils.data import DataLoader
import time

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )
        
        with torch.cuda.amp.autocast():
            print("")
            print(x.shape)
            print(x.get_device())
            print(next(model.parameters()).device)
            #print(x.type(torch.HalfTensor))
            time.sleep(5)
            out = model(x.permute(0,3,1,2))
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    # FIXA
    
    IMAGE_SIZE = 416
    TENSOR_DIR = "C:/coding/ExJobb/FourChannelYolo/data/rgbDistance"
    LABEL_DIR = "C:/coding/ExJobb/FourChannelYolo/data/labels"
    
    train_dataset = DistanceDataset(
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        tensor_dir=TENSOR_DIR,
        label_dir=LABEL_DIR,
        anchors=config.ANCHORS,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
    )
    
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1,3,2)
    ).to(config.DEVICE)

    
    for epoch in range(3):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    
    
    

if __name__ == "__main__":
    main()