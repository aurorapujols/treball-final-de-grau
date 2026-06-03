import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
import torch.nn.functional as F
import numpy as np
import math
import time
import copy
import pickle
from collections import Counter
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR_TRAIN  = r"../../../data/upftfg26/apujols/state_of_the_art/nightskyUCP/my_train_images/"
DATA_DIR_VAL    = r"../../../data/upftfg26/apujols/state_of_the_art/nightskyUCP/my_val_images/"
DATA_DIR_TEST   = r"../../../data/upftfg26/apujols/state_of_the_art/nightskyUCP/my_test_images/"
WEIGHTS_PATH    = "../../../data/upftfg26/apujols/models/spp_net_best_fold_grayscale.pth"
OUTPUT_DIR      = "logs/training/nightskyucp"
RESNET_WEIGHTS  = "/data/upftfg26/apujols/state_of_the_art/nightskyUCP/resnet18-f37072fd.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of warm-up passes before timing (avoids cold-start GPU overhead)
WARMUP_RUNS    = 10
# Number of timed passes for a stable inference-time estimate
TIMING_RUNS    = 100


# ── Model definition (must match training) ────────────────────────────────────

# https://github.com/revidee/pytorch-pyramid-pooling/blob/master/pyramidpooling.py
class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode   = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        num_sample          = previous_conv.size(0)
        previous_conv_size  = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil( (w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil( (h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])
            padded_input = F.pad(input=previous_conv,
                                 pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp


class SPP_Net(nn.Module):
    def __init__(self, feature_extractor):
        super(SPP_Net, self).__init__()
        self.extractor = feature_extractor
        self.fc1       = nn.Linear(in_features=5120, out_features=2)

    def forward(self, x):
        x      = self.extractor(x)
        x      = PyramidPooling.spatial_pyramid_pool(x, [2, 2, 1, 1], "max")
        output = self.fc1(x)
        return output

    def get_embedding(self, x):
        """Return the SPP feature vector (pre-fc1) for a single input."""
        x = self.extractor(x)
        x = PyramidPooling.spatial_pyramid_pool(x, [2, 2, 1, 1], "max")
        return x   # shape: (batch, 5120)


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

    def load_to_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def ema_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                _ = self(name, param.data)

    def register_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, batch_size=100, device='cuda:0'):
    print("Start training")
    since = time.time()

    val_acc_history = []
    ema = EMA(mu=.99999)
    ema.register_params(model.cuda())

    best_model_wts = copy.deepcopy(model.state_dict())
    test_model  = copy.deepcopy(model).to(device)
    train_model = copy.deepcopy(model).to(device)
    best_acc = 0.0
    model.to(device)
    lrs = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss     = 0.0
            running_corrects = 0
            idx    = 0
            outputs      = []
            labels       = []
            model_inputs = []

            if 'phase' == 'test':
                ema.load_to_model(test_model)
                model = test_model
            elif 'phase' == 'train':
                model = train_model

            with torch.set_grad_enabled(phase == 'train'):
                for idx, (inputs, label) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    label  = (label).to(device)
                    labels.append(label)
                    outputs.append(model(inputs))

                    if (idx + 1) % batch_size == 0:
                        o, l = torch.cat(outputs, dim=0), torch.stack(labels, dim=0).view(batch_size)
                        loss = criterion(o, l) + ((0.5 - o) ** 2).sum() * 1e-8
                        _, preds = torch.max(o, dim=1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            ema.ema_params(model)

                        running_loss     += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == l.data)
                        outputs      = []
                        labels       = []
                        model_inputs = []

            if 'phase' == 'test':
                test_model = model
            elif 'phase' == 'train':
                train_model = model

            epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
            epoch_acc  = float(running_corrects) / float(len(dataloaders[phase].dataset))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'test':
                val_acc_history.append(epoch_acc)
                print('Epoch: {} Loss: {:.4f} Acc: {:.4f} best: {:.4f}'.format(
                    epoch, epoch_loss, epoch_acc, best_acc))

            lrs.step()

    time_elapsed = time.time() - since
    print('Best val Acc: {:4f}'.format(best_acc))
    return val_acc_history, best_model_wts


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(all_labels, all_preds, class_names, output_dir,
                           filename="confusion_matrix.png"):
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - SPP_Net")
    plt.tight_layout()

    path = Path(output_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {path}")


def count_parameters(model):
    """Total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, input_tensor, warmup=WARMUP_RUNS, runs=TIMING_RUNS):
    """
    Returns mean and std of per-image inference time in milliseconds for both
    the original device and CPU.

    GPU timing uses CUDA events for accuracy (host-side perf_counter does not
    account for async kernel execution).  CPU timing moves the model and tensor
    to CPU regardless of the original device, so both figures are always
    reported even when running on GPU.  Warm-up passes are discarded in both
    cases to avoid cold-start overhead.
    """
    model.eval()

    def _time_on(m, x):
        use_cuda = x.device.type == "cuda"
        with torch.no_grad():
            for _ in range(warmup):
                m(x)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(runs):
                if use_cuda:
                    start_event.record()
                    m(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))   # ms
                else:
                    t0 = time.perf_counter()
                    m(x)
                    times.append((time.perf_counter() - t0) * 1e3)      # ms
        return float(np.mean(times)), float(np.std(times))

    # ── original device (GPU or CPU) ──────────────────────────────────────────
    mean_dev, std_dev = _time_on(model, input_tensor)

    # ── CPU ───────────────────────────────────────────────────────────────────
    cpu_model  = copy.deepcopy(model).to("cpu")
    cpu_tensor = input_tensor.to("cpu")
    mean_cpu, std_cpu = _time_on(cpu_model, cpu_tensor)

    return mean_dev, std_dev, mean_cpu, std_cpu


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluation():

    # ── Transform (must match training exactly) ───────────────────────────────
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.449], [0.226])
    ])

    # ── Rebuild model skeleton and load weights ───────────────────────────────
    resnet18 = models.resnet18(weights=None)
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    feature_extractor = nn.Sequential(*(list(resnet18.children())[0:8]))

    eval_model = SPP_Net(feature_extractor).to(DEVICE)
    eval_model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    eval_model.eval()
    print(eval_model)

    # ── 1. Parameter count ────────────────────────────────────────────────────
    total_params, trainable_params = count_parameters(eval_model)
    print("\n── Parameter count ──────────────────────────────────────────────────")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    # ── 2. Embedding dimensionality ───────────────────────────────────────────
    dummy = torch.zeros(1, 1, 128, 128).to(DEVICE)
    with torch.no_grad():
        embedding = eval_model.get_embedding(dummy)
    embed_dim = embedding.shape[1]
    print("\n── Embedding dimensionality ─────────────────────────────────────────")
    print(f"  SPP feature vector   : {embed_dim}-D  {list(embedding.shape)}")

    # ── 3. Inference time ─────────────────────────────────────────────────────
    mean_dev, std_dev, mean_cpu, std_cpu = measure_inference_time(eval_model, dummy)
    print("\n── Inference time (single image, 128×128) ───────────────────────────")
    print(f"  Warm-up runs         : {WARMUP_RUNS}")
    print(f"  Timed runs           : {TIMING_RUNS}")
    print(f"  Device ({str(DEVICE):<4})  Mean : {mean_dev:.3f} ms  Std : {std_dev:.3f} ms")
    print(f"  CPU          Mean : {mean_cpu:.3f} ms  Std : {std_cpu:.3f} ms")

    # ── 4. Classification accuracy + confusion matrix ─────────────────────────
    your_dataset = datasets.ImageFolder(root=DATA_DIR_TEST, transform=base_transform)
    your_loader  = torch.utils.data.DataLoader(
        your_dataset, batch_size=1, shuffle=False, num_workers=0)

    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in your_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = eval_model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n── Classification results ───────────────────────────────────────────")
    print(f"  Accuracy             : {correct/total:.4f}  ({correct}/{total})")
    print(f"  Class mapping        : {your_dataset.class_to_idx}")
    print(f"  Prediction dist.     : {Counter(all_preds)}")
    print(f"  Label dist.          : {Counter(all_labels)}")

    class_names = [k for k, v in sorted(
        your_dataset.class_to_idx.items(), key=lambda x: x[1])]
    save_confusion_matrix(all_labels, all_preds, class_names, OUTPUT_DIR)

    # ── Quick image sanity checks ─────────────────────────────────────────────
    img    = Image.open(your_dataset.samples[0][0])
    tensor = base_transform(img)
    print("\n── Input sanity check ───────────────────────────────────────────────")
    print(f"  PIL mode             : {img.mode}")
    print(f"  Tensor shape         : {list(tensor.shape)}")
    print(f"  Tensor range         : [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")


def training():

    # ── Transforms ────────────────────────────────────────────────────────────
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.449], [0.226])
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    dataset_train = datasets.ImageFolder(root=DATA_DIR_TRAIN, transform=base_transform)
    dataset_val   = datasets.ImageFolder(root=DATA_DIR_VAL,   transform=base_transform)

    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True,  num_workers=0),
        'test':  torch.utils.data.DataLoader(dataset_val,   batch_size=1, shuffle=True,  num_workers=0),
    }

    resnet18 = models.resnet18(weights=None)
    resnet18.load_state_dict(torch.load(RESNET_WEIGHTS, map_location="cpu", weights_only=True))

    pretrained_weight = resnet18.conv1.weight.data            # [64, 3, 7, 7]
    new_weight = pretrained_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet18.conv1.weight.data = new_weight

    feature_extractor = nn.Sequential(*(list(resnet18.children())[0:8]))
    model = SPP_Net(feature_extractor)
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    acc_hist, best_model = train_model(
        model, dataloaders_dict, criterion, optimizer, batch_size=100, num_epochs=50)

    torch.save(best_model, WEIGHTS_PATH)


def main():
    evaluation()


if __name__ == "__main__":
    main()
