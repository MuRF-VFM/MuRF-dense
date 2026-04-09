from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from .segmetric import SegmentationMetrics
from .utils import visualize_segmentation


def test(model: nn.Module,
         dataloader,
         all_metrics: bool = True,
         models=None):
    example = next(iter(dataloader))
    metric = SegmentationMetrics()
    frequency = 10000
    chosen_epochs = [0]
    callbacks = []

    # put model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    metric.to(device)

    # put model in evaluation mode
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            images = batch["pixel_values"].to(device, non_blocking=True)
            gt_seg = batch["labels"].to(device, non_blocking=True).squeeze(1)

            # forward
            with torch.amp.autocast(device):
                outputs = model(images)

            if i in chosen_epochs:
            # if i % frequency == 0:
                visualize_segmentation(images,
                                       models,
                                        model_names=["140", "266", "518", "MuRF"],
                                       epoch=i)

    return None

if __name__ == "__main__":
    from pathlib import Path
    from dataset.voc import get_voc_dataloader
    from train_mrf_voc import evaluate
    from .model import DinoV2SegmentationModelMRF

    models = []
    model_paths = ["weights/140.pth",
                   "weights/266.pth",
                   "weights/518.pth",
                   "weights/mrf.pth"]
    for path in model_paths:
        checkpoint = torch.load(path, map_location="cpu")
        seg_head = checkpoint['seg_head']
        resolutions = checkpoint['resolutions']
        model = DinoV2SegmentationModelMRF(resolutions=resolutions,
                                           num_classes=21)
        model.seg_head.load_state_dict(seg_head)
        models.append(model)

    path = Path("/home/ubuntu/coding/datasets/VOC2012")
    dataloader = get_voc_dataloader(split='val', batch_size=1, num_workers=4, root_dir=path, shuffle=False)
    import time

    for i, model in enumerate(models):
        start_time = time.time()
        evaluate(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu")
        end_time = time.time()
        print(f"Model {i} inference time: {end_time - start_time} seconds")

    # from pathlib import Path
    # from dataset.ade import get_ade20k_dataloader
    # from model import DinoV2SegmentationModelMRF
    #
    # models = []
    # model_paths = ["weights/266_ade.pth",
    #                "weights/518_ade.pth",
    #                "weights/784_ade.pth",
    #                "weights/mrf_ade.pth"]
    # resolutions = [[266], [518], [784], [266, 518, 784]]
    # for i, path in enumerate(model_paths):
    #     print(i)
    #     checkpoint = torch.load(path, map_location="cpu")
    #     seg_head = checkpoint['model_state_dict']
    #     model = DinoV2SegmentationModelMRF(resolutions=resolutions,
    #                                        num_classes=151)
    #     model.seg_head.load_state_dict(seg_head)
    #     models.append(model)
    #
    # path = Path("/home/ubuntu/coding/datasets/VOC2012")
    # dataloader = get_ade20k_dataloader(split='val', batch_size=2, num_workers=4, root_dir=path)
    # test(models[0], dataloader, all_metrics=True, models=models)
