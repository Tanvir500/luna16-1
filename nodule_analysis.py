import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader

from dsets import (
    CandidateInfoTuple,
    LunaDataset,
    getCandidateInfoDict,
    getCandidateInfoList,
    getCt,
)
from model import LunaModel, UNetWrapper
from seg_dsets import Lund2dSegentationDataset
from util import irc2xyz

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def print_confusion(label, confusions, do_mal):
    row_labels = ["Non-Nodules", "Benign", "Malignant"]

    if do_mal:
        col_labels = [
            "",
            "Complete Miss",
            "Filtered Out",
            "Pred. Benign",
            "Pred. Malignant",
        ]
    else:
        col_labels = ["", "Complete Miss", "Filtered Out", "Pred. Nodule"]
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = "{:>" + str(cell_width) + "}"
    print(label)
    print(" | ".join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ""
        print(" | ".join([f.format(i) for i in r]))


def match_and_score(detections, truth, threshold=0.5, threshold_mal=0.5):
    # Returns 3x4 confusion matrix for:
    # Rows: Truth: Non-Nodules, Benign, Malignant
    # Cols: Not Detected, Detected by Seg,
    #       Detected as Benign, Detected as Malignant
    # If one true nodule matches multiple detections,
    #           the "highest" detection is considered
    # If one detection matches several true nodule annotations,
    #           it counts for all of them
    true_nodules = [c for c in truth if c.isNodule_bool]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    detected_xyz = np.array([n[2] for n in detections])
    # detection classes will contain
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
    # 3 -> detected as malignant nodule (if applicable)
    detected_classes = np.array(
        [1 if d[0] < threshold else (2 if d[1] < threshold else 3) for d in detections]
    )

    confusion = np.zeros((3, 4), dtype=np.int32)
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion[2 if tn.isMal_bool else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    else:
        normalized_dists = (
            np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1)
            / truth_diams[:, None]
        )
        matches = normalized_dists < 0.7
        unmatched_detections = np.ones(len(detections), dtype=bool)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int32)
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(
                matched_true_nodules[i_tn], detected_classes[i_detection]
            )
            unmatched_detections[i_detection] = False

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1
    return confusion


def initModels():
    seg_dict = torch.load("data/models/seg_model_best0.state")
    seg_model = UNetWrapper(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode="upconv",
    )
    seg_model.load_state_dict(seg_dict["model_state"])
    seg_model.eval()

    cls_dict = torch.load("data/models/cls_model_best0.state")
    cls_model = LunaModel()
    cls_model.load_state_dict(cls_dict["model_state"])
    cls_model.eval()

    if USE_CUDA:
        if torch.cuda.device_count() > 1:
            seg_model = nn.DataParallel(seg_model)
            cls_model = nn.DataParallel(cls_model)
        seg_model.to(DEVICE)
        cls_model.to(DEVICE)

    return seg_model, cls_model


def initSegmentationDl(series_uid):
    seg_ds = Lund2dSegentationDataset(
        contextSlices_count=3,
        series_uid=series_uid,
        fullCt_bool=True,
    )
    seg_dl = DataLoader(
        seg_ds,
        batch_size=4,
        num_workers=4,
        pin_memory=USE_CUDA,
    )
    return seg_dl


def initClassificationDl(candidateInfo_list):
    cls_ds = LunaDataset(
        sortby_str="series_uid",
        candidateInfo_list=candidateInfo_list,
    )
    cls_dl = DataLoader(
        cls_ds,
        batch_size=4,
        num_workers=4,
        pin_memory=USE_CUDA,
    )
    return cls_dl


def segmentCt(ct, series_uid, seg_model):
    with torch.no_grad():
        output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
        seg_dl = initSegmentationDl(series_uid)
        for input_t, _, _, slice_ndx_list in seg_dl:

            input_g = input_t.to(DEVICE)
            prediction_g = seg_model(input_g)

            for i, slice_ndx in enumerate(slice_ndx_list):
                output_a[slice_ndx] = prediction_g[i].cpu().numpy()

        mask_a = output_a > 0.5
        mask_a = ndimage.binary_erosion(mask_a, iterations=1)

    return mask_a


def groupSegmentationOutput(series_uid, ct, clean_a):
    candidateLabel_a, candidate_count = ndimage.label(clean_a)
    centerIrc_list = ndimage.center_of_mass(
        ct.hu_a.clip(-1000, 1000) + 1001,
        labels=candidateLabel_a,
        index=np.arange(1, candidate_count + 1),
    )
    candidateInfo_list = []
    for i, center_irc in enumerate(centerIrc_list):
        center_xyz = irc2xyz(
            center_irc,
            ct.origin_xyz,
            ct.vxSize_xyz,
            ct.direction_a,
        )
        candiddateInfo_tup = CandidateInfoTuple(
            False, False, False, 0.0, series_uid, center_xyz
        )
        candidateInfo_list.append(candiddateInfo_tup)
    return candidateInfo_list


def classifyCandidates(ct, candidateInfo_list, cls_model):
    cls_dl = initClassificationDl(candidateInfo_list)
    classifications_list = []
    for batch_ndx, batch_tup in enumerate(cls_dl):
        input_t, _, _, series_list, center_list = batch_tup

        input_g = input_t.to(DEVICE)
        with torch.no_grad():
            _, probability_nodule_g = cls_model(input_g)
            probability_mal_g = torch.zeros_like(probability_nodule_g)

        zip_iter = zip(
            center_list,
            probability_nodule_g[:, 1].tolist(),
            probability_mal_g[:, 1].tolist(),
        )
        for center_irc, prob_nodule, prob_mal in zip_iter:
            center_xyz = irc2xyz(
                center_irc,
                direction_a=ct.direction_a,
                origin_xyz=ct.origin_xyz,
                vxSize_xyz=ct.vxSize_xyz,
            )
            cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
            classifications_list.append(cls_tup)
    return classifications_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "series_uid", nargs="?", default=None, help="Series UID to use."
    )
    cli_args = parser.parse_args()
    seg_model, cls_model = initModels()

    val_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=True,
    )

    val_set = set(
        candidateInfo_tup.series_uid for candidateInfo_tup in val_ds.candidateInfo_list
    )
    positive_set = set(
        candidateInfo_tup.series_uid
        for candidateInfo_tup in getCandidateInfoList()
        if candidateInfo_tup.isNodule_bool
    )
    if cli_args.series_uid:
        series_set = set(cli_args.series_uid.split(","))
    else:
        series_set = set(
            candidateInfo_tup.series_uid for candidateInfo_tup in getCandidateInfoList()
        )
    val_list = sorted(series_set & val_set)
    train_list = []
    candidateInfo_dict = getCandidateInfoDict()

    all_confusion = np.zeros((3, 4), dtype=np.int32)
    series_iter = val_list + train_list
    series_n = len(series_iter)

    for i, series_uid in enumerate(series_iter):
        print(f"{i}/{series_n}---------------------------------------")
        ct = getCt(series_uid)
        mask_a = segmentCt(ct, series_uid, seg_model)

        candidateInfo_list = groupSegmentationOutput(
            series_uid,
            ct,
            mask_a,
        )
        classifications_list = classifyCandidates(
            ct,
            candidateInfo_list,
            cls_model,
        )

        if cli_args.series_uid:
            print(f"found nodule candidates in {series_uid}:")
            for prob, prob_mal, center_xyz, center_irc in classifications_list:
                if prob > 0.5:
                    s = f"nodule prob {prob:.3f}, "
                    s += f"center xyz {center_xyz}"
                    print(s)

        if series_uid in candidateInfo_dict:
            one_confusion = match_and_score(
                classifications_list, candidateInfo_dict[series_uid]
            )
            all_confusion += one_confusion
            print_confusion(series_uid, one_confusion, False)
    print_confusion("Total", all_confusion, False)
