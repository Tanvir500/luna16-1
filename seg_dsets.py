import csv
import functools
import glob
import os.path
from collections import namedtuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from util import XyzTuple, xyz2irc

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple",
    (
        "isNodule_bool, hasAnnotation_bool, isMal_bool, "
        "diameter_mm, series_uid, center_xyz"
    ),
)


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob("data/luna/imgs/{}.mhd".format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (
            self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist()
        )

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while (
                    self.hu_a[ci + index_radius, cr, cc] > threshold_hu
                    and self.hu_a[ci - index_radius, cr, cc] > threshold_hu
                ):
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while (
                    self.hu_a[ci, cr + row_radius, cc] > threshold_hu
                    and self.hu_a[ci, cr - row_radius, cc] > threshold_hu
                ):
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while (
                    self.hu_a[ci, cr, cc + col_radius] > threshold_hu
                    and self.hu_a[ci, cr, cc - col_radius] > threshold_hu
                ):
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                ci - index_radius : ci + index_radius + 1,
                cr - row_radius : cr + row_radius + 1,
                cc - col_radius : cc + col_radius + 1,
            ] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert (
                center_val >= 0 and center_val < self.hu_a.shape[axis]
            ), repr(
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vxSize_xyz,
                    center_irc,
                    axis,
                ]
            )

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob("data-unversioned/part2/luna/subset*/*.mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    with open("data/luna/annotations_with_malignancy.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {"False": False, "True": True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open("data/luna/candidates.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(
            candidateInfo_tup
        )

    return candidateInfo_dict


def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


class Lund2dSegentationDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        isValSet_bool=None,
        series_uid=None,
        contextSlices_count=3,
        fullCt_bool=False,
    ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in range(index_count)
                ]
            else:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in positive_indexes
                ]

        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [
            cit
            for cit in self.candidateInfo_list
            if cit.series_uid in series_set
        ]

        self.pos_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx
