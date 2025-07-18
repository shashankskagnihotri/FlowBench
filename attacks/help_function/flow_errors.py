import numpy as np
from attacks.help_function.flow_utils import backproject_flow3d_target


def compute_AAE(flow, gt):
    """compute the average angular error (AAE) in degrees between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    return: AAE in [deg]
    """
    arg = flow[:, :, 0] * gt[:, :, 0] + flow[:, :, 1] * gt[:, :, 1] + 1

    # number of valid pixels:
    count = np.count_nonzero(~np.isnan(arg))

    arg /= np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2 + 1) * np.sqrt(
        gt[:, :, 0] ** 2 + gt[:, :, 1] ** 2 + 1
    )

    # set nan values to 1 since arccos(1)=0
    arg = np.nan_to_num(arg, nan=1.0)

    # clip to the arccos range [-1;1]
    arg[arg > 1.0] = 1.0
    arg[arg < -1.0] = -1.0

    angular_error = np.arccos(arg)

    return np.sum(angular_error) / count / (2 * np.pi) * 360.0


def compute_EE(flow, gt):
    """compute the endpoint error for every pixel location
    flow: estimated flow
    gt: ground truth flow
    return: 2D np array with pixel-wise endpoint error or nan if no groundtruth is present
    """
    diff = np.square(flow - gt)
    comp = np.sum(diff, axis=-1)
    comp = np.sqrt(comp)

    return comp


def compute_AEE(flow, gt, ee=None):
    """compute the average endpoint error (AEE, sometimes also EPE) between the estimated flow field and the groundtruth flow field
    flow: estimated flow
    gt: groundtruth flow
    ee: precomputed endpoint error
    """
    if ee is None:
        ee = compute_EE(flow, gt)
    count = np.count_nonzero(~np.isnan(ee))
    return np.nansum(ee) / count


def compute_BP(flow, gt, useKITTI15=False, ee=None, return_mask=False, t1=3.0, t2=0.05):
    """compute the bad pixel error (BP) between the estimated flow field and the groundtruth flow field.
    The bad pixel error is defined as the percentage of valid pixels.
    Valid pixel are generally defined as those whose endpoint is smaller than 3px.
    An extension to this definition used for the KITTI15 dataset is that a pixel is valid if
    the endpoint error is smaller than 3px OR less than 5% of the groundtruth vector length.
    This extension has an influence if the groundtruth vector length is > 60px.
    flow: estimated flow
    gt: groundtruth flow
    useKITTI15: boolean flag if the KITTI15 calculation method should be used (gives better results)
    ee: precomputed endpoint error
    return_mask: if True, return pixelwise boolean mask instead of aggregated number
    return: BP error as percentage [0;100], or mask if return_mask is True
    """
    if ee is None:
        ee = compute_EE(flow, gt)

    # number of valid pixels:
    count = np.count_nonzero(~np.isnan(ee))

    # set the ee of nan pixels to zero
    ee = np.nan_to_num(ee, nan=0.0)
    abs_err = ee > t1

    if useKITTI15:
        gt_vec_length = np.nan_to_num(
            np.sqrt(np.square(gt[..., 0]) + np.square(gt[..., 1])), nan=0.0
        )
        rel_err = ee > t2 * gt_vec_length

        bp_mask = abs_err & rel_err
    else:
        bp_mask = abs_err

    if return_mask:
        return bp_mask
    else:
        return 100 * np.sum(bp_mask) / count


def compute_Fl(flow, gt, ee=None, return_mask=False, t1=3.0, t2=0.05):
    """compute the bad pixel error (Fl) between the estimated flow field and the groundtruth flow field.
    The bad pixel error is defined as the percentage of valid pixels.
    Valid pixel are defined as those whose endpoint is smaller than 3px OR less than 5% of the groundtruth vector length.
    flow: estimated flow
    gt: groundtruth flow
    ee: precomputed endpoint error
    return_mask: if True, return pixelwise boolean mask instead of aggregated number
    return: Fl error as percentage [0;100], or mask if return_mask is True
    """
    return compute_BP(
        flow, gt, useKITTI15=True, ee=ee, return_mask=return_mask, t1=t1, t2=t2
    )


def printAllErrorMeasures(flow, gt):
    """print the AAE, AEE, BP and Fl error measures
    flow: estimated flow
    gt: groundtruth flow
    """
    for err, name in zip(
        [compute_AAE, compute_AEE, compute_BP, compute_Fl], ["AAE", "AEE", "BP", "Fl"]
    ):
        print(f"{name:3s}: {err(flow,gt):.2f}")


def getAllErrorMeasures(flow, gt):
    """create a dictionary with the AAE, AEE, BP and Fl error measures
    flow: estimated flow
    gt: groundtruth flow
    return: dictionary with keys AAE, AEE, BP, Fl and error values
    """
    result = {}
    result["AAE"] = compute_AAE(flow, gt)

    # precompute EE
    ee = compute_EE(flow, gt)
    for err, name in zip([compute_AEE, compute_BP, compute_Fl], ["AEE", "BP", "Fl"]):
        result[name] = err(flow, gt, ee=ee)
    return result


def getAllErrorMeasures_area(flow, gt, area):
    """compute all error measures only for a certain area of pixels and return them as a dict
    flow: estimated flow
    gt: groundtruth flow
    area: boolean array determining the evaluation area
    return: dictionary with keys AAE, AEE, BP, Fl and error values
    """
    gt_area = gt.copy()
    gt_area[np.invert(area)] = np.nan
    return getAllErrorMeasures(flow, gt_area)


def compute_SF(disp0, disp1, flow, gt_disp0, gt_disp1, gt_flow, t1=3.0, t2=0.05):
    disp0_mask = compute_DisparityError(disp0, gt_disp0, return_mask=True, t1=t1, t2=t2)
    disp1_mask = compute_DisparityError(disp1, gt_disp1, return_mask=True, t1=t1, t2=t2)
    flow_mask = compute_Fl(flow, gt_flow, return_mask=True, t1=t1, t2=t2)

    valid = (
        (~np.isnan(gt_disp0))
        & (~np.isnan(gt_disp1))
        & (~np.isnan(gt_flow[:, :, 0]))
        & (~np.isnan(gt_flow[:, :, 1]))
    )
    sf_mask = disp0_mask | disp1_mask | flow_mask
    sf_mask[~valid] = False

    d1_badcount = disp0_mask.sum()
    d1_pxcount = np.count_nonzero(~np.isnan(gt_disp0))
    d2_badcount = disp1_mask.sum()
    d2_pxcount = np.count_nonzero(~np.isnan(gt_disp1))
    fl_badcount = flow_mask.sum()
    fl_pxcount = np.count_nonzero(
        ~(np.isnan(gt_flow[:, :, 0]) | np.isnan(gt_flow[:, :, 1]))
    )
    sf_badcount = sf_mask.sum()
    sf_pxcount = np.count_nonzero(valid)

    return (
        d1_badcount,
        d1_pxcount,
        d2_badcount,
        d2_pxcount,
        fl_badcount,
        fl_pxcount,
        sf_badcount,
        sf_pxcount,
    )


def compute_SF_full(estimate, gt_noc, gt_occ, object_map, return_list=False):
    disp0, disp1, flow = estimate
    if return_list:
        output = []
    else:
        output = {}
    for obj, obj_name in zip([object_map, ~object_map, None], ["fg", "bg", "all"]):
        if not return_list:
            output[obj_name] = {}
        for occlusion, occ_name in zip([gt_noc, gt_occ], ["noc", "occ"]):
            if not return_list:
                output[obj_name][occ_name] = {}
            gt_disp_0, gt_disp_1, gt_flow = occlusion
            d1, d2, fl, sf = compute_SF(
                disp0,
                disp1,
                flow,
                gt_disp_0,
                gt_disp_1,
                gt_flow,
                return_all=True,
                eval_mask=obj,
            )
            for e, n in zip([d1, d2, fl, sf], ["d1", "d2", "fl", "sf"]):
                if return_list:
                    output.append(e)
                else:
                    output[obj_name][occ_name][n] = e

    return output


def compute_DisparityError(disp, gt, return_mask=False, t1=3.0, t2=0.05):
    error = np.abs(disp - gt)
    # number of valid pixels:
    count = np.count_nonzero(~np.isnan(error))

    # set the ee of nan pixels to zero
    error = np.nan_to_num(error, nan=0.0)
    abs_err = error > t1

    rel_err = error > t2 * np.nan_to_num(gt, nan=0.0)

    bp_mask = abs_err & rel_err

    if return_mask:
        return bp_mask
    else:
        return 100 * np.sum(bp_mask) / count


def compute_absDispError(disp, gt):
    valid = ~np.isnan(gt)
    return (np.abs(np.nan_to_num(gt) - disp) * valid).sum() / valid.sum()


def compute_epe3DError(disp2, flow, gt_disp2, gt_flow, intrinsics):
    target_gt = backproject_flow3d_target(gt_flow, intrinsics[0] / gt_disp2, intrinsics)
    target_est = backproject_flow3d_target(flow, intrinsics[0] / disp2, intrinsics)
    valid = ~np.isnan(target_gt.sum(axis=-1))
    return np.nansum(np.linalg.norm(target_gt - target_est, axis=-1)) / valid.sum()
