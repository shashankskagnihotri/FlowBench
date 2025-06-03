#coding=utf-8

def rescale_sintel_scenes(
    scene, gt1_depth, gt2_depth, rel_mat, OVERALL_SCALE=1.0, transl_scale=0.0
):

    GLOBAL_SCALE = 1.0
    # for mountain_1 --> scale 3D scene globally
    if scene in ["alley_1", "alley_2", "ambush_4", "bandage_2", "market_5", "market_6"]:
        GLOBAL_SCALE = 2.0
    elif scene in ["mountain_1"]:
        GLOBAL_SCALE = 0.05
    elif scene in ["market_2"]:
        GLOBAL_SCALE = 0.2
    elif scene in ["cave_2"]:
        GLOBAL_SCALE = 0.4
    elif scene in ["ambush_2", "ambush_6"]:
        GLOBAL_SCALE = 1.5
    elif scene in ["shaman_2"]:
        GLOBAL_SCALE = 2.3
    elif scene in ["bandage_1", "shaman_3"]:
        GLOBAL_SCALE = 2.5
    elif scene in ["sleeping_1"]:
        GLOBAL_SCALE = 5.0
    elif scene in ["ambush_7"]:
        GLOBAL_SCALE = 7.0

    if GLOBAL_SCALE != 1.0 or OVERALL_SCALE != 1.0 or transl_scale != 0:
        gt1_depth *= GLOBAL_SCALE * OVERALL_SCALE
        gt2_depth *= GLOBAL_SCALE * OVERALL_SCALE
        rel_mat[..., :3, 3] *= GLOBAL_SCALE * OVERALL_SCALE * transl_scale

    return gt1_depth, gt2_depth, rel_mat
