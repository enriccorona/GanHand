import os
import torch
from torch.optim import lr_scheduler

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == 'classify_resnet_25':
            from .classify_resnet_25 import Model
            model = Model(*args, **kwargs)
        elif model_name == 'baseline_direct_regression':
            from .baseline_direct_regression import Model
            model = Model(*args, **kwargs)
        elif model_name == 'baseline_direct_regression_reltrans':
            from .baseline_direct_regression_reltrans import Model
            model = Model(*args, **kwargs)
        elif model_name == 'baseline_wherewhat':
            from .baseline_wherewhat import Model
            model = Model(*args, **kwargs)
        elif model_name == 'baseline_wherewhat_reltrans':
            from .baseline_wherewhat_reltrans import Model
            model = Model(*args, **kwargs)
        elif model_name == 'pose_estimation_directjoints':
            from .pose_estimation_directjoints import Model
            model = Model(*args, **kwargs)
        elif model_name == 'pose_estimation_voxels':
            from .pose_estimation_voxels import Model
            model = Model(*args, **kwargs)
        elif model_name == 'classify_resnet_25_inputnoise':
            from .classify_resnet_25_inputnoise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'classify_resnet_25_weighted':
            from .classify_resnet_25_weighted import Model
            model = Model(*args, **kwargs)
        elif model_name == 'classify_resnet_25_inputnoise_weighted':
            from .classify_resnet_25_inputnoise_weighted import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_classify_resnet_25':
            from .double_classify_resnet_25 import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_classify_resnet_25_weighted':
            from .double_classify_resnet_25_weighted import Model
            model = Model(*args, **kwargs)
        elif model_name == 'rgb_pretrain_model_cvpr1':
            from .rgb_pretrain_model_cvpr1 import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2_rgb':
            from .l2_rgb import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2_depth_to_hand':
            from .l2_depth_to_parametric_hand import Model
            model = Model(*args, **kwargs)
        elif model_name == 'rgb_to_classification':
            from .rgb_to_classification_norot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'rgb_to_classification_wnoise':
            from .rgb_to_classification_norot_wnoise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification':
            from .depth_to_classification import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_wrot':
            from .depth_to_classification_wrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot':
            from .depth_to_classification_norot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine':
            from .depth_to_classification_norot_refine import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHgan':
            from .depth_to_classification_norot_refine_CHgan import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI':
            from .depth_to_classification_norot_refine_CHganI import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_HganCI':
            from .depth_to_classification_norot_refine_HganCI import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_wimg_relativepos':
            from .depth_to_classification_norot_refine_CHganI_relativepos import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_wimg_relativepos':
            from .depth_to_classification_norot_refine_CHganI_relativepos import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_residualpos':
            from .depth_to_classification_norot_refine_CHganI_residualpos import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot':
            from .depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot':
            from .depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot_winputnoise':
            from .depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot_winputnoise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_refine_diff_fk':
            from .double_depth_refine_diff_fk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_refine_diff_fk_doublepenalisation':
            from .double_depth_refine_diff_fk_doublepenalisation import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_refine_diff_fk_DseesrelativeT_higherlambdas':
            from .double_depth_refine_diff_fk_DseesrelativeT_higherlambdas import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_refine_diff_fk_theoreticalloss':
            from .double_depth_refine_diff_fk_theoreticalloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_theoretical_wBVH':
            from .fk_theoretical_wBVH import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_theoretical_wBVH_wfingersloss':
            from .fk_theoretical_wBVH_wfingersloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb':
            from .fk_practical_rgb import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_DseesCPdist':
            from .fk_practical_rgb_DseesCPdist import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_DseesCPdist_optim2dof':
            from .fk_practical_rgb_DseesCPdist_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist':
            from .fk_practical_rgb_alsominCPdist import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_nofk':
            from .fk_practical_rgb_alsominCPdist_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_deeperD':
            from .fk_practical_rgb_alsominCPdist_deeperD import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_optim2dof':
            from .fk_practical_rgb_alsominCPdist_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_optim2dof_deeperD':
            from .fk_practical_rgb_alsominCPdist_optim2dof_deeperD import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_optim3dof':
            from .fk_practical_rgb_alsominCPdist_optim3dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_optim3dof_deeperD':
            from .fk_practical_rgb_alsominCPdist_optim3dof_deeperD import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_originalFK':
            from .fk_practical_rgb_alsominCPdist_originalFK import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation import Model
            model = Model(*args, **kwargs)

        # GET DATASET WITH DIVERSE DELTA IN TEST SET
        elif model_name == 'fromatlas_optim1dof_deltaexperiment':
            from .fromatlas_optim1dof_deltaexperiment import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fromatlas_optim2dof_deltaexperiment':
            from .fromatlas_optim2dof_deltaexperiment import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fromatlas_optim3dof_deltaexperiment':
            from .fromatlas_optim3dof_deltaexperiment import Model
            model = Model(*args, **kwargs)
        elif model_name == 'noatlas_optim1dof_deltaexperiment':
            from .noatlas_optim1dof_deltaexperiment import Model
            model = Model(*args, **kwargs)
        elif model_name == 'noatlas_optim2dof_deltaexperiment':
            from .noatlas_optim2dof_deltaexperiment import Model
            model = Model(*args, **kwargs)
        elif model_name == 'noatlas_optim3dof_deltaexperiment':
            from .noatlas_optim3dof_deltaexperiment import Model
            model = Model(*args, **kwargs)

        # MODELS FROM SYNTHETIC DATA WITH RECONSTRUCTION
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_nofk':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim1dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_optim3dof import Model
            model = Model(*args, **kwargs)

        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_nofk':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim1dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_baselinedirect_optim3dof import Model
            model = Model(*args, **kwargs)

        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_nofk':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim1dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_fromatlas_wherewhat_optim3dof import Model
            model = Model(*args, **kwargs)



            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_wmask':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_wmask import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_optim3dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_nofk':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_nofk import Model
            model = Model(*args, **kwargs)

# OURS TRAINING G TOO:
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_optim2dof_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_optim2dof_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_optim3dof_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_optim3dof_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_nofk_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_nofk_trainGtoo import Model
            model = Model(*args, **kwargs)

# BASELINE DIRECT:
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim1dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_direct_optim3dof import Model
            model = Model(*args, **kwargs)

# BASELINE WHEREWHAT
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim1dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim2dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim3dof':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim3dof import Model
# BASELINE WHEREWHAT TRAINING G TOO
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim1dof_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim1dof_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim2dof_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim2dof_trainGtoo import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim3dof_trainGtoo':
            from .fk_practical_rgb_alsominCPdist_gradientaccumulation_baseline_wherewhat_optim3dof_trainGtoo import Model


# OLDER:
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_rgb_alsominCPdist_nonormals':
            from .fk_practical_rgb_alsominCPdist_nonormals import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH':
            from .fk_practical_wBVH import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH_completeD':
            from .fk_practical_wBVH_completeD import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH_completeD_twolosses':
            from .fk_practical_wBVH_completeD_twolosses import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH_single_oldinter':
            from .fk_practical_wBVH_single_oldinter import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH_efficient':
            from .fk_practical_wBVH_efficient import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_practical_wBVH_efficient_twolosses':
            from .fk_practical_wBVH_efficient_twolosses import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_iterative':
            from .fk_iterative import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_iterative_efficient':
            from .fk_iterative_efficient import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_iterative_efficient_real':
            from .fk_iterative_efficient_real import Model
            model = Model(*args, **kwargs)
        elif model_name == 'fk_iterative_efficient_usecpd':
            from .fk_iterative_efficient_usecpd import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_refine_diff_fk_doublepenalisation_DseesrelativeT_higherlambdas':
            from .double_depth_refine_diff_fk_doublepenalisation_DseesrelativeT_higherlambdas import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHgan_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_CHgan_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_pretrained_CHgan_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_pretrained_CHgan_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_pretrained_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_pretrained_argmaxClass_CHganI_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_pretrained_argmaxClass_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_pretrainedweighted_argmaxClass_CHganI_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_pretrainedweighted_argmaxClass_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot':
            from .double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHganI_wimg_residualpos':
            from .double_depth_to_classification_norot_refine_CHganI_wimg_residualpos import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_deeperD':
            from .double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_deeperD import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_deeperD_noiseR':
            from .double_depth_to_classification_norot_refine_CHganI_wimg_residualpos_deeperD_noiseR import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot_weighted':
            from .depth_to_classification_norot_refine_CHganI_wimg_residualpos_residualrot_weighted import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganI_wimg':
            from .depth_to_classification_norot_refine_CHganI_wimg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganIdC_wimg':
            from .depth_to_classification_norot_refine_CHganIdC_wimg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'double_depth_to_classification_norot_refine_CHganI_wimg':
            from .double_depth_to_classification_norot_refine_CHganI_wimg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_CHganL2':
            from .depth_to_classification_norot_refine_CHganL2 import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_Closs':
            from .depth_to_classification_norot_refine_Closs import Model
            model = Model(*args, **kwargs)
        elif model_name == 'depth_to_classification_norot_refine_wimg':
            from .depth_to_classification_norot_refine_wimg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2D_depth_to_hand':
            from .l2D_depth_to_parametric_hand import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2D_depth_to_hand_w_noise':
            from .l2D_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2DMinFi_depth_to_hand_w_noise':
            from .l2DMinFi_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2DMinFiMinAngle_depth_to_hand_w_noise':
            from .l2DMinFiMinAngle_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2DR_depth_to_hand_w_noise':
            from .l2DR_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2DCMinFiMinAngle_depth_to_parametric_hand_w_noise':
            from .l2DCMinFiMinAngle_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'l2DC_depth_to_hand_w_noise':
            from .l2DC_depth_to_parametric_hand_w_noise import Model
            model = Model(*args, **kwargs)
        elif model_name == 'model_predict_grasp_quality_from_hand_pose':
            from .model_predict_grasp_quality_from_hand_pose import Model
            model = Model(*args, **kwargs)
        elif model_name == 'model_predict_grasp_quality_from_contact_normals':
            from .model_predict_grasp_quality_from_contact_normals import Model
            model = Model(*args, **kwargs)
        elif model_name == 'Atlasnet':
            from .depth_atlasnet import Model
            model = Model(*args, **kwargs)
        elif model_name == 'Atlasnet_as_in_obman':
            from .depth_atlasnet_obman import Model
            model = Model(*args, **kwargs)
        elif model_name == 'rgb_atlasnet_obman':
            from .rgb_atlasnet_obman import Model
            model = Model(*args, **kwargs)
        elif model_name == 'rgb_atlasnet_obman_regularised':
            from .rgb_atlasnet_obman_regularised import Model
            model = Model(*args, **kwargs)

    ## YCB DATASET:
# OURS:
        elif model_name == 'ycb_ours_nofk':
            from .ycb_ours_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_nofk_b':
            from .ycb_ours_nofk_b import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof':
            from .ycb_ours_1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_b':
            from .ycb_ours_1dof_b import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_2dof':
            from .ycb_ours_2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_3dof':
            from .ycb_ours_3dof import Model
            model = Model(*args, **kwargs)

        elif model_name == 'ycb_ours_DFseesplane2_getting_no_and_optim':
            from .ycb_ours_DFseesplane2_getting_no_and_optim import Model
            model = Model(*args, **kwargs)

# OURS FOR DELTA:
        elif model_name == 'ycbdelta_1dof':
            from .ycbdelta_1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycbdelta_2dof':
            from .ycbdelta_1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycbdelta_3dof':
            from .ycbdelta_1dof import Model
            model = Model(*args, **kwargs)

# OURS WITHOUT SOME OF THE LOSSES FOR ABLATION:
        elif model_name == 'ycb_ours_nofk_b_noadversarialloss':
            from .ycb_ours_nofk_b_noadversarialloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_nofk_b_nointersectionloss':
            from .ycb_ours_nofk_b_nointersectionloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_nofk_b_nocontactloss':
            from .ycb_ours_nofk_b_nocontactloss import Model
            model = Model(*args, **kwargs)

        elif model_name == 'ycb_ours_1dof_b_noadversarialloss':
            from .ycb_ours_1dof_b_noadversarialloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_b_nointersectionloss':
            from .ycb_ours_1dof_b_nointersectionloss import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_b_nocontactloss':
            from .ycb_ours_1dof_b_nocontactloss import Model
            model = Model(*args, **kwargs)


# OURS MORE:
        elif model_name == 'ycb_ours_1dof_alsolossreg':
            from .ycb_ours_1dof_alsolossreg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_lrdiff':
            from .ycb_ours_1dof_lrdiff import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeobjtype':
            from .ycb_ours_1dof_DFseeobjtype import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq':
            from .ycb_ours_1dof_DFseeplaneeq import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq_deeperFC':
            from .ycb_ours_1dof_DFseeplaneeq_deeperFC import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq_alsolossreg_fromapproxrightrot.py':
            from .ycb_ours_1dof_DFseeplaneeq_alsolossreg_fromapproxrightrot.py import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq_deeperFC_fromapproxrightrot':
            from .ycb_ours_1dof_DFseeplaneeq_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq_alsolossreg':
            from .ycb_ours_1dof_DFseeplaneeq_alsolossreg import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeallpluspose':
            from .ycb_ours_1dof_DFseeallpluspose import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeallplusbbox':
            from .ycb_ours_1dof_DFseeallplusbbox import Model
            model = Model(*args, **kwargs)

        elif model_name == 'ycb_ours_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_ours_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_ours_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_ours_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_ours_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_ours_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
# Wherewhat Baseline
        elif model_name == 'ycb_wherewhat_nofk':
            from .ycb_wherewhat_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_1dof':
            from .ycb_wherewhat_1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_2dof':
            from .ycb_wherewhat_2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_3dof':
            from .ycb_wherewhat_3dof import Model
            model = Model(*args, **kwargs)

        elif model_name == 'ycb_wherewhat_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_wherewhat_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_wherewhat_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_wherewhat_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_wherewhat_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_wherewhat_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
# Direct Baseline
        elif model_name == 'ycb_bsl_nofk':
            from .ycb_bsl_nofk import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_1dof':
            from .ycb_bsl_1dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_2dof':
            from .ycb_bsl_2dof import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_3dof':
            from .ycb_bsl_3dof import Model
            model = Model(*args, **kwargs)


        elif model_name == 'ycb_bsl_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_bsl_nofk_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_bsl_1dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_bsl_2dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)
        elif model_name == 'ycb_bsl_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot':
            from .ycb_bsl_3dof_DFseeplaneeq2_deeperFC_fromapproxrightrot import Model
            model = Model(*args, **kwargs)

        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model_name)
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._gpu_ids = opt.gpu_ids
        self._is_train = opt.is_train

        self._Tensor = torch.cuda.FloatTensor if self._gpu_ids else torch.Tensor
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        network.load_state_dict(torch.load(load_path))
        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def _get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
