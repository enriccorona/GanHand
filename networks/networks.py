import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'image_to_hand_joints':
            from .image_to_hand_joints import Network
            network = Network(*args, **kwargs)
        elif network_name == 'vae':
            from .vae import Network
            network = Network(*args, **kwargs)
        elif network_name == 'vae_concatdec':
            from .vae_concatdec import Network
            network = Network(*args, **kwargs)
        elif network_name == 'rgb_to_hand_joints':
            from .rgb_to_hand_joints_and_imgrep import Network
            network = Network(*args, **kwargs)
        elif network_name == 'depth_to_hand_joints':
            from .depth_to_hand_joints import Network
            network = Network(*args, **kwargs)
        elif network_name == 'depth_to_hand_joints_and_imgrep':
            from .depth_to_hand_joints_and_imgrep import Network
            network = Network(*args, **kwargs)
        elif network_name == 'depth_to_hand_joints_and_imgrep_noisebeforebns':
            from .depth_to_hand_joints_and_imgrep_noisebeforebns import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_R':
            from .refine_from_R import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RT':
            from .refine_from_RT import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RTt':
            from .refine_from_RTt import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RI':
            from .refine_from_RI import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RTI':
            from .refine_from_RTI import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RTtI':
            from .refine_from_RTtI import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RTtIn':
            from .refine_from_RTtIn import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RrTtI':
            from .refine_from_RrTtI import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RrTtI_nonormals':
            from .refine_from_RrTtI_nonormals import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RrTtI_nonormals_deep':
            from .refine_from_RrTtI_nonormals_deep import Network
            network = Network(*args, **kwargs)
        elif network_name == 'refine_from_RrTtI_variableinput':
            from .refine_from_RrTtI_variableinput import Network
            network = Network(*args, **kwargs)
        elif network_name == 'discriminator_smpl+h':
            from .discriminator_smplh import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'discriminator_smpl+C':
            from .discriminator_smplC import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'discriminator_smpl+C_deeper':
            from .discriminator_smplC_deeper import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'discriminator_Dsmpl+C':
            from .discriminator_deeper_smplC import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'discriminator_image':
            from .discriminator_image import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'discriminator_wasserstein_gan':
            from .discriminator_wasserstein_gan import Discriminator
            network = Discriminator(*args, **kwargs)
        elif network_name == 'atlasnet':
            from .Atlasnet import Atlasnet
            network = Atlasnet(*args, **kwargs)
        elif network_name == 'atlasnet_as_in_obman':
            from .Atlasnet_obman import AtlasBranch
            network = AtlasBranch(*args, **kwargs)
        elif network_name == 'PoseNet':
            from .PoseNet import Network
            network = Network(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer
