from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self.is_train = False

        self._parser.add_argument('--lambda_D_prob', type=float, default=1, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')
        self._parser.add_argument('--lambda_G_contactloss', type=float, default=100, help='')
        self._parser.add_argument('--lambda_G_different_normals', type=float, default=1, help='')
        self._parser.add_argument('--lambda_G_normals_suit_object', type=float, default=3, help='')
        self._parser.add_argument('--lambda_G_l2', type=float, default=1, help='')
        self._parser.add_argument('--lambda_G_intersections', type=float, default=100, help='')

        self._parser.add_argument('--lambda_G_fk', type=float, default=0.1, help='')
        self._parser.add_argument('--lambda_G_angles', type=float, default=0.01, help='')
        self._parser.add_argument('--lambda_G_plane', type=float, default=1.0, help='')

        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=5, help='train G every n interations')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=20, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')


        self._parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for G adam')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self._parser.add_argument('--lr_D', type=float, default=0.0001, help='initial learning rate for D adam')
        self._parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')

        self._parser.add_argument('--Delta', type=float, default=0.0, help='beta2 for D adam')
