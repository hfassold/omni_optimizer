# FAH  taken and adapted from 'AdaMomentum' source code (file 'tp_adamomentum.py' in the same directory)

import math
import torch
from torch.optim.optimizer import Optimizer


from src.optimizer.tp_adabelief import AdaBelief
from src.optimizer.tp_eadam import EAdam
from src.optimizer.my_adafamily import AdaFamily
from src.optimizer.my_adamomentum import AdaMomentum

version_higher = (torch.__version__ >= "1.5.0")

# Get a linear mapping from interval [A, B] to interval [a, b], and evaluate this mapping fn. for value 'val'
def linear_interval_mapping_eval_at(val, A, B, a, b):
    # taken from https://stackoverflow.com/questions/12931115/algorithm-to-map-an-interval-to-a-smaller-interval/45426581
    return (val - A) * (b - a) / (B - A) + a


class OmniOptimizer:
    r"""
    This class contains multiple optimizers in one class.
    Important: Parameter 'myu' (for AdaFamily) must be set _explicitly_ at begin of each training epoch
    (as it can change during training) via function 'update_myu_for_training_epoch(...)' !
    The optimizer (which you give e.g. to pytorch or pytorch lightning) is accessable via 'self.opt' !
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            Note: for technical reasons, this must be a 'Callable' - usually you pass 'my_model.parameters' here
            (so you hand over sort of a 'pointer' to the class method 'parameters' itself - to speak in C++ syntax...)
            Do NOT pass 'my_model.parameters()' (so the already generated sequence) !!
        algorithm (str, optional): Desired optimizer algorithm (default: 'torch.adam')
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum for SGD optimizer (default: 0.9)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:False) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
        myu_a(float, optional): (default: 0.0)
        myu_b(float, optional): (default: 1.0)
            The myu-range for 'AdaFamily' algorithm
            Both values 'myu_a' and 'myu_b' must be in range [0.0 1.0]
            'myu_a' must be smaller than (or equal to) 'myu_b'
            myu_epoch_scheme: (str, optional): (default: 'constant')
            The myu-setting scheme,as a function of the current epoch 'curr_epoch' (and maximum epoch 'max_epoch')
            'myu' is the actual myu used for current epoch
            Possible values
            - 'constant': myu is always set to 'myu_a'
            - 'linear':   myu varies according to a linear function which maps
               interval [0, max_epoch] to interval [myu_a myu_b]
               So in epoch 0, myu will be 'myu_a' and in the final epoch, myu will be 'myu_b'
        epsilon_mode(int, optional): (default: 0)
            Defines - only for 'AdaFamily' algorithm' - on which places the 'epsilon' value
            is added in the statement where the denominator is calculated.
            In AdaMomentum and AdaFamily (with default epsilon_mode value '0'), we only add the first epsilon
            (and do not add the _last_ epsilon - compared to AdaBelief where the last epsilon is added).
            In EAdam it is done in the same way as in AdaMomentum.
            Note the _first_ epsilon is added in both AdaMomentum _and_ also in AdaBelief (and also in EAdam),
            but _not_ in original 'Adam' or 'AdamW' algorithm where only the last epsilon is added !
            Possible values:
            - 0: Add only first epsilon (like in AdaMomentum and EAdam)
            - 1: Add only last epsilon  (like in Adam and AdamW)
            - 2: Add first epsilon and also last epsilon (like in AdaBelief)
        use_past_denom_with_delay (int, optional): (default: 0)
            If set > 0, we use in 'AdaFamily' algorithm a 'past' version of the denominator (from time 't - k') instead the current version (time 't')
            This is inspired by the 'ACProp' paper, NeurIPS 2021, https://arxiv.org/pdf/2110.05454.pdf
            If set to '0', we disable this mechanism (so we have no 'delay').
            If set to a value 'k' with k > 0, a delay of 'k' is introduced, so we at timepoint 't'
            we use the denominator for time 't - k'.
            In the ACProp paper, k is set to '1'
            Note I did not test so far whether this 'denom delay' brings a benefit for AdaFamily algorithm - have to do some time...

    """

    def __init__(self, params, algorithm = 'torch.adam', lr=1e-3, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, eps=1e-8,
                 weight_decay = 0, amsgrad = False, weight_decouple = True, fixed_decay = False, rectify = False,
                 degenerated_to_sgd = False, myu_a = 0.0, myu_b = 1.0, myu_epoch_scheme = 'constant',
                 epsilon_mode = 0, use_past_denom_with_delay = 0):

        self.myu_a = myu_a
        self.myu_b = myu_b
        self.myu_epoch_scheme = myu_epoch_scheme

        #param_groups = list(params())
        # my_len should be non-zero
        #my_len = len(param_groups)

        # Note the '()' at the end of 'params' - it invokes the class method handed over via 'params' !!
        self.algorithm = algorithm
        if algorithm == 'torch.sgdm':
            self.opt = torch.optim.SGD(params(), lr = lr, momentum = momentum, weight_decay = weight_decay)
        elif algorithm == 'torch.adam':
            self.opt = torch.optim.Adam(params(), lr = lr, betas = (beta1, beta2), eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
        elif algorithm == 'torch.adamw':
            self.opt = torch.optim.AdamW(params(), lr = lr, betas = (beta1, beta2), eps=eps, weight_decay = weight_decay, amsgrad = amsgrad)
        elif algorithm == 'tp.eadam':
            self.opt = EAdam(params(), lr = lr, betas = (beta1, beta2), eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
        elif algorithm == 'tp.adabelief':
            self.opt = AdaBelief(params(), lr = lr, betas = (beta1, beta2), eps=eps, weight_decay = weight_decay, amsgrad = amsgrad,
                                 weight_decouple = weight_decouple, fixed_decay = fixed_decay, rectify = rectify,
                                 degenerated_to_sgd = degenerated_to_sgd, print_change_log = False)
        elif algorithm == 'my.adamomentum':
            self.opt = AdaMomentum(params(), lr = lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                                 weight_decouple=weight_decouple, fixed_decay=fixed_decay, rectify=rectify,
                                 degenerated_to_sgd=degenerated_to_sgd, print_change_log=False)
        elif algorithm == 'my.adafamily':
            # Note: we set an arbitrary myu, as it will be updated every epoch ...
            self.opt = AdaFamily(params(), lr = lr, myu = 0.0, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                                 weight_decouple=weight_decouple, fixed_decay=fixed_decay, rectify=rectify,
                                 degenerated_to_sgd=degenerated_to_sgd, print_change_log=False,
                                 epsilon_mode=epsilon_mode, use_past_denom_with_delay=use_past_denom_with_delay)
        xyz = 123

    def update_myu_for_training_epoch(self, current_epoch, max_epochs):
        if self.algorithm == 'my.adafamily':
            # calculate myu to use for current epoch
            myu_a = self.myu_a
            myu_b = self.myu_b
            myu_epoch_scheme = self.myu_epoch_scheme
            if myu_epoch_scheme == 'constant':
                myu = myu_a
            elif myu_epoch_scheme == 'linear':
                # myu varies according to a linear function which maps interval [0, max_epochs - 1] to interval [myu_a, myu_b]
                myu = linear_interval_mapping_eval_at(current_epoch, 0, max_epochs - 1, myu_a, myu_b)
            elif myu_epoch_scheme == 'linear_rev':
                # myu varies according to a linear function which maps interval [0, max_epochs - 1] to interval [myu_b, myu_a]
                myu = linear_interval_mapping_eval_at(current_epoch, 0, max_epochs - 1, myu_b, myu_a)
            else:
                raise ValueError("Invalid value for parameter myu_epoch_scheme")

            # set current myu now for AdaFamily algorithm
            self.opt.set_myu(myu)

class OmniOptimizerLazyConstruct:
    r"""
    Is more or less the same as 'OmniOptimizer' class, but with _lazy_ construction of 'self.core' object,
    when 'set_model_params' function is called (should be called only _once_).
    So 'params' is not handed over as constructor parameter, but later in 'set_model_params' function.
    Why is this class necessary ? for technical reasons ('params' cannot be specified in optimizer's config file).
    And whe are using config files for Omni-Optimizer parameters (specifically, we are using the 'hydra' python package)
    because config files are much nicer than command line arguments..
    """

    def __init__(self, algorithm = 'torch.adam', lr=1e-3, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                 weight_decay = 0, amsgrad = False, weight_decouple = True, fixed_decay = False, rectify = False,
                 degenerated_to_sgd = False, myu_a = 0.0, myu_b = 1.0, myu_epoch_scheme = 'constant',
                 epsilon_mode = 0, use_past_denom_with_delay = 0):

        # Get all constructor arguments as a _dictionary_ and save them in 'self.constructor_args'
        # Note this statements must be at the VERY BEGIN of this function !!
        # Taken from answer of 'isaiah' at https://stackoverflow.com/questions/1408818/getting-the-keyword-arguments-actually-passed-to-a-python-method
        # see also https://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function
        self.constructor_args =  locals().copy()
        # delete 'self' key from list of constructor args
        self.constructor_args.pop('self', None)
        xyz = 123

    # Constructs the 'OmniOptimizer' object 'lazily' in 'self.core'
    # 'params' are the model params (you get via fn. 'my_model.params()')
    def set_model_params_from_model(self, model):
        # Now we have all parameters and can construct the OmniOptimizer object
        # The '**' operators spreads out the dictionary items in 'self.constructor_args'
        # See answer of 'mark' at https://stackoverflow.com/questions/64831569/passing-a-dictionary-into-init-in-python-and-having-issues
        #self.constructor_args['params'] = params
        # Important: We hand over only the 'pointer' to the 'model.parameters()' function (which is sort of a generator function),
        # not the (generated) parameters itself !
        self.core = OmniOptimizer(model.parameters, **self.constructor_args)
        xyz = 123



