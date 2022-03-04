# FAH  taken and adapted from 'AdaMomentum' source code (file 'tp_adamomentum.py' in the same directory)

import math
import torch
from torch.optim.optimizer import Optimizer
from tabulate import tabulate
from colorama import Fore, Back, Style

version_higher = (torch.__version__ >= "1.5.0")


class AdaFamily(Optimizer):
    r"""Implements AdaFamily algorithm - 'don't torture yourself, Gomez - that's my job'
    AdaFamily is a 'family' of Adam-like Algorithms and can interpolate (via parameter 'myu') smoothly between Adam,
    AdaBelief and AdaMomentum (note it's not exactly Adam and Adabelief because 'eps' is on different positions).
    Code is modified from my AdaMomentum pytorch implementation (which in turn is based on AdaBelief code).
    My (Hannes Fassold) changes (from AdaBelief to AdaMomentum)e are marked with the tag '[FAH]' or 'FAH'.
    My changes (from AdaMomentum to AdaFamily) are marked with the tag '[FAMILY]' or 'FAMILY'.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        myu (float, optional): Interpolation parameter (default: 1.0)
            With myu = 0.0 you get (sort of) Adam algorithm,
            with myu = 0.5 you get (sort of) AdaBelief algorithm, and
            with myu = 1.0 you get (sort of) the AdaMomentum algorithm.
            Why 'sort of' ? Because the position of the epsilon differs in Adam and AdaBelief (see comments in code).
            Other myu values mean that are using a 'blend'of two algorithms (either Adam&AdaBelief or AdaBelief&AdaMomentum)
			Note via the 'set_myu' member function, you can set/change the parameter 'myu' later on also if you want.
        lr (float, optional): learning rate (default: 1e-3)
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

        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    """

    def __init__(self, params, myu = 1.0, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=False,
                 degenerated_to_sgd=False, epsilon_mode=0, use_past_denom_with_delay = 0, print_change_log=False):

        # ------------------------------------------------------------------------------
        # Print modifications to default arguments
        if print_change_log:
            print(Fore.RED + 'Please check your arguments if you have upgraded adabelief-pytorch from version 0.0.5.')
            print(Fore.RED + 'Modifications to default arguments:')
            default_table = tabulate([
                ['adabelief-pytorch=0.0.5', '1e-8', 'False', 'False'],
                ['>=0.1.0 (Current 0.2.0)', '1e-16', 'True', 'True']],
                headers=['eps', 'weight_decouple', 'rectify'])
            print(Fore.RED + default_table)

            recommend_table = tabulate([
                ['Recommended eps = 1e-8', 'Recommended eps = 1e-16'],
            ],
                headers=['SGD better than Adam (e.g. CNN for Image Classification)',
                         'Adam better than SGD (e.g. Transformer, GAN)'])
            print(Fore.BLUE + recommend_table)

            print(Fore.BLUE + 'For a complete table of recommended hyperparameters, see')
            print(Fore.BLUE + 'https://github.com/juntang-zhuang/Adabelief-Optimizer')

            print(
                Fore.GREEN + 'You can disable the log message by setting "print_change_log = False", though it is recommended to keep as a reminder.')

            print(Style.RESET_ALL)
        # ------------------------------------------------------------------------------

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(AdaFamily, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay

        # [FAMILY]
        self.epsilon_mode = epsilon_mode
        self.use_past_denom_with_delay = use_past_denom_with_delay
        # Set variable 'myu' properly (and calculate constant 'C' from it)
        self.set_myu(myu)
        # [~FAMILY]

        if self.weight_decouple:
            print('Weight decoupling enabled in AdaFamily')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaFamily')
        if amsgrad:
            print('AMSGrad enabled in AdaFamily')

    def __setstate__(self, state):
        super(AdaFamily, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

                # [FAMILY]
                state['past_denom'] = -1.0
                # [~FAMILY]
		
    # [FAMILY]
    # Set/modify 'myu' parameter during the training process (e.g. modify it every epoch).
    # This can be used e.g. to make an AdaFamily optimizer which 'morphes' during training
    # from 'Adam' (first epochs) to 'AdaBelief' (middle epochs) and then to 'AdaMomentum' (final epochs).
    def set_myu (self, myu):
        if not 0.0 <= myu <= 1.0:
            raise ValueError("Invalid myu parameter: {}".format(myu))
        self.myu = myu
        # Calculate the scaling constant 'C' from interpolation parameter 'myu'
        # C(myu) is a (slightly modified) triangle function, varying between 1.0 and 2.0.
        # myu = 0.0 and myu = 1.0 -> C = 1.0
        # myu = 0.5 -> C = 2.0
        self.C = 2.0 * (1.0 - abs(myu - 0.5))
    # [~FAMILY]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # FAH Regarding iterating over 'param groups', see https://discuss.pytorch.org/t/a-problem-about-optimizer-param-groups-in-step-function/14463
        # and https://discuss.pytorch.org/t/is-using-separate-optimizers-equivalent-to-specifying-different-parameter-groups-for-the-same-optimizer/95075
        # and https://stackoverflow.com/questions/62260985/what-are-saved-in-optimizers-state-dict-what-state-param-groups-stands-for
        # Note usually only one param group is used, containing _all_ parameters (to be optimized) of the model
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                # FAH 'p.grad' is the current gradient (is calculated 'lazily' on demand I suppose)
                # see https://stackoverflow.com/questions/65876372/pytorch-using-param-grad-affects-learning-process-solved
                # and https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
                # and https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaFamily does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                # FAH Statement means: exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # [FAMILY]
                # [FAH] In 'AdaMomentum' algorithm, we take 'exp_avg' instead of difference 'grad - exp_avg' (in AdaBelief)
                # Due to the (element-wise) squaring operation, it does not matter if we take 'exp_avg' or '-exp_avg'
                ##grad_residual = grad - exp_avg
                #grad_residual = exp_avg
                # [~FAH]
                # In 'AdaFamily' algorithm, we take a linear combination (weighted difference) of the vectors 'exp_avg'and 'grad'.
                # The calculation of 'grad_residual' is the _only_ difference (except for 'epsilon_mode') between AdaFamily and AdaMomentum algorithm
                grad_residual = self.C * (self.myu * exp_avg - (1 - self.myu) * grad)
                # Alternatively, calculate 'grad_residual' in a (possibly) more performant way using inplace functions.
                # Note we have to make a deep copy as we need 'exp_avg' still (do we really need it later on ?)
                #grad_residual = exp_avg.clone().mul_(self.C * myu)
                #grad_residual.add_(grad, alpha = self.C * (1 - myu))
                # FAH Statement means: exp_avg_var = beta2 * exp_avg_var + (1 - beta2) * grad_residual^2
                # Note that here, "^2" denotes _elementwise_ squaring (and not a scalar-product)
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value= 1 - beta2)
                # [~FAMILY]

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    # [FAH] Note for 'amsgrad' variant (which is inferior I suppose), I did not implement out 'epsilon_mode' [~FAH]
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # [FAH] In AdaFamily (with default epsilon_mode=0), we do not add the _last_ epsilon (compared to AdaBelief where it is added)
                    # In AdaMomentum & EAdam (see respective files in this directory) it is done in the same way as in AdaFamily.
                    # Note the _first_ epsilon is added in both AdaFamily _and_ also in AdaMomentum & AdaBelief (and also in EAdam),
                    # but _not_ in original 'Adam' or 'AdamW' algorithm where only the last epsilon is added !
                    if self.epsilon_mode == 0:
                        # epsilon mode 0 - add only first epsilon (like in EAdam and AdaMomentum)
                        # FAH Statement means: denom = sqrt( (exp_avg_var + eps) / (1 - beta2^step) )
                        denom = exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)
                    elif self.epsilon_mode == 1:
                        # epsilon mode 1 - add only last epsilon (like in Adam and AdamW)
                        # FAH Statement means: denom = sqrt(exp_avg_var) / sqrt(1 - beta2^step) + eps
                        denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    elif self.epsilon_mode == 2:
                        # epsilon mode 2 - add both first and last epsilon (like in AdaBelief)
                        # FAH Statement means: denom = sqrt(exp_avg_var + eps) / sqrt(1 - beta2^step) + eps
                        denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        raise ValueError("Invalid value for parameter epsilon_mode")
                    # [~FAH]

                #[FAMILY]
                # If we shall use the 'past version' of the denominator, do this here
                if self.use_past_denom_with_delay > 0:
                    if self.use_past_denom_with_delay > 1:
                        raise ValueError("Currently implemented only for a lag of 1 !!")
                    current_denom = denom
                    # We use the 'past' version of denom (for time 't - k'). Currently we support only value k = 1.
                    if not state['past_denom'] == -1:
                        denom = state['past_denom']
                    state['past_denom'] = current_denom
                #[~FAMILY]

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss