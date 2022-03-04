# FAH  taken and adapted from 'AdaBelief' source code (file 'tp_adabélief.py' in the same directory)
# FAH:

import math
import torch
from torch.optim.optimizer import Optimizer
from tabulate import tabulate
from colorama import Fore, Back, Style

version_higher = (torch.__version__ >= "1.5.0")


class AdaMomentum(Optimizer):
    r"""Implements AdaMomentum algorithm.
    AdaMomentum paper can be found at https://arxiv.org/abs/2106.11514
    Code is modified from AdaBelief pytorch implementation (pip package, version 0.2.1)
    My (Hannes Fassold) changes in the code are marked with the tag '[FAH]' or 'FAH'
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
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
        print_change_log (boolean, optional) (default: False) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaMomentum Optimizer
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=False,
                 degenerated_to_sgd=False, print_change_log=False):

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
        super(AdaMomentum, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaMomentum')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaMomentum')
        if amsgrad:
            print('AMSGrad enabled in AdaMomentum')

    def __setstate__(self, state):
        super(AdaMomentum, self).__setstate__(state)
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
                        'AdaMomentum does not support sparse gradients, please consider SparseAdam instead')
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
                # [FAH] In 'AdaMomentum' algorithm, we take 'exp_avg' instead of difference 'grad - exp_avg' (in AdaBelief)
                # Due to the squaring operation, it does not matter if we take 'exp_avg' or '-exp_avg'
                #grad_residual = grad - exp_avg
                grad_residual = exp_avg
                # [~FAH]
                # FAH Statement means: exp_avg_var = beta2 * exp_avg_var + (1 - beta2) * grad_residual^2,
                # with "^2" being the _elementwise_ squaring operation
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # [FAH] In AdaMomentum, we do not add the _last_ epsilon (compared to AdaBelief where it is added)
                    # In EAdam (see file 'tp_eadam.py' in the same directory) it is done in the same way as in AdaMomentum.
                    # Note the _first_ epsilon is added in both AdaMomentum _and_ also in AdaBelief (and also in EAdam),
                    # but _not_ in original 'Adam' or 'AdamW' algorithm where only the last epsilon is added !
                    #denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # FAH Statement means: denom = sqrt( (exp_avg_var + eps) / (1 - beta2^step) )
                    denom = exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)
                    # [~FAH]

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