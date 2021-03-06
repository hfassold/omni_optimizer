# omni_optimizer
A wrapper for several SoA adaptive-gradient optimizer (Adam/AdamW/EAdam/AdaBelief/AdaMomentum/AdaFamily) via one API, including my novel 'AdaFamily' algorithm.
Regarding 'AdaFamily', see the arxiv preprint at https://arxiv.org/abs/2203.01603 (submitted to ISPR 2022 conference).
Setting 'myu' to either 0.25 oder 0.75 might be the best choice according to experiments (see preprint).

The main class to use is the class 'OmniOptimizer' (file 'omni_optimizer.py').

See https://github.com/hfassold/nlp_finetuning_adafamily for an application where AdaFamily is used for NLP finetuning.
