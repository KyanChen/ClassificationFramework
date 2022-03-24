from mmcv.runner import HOOKS, Hook, PolyLrUpdaterHook


@HOOKS.register_module()
class InnerPolyLrUpdaterHook(PolyLrUpdaterHook):
    def __init__(self, apply_model=None, inner_loop_num=10, **kwargs):
        assert apply_model is not None
        self.apply_model = apply_model
        self.inner_loop_num = inner_loop_num
        super(InnerPolyLrUpdaterHook, self).__init__(**kwargs)

    def _set_lr(self, runner, lr_groups):
        param_groups = runner.optimizer[self.apply_model].param_groups

        for param_group, lr in zip(param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr, cur_iter=0):
        max_progress = self.inner_loop_num
        progress = cur_iter
        coeff = (1 - progress / max_progress) ** self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr

    def get_regular_lr(self, runner, cur_iter):
        # param_groups = runner.optimizer[self.apply_model].param_groups
        return [self.get_lr(runner, _base_lr, cur_iter) for _base_lr in self.base_lr]

    def before_run(self, runner):
        param_groups = runner.optimizer[self.apply_model].param_groups
        for group in param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in param_groups
        ]

    def before_train_inner_iter(self, runner, cur_iter):
        self.regular_lr = self.get_regular_lr(runner, cur_iter)
        self._set_lr(runner, self.regular_lr)

    def before_train_epoch(self, runner):
        pass

    def before_train_iter(self, runner):
        pass




@HOOKS.register_module()
class OuterPolyLrUpdaterHook(PolyLrUpdaterHook):
    def __init__(self, apply_model=None, **kwargs):
        assert apply_model is not None
        self.apply_model = apply_model
        super(OuterPolyLrUpdaterHook, self).__init__(**kwargs)

    def before_run(self, runner):
        param_groups = runner.optimizer[self.apply_model].param_groups
        for group in param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in param_groups
        ]

    def _set_lr(self, runner, lr_groups):
        param_groups = runner.optimizer[self.apply_model].param_groups

        for param_group, lr in zip(param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress) ** self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr

    def get_regular_lr(self, runner):
        # param_groups = runner.optimizer[self.apply_model].param_groups
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]