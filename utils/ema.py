class EMA_classifier():
    def __init__(self, model, decay, args):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # the flag for whether register ema
        self.flag = True
        self.args = args

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        if self.flag:
            self.register()
            self.flag = False
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    if self.args.adapt_ema_c:
                        new_average = self.model.gamma * self.shadow[name] + (1.0 - self.model.gamma) * param.data 
                    else:
                        new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data 
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self, w=0.5):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                if self.args.test_fuse_c:
                    param.data = self.shadow[name] * w + param.data * (1-w)
                else:
                    param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                if self.args.train_fuse_c:
                    new_average = (self.backup[name] + self.shadow[name]) / 2 
                    param.data = new_average.clone()
                else:
                    param.data = self.backup[name]
        self.backup = {}