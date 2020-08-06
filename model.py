import torch
import torch.nn as nn

import wideresnet


class F(nn.Module):
    def __init__(self, args):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth=args.depth, widen_factor=args.width, norm=args.norm, dropout_rate=args.dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, args.n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, args):
        super(CCF, self).__init__(args)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])


class HYM(CCF):
    def __init__(self, args):
        super(HYM, self).__init__(args)

        self.K = args.contrast_k
        self.T = args.contrast_t
        self.dim = args.n_classes

        # create the queue
        init_logit = torch.randn(args.n_classes, args.contrast_k)
        self.register_buffer("queue_logit", init_logit)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, logits):
        # gather logits before updating queue
        batch_size = logits.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the logits at ptr (dequeue and enqueue)
        self.queue_logit[:, ptr : ptr + batch_size] = logits.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def joint(self, img, dist=None, evaluation=False):
        f_logit = self.class_output(self.f(img))  # queries: NxC
        ce_logit = f_logit  # cross-entropy loss logits
        prob = nn.functional.normalize(f_logit, dim=1)
        # positive logits: Nx1
        l_pos = dist * prob  # NxC
        l_pos = torch.logsumexp(l_pos, dim=1, keepdim=True)  # Nx1
        # negative logits: NxK
        buffer = nn.functional.normalize(self.queue_logit.clone().detach(), dim=0)
        l_neg = torch.einsum("nc,ck->nck", [dist, buffer])  # NxCxK
        l_neg = torch.logsumexp(l_neg, dim=1)  # NxK

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if not evaluation:
            self._dequeue_and_enqueue(f_logit)

        return logits, labels, ce_logit, l_neg.size(1)
