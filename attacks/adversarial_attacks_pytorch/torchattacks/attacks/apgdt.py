import time

import numpy as np

import torch

from ..attack import Attack


class APGDT(Attack):
    r"""
    APGD-Targeted in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.'
    Targeted attack for every wrong classes.
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)
        n_classes (int): number of classes. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        steps=10,
        n_restarts=1,
        seed=0,
        eot_iter=1,
        rho=0.75,
        verbose=False,
        n_classes=1,
    ):
        super().__init__("APGDT", model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.n_target_classes = n_classes - 1
        self.supported_mode = ["default"]

    def forward(self, images, labels, target):
        r"""
        Overridden.
        """

        images = images.squeeze(0)
        labels = labels.squeeze(0)
        self.target_class = target.squeeze(0)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (
            x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-12
        )

    def attack_single_run(self, images_orig, labels_orig):
        images = images_orig.clone() if len(images_orig.shape) == 4 else images_orig.clone().unsqueeze(0)
        lables = labels_orig.clone() if len(labels_orig.shape) == 4 else labels_orig.clone().unsqueeze(0)

        image_1 = images[0].unsqueeze(0)
        image_2 = images[1].unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )  # nopep8
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )  # nopep8

        if self.norm == "Linf":
            # image 1 Linf-norm
            t = 2 * torch.rand(image_1.shape).to(self.device).detach() - 1
            image_1_adv = image_1.detach() + self.eps * torch.ones([image_1.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
            # image 2 Linf-norm
            t = 2 * torch.rand(image_2.shape).to(self.device).detach() - 1
            image_2_adv = image_2.detach() + self.eps * torch.ones([image_2.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
            # image 1 L2-norm
        elif self.norm == "L2":
            t = torch.randn(image_1.shape).to(self.device).detach()
            image_1_adv = image_1.detach() + self.eps * torch.ones([image_1.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
            t = torch.randn(image_2.shape).to(self.device).detach()
            image_2_adv = image_2.detach() + self.eps * torch.ones([image_2.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
        image_1_adv = image_1_adv.clamp(0.0, 1.0)
        image_2_adv = image_2_adv.clamp(0.0, 1.0)

        images_best = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)
        images_best_adv = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)

        loss_steps = torch.zeros([self.steps, 1])
        loss_best_steps = torch.zeros([self.steps + 1, 1])
        epe_steps = torch.zeros_like(loss_best_steps)

        images_adv = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)
        images_adv_dic = {"images": images_adv}
        preds_dic = self.get_logits(images_adv_dic)
        preds = preds_dic["flows"].squeeze(0)

        import pdb
        pdb.set_trace()

        # y_target = output.sort(dim=1)[1][:, -self.target_class]

        images_adv.requires_grad_(True)
        grad = torch.zeros_like(images.unsqueeze(0))
        images_adv_dic = {"images": images_adv}

        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        # counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size[0] * grad / (
                        (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )  # nopep8
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        # n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, images_orig, labels_orig, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        images = images_orig.clone() if len(images_orig.shape) == 4 else images_orig.clone().unsqueeze(0)
        labels = labels_orig.clone() if len(labels_orig.shape) == 4 else labels_orig.clone().unsqueeze(0)

        images_adv = images.clone()
        pred_dic = self.get_logits(get_input_format(images))

        epe_score = epe(labels, pred_dic["flows"].squeeze(0))
        epe_score = epe_score.mean()
        # acc = self.get_logits(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            # print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        if not cheap:
            raise ValueError("not implemented yet")

        else:
            # for target_class in range(2, self.n_target_classes + 2):
                # self.target_class = target_class
            for counter in range(self.n_restarts):
                # ind_to_fool = acc.nonzero().squeeze()
                # if len(ind_to_fool.shape) == 0:
                #     ind_to_fool = ind_to_fool.unsqueeze(0)
                # if ind_to_fool.numel() != 0:
                images_to_fool, lables_to_fool = (
                    images.clone(),
                    labels.clone(),
                )  # nopep8
                (
                    images_best_curr,
                    epe_curr,
                    loss_curr,
                    images_adv_curr,
                ) = self.attack_single_run(
                    images_to_fool, lables_to_fool
                )  # nopep8
                # ind_curr = (acc_curr == 0).nonzero().squeeze()

                # acc[ind_to_fool[ind_curr]] = 0
                # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print(
                        "restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s".format(
                            counter,
                            self.target_class,
                            # acc.float().mean(),
                            self.eps,
                            time.time() - startt,
                        )
                    )

        return epe_score, images_adv
    

@staticmethod
def get_input_format(input):
    if isinstance(input, dict):
        return input
    elif torch.is_tensor(input) and len(input.size()) == 4:
        input_dic = {"images": input.unsqueeze(0)}
        return input_dic
    elif torch.is_tensor(input) and len(input.size()) == 5:
        input_dic = {"images": input}
        return input_dic
    
# From FlowUnderAttack
@staticmethod
def epe(flow1, flow2):
    """"
    Compute the  endpoint errors (EPEs) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error
    """
    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=0).sqrt()
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=1).sqrt()
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(flow1.size()) + " and " + str(flow1.size()))
    return epe
