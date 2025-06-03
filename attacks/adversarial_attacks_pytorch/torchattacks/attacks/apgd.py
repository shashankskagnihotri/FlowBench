import time

import numpy as np

import torch
import torch.nn as nn

from ..attack import Attack
from ....attack_utils.loss_criterion import LossCriterion


class APGD(Attack):
    r"""
    APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
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
        loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
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
        loss="epe",
        eot_iter=1,
        rho=0.75,
        verbose=False,
    ):
        super().__init__("APGD", model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.squeeze(0)
        labels = labels.squeeze(0)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, images_orig, lables_orig):
        images = (
            images_orig.clone()
            if len(images_orig.shape) == 4
            else images_orig.clone().unsqueeze(0)
        )
        lables = (
            lables_orig.clone()
            if len(lables_orig.shape) == 4
            else lables_orig.clone().unsqueeze(0)
        )

        image_1 = images[0].unsqueeze(0)
        image_2 = images[1].unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )

        if self.norm == "Linf":
            # image 1 Linf-norm
            t = 2 * torch.rand(image_1.shape).to(self.device).detach() - 1
            image_1_adv = image_1.detach() + self.eps * torch.ones(
                [image_1.shape[0], 1, 1, 1]
            ).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
            # image 2 Linf-norm
            t = 2 * torch.rand(image_2.shape).to(self.device).detach() - 1
            image_2_adv = image_2.detach() + self.eps * torch.ones(
                [image_2.shape[0], 1, 1, 1]
            ).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
            # image 1 L2-norm
        elif self.norm == "L2":
            t = torch.randn(image_1.shape).to(self.device).detach()
            image_1_adv = image_1.detach() + self.eps * torch.ones(
                [image_1.shape[0], 1, 1, 1]
            ).to(self.device).detach() * t / (
                (t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
            t = torch.randn(image_2.shape).to(self.device).detach()
            image_2_adv = image_2.detach() + self.eps * torch.ones(
                [image_2.shape[0], 1, 1, 1]
            ).to(self.device).detach() * t / (
                (t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
        image_1_adv = image_1_adv.clamp(0.0, 1.0)
        image_2_adv = image_2_adv.clamp(0.0, 1.0)

        images_best = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)
        images_best_adv = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)

        # loss_steps = torch.zeros([self.steps, x.shape[0]])
        # loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        # acc_steps = torch.zeros_like(loss_best_steps)
        loss_steps = torch.zeros([self.steps, 1])
        loss_best_steps = torch.zeros([self.steps + 1, 1])
        # epe_steps = torch.zeros_like(loss_best_steps)

        # TODO: implement losses correctly
        if self.loss == "ce":
            # criterion_indiv = nn.CrossEntropyLoss(reduction="none")
            raise ValueError("only epe or mse")
        elif self.loss == "dlr":
            # TODO: is dlr applicable for Optical Flow?
            # criterion_indiv = self.dlr_loss
            raise ValueError("dlr not implemented")
        criterion_indiv = LossCriterion(self.loss)

        images_adv = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)
        images_adv.requires_grad_(True)
        grad = torch.zeros_like(images.unsqueeze(0))
        images_adv_dic = {"images": images_adv}
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                preds_dic = self.get_logits(images_adv_dic)
                preds = preds_dic["flows"].squeeze(0)
                loss_indiv = criterion_indiv.loss(preds, lables)
                # TODO:SUM correct? Not Mean?
                loss = loss_indiv.mean()

            # 1 backward pass (eot_iter = 1)

            grad += torch.autograd.grad(loss, images_adv)[0].detach()
        grad /= float(self.eot_iter)
        if self.targeted:
            grad = grad * -1
        grad_best = grad.clone()

        # if self.loss == "epe":
        #     epe_score = loss
        # else:
        #     epe_score = epe(preds, lables)
        # TODO: what is this used for? Was acc steps
        # epe_steps[0] = epe_score + 0
        loss_best = loss_indiv.detach().view(len(loss_indiv), -1).mean(dim=1)
        step_size = (
            self.eps
            * torch.ones([1, 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        images_adv_old = images_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(1)
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                images_adv = images_adv.detach()
                grad2 = images_adv - images_adv_old
                images_adv_old = images_adv.clone()

                a = 0.75 if i > 0 else 1.0

                image_1_adv = images_adv.squeeze(0)[0].unsqueeze(0)
                image_2_adv = images_adv.squeeze(0)[1].unsqueeze(0)

                grad2_i1 = grad2.squeeze(0)[0].unsqueeze(0)
                grad2_i2 = grad2.squeeze(0)[1].unsqueeze(0)

                grad_i1 = grad.squeeze(0)[0].unsqueeze(0)
                grad_i2 = grad.squeeze(0)[1].unsqueeze(0)

                if self.norm == "Linf":
                    # image_1_adv Linf-norm
                    image_1_adv_v1 = image_1_adv + step_size * torch.sign(grad_i1)
                    image_1_adv_v1 = torch.clamp(
                        torch.min(
                            torch.max(image_1_adv_v1, image_1 - self.eps),
                            image_1 + self.eps,
                        ),
                        0.0,
                        1.0,
                    )
                    image_1_adv_v1 = torch.clamp(
                        torch.min(
                            torch.max(
                                image_1_adv
                                + (image_1_adv_v1 - image_1_adv) * a
                                + grad2_i1 * (1 - a),
                                image_1 - self.eps,
                            ),
                            image_1 + self.eps,
                        ),
                        0.0,
                        1.0,
                    )
                    # image_2_adv Linf-norm
                    image_2_adv_v1 = image_2_adv + step_size * torch.sign(grad_i2)
                    image_2_adv_v1 = torch.clamp(
                        torch.min(
                            torch.max(image_2_adv_v1, image_2 - self.eps),
                            image_2 + self.eps,
                        ),
                        0.0,
                        1.0,
                    )
                    image_2_adv_v1 = torch.clamp(
                        torch.min(
                            torch.max(
                                image_2_adv
                                + (image_2_adv_v1 - image_2_adv) * a
                                + grad2_i2 * (1 - a),
                                image_2 - self.eps,
                            ),
                            image_2 + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    # image_1_adv L2-norm
                    # TODO: are the dimensions here correct? What is exaclty happening?
                    image_1_adv_v1 = image_1_adv + step_size * grad_i1 / (
                        (grad_i1**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    image_1_adv_v1 = torch.clamp(
                        image_1
                        + (image_1_adv_v1 - image_1)
                        / (
                            ((image_1_adv_v1 - image_1) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(image_1.shape).to(self.device).detach(),
                            ((image_1_adv_v1 - image_1) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    image_1_adv_v1 = (
                        image_1_adv
                        + (image_1_adv_v1 - image_1_adv) * a
                        + grad2_i1 * (1 - a)
                    )
                    image_1_adv_v1 = torch.clamp(
                        image_1
                        + (image_1_adv_v1 - image_1)
                        / (
                            ((image_1_adv_v1 - image_1) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(image_1.shape).to(self.device).detach(),
                            ((image_1_adv_v1 - image_1) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                    # image_2_adv L2-norm
                    image_2_adv_v1 = image_2_adv + step_size * grad_i2 / (
                        (grad_i2**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    image_2_adv_v1 = torch.clamp(
                        image_2
                        + (image_2_adv_v1 - image_2)
                        / (
                            ((image_2_adv_v1 - image_2) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(image_2.shape).to(self.device).detach(),
                            ((image_2_adv_v1 - image_2) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    image_2_adv_v1 = (
                        image_2_adv
                        + (image_2_adv_v1 - image_2_adv) * a
                        + grad2_i2 * (1 - a)
                    )
                    image_2_adv_v1 = torch.clamp(
                        image_2
                        + (image_2_adv_v1 - image_2)
                        / (
                            ((image_2_adv_v1 - image_2) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(image_2.shape).to(self.device).detach(),
                            ((image_2_adv_v1 - image_2) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                # x_adv = x_adv_1 + 0.0
                image_1_adv = image_1_adv_v1 + 0.0
                image_2_adv = image_2_adv_v1 + 0.0
                images_adv = torch.cat((image_1_adv, image_2_adv)).unsqueeze(0)

            # get gradient
            images_adv.requires_grad_(True)
            grad = torch.zeros_like(images.unsqueeze(0))
            images_adv_dic = {"images": images_adv}
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    preds_dic = self.get_logits(images_adv_dic)
                    preds = preds_dic["flows"].squeeze(0)
                    loss_indiv = criterion_indiv.loss(preds, lables)
                    loss = loss_indiv.mean()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, images_adv)[0].detach()

            grad /= float(self.eot_iter)
            if self.targeted:
                grad = grad * -1

            # if self.loss == "epe":
            #     epe_score = loss
            # else:
            #     epe_score = epe(preds, lables)
            # TODO: what is this used for? Was acc steps
            # epe_steps[0] = epe_score + 0

            # x_best_adv[(pred == 0).nonzero().squeeze()] = (
            #    x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            # )  # nopep8
            images_best_adv = images_adv
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone().view(len(loss_indiv), -1).mean(dim=1)

                loss_steps[i] = y1.cpu() + 0
                if self.targeted:
                    ind = (y1 < loss_best).nonzero().squeeze()
                else:
                    ind = (y1 > loss_best).nonzero().squeeze()
                images_best[ind] = images_adv[ind].clone()
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
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        images_adv[fl_oscillation] = images_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return images_best, None, loss_best, images_best_adv

    def perturb(self, images_orig, labels_orig, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        images = (
            images_orig.clone()
            if len(images_orig.shape) == 4
            else images_orig.clone().unsqueeze(0)
        )
        lables = (
            labels_orig.clone()
            if len(labels_orig.shape) == 4
            else labels_orig.clone().unsqueeze(0)
        )

        images_adv = images.clone()
        # pred_dic = self.get_logits(get_input_format(images))
        epe_score = None
        # epe_score = epe(lables, pred_dic["flows"].squeeze(0))
        # epe_score = epe_score.mean()
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

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError("not implemented yet")

            else:
                for counter in range(self.n_restarts):
                    # ind_to_fool = acc.nonzero().squeeze()
                    # if len(ind_to_fool.shape) == 0:
                    # ind_to_fool = ind_to_fool.unsqueeze(0)

                    images_to_fool, lables_to_fool = (
                        images.clone(),
                        lables.clone(),
                    )  # nopep8
                    (
                        images_best_curr,
                        epe_curr,
                        loss_curr,
                        images_adv_curr,
                    ) = self.attack_single_run(
                        images_to_fool, lables_to_fool
                    )  # nopep8
                    images_adv = images_adv_curr.clone()
                    # ind_curr = (acc_curr == 0).nonzero().squeeze()
                    # acc[ind_to_fool[ind_curr]] = 0
                    # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print(
                            "restart {}  - cum. time: {:.1f} s".format(
                                counter, time.time() - startt
                            )
                        )
                    epe_score = None
            return epe_score, images_adv

        else:
            images_adv_best = images.detach().clone()
            loss_best = torch.ones([1]).to(self.device) * (-float("inf"))  # nopep8
            import pdb

            pdb.set_trace()
            for counter in range(self.n_restarts):
                images_best_curr, _, loss_curr, _ = self.attack_single_run(
                    images, lables
                )
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                images_adv_best[ind_curr] = images_best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print("restart {} - loss: {:.5f}".format(counter, loss_best.sum()))

            return loss_best, images_adv_best


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
