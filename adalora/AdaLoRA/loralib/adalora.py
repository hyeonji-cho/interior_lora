#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List 

class RBFNonlinearity(nn.Module):
    def __init__(self, num_basis: int, reg_lambda: float = 1e-3, init_range: float = 2.0, device=None):
        super().__init__()
        self.num_basis = num_basis
        self.reg_lambda = reg_lambda

        # B-spline 근사: RBF 중심 및 너비
        centers = torch.linspace(-init_range, init_range, num_basis)
        self.centers = nn.Parameter(centers.to(device))  # (num_basis,)
        self.widths_raw = nn.Parameter(torch.ones(num_basis).to(device))  # (num_basis,)
        self.coeffs = nn.Parameter(torch.zeros(num_basis).to(device))  # 학습 가능한 계수

        self.regularization_loss = torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor):  # x: (r,) or (r, 1)
        x = x.view(-1)  # (r,)
        widths = F.softplus(self.widths_raw) + 1e-6  # ensure positivity

        # RBF 계산: exp(-((x - c)/w)^2)
        diff = (x.unsqueeze(1) - self.centers) / widths  # (r, num_basis)
        rbf = torch.exp(-diff.pow(2))  # (r, num_basis)

        # B-spline 근사 결과
        L = rbf @ self.coeffs  # (r,)

        # 정규화 손실: L2 norm of coeffs
        if self.training:
            self.regularization_loss = self.reg_lambda * (self.coeffs ** 2).mean()

        return L

class ChebyshevNonlinearity(nn.Module):
    def __init__(self, num_basis: int, reg_lambda: float = 1e-3, domain: tuple = (-1.0, 1.0), device=None):
        super().__init__()
        self.num_basis = num_basis
        self.reg_lambda = reg_lambda
        self.domain = domain
        self.coeffs = nn.Parameter(torch.zeros(num_basis).to(device))  # 학습할 계수
        self.regularization_loss = torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor):
        # 입력 정규화: domain 범위로 스케일링
        a, b = self.domain
        x = 2 * (x - a) / (b - a) - 1  # x in [-1, 1]
        x = x.view(-1)  # (r,)

        # Chebyshev 다항식 계산 (재귀)
        T = [torch.ones_like(x), x]  # T_0(x), T_1(x)
        for k in range(2, self.num_basis):
            T_k = 2 * x * T[-1] - T[-2]  # T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            T.append(T_k)

        # 각 basis에 대한 선형 조합
        T_stack = torch.stack(T[:self.num_basis], dim=1)  # (r, num_basis)
        L = T_stack @ self.coeffs  # (r,)

        if self.training:
            self.regularization_loss = self.reg_lambda * (self.coeffs ** 2).mean()

        return L

class LegendreNonlinearity(nn.Module):
    def __init__(self, num_basis: int, reg_lambda: float = 1e-3, domain: tuple = (-1.0, 1.0), device=None):
        super().__init__()
        self.num_basis = num_basis
        self.reg_lambda = reg_lambda
        self.domain = domain
        self.coeffs = nn.Parameter(torch.zeros(num_basis).to(device))  # 학습할 계수
        self.regularization_loss = torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor):
        # 입력 정규화: domain을 [-1, 1]로 매핑
        a, b = self.domain
        x = 2 * (x - a) / (b - a) - 1
        x = x.view(-1)  # (r,)

        # Legendre 다항식 계산 (재귀 정의)
        P = [torch.ones_like(x)]  # P0(x) = 1
        if self.num_basis > 1:
            P.append(x)  # P1(x) = x
        for k in range(2, self.num_basis):
            P_k = ((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k
            P.append(P_k)

        P_stack = torch.stack(P, dim=1)  # (r, num_basis)
        L = P_stack @ self.coeffs  # (r,)

        if self.training:
            self.regularization_loss = self.reg_lambda * (self.coeffs ** 2).mean()

        return L

class MLPNonlinearity(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, reg_lambda: float = 1e-3, device=None):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  # scalar output for each input
        ).to(device)
        self.regularization_loss = torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1)  # (r, 1)
        L = self.model(x).view(-1)  # (r,)

        if self.training:
            # L2 정규화 항 추가
            reg = sum((param ** 2).mean() for param in self.model.parameters())
            self.regularization_loss = self.reg_lambda * reg

        return L

class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        nonlinearity: Optional[nn.Module] = None,
        num_basis=8,
        r=0,  # Add r as a default argument
        **kwargs
    ):
        # Initialize nn.Linear with only supported arguments
        super(SVDLinear, self).__init__(in_features, out_features, bias=kwargs.get('bias', True))
        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.fixed_nonlinearity = nn.GELU()
        self.learnable_nonlinearity = MLPNonlinearity(input_dim=1, device=self.weight.device)
        # alpha, beta를 학습 가능한 파라미터로 선언
        self.alpha = nn.Parameter(torch.tensor(1.0, device=self.weight.device))
        self.beta = nn.Parameter(torch.tensor(1.0, device=self.weight.device))

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r,)
            ) 
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                torch.tensor(float(r), device=self.weight.device), requires_grad=False
            )
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        # print("SVDLinear input shape:", x.shape)
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:          
                # lora_part = self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                

                # f(PΛQ) 구조
                # lora_part = self.nonlinearity(lora_part)
                # result += lora_part * self.scaling / (self.ranknum+1e-5)

                # # P·f(Λ)·Q 구조: lora_B @ f(lora_E) @ lora_A
                # # x @ Q^T = x @ lora_A.T
                # q_proj = self.lora_dropout(x) @ self.lora_A.T  # (batch, r)
                # # f(Λ): (r, 1) -> (r,) for broadcasting
                # f_lambda = self.nonlinearity(self.lora_E).view(-1)  # (r,)
                # # f(Λ) @ (x @ Q^T): (r,) * (batch, r) -> (batch, r)
                # q_proj = q_proj * f_lambda.unsqueeze(0)  # broadcasting
                # # P @ ...: (out_features, r) @ (batch, r).T -> (out_features, batch)
                # lora_part = (self.lora_B @ q_proj.T).T  # (batch, out_features)
                # result += lora_part * self.scaling / (self.ranknum+1e-5)

                # f = F + L
                q_proj = self.lora_dropout(x) @ self.lora_A.T  # (batch, r)
                # F: 고정 비선형성
                F_lambda = self.fixed_nonlinearity(self.lora_E).view(-1)  # (r,)
                # L: 학습 가능한 비선형성 (B-spline basis, RBF 근사)
                L_lambda = self.learnable_nonlinearity(self.lora_E)  # (r,)
                # f(λ) = α·F(λ) + β·L(λ)
                f_lambda = self.alpha * F_lambda + self.beta * L_lambda  # (r,)
                q_proj = q_proj * f_lambda.unsqueeze(0)  # broadcasting
                # lora_part = (self.lora_B @ q_proj.T).T  # (batch, out_features)
                lora_part = q_proj @ self.lora_B.T
                result += lora_part * self.scaling / (self.ranknum+1e-5)

                # # gated blending
                # q_proj = self.lora_dropout(x) @ self.lora_A.T  # (batch, r)
                # # F: 고정 비선형성
                # F_lambda = self.fixed_nonlinearity(self.lora_E).view(-1)  # (r,)
                # # L: 학습 가능한 비선형성
                # L_lambda = self.learnable_nonlinearity(self.lora_E)  # (r,)
                # # g(λ): λ-dependent gate (예시: sigmoid)
                # # g_lambda = torch.sigmoid(self.lora_E.view(-1))  # (r,)
                # g_lambda = sigmoid(MLP(self.lora_E))
                # # f(λ) = g(λ)·F(λ) + (1−g(λ))·L(λ)
                # f_lambda = g_lambda * F_lambda + (1 - g_lambda) * L_lambda  # (r,)
                # q_proj = q_proj * f_lambda.unsqueeze(0)  # broadcasting
                # lora_part = (self.lora_B @ q_proj.T).T  # (batch, out_features)
                # result += lora_part * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 

    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            # Initial warmup 
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # Final fine-tuning 
            curr_rank = self.target_rank 
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
        else: 
            # Budget decreasing 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Check if gradient exists
                    if p.grad is not None:
                        # Calculate sensitivity 
                        self.ipt[n] = (p * p.grad).abs().detach()
                        # Update sensitivity 
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                            (1-self.beta1)*self.ipt[n]
                        # Update uncertainty 
                        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                            (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                    else:
                        # If gradient is None, keep the previous values or use zeros
                        # This can happen in the first few steps or if the parameter doesn't contribute to loss
                        pass

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank): 
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores 
        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"lora_E"
            is_dict[name_E] = sum_ipt.view(-1)  # Store as 1D tensor to match lora_E parameter shape
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold 
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()

        # Mask out unimportant singular values 
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    # Now is_dict[n] should have the same shape as p.data (both are 1D)
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item() 

                    if self.tb_writter is not None and self.global_step%self.log_interval==0:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = ranknum 
                        curr_sum_rank += ranknum 
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]  

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            # do not update ipt during final fine-tuning 
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget 
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param

def replace_with_svdlinear(module, r=8, lora_alpha=32, lora_dropout=0.1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 안정 조건: LoRA는 일반적으로 in = out 인 구조에서만 안전
            if child.in_features == child.out_features and child.in_features >= r:
                setattr(module, name, SVDLinear(
                    child.in_features, child.out_features,
                    r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                ))
            else:
                print(f"⚠️ Skip {name} (in={child.in_features}, out={child.out_features})")
        else:
            replace_with_svdlinear(child, r, lora_alpha, lora_dropout)
