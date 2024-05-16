import torch


def broadcast_stack(a: torch.Tensor, b: torch.Tensor):
    '''
    a: M_0×……×M_m
    b: N_0×……×N_n

    broadcast to M_0×……×M_m × N_0×……×N_n

    return (a,b)
    '''
    a = a.unsqueeze(0)  # 1×M_0×……×M_m
    b = b.unsqueeze(0)  # 1×N_0×……×N_n

    a_shape = list(a.size())  # [1,M_0,...,M_m]
    b_shape = list(b.size())  # [1,N_0,...,N_n]

    a_new_shape = a_shape+(len(b_shape)-1)*[1]  # [1,M_0,...,M_m, 1,...,1]
    b_new_shape = len(a_shape)*[1]+b_shape[1:]  # [1,1,...,1, N_0,...,N_n]
    a = a.view(a_new_shape)  # 1 × M_0×……×M_m × 1×……×1
    b = b.view(b_new_shape)  # 1 × 1×……×1     × N_0×……×N_n

    a, b = torch.broadcast_tensors(a, b)  # 1 × M_0×……×M_m × N_0×……×N_n

    # M_0×……×M_m × N_0×……×N_n
    a = a.squeeze(0)
    b = b.squeeze(0)

    return (a, b)
# def broadcast


def stepfunc(input: torch.Tensor):
    '''
    output = 1, where input>=0
             0, where input<0

    input: no grad.

    return int Tenser, with same shape as input.
    '''
    return (input >= 0).int()


def grad1(seq: torch.Tensor):
    '''
    `seq`: 1-d tensor with at least 3 elements
    '''
    return seq[2:]-seq[:-2]


def grad2(seq: torch.Tensor):
    '''
    `seq`: 1-d tensor with at least 3 elements
    '''
    return seq[:-2]+seq[2:]-2*seq[1:-1]


def compute_loss_forward(
    control_points_x: torch.Tensor, control_points_y: torch.Tensor, t_seq: torch.Tensor,
    pixels_x: torch.Tensor, pixels_y: torch.Tensor, pixels_foreground_values: torch.Tensor,
    f_star: float,
    untrailed_unbinned_psf,
    regularization_weight_norm: float, regularization_weight_tang: float,
    numerical_integration_steps_number: int,
    binning_grid_x: torch.Tensor, binning_grid_y: torch.Tensor
):
    '''
    `control_points_x`,`control_points_y`:
        size=(Q+1,); grad required; to be optimized.
    `t_seq`:
        size=(Q+1,); no grad; an uniformly ascending float-type sequence, the time sequence of the control points.
    `pixels_x`,`pixels_y`,pixels_foreground_values:
        size=(pixels_number,); no grad.
    `binning_grid_x`,`binning_grid_y`:
        size=(β_x,β_y); no grad.
    `untrailed_unbinned_psf`:
        auto-grad functor (Δx,Δy) -> psf_values; `Δx`,`Δy`,`psf_values` share the same shape.
        Do not wrap numpy.ndarray but torch.Tensor.
    `numerical_integration_steps_number`:
        `2N` in our paper, an even-positive integer; not giving an even-positive integer is an undefined behavior.

    return (intensity_square_residual_sum,regularization_term)

    NOTE: for convenience, here we refer to `x`, `y` No.0 axis and No.1 axis of array (i.e. Tensor) respectively, which contrasts the denotion in our paper.
    '''
    T = (t_seq[-1]-t_seq[0])/2
    N = numerical_integration_steps_number//2
    Q = t_seq.size()-1
    pixels_number = pixels_foreground_values.size()

    n = torch.arange(-N, N, 1)
    t = (n+0.5)*T/N
    broadcast_shape = (2*N, Q)
    t = torch.broadcast_to(torch.unsqueeze(t, dim=1),
                           broadcast_shape)

    t_seq_curr = t_seq[:-1]
    t_seq_next = t_seq[1:]
    t_seq_curr = torch.broadcast_to(torch.unsqueeze(
        t_seq_curr, dim=0), broadcast_shape)
    t_seq_next = torch.broadcast_to(torch.unsqueeze(
        t_seq_next, dim=0), broadcast_shape)
    delta_t = t-t_seq_curr
    stepfunc_input_curr = delta_t
    stepfunc_input_next = t-t_seq_next
    gate_values = stepfunc(stepfunc_input_curr) - \
        stepfunc(stepfunc_input_next)

    x_curr = control_points_x[:-1]
    y_curr = control_points_y[:-1]
    x_next = control_points_x[1:]
    y_next = control_points_x[1:]
    x_curr = torch.broadcast_to(torch.unsqueeze(
        x_curr, dim=0), broadcast_shape)
    y_curr = torch.broadcast_to(torch.unsqueeze(
        y_curr, dim=0), broadcast_shape)
    x_next = torch.broadcast_to(torch.unsqueeze(
        x_next, dim=0), broadcast_shape)
    y_next = torch.broadcast_to(torch.unsqueeze(
        y_next, dim=0), broadcast_shape)
    piecewise_x = x_curr+delta_t*(x_next-x_curr)*Q/(2*T)
    piecewise_y = y_curr+delta_t*(y_next-y_curr)*Q/(2*T)
    gated_piecewise_x = gate_values*piecewise_x
    gated_piecewise_y = gate_values*piecewise_y
    traj_x = torch.sum(gated_piecewise_x, dim=1)
    traj_y = torch.sum(gated_piecewise_y, dim=1)

    pixels_x, traj_x = broadcast_stack(
        pixels_x, traj_x)
    pixels_y, traj_y = broadcast_stack(
        pixels_y, traj_y)
    delta_x_for_untrailed_psf = pixels_x-traj_x
    delta_y_for_untrailed_psf = pixels_y-traj_y

    delta_x_for_untrailed_psf, binning_grid_x = broadcast_stack(
        delta_x_for_untrailed_psf, binning_grid_x)
    delta_y_for_untrailed_psf, binning_grid_y = broadcast_stack(
        delta_y_for_untrailed_psf, binning_grid_y)
    unbinned_delta_x = delta_x_for_untrailed_psf+binning_grid_x
    unbinned_delta_y = delta_y_for_untrailed_psf+binning_grid_y
    psf_values = untrailed_unbinned_psf(unbinned_delta_x, unbinned_delta_y)

    binned_psf_values = torch.sum(
        psf_values, dim=(2, 3))
    trailed_psf_values = torch.sum(
        binned_psf_values, dim=1)*T/N
    residual = pixels_foreground_values-f_star*trailed_psf_values
    square_residual = torch.pow(residual, 2)
    intensity_square_residual_sum = torch.sum(square_residual, dim=0)

    amplified_points_x = control_points_x*f_star
    amplified_points_y = control_points_y*f_star

    grad1_x = grad1(amplified_points_x)
    grad1_y = grad1(amplified_points_y)
    grad1_mag = torch.sqrt(torch.pow(grad1_x, 2)+torch.pow(grad1_y, 2))
    tang_x = grad1_x/grad1_mag
    tang_y = grad1_y/grad1_mag
    norm_x = (-1.0)*tang_y
    norm_y = tang_x

    grad2_x = grad2(amplified_points_x)
    grad2_y = grad2(amplified_points_y)
    temp = (2*T/Q)**2
    deri2_x = grad2_x/temp
    deri2_y = grad2_y/temp

    tang_penalty_x = torch.mul(deri2_x, tang_x)
    tang_penalty_y = torch.mul(deri2_y, tang_y)
    tang_square_penalty = torch.pow(
        tang_penalty_x, 2)+torch.pow(tang_penalty_y, 2)
    mean_tang_square_penalty = torch.mean(tang_square_penalty)

    norm_penalty_x = torch.mul(deri2_x, norm_x)
    norm_penalty_y = torch.mul(deri2_y, norm_y)
    norm_square_penalty = torch.pow(
        norm_penalty_x, 2)+torch.pow(norm_penalty_y, 2)
    mean_norm_square_penalty = torch.mean(norm_square_penalty)

    regularization_term = pixels_number * \
        (regularization_weight_norm*mean_norm_square_penalty +
         regularization_weight_tang*mean_tang_square_penalty)

    return (intensity_square_residual_sum, regularization_term)
