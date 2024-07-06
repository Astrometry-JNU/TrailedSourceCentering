import torch


PRIME_NUM_LIST = torch.tensor(
    [3, 5, 7, 11, 19, 31, 59, 113, 223, 443, 883, 1765, 3527],
    dtype=torch.int32
)


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


def locate(seq: torch.Tensor, data: torch.Tensor, inverse=False):
    '''
    `seq`:
        non-decreasing 1-D array.

    return:
        inverse=False ---- size alike to `data`; how many element in the front of `seq` are smaller than `data`.
        inverse=True ---- size alike to `data`; how many element in the rear of `seq` are greater than `data`.
    '''
    data = torch.unsqueeze(data, dim=0)

    data, seq = broadcast_stack(data, seq)

    if inverse:
        where_data_lt_seq = data.lt(seq).int()
        each_data_lt_seq_count = torch.sum(where_data_lt_seq, dim=-1)
        result = each_data_lt_seq_count.squeeze(dim=0)
    else:
        where_data_gt_seq = data.gt(seq).int()
        each_data_gt_seq_count = torch.sum(where_data_gt_seq, dim=-1)
        result = each_data_gt_seq_count.squeeze(dim=0)
    return result


def piecewise_linear(control_points_x_or_y: torch.Tensor, t_seq: torch.Tensor, t: torch.Tensor):
    '''
    `control_points_x_or_y`,`t_seq`:
        size=(Q+1,).
    `t`:
        any size.

    return: `s(t)`; with the same shape as `t`.
    '''
    _2T = (t_seq[-1]-t_seq[0])
    Q = t_seq.size(dim=0)-1

    endpointmask = torch.eq(t, t_seq[-1])

    x_curr = control_points_x_or_y[:-1]
    x_next = control_points_x_or_y[1:]
    _, x_curr = broadcast_stack(t, x_curr)
    _, x_next = broadcast_stack(t, x_next)

    t_curr = t_seq[:-1]
    t_next = t_seq[1:]
    _, t_curr = broadcast_stack(t, t_curr)
    t, t_next = broadcast_stack(t, t_next)

    delta_t_curr = t-t_curr
    delta_t_next = t-t_next
    gate = stepfunc(delta_t_curr)-stepfunc(delta_t_next)

    gated_pieces = gate*(x_curr+delta_t_curr*(x_next-x_curr)*Q/_2T)
    trajectory_x_or_y = torch.sum(gated_pieces, dim=-1)

    # fill the endpoint manually
    trajectory_x_or_y[endpointmask] = control_points_x_or_y[-1]

    return trajectory_x_or_y


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
    `pixels_x`,`pixels_y`,`pixels_foreground_values`:
        size=(pixels_number,); no grad; ROI pixels' coordinates and foreground values.
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
    Q = t_seq.size(dim=0)-1
    pixels_number = pixels_foreground_values.size(dim=0)

    n = torch.arange(-N, N, 1)
    t = (n+0.5)*T/N

    traj_x = piecewise_linear(control_points_x, t_seq, t)
    traj_y = piecewise_linear(control_points_y, t_seq, t)

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


def rmsprop_optimize(
    control_points_x: torch.Tensor, control_points_y: torch.Tensor, t_seq: torch.Tensor,
    pixels_x: torch.Tensor, pixels_y: torch.Tensor, pixels_foreground_values: torch.Tensor,
    f_star: float,
    untrailed_unbinned_psf,
    regularization_weight_norm: float, regularization_weight_tang: float,
    numerical_integration_steps_number: int,
    binning_grid_x: torch.Tensor, binning_grid_y: torch.Tensor,
    max_epochs_number=1024, learning_rate=0.001, control_point_tolerance=0.001,
    each_epoch_callback=lambda epoch, control_points_x, control_points_y, intensity_square_residual_sum, regularization_term: None
):
    '''
    `control_points_x`,`control_points_y`:
        size=(Q+1,); NO GRAD; to be optimized.
    `t_seq`:
        size=(Q+1,); no grad; an uniformly ascending float-type sequence, the time sequence of the control points.
    `pixels_x`,`pixels_y`,`pixels_foreground_values`:
        size=(pixels_number,); no grad; ROI pixels' coordinates and foreground values.
    `binning_grid_x`,`binning_grid_y`:
        size=(β_x,β_y); no grad.
    `untrailed_unbinned_psf`:
        auto-grad functor (Δx,Δy) -> psf_values; `Δx`,`Δy`,`psf_values` share the same shape.
        Do not wrap numpy.ndarray but torch.Tensor.
    `numerical_integration_steps_number`:
        `2N` in our paper, an even-positive integer; not giving an even-positive integer is an undefined behavior.

    `max_epochs_number`:
        max steps number of the optimizer; ≤0 ---- unlimited.
    `control_point_tolerance`:
        stop optimizing if none of the control points is moved by over `control_point_tolerance` in optimizer.step(); ≤0 ---- loop till `max_epochs_number`.

    return (optimized_control_points_x,optimized_control_points_y); no grad
    '''
    control_points_x.requires_grad_(True)
    control_points_y.requires_grad_(True)

    tol2 = control_point_tolerance**2

    optimizer = torch.optim.RMSprop(
        [control_points_x, control_points_y],
        lr=learning_rate
    )

    epoch = 0
    while True:
        if 0 < max_epochs_number <= epoch:
            break

        old_control_points_x = torch.detach(torch.clone(control_points_x))
        old_control_points_y = torch.detach(torch.clone(control_points_y))

        optimizer.zero_grad()
        intensity_square_residual_sum, regularization_term = compute_loss_forward(
            control_points_x, control_points_y, t_seq,
            pixels_x, pixels_y, pixels_foreground_values,
            f_star,
            untrailed_unbinned_psf,
            regularization_weight_norm, regularization_weight_tang,
            numerical_integration_steps_number,
            binning_grid_x, binning_grid_y)
        each_epoch_callback(epoch, control_points_x, control_points_y,
                            intensity_square_residual_sum, regularization_term)

        loss = intensity_square_residual_sum+regularization_term
        loss.backward()
        optimizer.step()

        if control_point_tolerance <= 0:
            continue
        delta_control_points_x = control_points_x-old_control_points_x
        delta_control_points_y = control_points_y-old_control_points_y
        delta_control_points_mag2 = delta_control_points_x**2+delta_control_points_y**2
        if torch.max(delta_control_points_mag2) <= tol2:
            break

        epoch += 1

    control_points_x.requires_grad_(False)
    control_points_y.requires_grad_(False)
    return (control_points_x, control_points_y)


def accumulate_along_trajectory(
    control_points_x: torch.Tensor, control_points_y: torch.Tensor, t_seq: torch.Tensor,
    pixels_x: torch.Tensor, pixels_y: torch.Tensor, pixels_foreground_values: torch.Tensor,
    accumulative_steps_number: int
):
    '''
    `control_points_x`,`control_points_y`:
        size=(Q+1,).
    `t_seq`:
        size=(Q+1,); an uniformly ascending float-type sequence, the time sequence of the control points.
    `pixels_x`,`pixels_y`,`pixels_foreground_values`:
        size=(pixels_number,); no grad; ROI pixels' coordinates and foreground values.
    `accumulative_steps_number`:
        similar to `numerical_integration_steps_number`.

    return (accu,accu_t_seq)
        both size=(accumulative_steps_number+1,); the accumulations and corresponding time points [after 0 steps, after 1 steps, …, after `accumulative_steps_number` steps]
    '''
    accu_t_seq = torch.linspace(
        t_seq[0], t_seq[-1], accumulative_steps_number+1)

    trajectory_x = piecewise_linear(control_points_x, t_seq, accu_t_seq)
    trajectory_y = piecewise_linear(control_points_y, t_seq, accu_t_seq)
    pixels_x, trajectory_x = broadcast_stack(pixels_x, trajectory_x)
    pixels_y, trajectory_y = broadcast_stack(pixels_y, trajectory_y)
    distance_x = pixels_x-trajectory_x
    distance_y = pixels_y-trajectory_y
    distance2 = distance_x**2+distance_y**2
    which_closest = torch.argmin(distance2, dim=1)

    accu = torch.zeros((accumulative_steps_number+1,), dtype=torch.float64)
    accu[0] = 0.0
    for step in range(accumulative_steps_number):
        mask = which_closest.eq(step)
        selected_pixels_foreground_values = torch.masked_select(
            pixels_foreground_values, mask)
        accu[step+1] = accu[step]+torch.sum(selected_pixels_foreground_values)
    # for step in range(accumulative_steps_number)

    # 强制矫正浮点计算导致的误差
    F = torch.sum(pixels_foreground_values)
    accu = torch.clip(accu, None, F)
    accu[-1] = F

    return (accu, accu_t_seq)


def refine_control_points(
    control_points_x: torch.Tensor, control_points_y: torch.Tensor, t_seq: torch.Tensor,
    pixels_x: torch.Tensor, pixels_y: torch.Tensor, pixels_foreground_values: torch.Tensor,
    accumulative_steps_number: int,
    control_points_number_seq=PRIME_NUM_LIST
):
    '''
    `control_points_x`,`control_points_y`:
        size=(Q+1,).
    `t_seq`:
        size=(Q+1,); an uniformly ascending float-type sequence, the time sequence of the control points.
    `pixels_x`,`pixels_y`,`pixels_foreground_values`:
        size=(pixels_number,); ROI pixels' coordinates and foreground values.

    return (new_control_points_x, new_control_points_y, new_t_seq)
    '''
    accu, accu_t_seq = accumulate_along_trajectory(
        control_points_x, control_points_y, t_seq,
        pixels_x, pixels_y, pixels_foreground_values,
        accumulative_steps_number
    )

    old_control_points_number = t_seq.size(dim=0)
    old_control_points_number_gt_prime_count = locate(
        control_points_number_seq, torch.tensor([old_control_points_number], dtype=torch.int32))[0]
    new_control_points_number = control_points_number_seq[old_control_points_number_gt_prime_count+1]

    flux_inc_seq = torch.linspace(0.0, accu[-1], new_control_points_number)
    flux_inc_gt_count = locate(accu, flux_inc_seq, False)
    flux_inc_lt_count = locate(accu, flux_inc_seq, True)
    which_coincide = torch.add(flux_inc_gt_count, flux_inc_lt_count).lt(
        accumulative_steps_number+1)
    which_coincide = which_coincide.int()
    flux_inc_ndx_in_accu_0 = flux_inc_gt_count-1+which_coincide
    flux_inc_ndx_in_accu_1 = (accumulative_steps_number+1) - \
        flux_inc_lt_count-which_coincide
    new_control_points_at_which_time_points_in_old_trajectory_0 = accu_t_seq[
        flux_inc_ndx_in_accu_0]
    new_control_points_at_which_time_points_in_old_trajectory_1 = accu_t_seq[
        flux_inc_ndx_in_accu_1]
    new_control_points_x_0 = piecewise_linear(
        control_points_x, t_seq, new_control_points_at_which_time_points_in_old_trajectory_0)
    new_control_points_x_1 = piecewise_linear(
        control_points_x, t_seq, new_control_points_at_which_time_points_in_old_trajectory_1)
    new_control_points_y_0 = piecewise_linear(
        control_points_y, t_seq, new_control_points_at_which_time_points_in_old_trajectory_0)
    new_control_points_y_1 = piecewise_linear(
        control_points_y, t_seq, new_control_points_at_which_time_points_in_old_trajectory_1)
    new_control_points_x = (new_control_points_x_0+new_control_points_x_1)/2
    new_control_points_y = (new_control_points_y_0+new_control_points_y_1)/2

    ndx_left = 0
    ndx_right = 0  # [,)
    while True:  # disperse coinciding control points
        if ndx_left >= new_control_points_number:
            break
        ndx_right = ndx_left+1
        while True:  # find ndx_right
            if ndx_right >= new_control_points_number:
                break
            if (new_control_points_x[ndx_right] != new_control_points_x[ndx_left] or
                    new_control_points_y[ndx_right] != new_control_points_y[ndx_left]):
                break
            ndx_right += 1
        win_size = ndx_right-ndx_left

        # (,)
        arange_ndx_left = ndx_left-1
        arange_ndx_right = ndx_right

        if arange_ndx_left < 0:  # dispersed to the right
            arange_x = torch.linspace(
                new_control_points_x[0],
                new_control_points_x[arange_ndx_right],
                win_size+1)
            arange_y = torch.linspace(
                new_control_points_y[0],
                new_control_points_y[arange_ndx_right],
                win_size+1)
            new_control_points_x[ndx_left:ndx_right] = arange_x[:-1]
            new_control_points_y[ndx_left:ndx_right] = arange_y[:-1]
        elif arange_ndx_right >= new_control_points_number:  # dispersed to the left
            arange_x = torch.linspace(
                new_control_points_x[arange_ndx_left],
                new_control_points_x[-1],
                win_size+1)
            arange_y = torch.linspace(
                new_control_points_y[arange_ndx_left],
                new_control_points_y[-1],
                win_size+1)
            new_control_points_x[ndx_left:ndx_right] = arange_x[1:]
            new_control_points_y[ndx_left:ndx_right] = arange_y[1:]
        else:  # dispersed to both sides
            if win_size % 2 == 0:
                # left half
                arange_x_step = (
                    new_control_points_x[ndx_left]-new_control_points_x[arange_ndx_left])*2/win_size
                arange_y_step = (
                    new_control_points_y[ndx_left]-new_control_points_y[arange_ndx_left])*2/win_size
                arange_x = torch.arange(
                    new_control_points_x[arange_ndx_left]+arange_x_step/2,
                    new_control_points_x[ndx_left],
                    arange_x_step)
                arange_y = torch.arange(
                    new_control_points_y[arange_ndx_left]+arange_y_step/2,
                    new_control_points_y[ndx_left],
                    arange_y_step)
                new_control_points_x[ndx_left: ndx_left +
                                     win_size//2] = arange_x
                new_control_points_y[ndx_left: ndx_left +
                                     win_size//2] = arange_y

                # right half
                arange_x_step = (
                    new_control_points_x[arange_ndx_right]-new_control_points_x[ndx_left])*2/win_size
                arange_y_step = (
                    new_control_points_y[arange_ndx_right]-new_control_points_y[ndx_left])*2/win_size
                arange_x = torch.arange(
                    new_control_points_x[ndx_left]+arange_x_step/2,
                    new_control_points_x[arange_ndx_right],
                    arange_x_step)
                arange_y = torch.arange(
                    new_control_points_y[ndx_left]+arange_y_step/2,
                    new_control_points_y[arange_ndx_right],
                    arange_y_step)
                new_control_points_x[ndx_left +
                                     win_size//2: ndx_right] = arange_x
                new_control_points_y[ndx_left + win_size //
                                     2: ndx_right] = arange_y
            else:  # win_size%2!=0
                # left half
                arange_x = torch.linspace(
                    new_control_points_x[arange_ndx_left],
                    new_control_points_x[ndx_left],
                    win_size//2 + 2)
                arange_y = torch.linspace(
                    new_control_points_y[arange_ndx_left],
                    new_control_points_y[ndx_left],
                    win_size//2 + 2)
                new_control_points_x[ndx_left: ndx_left +
                                     win_size//2] = arange_x[1:-1]
                new_control_points_y[ndx_left: ndx_left +
                                     win_size//2] = arange_y[1:-1]

                # right half
                arange_x = torch.linspace(
                    new_control_points_x[ndx_left],
                    new_control_points_x[arange_ndx_right],
                    win_size//2 + 2)
                arange_y = torch.linspace(
                    new_control_points_y[ndx_left],
                    new_control_points_y[arange_ndx_right],
                    win_size//2 + 2)
                new_control_points_x[ndx_right - win_size //
                                     2: ndx_right] = arange_x[1:-1]
                new_control_points_y[ndx_right - win_size //
                                     2: ndx_right] = arange_y[1:-1]

        ndx_left = ndx_right
    # disperse coinciding control points

    new_t_seq = torch.linspace(t_seq[0], t_seq[-1], new_control_points_number)

    return (new_control_points_x, new_control_points_y, new_t_seq)


def get_default_gaussian_psf_func(fwhn=1.29):
    '''
    return functor (delta_x:torch.Tensor, delta_y:torch.Tensor) -> psf_values
    '''
    sigma2 = ((fwhn**2) * 1.4426950408889634) / 4

    def get_gaussian_psf(delta_x, delta_y):
        return 0.5275796705272378*torch.exp(-(delta_x**2 + delta_y**2) / sigma2)
    return get_gaussian_psf


def main_loop(
    control_points_x: torch.Tensor, control_points_y: torch.Tensor, t_seq: torch.Tensor,
    pixels_x: torch.Tensor, pixels_y: torch.Tensor, pixels_foreground_values: torch.Tensor,
    f_star: float,
    untrailed_unbinned_psf,
    regularization_weight_norm: float, regularization_weight_tang: float,
    numerical_integration_steps_number: int,
    binning_grid_x: torch.Tensor, binning_grid_y: torch.Tensor,
    loop_exit_predicate,
    max_epochs_number=1024, learning_rate=0.001, control_point_tolerance=0.001,
    each_epoch_callback=lambda epoch, control_points_x, control_points_y, intensity_square_residual_sum, regularization_term: None,
    control_points_number_seq=PRIME_NUM_LIST
):
    '''
    `control_points_x`,`control_points_y`:
        size=(Q+1,); NO GRAD; to be optimized.
    `t_seq`:
        size=(Q+1,); no grad; an uniformly ascending float-type sequence, the time sequence of the control points.
    `pixels_x`,`pixels_y`,`pixels_foreground_values`:
        size=(pixels_number,); no grad; ROI pixels' coordinates and foreground values.
    `binning_grid_x`,`binning_grid_y`:
        size=(β_x,β_y); no grad.
    `untrailed_unbinned_psf`:
        auto-grad functor (Δx,Δy) -> psf_values; `Δx`,`Δy`,`psf_values` share the same shape.
        Do not wrap numpy.ndarray but torch.Tensor.
    `numerical_integration_steps_number`:
        `2N` in our paper, an even-positive integer; not giving an even-positive integer is an undefined behavior.

    `loop_exit_predicate`:
        functor (optimized_control_points_x,optimized_control_points_y,t_seq) -> bool;
        return True  ---- exit the loop, and return current (optimized_control_points_x,optimized_control_points_y,t_seq);
        return False ---- refine the control points, and continue looping.

    `max_epochs_number`:
        max steps number of the optimizer; ≤0 ---- unlimited.
    `control_point_tolerance`:
        stop optimizing if none of the control points is moved by over `control_point_tolerance` in optimizer.step(); ≤0 ---- loop till `max_epochs_number`.

    return (optimized_control_points_x,optimized_control_points_y,refined_t_seq); no grad
    '''

    new_control_points_x, new_control_points_y, new_t_seq = refine_control_points(
        control_points_x, control_points_y, t_seq,
        pixels_x, pixels_y, pixels_foreground_values,
        numerical_integration_steps_number,
        control_points_number_seq)
    control_points_x = piecewise_linear(new_control_points_x, new_t_seq, t_seq)
    control_points_y = piecewise_linear(new_control_points_y, new_t_seq, t_seq)

    while True:  # main loop
        optimized_control_points_x, optimized_control_points_y = rmsprop_optimize(
            control_points_x, control_points_y, t_seq,
            pixels_x, pixels_y, pixels_foreground_values,
            f_star,
            untrailed_unbinned_psf,
            regularization_weight_norm, regularization_weight_tang,
            numerical_integration_steps_number,
            binning_grid_x, binning_grid_y,
            max_epochs_number, learning_rate, control_point_tolerance,
            each_epoch_callback)
        if loop_exit_predicate(optimized_control_points_x, optimized_control_points_y, t_seq):
            break
        control_points_x, control_points_y, t_seq = refine_control_points(
            optimized_control_points_x, optimized_control_points_y, t_seq,
            pixels_x, pixels_y, pixels_foreground_values,
            numerical_integration_steps_number,
            control_points_number_seq)
    # main loop

    return optimized_control_points_x, optimized_control_points_y, t_seq
