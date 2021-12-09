import os

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import derivatives
import get_param
from Logger import Logger
from derivatives import params, toCuda, toCpu
from pde_cnn import get_Net


def benchmark():
    folder = Path('benchmark')
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/p', exist_ok=True)
    os.makedirs(f'{folder}/v', exist_ok=True)
    os.makedirs(f'{folder}/u', exist_ok=True)
    os.makedirs(f'{folder}/plots', exist_ok=True)

    dt = 0.01
    space_unit = 0.0025
    num_frames = 300

    ### Set up the Hydronet
    # params.mu = 0.00125
    params.rho = 4.0
    # params.dt = dt
    # params.space_unit = space_unit
    params.mu = 0.1
    params.dt = 4.0
    params.space_unit = 1.0
    params.load_date_time = "2020-05-28 16:13:12"

    n_cuda_devices = torch.cuda.device_count()

    live_visualize = False
    save_movie = True

    w, h = params.width, params.height
    n_warmup_time_steps = 30 * int(4 / dt)

    logger = Logger(get_param.get_hyperparam(params), use_csv=False, use_tensorboard=True)
    pde_cnn_copies = []
    for cuda_i in range(n_cuda_devices):
        pde_cnn = toCuda(get_Net(params), cuda_i)
        date_time, index = logger.load_state(pde_cnn, None, datetime=params.load_date_time, index=params.load_index)
        for p in pde_cnn.parameters():
            p.requires_grad = False
        pde_cnn.eval()
        pde_cnn_copies.append(pde_cnn)
    print(f"loaded {params.net}: {date_time}, index: {index}")

    # setup opencv windows for in depth visualizations
    if live_visualize:
        cv2.namedWindow('p', cv2.WINDOW_NORMAL)
        cv2.namedWindow('a', cv2.WINDOW_NORMAL)
        cv2.namedWindow('v', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
        cv2.namedWindow('debug2', cv2.WINDOW_NORMAL)
        cv2.namedWindow('debug3', cv2.WINDOW_NORMAL)

    if save_movie:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        movie_p = cv2.VideoWriter(f'{folder}/plots/p_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0, (w, h))
        movie_v = cv2.VideoWriter(f'{folder}/plots/v_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0,
                                  (w - 3, h - 3))
        movie_a = cv2.VideoWriter(f'{folder}/plots/a_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0, (w, h))
        movie_vv = cv2.VideoWriter(f'{folder}/plots/vv_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0,
                                  (w - 3, h - 3))
        movie_vu = cv2.VideoWriter(f'{folder}/plots/vu_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0,
                                  (w - 3, h - 3))

    derivatives.current_cuda = 0

    flow_v = 1.0

    def get_problem(w, h, flow_v):
        v_cond = toCuda(torch.zeros(1, 2, h, w))
        v_cond[0, 1, 10:(h - 10), 0:5] = flow_v
        v_cond[0, 1, 10:(h - 10), (w - 5):w] = flow_v

        cond_mask = toCuda(torch.zeros(1, 1, h, w))
        cond_mask[0, :, 0:3, :] = 1
        cond_mask[0, :, (h - 3):h, :] = 1
        cond_mask[0, :, :, 0:5] = 1
        cond_mask[0, :, :, (w - 5):w] = 1

        flow_mask = 1 - cond_mask
        a_old = toCuda(torch.zeros(1, 1, h, w))
        p_old = toCuda(torch.zeros(1, 1, h, w))
        return v_cond, cond_mask, flow_mask, a_old, p_old

    bg_v_cond, bg_cond_mask, flow_mask, a_old, p_old = get_problem(w, h, flow_v)

    object_y = h // 2
    object_x = w // 2
    object_r = 21
    object_vy = 0
    object_vx = 0
    object_w = 0
    y_mesh, x_mesh = torch.meshgrid([torch.arange(-object_r, object_r + 1), torch.arange(-object_r, object_r + 1)])
    y_mesh = toCuda(y_mesh)
    x_mesh = toCuda(x_mesh)
    mask_ball = ((x_mesh ** 2 + y_mesh ** 2) < object_r ** 2).float().unsqueeze(0)
    v_ball = object_w * torch.cat([x_mesh.unsqueeze(0), -y_mesh.unsqueeze(0)]) * mask_ball

    cond_mask = bg_cond_mask.clone()
    v_cond = bg_v_cond.clone()

    cond_mask[0, :, (object_y - object_r):(object_y + object_r + 1),
    (object_x - object_r):(object_x + object_r + 1)] += mask_ball
    v_cond[0, 0, (object_y - object_r):(object_y + object_r + 1), (object_x - object_r):(object_x + object_r + 1)] += \
    v_ball[0] + object_vy
    v_cond[0, 1, (object_y - object_r):(object_y + object_r + 1), (object_x - object_r):(object_x + object_r + 1)] += \
    v_ball[1] + object_vx

    cond_mask = torch.clamp(cond_mask, 0, 1)
    flow_mask = 1 - cond_mask

    v_cond_mac = derivatives.normal2staggered(bg_v_cond)
    with torch.no_grad():
        for t in range(n_warmup_time_steps):
            a_new, p_new = pde_cnn_copies[0](a_old, p_old, flow_mask, v_cond_mac, cond_mask)
            # p_new = (p_new - torch.mean(p_new, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
            # a_new = (a_new - torch.mean(a_new, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
            a_old, p_old = a_new, p_new

    for frame in range(num_frames):
        cuda_i = int(np.floor((frame / num_frames) * n_cuda_devices))
        derivatives.current_cuda = cuda_i
        flow_mask = flow_mask.cuda(cuda_i)
        v_cond_mac = v_cond_mac.cuda(cuda_i)
        cond_mask = cond_mask.cuda(cuda_i)

        a_new, p_new = pde_cnn_copies[cuda_i](a_old.cuda(cuda_i), p_old.cuda(cuda_i), flow_mask, v_cond_mac, cond_mask)

        cond_mask_mac = derivatives.normal2staggered(cond_mask.repeat(1, 2, 1, 1))
        flow_mask_mac = derivatives.normal2staggered(flow_mask.repeat(1, 2, 1, 1))
        v_new = derivatives.rot_mac(a_new)
        v_new = cond_mask_mac * v_cond_mac + flow_mask_mac * v_new

        pmin = -4.0
        pmax = 1.0

        with torch.no_grad():
            p = flow_mask[0, 0] * p_new[0, 0].clone()
            p_df = pd.DataFrame(toCpu(p).numpy())
            p_df.to_csv(f'{folder}/p/{frame}.csv')
            # pmin = torch.min(p)
            # pmax = torch.max(p)
            pmin += 3  # adjust brightness
            p = p - pmin
            p = p / (pmax - pmin)
            p = torch.clamp(p, 0.0, 1.0)
            if live_visualize:
                cv2.imshow('p', toCpu(p).numpy())
            p = toCpu(p).unsqueeze(2).repeat(1, 1, 3).numpy()
            if save_movie:
                movie_p.write((255 * p).astype(np.uint8))

            vector = derivatives.staggered2normal(v_new.clone())[0, :, 2:-1, 2:-1]
            u_df = pd.DataFrame(toCpu(vector[0, :, :]).numpy())
            u_df.to_csv(f'{folder}/v/{frame}.csv')
            v_df = pd.DataFrame(toCpu(vector[1, :, :]).numpy())
            v_df.to_csv(f'{folder}/u/{frame}.csv')
            vvmin = -0.5
            vvmax = 0.5
            vv = vector[0, :, :] - vvmin
            vv = vv / (vvmax - vvmin)
            vv = torch.clamp(vv, 0.0, 1.0)
            vv= toCpu(vv).unsqueeze(2).repeat(1, 1, 3).numpy()
            if save_movie:
                movie_vv.write((255 * vv).astype(np.uint8))
            vumin = -0.5
            vumax = 0.5
            vu = vector[1, :, :] - vumin
            vu = vu / (vumax - vumin)
            vu = torch.clamp(vu, 0.0, 1.0)
            vu = toCpu(vu).unsqueeze(2).repeat(1, 1, 3).numpy()
            if save_movie:
                movie_vu.write((255 * vu).astype(np.uint8))

            image = derivatives.vector2HSV(vector.detach())
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            if live_visualize:
                cv2.imshow('v', image)
            if save_movie:
                movie_v.write((255 * image).astype(np.uint8))

            a = a_new[0, 0].clone()
            a = a - torch.min(a)
            a = toCpu(a / torch.max(a)).unsqueeze(2).repeat(1, 1, 3).numpy()
            if save_movie:
                movie_a.write((255 * a).astype(np.uint8))
            if live_visualize:
                cv2.imshow('a', a)

        a_old, p_old = a_new, p_new

    if save_movie:
        movie_p.release()
        movie_v.release()
        movie_a.release()
        movie_vv.release()
        movie_vu.release()


if __name__ == '__main__':
    benchmark()
