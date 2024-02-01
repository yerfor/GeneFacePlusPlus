import numpy as np
from data_util.face3d_helper import Face3DHelper
from utils.commons.tensor_utils import convert_to_tensor

def polygon_area(x, y):
    """
    x: [T, K=6]
    y: [T, K=6]
    return: [T,]
    """
    x_ = x - x.mean(axis=-1, keepdims=True)
    y_ = y - y.mean(axis=-1, keepdims=True)
    correction = x_[:,-1] * y_[:,0] - y_[:,-1]* x_[:,0]
    main_area = (x_[:,:-1] * y_[:,1:]).sum(axis=-1) - (y_[:,:-1] * x_[:,1:]).sum(axis=-1)
    return 0.5 * np.abs(main_area + correction)

def get_eye_area_percent(id, exp, face3d_helper):
    id = convert_to_tensor(id)
    exp = convert_to_tensor(exp)
    cano_lm3d = face3d_helper.reconstruct_cano_lm3d(id, exp)
    cano_lm2d = (cano_lm3d[..., :2] + 1) / 2
    lms = cano_lm2d.cpu().numpy()
    eyes_left = slice(36, 42)
    eyes_right = slice(42, 48)
    area_left = polygon_area(lms[:, eyes_left, 0], lms[:, eyes_left, 1])
    area_right = polygon_area(lms[:, eyes_right, 0], lms[:, eyes_right, 1])
    # area percentage of two eyes of the whole image...
    area_percent = (area_left + area_right) / 1 * 100 # recommend threshold is 0.25%
    return area_percent # [T,]


if __name__ == '__main__':
    import numpy as np
    import imageio
    import cv2
    import torch
    from data_gen.utils.process_video.extract_lm2d import extract_lms_mediapipe_job, read_video_to_frames, index_lm68_from_lm468
    from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
    from data_util.face3d_helper import Face3DHelper

    face3d_helper = Face3DHelper()
    video_name = 'data/raw/videos/May_10s.mp4'
    frames = read_video_to_frames(video_name)
    coeff = fit_3dmm_for_a_video(video_name, save=False)
    area_percent = get_eye_area_percent(torch.tensor(coeff['id']), torch.tensor(coeff['exp']), face3d_helper)
    writer = imageio.get_writer("1.mp4", fps=25)
    for idx, frame in enumerate(frames):
        frame = cv2.putText(frame, f"{area_percent[idx]:.2f}", org=(128,128), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)
        writer.append_data(frame)
    writer.close()