import os
import copy
import numpy as np
import tqdm
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.commons.multiprocess_utils import multiprocess_run_tqdm, multiprocess_run
from utils.commons.tensor_utils import convert_to_np
from sklearn.neighbors import NearestNeighbors

def scatter_np(condition_img, classSeg=5):
# def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.shape
    # if height != label_size[0] or width != label_size[1]:
        # condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = np.zeros([batch, classSeg, condition_img.shape[2], condition_img.shape[3]]).astype(np.int_)
    # input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    np.put_along_axis(input_label, condition_img, 1, 1)
    return input_label

def scatter(condition_img, classSeg=19):
# def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.size()
    # if height != label_size[0] or width != label_size[1]:
        # condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = torch.zeros(batch, classSeg, condition_img.shape[2], condition_img.shape[3], device=condition_img.device)
    # input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    return input_label.scatter_(1, condition_img.long(), 1)

def encode_segmap_mask_to_image(segmap):
    # rgb
    _,h,w = segmap.shape
    encoded_img = np.ones([h,w,3],dtype=np.uint8) * 255
    colors = [(255,255,255),(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0)]
    for i, color in enumerate(colors):
        mask = segmap[i].astype(int)
        index = np.where(mask != 0)
        encoded_img[index[0], index[1], :] = np.array(color)
    return encoded_img.astype(np.uint8)
        
def decode_segmap_mask_from_image(encoded_img):
    # rgb
    colors = [(255,255,255),(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0)]
    bg = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 255)
    hair = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 0)
    body_skin = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 0) & (encoded_img[..., 2] == 255)
    face_skin = (encoded_img[..., 0] == 0) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 255)
    clothes = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 0) & (encoded_img[..., 2] == 0)
    others = (encoded_img[..., 0] == 0) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 0)
    segmap = np.stack([bg, hair, body_skin, face_skin, clothes, others], axis=0)
    return segmap.astype(np.uint8)

def read_video_frame(video_name, frame_id):
    # https://blog.csdn.net/bby1987/article/details/108923361
    # frame_num = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) # ==> 总帧数
    # fps = video_capture.get(cv2.CAP_PROP_FPS)               # ==> 帧率
    # width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)     # ==> 视频宽度
    # height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   # ==> 视频高度
    # pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)        # ==> 句柄位置
    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1000)        # ==> 设置句柄位置
    # pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)        # ==> 此时 pos = 1000.0
    # video_capture.release()
    vr = cv2.VideoCapture(video_name)
    vr.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = vr.read()
    return frame

def decode_segmap_mask_from_segmap_video_frame(video_frame):
    # video_frame: 0~255 BGR, obtained by read_video_frame
    def assign_values(array):
        remainder = array % 40  # 计算数组中每个值与40的余数
        assigned_values = np.where(remainder <= 20, array - remainder, array + (40 - remainder))
        return assigned_values
    segmap = video_frame.mean(-1)
    segmap = assign_values(segmap) // 40 # [H, W] with value 0~5 
    segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
    return segmap.astype(np.uint8)

def extract_background(img_lst, segmap_lst=None):
    """
    img_lst: list of rgb ndarray
    """
    # only use 1/20 images
    num_frames = len(img_lst)
    img_lst = img_lst[::20] if num_frames > 20 else img_lst[0:1]
        
    if segmap_lst is not None:
        segmap_lst = segmap_lst[::20] if num_frames > 20 else segmap_lst[0:1]
        assert len(img_lst) == len(segmap_lst)
    # get H/W
    h, w = img_lst[0].shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for idx, img in enumerate(img_lst):
        if segmap_lst is not None:
            segmap = segmap_lst[idx]
        else:
            segmap = seg_model._cal_seg_map(img)
        bg = (segmap[0]).astype(bool)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 10 # 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    num_pixs = distss.shape[1]
    imgs = np.stack(img_lst).reshape(-1, num_pixs, 3)

    bg_img = np.zeros((h*w, 3), dtype=np.uint8)
    bg_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bg_img = bg_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 10 # 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bg_img[bg_xys[:, 0], bg_xys[:, 1], :] = bg_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    return bg_img


global_segmenter = None
def job_cal_seg_map_for_image(img, segmenter_options=None, segmenter=None):
    """
    被 MediapipeSegmenter.multiprocess_cal_seg_map_for_a_video所使用, 专门用来处理单个长视频.
    """
    global global_segmenter
    if segmenter is not None:
        segmenter_actual = segmenter
    else:
        global_segmenter = vision.ImageSegmenter.create_from_options(segmenter_options) if global_segmenter is None else global_segmenter
        segmenter_actual = global_segmenter
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    out = segmenter_actual.segment(mp_image)
    segmap = out.category_mask.numpy_view().copy() # [H, W]

    segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
    segmap_image = segmap[:, :, None].repeat(3, 2).astype(float)
    segmap_image = (segmap_image * 40).astype(np.uint8)

    return segmap_mask, segmap_image

class MediapipeSegmenter:
    def __init__(self):
        model_path = 'data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("downloading segmenter model from mediapipe...")
            os.system(f"wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
            os.system(f"mv selfie_multiclass_256x256.tflite {model_path}")
            print("download success")
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=vision.RunningMode.IMAGE, output_category_mask=True)
        self.video_options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO, output_category_mask=True)
    
    def multiprocess_cal_seg_map_for_a_video(self, imgs, num_workers=4):
        """
        并行处理单个长视频
        imgs: list of rgb array in 0~255
        """
        segmap_masks = []
        segmap_images = []
        img_lst = [(self.options, imgs[i]) for i in range(len(imgs))]
        for (i, res) in multiprocess_run_tqdm(job_cal_seg_map_for_image, args=img_lst, num_workers=num_workers, desc='extracting from a video in multi-process'):
            segmap_mask, segmap_image = res
            segmap_masks.append(segmap_mask)
            segmap_images.append(segmap_image)
        return segmap_masks, segmap_images
        
    def _cal_seg_map_for_video(self, imgs, segmenter=None, return_onehot_mask=True, return_segmap_image=True):
        segmenter = vision.ImageSegmenter.create_from_options(self.video_options) if segmenter is None else segmenter
        assert return_onehot_mask or return_segmap_image # you should at least return one
        segmap_masks = []
        segmap_images = []
        for i in tqdm.trange(len(imgs), desc="extracting segmaps from a video..."):
            img = imgs[i]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            out = segmenter.segment_for_video(mp_image, 40 * i)
            segmap = out.category_mask.numpy_view().copy() # [H, W]

            if return_onehot_mask:
                segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
                segmap_masks.append(segmap_mask)
            if return_segmap_image:
                segmap_image = segmap[:, :, None].repeat(3, 2).astype(float)
                segmap_image = (segmap_image * 40).astype(np.uint8)
                segmap_images.append(segmap_image)
        
        if return_onehot_mask and return_segmap_image:
            return segmap_masks, segmap_images
        elif return_onehot_mask:
            return segmap_masks
        elif return_segmap_image:
            return segmap_images
    
    def _cal_seg_map(self, img, segmenter=None, return_onehot_mask=True):
        """
        segmenter: vision.ImageSegmenter.create_from_options(options)
        img: numpy, [H, W, 3], 0~255
        segmap: [C, H, W]
        0 - background
        1 - hair
        2 - body-skin
        3 - face-skin
        4 - clothes
        5 - others (accessories)
        """
        assert img.ndim == 3
        segmenter = vision.ImageSegmenter.create_from_options(self.options) if segmenter is None else segmenter 
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        out = segmenter.segment(image) 
        segmap = out.category_mask.numpy_view().copy() # [H, W]
        if return_onehot_mask:
            segmap = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
        return segmap

    def _seg_out_img_with_segmap(self, img, segmap, mode='head'):
        """
        img: [h,w,c], img is in 0~255, np
        """
        # 
        img = copy.deepcopy(img)
        if mode == 'head':
            selected_mask = segmap[[1,3,5] , :, :].sum(axis=0)[None,:] > 0.5 # glasses 也属于others
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
            # selected_mask = segmap[[1,3] , :, :].sum(dim=0, keepdim=True) > 0.5
        elif mode == 'person':
            selected_mask = segmap[[1,2,3,4,5], :, :].sum(axis=0)[None,:] > 0.5 
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'torso':
            selected_mask = segmap[[2,4], :, :].sum(axis=0)[None,:] > 0.5
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'torso_with_bg':
            selected_mask = segmap[[0, 2,4], :, :].sum(axis=0)[None,:] > 0.5
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'bg':
            selected_mask = segmap[[0], :, :].sum(axis=0)[None,:] > 0.5  # only seg out 0, which means background
            img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
        elif mode == 'full':
            pass
        else:
            raise NotImplementedError()
        return img, selected_mask
    
    def _seg_out_img(self, img, segmenter=None, mode='head'):
        """
        imgs [H, W, 3] 0-255
        return : person_img [B, 3, H, W]
        """
        segmenter = vision.ImageSegmenter.create_from_options(self.options) if segmenter is None else segmenter 
        segmap = self._cal_seg_map(img, segmenter=segmenter, return_onehot_mask=True) # [B, 19, H, W]
        return self._seg_out_img_with_segmap(img, segmap, mode=mode)

    def seg_out_imgs(self, img, mode='head'):
        """
        api for pytorch img, -1~1
        img: [B, 3, H, W], -1~1
        """
        device = img.device
        img = convert_to_np(img.permute(0, 2, 3, 1)) # [B, H, W, 3]
        img = ((img + 1) * 127.5).astype(np.uint8)
        img_lst = [copy.deepcopy(img[i]) for i in range(len(img))]
        out_lst = []
        for im in img_lst:
            out = self._seg_out_img(im, mode=mode)
            out_lst.append(out)
        seg_imgs = np.stack(out_lst) # [B, H, W, 3]
        seg_imgs = (seg_imgs - 127.5) / 127.5
        seg_imgs = torch.from_numpy(seg_imgs).permute(0, 3, 1, 2).to(device)
        return seg_imgs

if __name__ == '__main__':
    import imageio, cv2, tqdm
    import torchshow as ts
    img = imageio.imread("1.png")
    img = cv2.resize(img, (512,512))

    seg_model = MediapipeSegmenter()
    img = torch.tensor(img).unsqueeze(0).repeat([1, 1, 1, 1]).permute(0, 3,1,2)
    img = (img-127.5)/127.5
    out = seg_model.seg_out_imgs(img, 'torso')
    ts.save(out,"torso.png")
    out = seg_model.seg_out_imgs(img, 'head')
    ts.save(out,"head.png")
    out = seg_model.seg_out_imgs(img, 'bg')
    ts.save(out,"bg.png")
    img = convert_to_np(img.permute(0, 2, 3, 1)) # [B, H, W, 3]
    img = ((img + 1) * 127.5).astype(np.uint8)
    bg = extract_background(img)
    ts.save(bg,"bg2.png")
