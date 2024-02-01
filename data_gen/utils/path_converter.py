import os


class PathConverter():
    def __init__(self):
        self.prefixs = {
            "vid": "/video/",
            "gt": "/gt_imgs/",
            "head": "/head_imgs/", 
            "torso": "/torso_imgs/", 
            "person": "/person_imgs/", 
            "torso_with_bg": "/torso_with_bg_imgs/", 
            "single_bg": "/bg_img/",
            "bg": "/bg_imgs/",
            "segmaps": "/segmaps/",
            "inpaint_torso": "/inpaint_torso_imgs/",
            "com": "/com_imgs/",
            "inpaint_torso_with_com_bg": "/inpaint_torso_with_com_bg_imgs/",
        }
        
    def to(self, path: str, old_pattern: str, new_pattern: str):
        return path.replace(self.prefixs[old_pattern], self.prefixs[new_pattern], 1)

pc = PathConverter()