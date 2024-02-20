import os
import sys
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()

def clear_gpus():
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    for d in devices:
        os.system(f'pkill -f "voidgpu{d}"')

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        set_hparams()
        run_task()
    except KeyboardInterrupt:
        if hparams['init_method'] == 'file':
            # on exit, remove the shared file in nfs for DDP
            exp_name = hparams['exp_name']
            shared_file_name = f'/home/tiger/nfs/pytorch_ddp_sharedfile/{exp_name}'
            if os.path.exists(shared_file_name):
                os.system(f"rm -r {shared_file_name}")

