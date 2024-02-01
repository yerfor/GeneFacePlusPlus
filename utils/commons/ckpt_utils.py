import glob
import os
import re
import torch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    if work_dir.endswith(".ckpt"):
        ckpt_paths = [work_dir]
    else:
        ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True, steps=None, verbose=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if '.' in k]) > 0:
            state_dict = {k[len(model_name) + 1:]: v for k, v in state_dict.items()
                          if k.startswith(f'{model_name}.')}
        else:
            if '.' not in model_name:
                state_dict = state_dict[model_name]
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v for k, v in state_dict[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')}
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys (shape mismatch): ", key, new_param.shape, param.shape)
                else:
                    print(f"Skipping unmatched keys (in state_dict but not in cur_model): {key}")
            for key in unmatched_keys:
                if verbose:
                    print(f"Del unmatched keys {key}")
                del state_dict[key]
        if hasattr(cur_model, 'load_state_dict'):
            cur_model.load_state_dict(state_dict, strict=strict)
        else: # when cur_model is nn.Parameter
            cur_model.data = state_dict
        print(f"| load '{model_name}' from '{ckpt_path}', strict={strict}")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)

def restore_weights(task_ref, checkpoint):
    # load model state
    for k, v in checkpoint['state_dict'].items():
        if hasattr(task_ref, k):
            getattr(task_ref, k).load_state_dict(v, strict=True)
            print(f"| resotred {k} from pretrained checkpoints")
        else:
            print(f"| the checkpoint has unmatched keys {k}")

def restore_opt_state(optimizers, checkpoint):
    # restore the optimizers
    optimizer_states = checkpoint['optimizer_states']
    for optimizer, opt_state in zip(optimizers, optimizer_states):
        if optimizer is None:
            return
        try:
            optimizer.load_state_dict(opt_state)
            # move optimizer to GPU 1 weight at a time
            # if self.on_gpu:
            #     for state in optimizer.state.values():
            #         for k, v in state.items():
            #             if isinstance(v, torch.Tensor):
            #                 state[k] = v.cuda(self.root_gpu)
        except ValueError:
            print("| WARMING: optimizer parameters not match !!!")
    