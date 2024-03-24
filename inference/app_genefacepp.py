import os, sys
sys.path.append('./')
import argparse
import gradio as gr
from inference.genefacepp_infer import GeneFace2Infer
from utils.commons.hparams import hparams
import random
import time

class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        assert len(kargs) == 0
        keys = [
            'drv_audio_name',
            'blink_mode',
            'temperature',
            'lle_percent',
            'mouth_amp',
            'raymarching_end_threshold',
            'fp16',
            'a2m_ckpt',
            'postnet_ckpt',
            'head_ckpt',
            'torso_ckpt',
            'low_memory_usage',
        ]
        inp = {}
        out_name = None
        info = ""
        
        print(f"args {args}")
        try: # try to catch errors and jump to return 
            for key_index in range(len(keys)):
                key = keys[key_index]
                inp[key] = args[key_index]
                if '_name' in key or '_ckpt' in key:
                    inp[key] = inp[key] if inp[key] is not None else ''
                    
            if inp['drv_audio_name'] == '':
                info = "Input Error: Driving audio is REQUIRED!"
                raise ValueError
                
            # if inp['drv_pose'] == '':
            inp['drv_pose'] = 'nearest'    
            
            reload_flag = False
            if inp['a2m_ckpt'] != self.audio2secc_dir:
                print("Changes of a2m_ckpt detected, reloading model")
                reload_flag = True
            if inp['postnet_ckpt'] != self.postnet_dir:
                print("Changes of postnet_ckpt detected, reloading model")
                reload_flag = True
            if inp['head_ckpt'] != self.head_model_dir:
                print("Changes of head_ckpt detected, reloading model")
                reload_flag = True
            if inp['torso_ckpt'] != self.torso_model_dir:
                print("Changes of torso_ckpt detected, reloading model")
                reload_flag = True

            inp['debug'] = False
            inp['out_name'] = f"temp/out_{generate_random_uuid()}.mp4"
            # inp['seed'] = 42
            
            print(f"infer inputs : {inp}")
                
            try:
                if reload_flag:
                    self.__init__(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'])
            except Exception as e:
                content = f"{e}"
                info = f"Reload ERROR: {content}"
                
            try:
                out_name = self.infer_once(inp)
            except Exception as e:
                content = f"{e}"
                info = f"Inference ERROR: {content}"
        except Exception as e:
            if info == "": # unexpected errors
                content = f"{e}"
                info = f"WebUI ERROR: {content}"
        
        # output part
        
        if len(info) > 0 : # there is errors    
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else: # no errors
            info_gr = gr.update(visible=False, value=info)
            
        if out_name is not None and len(out_name) > 0 and os.path.exists(out_name): # good output
            print(f"Succefully generated in {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"Failed to generate")
            video_gr = gr.update(visible=True, value=out_name)
            
        return video_gr, info_gr

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

# generate random uuid and do not disturb global random state
def generate_random_uuid(len_uuid = 16):
    prev_state = random.getstate()
    random.seed(time.time())
    s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = ''.join(random.choices(s, k=len_uuid))
    random.setstate(prev_state)
    return res

def genefacepp_demo(
    audio2secc_dir,
    postnet_dir,
    head_model_dir,
    torso_model_dir, 
    device          = 'cuda',
    warpfn          = None,
    ):

    sep_line = "-" * 40

    infer_obj = Inferer(
        audio2secc_dir=audio2secc_dir, 
        postnet_dir=postnet_dir,
        head_model_dir=head_model_dir,
        torso_model_dir=torso_model_dir,
        device=device,
    )

    print(sep_line)
    print("Model loading is finished.")
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
        gr.Markdown("\
            <div align='center'> <h2> GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation </span> </h2> \
            <a style='font-size:18px;color: #a0a0a0' href='https://arxiv.org/pdf/2305.00787.pdf'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'> Github </div>")
        
        sources = None
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            drv_audio_name = gr.Audio(label="Input audio (required)", sources=sources, type="filepath", value='data/raw/val_wavs/MacronSpeech.wav')
                             
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('General Settings'):
                        gr.Markdown("need help? please visit our [best practice page](https://baidu.com) for more detials")
                        with gr.Column(variant='panel'):

                            blink_mode = gr.Radio(['none', 'period'], value='none', label='blink mode', info="whether to blink periodly")       
                            # blink_mode = gr.Radio(['none', 'period'], value='period', label='blink mode', info="whether to blink periodly")       
                            lle_percent = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="lle_percent",  value=0.0, info='higher-> drag pred.landmark closer to the training video\'s landmark set',)
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature",  value=0.0, info='audio to secc temperature',)
                            mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude",  value=0.4, info='higher -> mouth will open wider, default to be 0.4',)
                            raymarching_end_threshold = gr.Slider(minimum=0.0, maximum=0.1, step=0.0025, label="ray marching end-threshold",  value=0.01, info='increase it to improve inference speed',)
                            fp16 = gr.Checkbox(label="Whether to utilize fp16 to speed up inference")
                            low_memory_usage = gr.Checkbox(label="Low Memory Usage Mode: save memory at the expense of lower inference speed. Useful when running a low audio (minutes-long).", value=False)
                            
                            submit = gr.Button('Generate', elem_id="generate", variant='primary')
                        
                    with gr.Tabs(elem_id="genearted_video"):
                        info_box = gr.Textbox(label="Error", interactive=False, visible=False)
                        gen_video = gr.Video(label="Generated video", format="mp4", visible=True)
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('Checkpoints'):
                        with gr.Column(variant='panel'):
                            ckpt_info_box = gr.Textbox(value="Please select \"ckpt\" under the checkpoint folder ", interactive=False, visible=True, show_label=False)
                            
                            audio2secc_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=audio2secc_dir, file_count='single', label='audio2secc model ckpt path or directory')
                            torso_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=torso_model_dir, file_count='single', label='torso model ckpt path or directory')
                            postnet_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=postnet_dir, file_count='single', label='(optional) pose net model ckpt path or directory')
                            head_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=head_model_dir, file_count='single', label='(optional) head model ckpt path or directory (will be ignored if torso model is set)')

        fn = infer_obj.infer_once_args
        if warpfn:
            fn = warpfn(fn)
        submit.click(
                    fn=fn, 
                    inputs=[ 
                        drv_audio_name,
                        blink_mode,
                        temperature,
                        lle_percent,
                        mouth_amp,
                        raymarching_end_threshold,
                        fp16,
                        audio2secc_dir,
                        postnet_dir,
                        head_model_dir,
                        torso_model_dir,
                        low_memory_usage
                    ], 
                    outputs=[
                        gen_video,
                        info_box,
                    ],
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return genefacepp_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--postnet_ckpt", type=str, default='')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1') 

    args = parser.parse_args()
    demo = genefacepp_demo(
        audio2secc_dir=args.a2m_ckpt,
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        device='cuda:0',
        warpfn=None,
    )
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)


