from shutil import register_unpack_format
import cv2
from tqdm import trange
import os
import numpy as np

FRAME_RATE = 20

def convert_file_to_video(file_dir, video_dir, no_alpha=False):
    i = 0
    print('in:',file_dir)
    print('out:',video_dir)
    format_tile = '.mp4'#'.avi'
    format_name = 'mp4v'#'MJPG'
    zero_img_dir = os.path.join(file_dir, '0.png')
    if os.path.exists(zero_img_dir):
        img = cv2.imread(zero_img_dir)
    else:
        return
    video = cv2.VideoWriter(video_dir,cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(img.shape[1], img.shape[0]))
    while True:
        img_dir = os.path.join(file_dir, str(i)+'.png')
        if os.path.exists(img_dir):
            img = cv2.imread(img_dir)
            #img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
            if no_alpha:
                img = img.astype(np.float32)
                #print(img.shape)
                img/=255.00
                img = img[...,:3]*img[...,-1:] + (1.-img[...,-1:])
                #img = img[...,:3] + (1.-img[...,-1:])
                img = (255*np.clip(img,0,1)).astype(np.uint8)
            video.write(img)
            i += 1
        else:
            break
    print(i)
    video.release()
    return

def generate_one_replica(file_dir, out_dir):
    heads = ['albedo_','c','edit','residual_','rgb_','shading_','vis_label_','vis_depth_']
    out_heads = ['albedo', 'cluster_albedo', 'reconstruct', 'residual', 'render','shading','semantic','depth']
    format_tile = '.mp4'#'.avi'
    format_name = 'mp4v'#'MJPG'
    for n, head in enumerate(heads):
        out_head = out_heads[n]
        print('generate '+out_head+' video')
        video = cv2.VideoWriter(os.path.join(out_dir, out_head+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(320, 240))
        for i in trange(180):
            img_dir = os.path.join(file_dir, head+'{:03d}.png'.format(i))
            #print(img_dir)
            img = cv2.imread(img_dir)
            #print(img.shape)
            video.write(img)
        video.release()
    return

def generate_all_ori_replica(in_base_dir, out_base_dir):
    data_names = ['room_0','room_1','room_2','office_0','office_1','office_2','office_3','office_4']
    for data_name in data_names:
        print(data_name)
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        convert_file_to_video(os.path.join(in_dir, 'test'), out_dir+'/test.mp4')
        convert_file_to_video(os.path.join(in_dir, 'train'), out_dir+'/train.mp4')
    return

def generate_all_replica(in_base_dir, out_base_dir):
    data_names = ['room_0','room_1','room_2','office_0','office_1','office_2','office_3','office_4']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        out_test_dir = os.path.join(out_dir, 'test_render')
        out_train_dir = os.path.join(out_dir, 'train_render')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_test_dir, exist_ok=True)
        os.makedirs(out_train_dir, exist_ok=True)
        generate_one_replica(os.path.join(in_dir, 'test'), out_test_dir)
        generate_one_replica(os.path.join(in_dir, 'train_render/step_200000'), out_train_dir)
    return

def generate_one_blender(file_dir, out_dir):
    #heads = ['','a','c','edit','res','s']
    #out_heads = ['render','albedo','cluster_albedo','reconstruct','residual','shading']
    heads = ['','a','res','s']
    out_heads = ['render','albedo','residual','shading']
    format_tile = '.mp4'#'.avi'
    format_name = 'mp4v'#'MJPG'
    for n, head in enumerate(heads):
        out_head = out_heads[n]
        print('generate '+out_head+' video')
        video = cv2.VideoWriter(os.path.join(out_dir, out_head+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(400, 400))
        for i in trange(200):
            img_dir = os.path.join(file_dir, head+'{:03d}.png'.format(i))
            #print(img_dir)
            img = cv2.imread(img_dir)
            #print(img.shape)
            video.write(img)
        video.release()
    return

def generate_all_ori_blender(in_base_dir, out_base_dir):
    data_names = ['chair','drums','ficus','lego', 'hotdog','air_baloons','chair2','jugs']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        convert_file_to_video(os.path.join(in_dir, 'rgb'), out_dir+'/rgb.mp4',True)
        convert_file_to_video(os.path.join(in_dir, 'GT_albedo'), out_dir+'/albedo.mp4',True)
    return

def generate_all_blender(in_base_dir, out_base_dir):
    #data_names = ['chair','drums','ficus','lego', 'hotdog','air_baloons','chair2','jugs','lego_new']
    data_names = ['hotdog']
    #data_names = ['chair','drums','ficus','lego']
    #data_names = ['hotdog','air_baloons','chair2','jugs']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        #out_test_dir = os.path.join(out_dir, 'test_render')
        #out_train_dir = os.path.join(out_dir, 'train_render')
        os.makedirs(out_dir, exist_ok=True)
        #os.makedirs(out_test_dir, exist_ok=True)
        #os.makedirs(out_train_dir, exist_ok=True)
        generate_one_blender(os.path.join(in_dir, 'renderonly_test_199999'), out_dir)
        #generate_one_blender(os.path.join(in_dir, 'train_render/step_200000'), out_train_dir)
    return

def generate_all_IIW(in_base_dir, out_base_dir):
    data_names = ['chair','drums','ficus','lego', 'hotdog','air_baloons','chair2','jugs']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        convert_file_to_video(os.path.join(in_dir, 'r'), out_dir+'/r.mp4')
        convert_file_to_video(os.path.join(in_dir, 's'), out_dir+'/s.mp4')

def generate_all_replica_IIW(in_base_dir, out_base_dir):
    data_names = ['room_0','room_1','room_2','office_0','office_1','office_2','office_3','office_4']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        modes = ['test', 'train']
        for mode in modes:
            in_mode_dir = os.path.join(in_dir, mode)
            out_mode_dir = os.path.join(out_dir, mode)
            os.makedirs(out_mode_dir, exist_ok=True)
            convert_file_to_video(os.path.join(in_mode_dir, 'r'), out_mode_dir+'/r.mp4')
            convert_file_to_video(os.path.join(in_mode_dir, 's'), out_mode_dir+'/s.mp4')

def generate_all_replica_USI3D(in_base_dir, out_base_dir):
    #data_names = ['room_0','room_1','room_2','office_0','office_1','office_2','office_3','office_4']
    data_names = ['chair','drums','ficus','lego', 'hotdog','air_baloons','chair2','jugs']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        convert_file_to_video(in_dir, out_dir+'/r.mp4')
        #convert_file_to_video(os.path.join(in_dir, 'reflectance'), out_dir+'/r.mp4')
        #convert_file_to_video(os.path.join(in_dir, 'shading'), out_dir+'/s.mp4')

def generate_all_InvRender(in_base_dir, out_base_dir):
    data_names = ['hotdog','air_baloons','chair2','jugs']
    for data_name in data_names:
        in_dir = os.path.join(in_base_dir, data_name)
        out_dir = os.path.join(out_base_dir, data_name)
        os.makedirs(out_dir, exist_ok=True)
        convert_file_to_video(os.path.join(in_dir, 'albedo'), out_dir+'/r.mp4')
        #convert_file_to_video(os.path.join(in_dir, 'rgb'), out_dir+'/render.mp4')


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str) 
    parser.add_argument('--out_dir', type=str)
    parser.add_argument("--rate", type=int, default=20, help='frame rate')
    parser.add_argument("--mode", type=str, default='normal', help='is replica data')
    return parser

if __name__=='__main__':
    '''
    parser = config_parser()
    args = parser.parse_args()
    FRAME_RATE = args.rate
    if not os.path.exists(args.img_dir) or not os.path.exists(args.out_dir):
        print('invalid path dir')
        exit

    if(args.mode=='replica'):
        generate_one_replica(args.img_dir, args.out_dir)
    elif(args.mode=='replica_all'):
        generate_all_replica(args.img_dir, args.out_dir)
    elif(args.mode=='blender'):
        generate_one_blender(args.img_dir, args.out_dir)
        pass
    else:
        pass
    '''