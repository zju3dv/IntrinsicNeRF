from tkinter import *
from tkinter.colorchooser import *
from tkinter import filedialog
from SSR.training.cluster import * 
import time
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import torch
import copy
import colorsys
import datetime
FRAME_RATE = 20

def Hexstr2Int(colors: str) -> int:
    ret = 0
    pt = list(colors) #用列表一个字符一个字符存贮
    pt.reverse() #翻转列表
    idx = 0

    for i in pt:
        if i == 'A' or i == 'a':
            ret += int(i) * pow(16, idx)
        elif i == 'B' or i == 'b':
            ret += int(i) * pow(16, idx)
        elif i == 'C' or i == 'c':
            ret += int(i) * pow(16, idx)
        elif i == 'D' or i == 'd':
            ret += int(i) * pow(16, idx)
        elif i == 'E' or i == 'e':
            ret += int(i) * pow(16, idx)
        elif i == 'F' or i == 'f':
            ret += int(i) * pow(16, idx)
        else:
            ret += int(i) * pow(16, idx)
        idx += 1
    return ret

def bgUpdate(source):
    global curr_class_idx, curr_semantic_idx
    if curr_class_idx<0 or curr_semantic_idx<0:
        return
    red = rSlider.get()
    green = gSlider.get()
    blue = bSlider.get()
    update_color(red,green,blue)

def update_color(red,green,blue):
    global curr_class_idx, curr_semantic_idx, curr_class_color
    if curr_class_idx<0 or curr_semantic_idx<0:
        return
    myColor = "#%02x%02x%02x" % (red, green, blue) # 十六进制化
    curr_class_color[0] = float(red)/255.0
    curr_class_color[1] = float(green)/255.0
    curr_class_color[2] = float(blue)/255.0

    rgb_centers[curr_semantic_idx][curr_class_idx,0] = float(red)/255.0
    rgb_centers[curr_semantic_idx][curr_class_idx,1] = float(green)/255.0
    rgb_centers[curr_semantic_idx][curr_class_idx,2] = float(blue)/255.0

    color_lab.config(bg=myColor) #更改界面背景色
    global curr_img_idx
    update_img(curr_img_idx)

def askcc():
    myColor = askcolor()
    print(myColor)
    tmps = myColor[1]
    tmps = tmps[1:] #将后面的颜色值取出

    redc = Hexstr2Int(tmps[0:2]) # 设置红色int
    greenc = Hexstr2Int(tmps[2:4]) # 设置绿色int
    bluec = Hexstr2Int(tmps[4:6]) # 设置蓝色int

    rSlider.set(redc) #重设
    gSlider.set(greenc)
    bSlider.set(bluec)
    
    root.config(bg=myColor[1]) # 重设

# python gui.py --img_dir OURS_intrinsic_replica/office_1/train_render/step_200000 --cluster_config OURS_intrinsic_replica/office_1/train_render/step_200000/cluster/ --rate 20 --replica --head_name test
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str) 
    parser.add_argument('--cluster_config', type=str)
    parser.add_argument("--rate", type=int, default=10, help='frame rate')
    parser.add_argument("--replica", action='store_true', help='is replica data')
    parser.add_argument("--head_name", type=str, default='', help='is replica data')
    return parser

def init_window():
    root = Tk()
    root.title("Color Chooser")
    root.geometry("500x500")
    root.resizable(False,False)
    root.config(background="white")
    width  = 500
    height = 500
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size_geo = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
    root.geometry(size_geo)
    return root

def gettime():
    global is_play, is_record
    global result_buffer, albedo_buffer, shading_buffer, residual_buffer, color_buffer,c_albedo_buffer, show_buffer, class_color_buffer, semantic_buffer, render_buffer
    global curr_result, curr_albedo, curr_shading, curr_residual, curr_c_albedo, curr_show, curr_color_img, curr_class_color, curr_semantic, curr_render
    if is_record:
        result_buffer.append(to8b(curr_result))
        albedo_buffer.append(to8b(curr_albedo))
        shading_buffer.append(to8b(curr_shading))
        residual_buffer.append(to8b(curr_residual))
        c_albedo_buffer.append(to8b(curr_c_albedo.cpu().numpy()))
        show_buffer.append(to8b(curr_show))
        color_buffer.append(curr_color_img)
        class_color_buffer.append(curr_class_color.copy())
        semantic_buffer.append(curr_semantic)
        render_buffer.append(curr_render)
    if not is_play:
        root.after(int(1000/FRAME_RATE), gettime)
        return 
    global curr_img_idx
    curr_img_idx += 1
    curr_img_idx = curr_img_idx%imgs.shape[0]
    update_img(curr_img_idx)
    
    root.after(int(1000/FRAME_RATE), gettime)

def update_color_img():
    global curr_color_img
    color_photo = ImageTk.PhotoImage(image=Image.fromarray(curr_color_img))
    color_choose_label.configure(image=color_photo)
    color_choose_label.image = color_photo
    return

def update_img(idx):
    global curr_result, curr_albedo, curr_shading, curr_residual, curr_c_albedo, curr_show, curr_color_img, curr_semantic, curr_render
    # result_buffer.append(curr_result)
    # albedo_buffer.append(curr_albedo)
    # shading_buffer.append(curr_shading)
    # residual_buffer.append(curr_residual)
    # c_albedo_buffer.append(curr_c_albedo)
    # show_buffer.append(curr_show)
    global global_shading_scale, global_residual_scale
    curr_albedo = albedos[idx]
    curr_shading = shadings[idx]
    curr_residual = residuals[idx]
    curr_semantic = semantics[idx]
    curr_render = renders[idx]
    curr_show = None

    cluster_albedo = albedos_gpu[idx].clone()
    for i in range(len(rgb_centers)):
        if rgb_centers[i] is None:
            continue
        class_idx = torch.squeeze(labels_gpu[idx]==i)
        albedo_class = cluster_albedo_class[idx][class_idx]
        cluster_albedo[class_idx] = torch.squeeze(rgb_centers[i][albedo_class])
    curr_c_albedo = cluster_albedo
    img = cluster_albedo*t_shading(shadings_gpu[idx])*global_shading_scale+t_residual(residuals_gpu[idx])*global_residual_scale
    #img = cluster_albedo*shadings_gpu[idx]*global_shading_scale+residuals_gpu[idx]*global_residual_scale
    img = img.cpu().numpy()
    curr_result = img
    if show_status.get() == 0:
        curr_show = img
        photo = numpy_to_photo(img)
    elif show_status.get() == 1:
        cluster_albedo = cluster_albedo.cpu().numpy()
        curr_show = cluster_albedo
        photo = numpy_to_photo(cluster_albedo)
    elif show_status.get() == 2:
        curr_show = shadings[idx]
        photo = numpy_to_photo(shadings[idx])
    elif show_status.get() == 3:
        curr_show = residuals[idx]
        photo = numpy_to_photo(residuals[idx])
    lab.configure(image=photo)
    lab.image = photo
    return


def load_all_image(img_dir, device, is_replica=False):
    albedos   = []
    shadings  = []
    residuals = []
    imgs = []
    labels = []
    semantics = []
    renders = []
    i = 0
    print("loading images...")
    while True:
        if not is_replica:
            albedo_name = os.path.join(img_dir, 'a{:03d}.png'.format(i))
            shading_name  = os.path.join(img_dir, 's{:03d}.png'.format(i))
            residual_name = os.path.join(img_dir, 'res{:03d}.png'.format(i))
            label_name = os.path.join(img_dir, 'acc{:03d}.png'.format(i))
            semantic_name = os.path.join(img_dir, 'acc{:03d}.png'.format(i))
            render_name = os.path.join(img_dir, '{:03d}.png'.format(i))
        else:
            albedo_name = os.path.join(img_dir, 'albedo_{:03d}.png'.format(i))
            shading_name  = os.path.join(img_dir, 'shading_{:03d}.png'.format(i))
            residual_name = os.path.join(img_dir, 'residual_{:03d}.png'.format(i))
            label_name = os.path.join(img_dir, 'label_{:03d}.png'.format(i))
            semantic_name = os.path.join(img_dir, 'vis_label_{:03d}.png'.format(i))
            render_name = os.path.join(img_dir, 'rgb_{:03d}.png'.format(i))

        if not os.path.exists(albedo_name) or not os.path.exists(shading_name) or not os.path.exists(residual_name) or not os.path.exists(label_name) or not os.path.exists(semantic_name) or not os.path.exists(render_name):
            break
        render = cv2.imread(render_name)
        renders.append(render)

        albedo = cv2.imread(albedo_name)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        albedo = (np.array(albedo) / 255.).astype(np.float32)
        albedos.append(albedo[None,:])

        shading = cv2.imread(shading_name, cv2.IMREAD_GRAYSCALE)
        shading = (np.array(shading) / 255.).astype(np.float32)
        shadings.append(np.repeat(shading[None,:,:,None],3,axis=3))

        residual = cv2.imread(residual_name)
        residual = cv2.cvtColor(residual, cv2.COLOR_BGR2RGB)
        residual = (np.array(residual) / 255.).astype(np.float32)
        residuals.append(residual[None,:])

        semantic = cv2.imread(semantic_name)
        semantics.append(semantic)


        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label = np.array(label).astype(np.long)
        #label = (np.array(label) > 0).astype(np.long)
        labels.append(label[None,:,:,None])
        print(i,albedo_name)
        i += 1
    albedos = np.concatenate(albedos, 0)
    shadings = np.concatenate(shadings, 0)
    residuals = np.concatenate(residuals, 0)
    labels = np.concatenate(labels, 0)
    print("albedos:",albedos.shape,albedos.dtype)
    print("shadings:",shadings.shape,shadings.dtype)
    print("residuals:",residuals.shape,residuals.dtype)
    print("labels:",labels.shape,labels.dtype)
    imgs = albedos*shadings+residuals
    print("imgs:",imgs.shape,imgs.dtype)

    albedos_gpu = torch.from_numpy(albedos).to(device)
    shadings_gpu = torch.from_numpy(shadings).to(device)
    residuals_gpu = torch.from_numpy(residuals).to(device)
    labels_gpu = torch.from_numpy(labels).to(device)

    return imgs, albedos, shadings, residuals, labels, albedos_gpu, shadings_gpu, residuals_gpu, labels_gpu, semantics, renders

def numpy_to_photo(img):
    img = to8b(img)
    im = Image.fromarray(img)
    photo = ImageTk.PhotoImage(image=im)
    return photo

def switch():
    global is_play
    is_play = not is_play

def cluster_img(cluster_manager, albedos_gpu, labels_gpu):
    ori_shape = (albedos_gpu.shape[0], albedos_gpu.shape[1], albedos_gpu.shape[2],1)
    pixel = albedos_gpu.reshape(-1,3)
    cluster_albedo_class = cluster_manager.dest_class(pixel, labels_gpu.reshape(-1,1))#.cpu().numpy()
    rgb_centers = []
    for cluster in cluster_manager.clusters:
        if cluster is None:
            rgb_centers.append(None)
            print("None")
        else:
            rgb_centers.append(cluster.rgb_centers)#.cpu().numpy())
            print("rgb_centers",rgb_centers[-1].shape)
    
    return cluster_albedo_class.reshape(ori_shape), rgb_centers
    

    edit_imgs = cluster_albedo*shadings_gpu.reshape(-1,1) + residuals_gpu.reshape(-1,3)
    print(edit_imgs.shape)
    edit_imgs = edit_imgs.reshape(albedos_gpu.shape).cpu().numpy()
    return edit_imgs

def draw_X(center_x,center_y):
    global color_img, curr_color_img
    curr_color_img = color_img.copy()

    for bias_x in range(-5,6):
        for bias_y in range(-1,2):
            x = center_x+bias_x
            y = center_y+bias_y
            if x<0 or x>=H or y <0 or y>=W or (abs(bias_x)<=1 and abs(bias_y)<=1):
                continue
            curr_color_img[y,x] = 0,0,0
    
    for bias_x in range(-1,2):
        for bias_y in range(-5,6):
            x = center_x+bias_x
            y = center_y+bias_y
            if x<0 or x>=H or y <0 or y>=W or (abs(bias_x)<=1 and abs(bias_y)<=1):
                continue
            curr_color_img[y,x] = 0,0,0
    update_color_img()

    


def save_cluster_config():
    global rgb_centers
    device = cluster_manager.rgb_centers.device
    cluster_manager.rgb_centers = torch.from_numpy(rgb_centers).to(device)
    filename = filedialog.askdirectory()
    print("save to:",filename)
    cluster_manager.save(filename)
    return filename

def get_cluster_num(event):
    global curr_class_idx, curr_semantic_idx
    global curr_h, curr_s, curr_i
    choose_img_i = curr_img_idx
    x, y = event.x, event.y
    curr_class_idx = cluster_albedo_class[curr_img_idx,y,x]
    curr_semantic_idx = labels[curr_img_idx,y,x,0]
    print("img idx:",choose_img_i, ' semantic_idx:', curr_semantic_idx, " albedo class:",curr_class_idx, imgs[curr_img_idx,y,x])
    r = rgb_centers[curr_semantic_idx][curr_class_idx,0]
    g = rgb_centers[curr_semantic_idx][curr_class_idx,1]
    b = rgb_centers[curr_semantic_idx][curr_class_idx,2]
    curr_h, curr_i, curr_s = colorsys.rgb_to_hls(r,g,b)
    draw_X(int(curr_h*float(H)),W-int(curr_s*float(W)))
    i_Slider.set(float(curr_i*255.0))

def reset_rgb_center():
    global rgb_centers, ori_rgb_centers
    global curr_img_idx
    global global_shading_scale, global_residual_scale
    global global_change_shading, global_change_residual
    rgb_centers = copy.deepcopy(ori_rgb_centers)
    global_change_shading = False
    global_change_residual = False
    update_img(curr_img_idx)
    rSlider.set(float(rgb_centers[curr_semantic_idx][curr_class_idx,0]*255))
    gSlider.set(float(rgb_centers[curr_semantic_idx][curr_class_idx,1]*255))
    bSlider.set(float(rgb_centers[curr_semantic_idx][curr_class_idx,2]*255))
    global_shading_scale = 1.0
    global_residual_scale = 1.0
    s_r_Slider.set(100)
    s_s_Slider.set(100)


def update_show_mode():
    global curr_img_idx
    update_img(curr_img_idx)
    return

def draw_color_label(W,H):
    label = np.zeros(shape=(W,H,3), dtype=np.uint8)
    for w in range(W):
        for h in range(H):
            h_c = float(h)/float(H)
            s_c = float(W-w)/float(W)
            r,g,b = colorsys.hls_to_rgb(h_c,0.5,s_c)
            label[w,h,0] = int(r*255.0)
            label[w,h,1] = int(g*255.0)
            label[w,h,2] = int(b*255.0)
    #cv2.imwrite('label.png',label)
    return label

def pick_color(event):
    global curr_class_idx, curr_semantic_idx
    global curr_h, curr_s, curr_i
    x, y = event.x, event.y
    curr_h = float(x)/float(H)
    curr_s = float(W-y)/float(W)
    r,g,b = colorsys.hls_to_rgb(curr_h,curr_i,curr_s) 
    r = int(r*255.0)
    g = int(g*255.0)
    b = int(b*255.0)
    update_color(r, g, b)
    draw_X(x, y)
    return

def update_i(source):
    global curr_h, curr_s, curr_i
    curr_i = float(i_Slider.get())/255.0
    r,g,b = colorsys.hls_to_rgb(curr_h,curr_i,curr_s) 
    r = int(r*255.0)
    g = int(g*255.0)
    b = int(b*255.0)
    update_color(r, g, b)
    return

def record_frame():
    global curr_result, curr_albedo, curr_shading, curr_residual, curr_c_albedo, curr_show, curr_color_img, curr_class_color, curr_semantic, curr_render
    global result_buffer, albedo_buffer, shading_buffer, residual_buffer, color_buffer,c_albedo_buffer, show_buffer, class_color_buffer

    curr_time = datetime.datetime.now()
    folder_name = datetime.datetime.strftime(curr_time,'frame_%m-%d_%H-%M-%S')
    if args.head_name != '':
            folder_name = args.head_name + '_' + folder_name
    folder_name = os.path.join('gui_out',folder_name)
    os.makedirs(folder_name)
    print("save to:",folder_name)
    cv2.imwrite(os.path.join(folder_name, 'result'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_result), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'albedo'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_albedo), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'shading'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_shading), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'residual'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_residual), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'c_albedo'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_c_albedo.cpu().numpy()), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'show'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_show), cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_name, 'color'+str(curr_img_idx)+'.png'),cv2.cvtColor(curr_color_img, cv2.COLOR_BGR2RGB))

    curr_class_color_img = np.ones(shape=(100,100,3))*curr_class_color
    cv2.imwrite(os.path.join(folder_name, 'class_color'+str(curr_img_idx)+'.png'),cv2.cvtColor(to8b(curr_class_color_img), cv2.COLOR_BGR2RGB))

    cv2.imwrite(os.path.join(folder_name, 'semantic'+str(curr_img_idx)+'.png'),curr_semantic)
    cv2.imwrite(os.path.join(folder_name, 'render'+str(curr_img_idx)+'.png'),curr_render)

def record_video():
    global is_record
    global result_buffer, albedo_buffer, shading_buffer, residual_buffer, color_buffer,c_albedo_buffer, show_buffer, class_color_buffer, semantic_buffer
    if not is_record:
        print("start recording...")
        result_buffer = []
        albedo_buffer = []
        shading_buffer = []
        residual_buffer = []
        color_buffer = []
        is_record = True
    else:
        curr_time = datetime.datetime.now()
        folder_name = datetime.datetime.strftime(curr_time,'video_%m-%d_%H-%M-%S')
        if args.head_name != '':
            folder_name = args.head_name + '_' + folder_name
        folder_name = os.path.join('gui_out',folder_name)
        os.makedirs(folder_name)
        print("end recording...")
        print("save to:",folder_name)
        is_record = False
        print(albedo_buffer[0].shape[0], albedo_buffer[0].shape[1])
        print(len(albedo_buffer), len(color_buffer))
        format_tile = '.mp4'#'.avi'
        format_name = 'mp4v'#'MJPG'
        result_video = cv2.VideoWriter(os.path.join(folder_name, 'result'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(result_buffer[0].shape[1], result_buffer[0].shape[0]))
        albedo_video = cv2.VideoWriter(os.path.join(folder_name, 'albedo'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(albedo_buffer[0].shape[1], albedo_buffer[0].shape[0]))
        shading_video = cv2.VideoWriter(os.path.join(folder_name, 'shading'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(shading_buffer[0].shape[1], shading_buffer[0].shape[0]))
        residual_video = cv2.VideoWriter(os.path.join(folder_name, 'residual'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(residual_buffer[0].shape[1], residual_buffer[0].shape[0]))
        color_video = cv2.VideoWriter(os.path.join(folder_name, 'color'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(color_buffer[0].shape[1], color_buffer[0].shape[0]))
        c_albedo_video = cv2.VideoWriter(os.path.join(folder_name, 'c_albedo'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(c_albedo_buffer[0].shape[1], c_albedo_buffer[0].shape[0]))
        show_video = cv2.VideoWriter(os.path.join(folder_name, 'show'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(show_buffer[0].shape[1], show_buffer[0].shape[0]))
        class_color_video = cv2.VideoWriter(os.path.join(folder_name, 'class_color'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(100,100))
        semantic_video = cv2.VideoWriter(os.path.join(folder_name, 'semantic'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(semantic_buffer[0].shape[1], semantic_buffer[0].shape[0]))
        render_video = cv2.VideoWriter(os.path.join(folder_name, 'render'+format_tile),cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(render_buffer[0].shape[1], render_buffer[0].shape[0]))
        for i in range(len(color_buffer)):
            result_video.write(cv2.cvtColor(result_buffer[i], cv2.COLOR_BGR2RGB))
            albedo_video.write(cv2.cvtColor(albedo_buffer[i], cv2.COLOR_BGR2RGB))
            shading_video.write(cv2.cvtColor(shading_buffer[i], cv2.COLOR_BGR2RGB))
            residual_video.write(cv2.cvtColor(residual_buffer[i], cv2.COLOR_BGR2RGB))
            color_video.write(cv2.cvtColor(color_buffer[i], cv2.COLOR_BGR2RGB))
            c_albedo_video.write(cv2.cvtColor(c_albedo_buffer[i], cv2.COLOR_BGR2RGB))
            show_video.write(cv2.cvtColor(show_buffer[i], cv2.COLOR_BGR2RGB))
            curr_class_color_img = np.ones(shape=(100,100,3))*class_color_buffer[i]
            class_color_video.write(cv2.cvtColor(to8b(curr_class_color_img), cv2.COLOR_BGR2RGB))
            semantic_video.write(semantic_buffer[i])
            render_video.write(render_buffer[i])

        result_video.release()
        albedo_video.release()
        shading_video.release()
        residual_video.release()
        color_video.release()
        c_albedo_video.release()
        show_video.release()
        
def scale_shading(source):
    global global_shading_scale
    global_shading_scale = float(s_s_Slider.get())/100.0
    update_img(curr_img_idx)
    return

def scale_residual(source):
    global global_residual_scale
    global_residual_scale = float(s_r_Slider.get())/100.0
    update_img(curr_img_idx)
    return

def t_shading(s):
    global global_change_shading
    if global_change_shading:
        return s**2
        #return ( torch.pow(torch.sin(s*torch.pi-torch.pi/2), 2)+1)/2
        return (torch.sin(s*torch.pi-torch.pi/2)+1)/2
    return s

def t_residual(r):
    global global_change_residual
    if global_change_residual:
        return (torch.sin(r*torch.pi-torch.pi/2)+1)/2
    return r

def f_shading():
    global global_change_shading
    global_change_shading = not global_change_shading
    update_img(curr_img_idx)
    return

def f_residual():
    global global_change_residual
    global_change_residual = not global_change_residual
    update_img(curr_img_idx)
    return        

        

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()

    cluster_manager = Cluster_Manager(cluster_config_file = args.cluster_config)
    print("load cluster config:",args.cluster_config)
    root = init_window()
    ori_imgs, albedos, shadings, residuals, labels, albedos_gpu, shadings_gpu, residuals_gpu, labels_gpu, semantics, renders = load_all_image(args.img_dir,torch.device('cuda'),args.replica)
    FRAME_RATE = args.rate
    #imgs = cluster_img(cluster, albedos_gpu, shadings_gpu, residuals_gpu)
    cluster_albedo_class, rgb_centers = cluster_img(cluster_manager, albedos_gpu, labels_gpu)
    print(rgb_centers)
    tic = time.time()
    cluster_albedo = albedos_gpu.clone()
    for i in range(len(rgb_centers)):
        if rgb_centers[i] is None:
                continue
        class_idx = torch.squeeze(labels_gpu==i)
        albedo_class = cluster_albedo_class[class_idx]
        cluster_albedo[class_idx] = torch.squeeze(rgb_centers[i][albedo_class])
    imgs = cluster_albedo*shadings_gpu+residuals_gpu
    imgs = imgs.cpu().numpy()
    toc = time.time()
    print("cluster time:",toc-tic)
    #os._exit(0)
    ori_rgb_centers = copy.deepcopy(rgb_centers)

    curr_result = imgs[0]
    curr_albedo = albedos[0]
    curr_shading = shadings[0]
    curr_residual = residuals[0]
    curr_c_albedo = cluster_albedo[0]
    curr_show = imgs[0]
    curr_class_color = np.ones(3)
    curr_semantic = semantics[0]
    curr_render = renders[0]
    
    color_buffer = []
    result_buffer = []
    albedo_buffer = []
    shading_buffer = []
    residual_buffer = []
    c_albedo_buffer = []
    show_buffer = []
    class_color_buffer = []
    semantic_buffer = []
    render_buffer = []


    photo = numpy_to_photo(imgs[0])
    lab =Label(root,image=photo)
    lab.borderwidth = 0
    #lab.pack(side="top")
    lab.place(x = 90, y = 130)

    #color_lab = Label(root, height=3, width=6, bg='white')
    color_lab = Label(root, bg='white')
    #color_lab.pack(side = "top")
    color_lab.place(x = 220, y = 400, height = 60, width = 60)
    #color_lab.place(x = 300, y = 465, height = 35, width = 70)


    cc = Frame(root)
    cc.place(y = 400, x = 50)
    #cc.pack(side=TOP, anchor=NW)
    rSlider = Scale(cc, from_=0, to=255, troughcolor="pink", command=bgUpdate)
    gSlider = Scale(cc, from_=0, to=255, troughcolor="lightgreen", command=bgUpdate)
    bSlider = Scale(cc, from_=0, to=255, troughcolor="lightblue", command=bgUpdate)
    
    is_play = False
    stop_btn = Button(root, text="▶", command=switch)
    stop_btn = stop_btn.pack(side="bottom")

    save_btn = Button(root, text="save", command=save_cluster_config)
    save_btn.place(x = 300, y = 465, height = 35, width = 70)
    #save_btn = save_btn.pack(side="bottom")

    reset_btn = Button(root, text="reset", command=reset_rgb_center)
    reset_btn.place(x = 380, y = 465, height = 35, width = 70)

    record_btn = Button(root, text="●", command=record_video)
    record_btn.place(x = 460, y = 465, height = 35, width = 35)

    frame_btn = Button(root, text="f", command=record_frame)
    frame_btn.place(x = 460, y = 425, height = 35, width = 35)

    show_status = IntVar()
    show_status.set(0)

    Radiobutton(root, text="render",bg='white',highlightbackground='white',command = update_show_mode,variable=show_status, value=0).place(x = 300, y = 405, height = 30, width = 70)
    Radiobutton(root, text="albedo",bg='white',highlightbackground='white',command = update_show_mode,variable=show_status, value=1).place(x = 380, y = 405, height = 30, width = 70)
    Radiobutton(root, text="shading",bg='white',highlightbackground='white',command = update_show_mode,variable=show_status, value=2).place(x = 300, y = 435, height = 30, width = 70)
    Radiobutton(root, text="residual",bg='white',highlightbackground='white',command = update_show_mode,variable=show_status, value=3).place(x = 380, y = 435, height = 30, width = 70)

    #reset_btn = reset_btn.pack(side="bottom")

    #askBtn = Button(root, text="Using\nAskcolor", relief=SOLID, fg="royalblue",
    #                font=("Microsoft Yahei", 8, "normal"), command=askcc)
    
    ############
    W = 100
    H = 100
    color_img = draw_color_label(W,H)
    curr_color_img = color_img.copy()
    color_photo = ImageTk.PhotoImage(image=Image.fromarray(color_img))
    color_choose_label = Label(cc,image=color_photo)
    color_choose_label.borderwidth = 0
   # color_choose_label.pack(side="top")
    curr_h, curr_s, curr_i =0.0, 0.0, 0.0
    i_Slider = Scale(cc, from_=0, to=255, troughcolor="white", command=update_i)
    ############
    color_choose_label.pack(side=LEFT)
    i_Slider.pack(side=LEFT)
    #gSlider.pack(side=LEFT)
    #bSlider.pack(side=LEFT)
    #askBtn.pack(side=RIGHT,anchor=SE, fill=Y)
    curr_img_idx = 0
    curr_class_idx = -1
    curr_semantic_idx = -1

    ##################################transform################################
    global_shading_scale = 1.0
    s_s_Slider = Scale(root, from_=0, to=400, showvalue=0, troughcolor="white", command=scale_shading)
    #s_s_Slider.place(x = 5, y = 245, height = 80, width = 35)
    s_s_Slider.place(x = 460, y = 335, height = 80, width = 35)
    s_s_Slider.set(100)
    global_residual_scale = 1.0
    s_r_Slider = Scale(root, from_=0, to=400, showvalue=0, troughcolor="white", command=scale_residual)
    s_r_Slider.place(x = 5, y = 335, height = 80, width = 35)
    s_r_Slider.set(100)
    global_change_shading = False
    t_s_Btn = Button(root, text="S", command=f_shading)
    t_s_Btn.place(x = 5, y = 465, height = 35, width = 35)
    global_change_residual = False
    t_r_Btn = Button(root, text="R", command=f_residual)
    t_r_Btn.place(x = 5, y = 425, height = 35, width = 35)

    ###########################################################################


    is_record = False
    gettime()



    lab.bind('<Button-1>', get_cluster_num)
    color_choose_label.bind('<Button-1>', pick_color)
    root.mainloop()