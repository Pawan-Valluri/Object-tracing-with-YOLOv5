import sys
import tkinter
import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use( 'tkagg' )

def print_img(im0):
    matplotlib.use( 'tkagg' )
    plt.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  )
    plt.show()

from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.plots import Annotator, colors, save_one_box
from utils.general import (increment_path, non_max_suppression, check_file, xyxy2xywh, scale_coords, check_imshow, colorstr, strip_optimizer, check_img_size)


@torch.no_grad()
def traffic_main():
    update = False
    hide_labels = False
    view_img = True
    save_conf = False
    nosave = False
    save_txt = False
    visualize = False
    dnn=False
    augment = False
    source = "/mnt/shared/Downloads/road01.mp4"
    # source = "/mnt/shared/work_space/yolov5/data/images/bus.jpg"
    weights = "/mnt/shared/work_space/yolov5s.pt"
    line_thickness = 2
    save_crop = False
    hide_conf = False
    imgsz=640

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH 
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
    project=ROOT / 'runs/detect'
    save_dir = increment_path(Path(project) / "exp", exist_ok=False)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download



    device = select_device()
    half = True if device.type != 'cpu' else False

    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # imgsz=(1, 3, *imgsz)
    model.model.half() if half else model.model.float()
    dt, seen = [0.0, 0.0, 0.0], 0

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs


    def con_3xy_xy3(im):
        ans = []
        # print(im.shape)
        for i in range(im.shape[1]):
            for j in range(im.shape[2]):
                ans.append(im[:,i,j])
        return np.array(ans).reshape(im.shape[1], im.shape[2], im.shape[0])

    for path, im, im0s, vid_cap, s in dataset:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) ==3:
            im = im[None]
        pred = model(im, augment, visualize)
        pred = non_max_suppression(pred,)
        # print(len(pred[0]))
        for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                sys.stdout.write('\r'+f'{s}Done. ({1.2:.3f}s)')
                sys.stdout.flush()

                # Stream results
                im0 = annotator.result()

                if view_img:
                    print_img(im0)
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)



traffic_main()



