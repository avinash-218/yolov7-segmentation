import cv2
import os
import numpy as np
import onnxruntime
from pathlib import Path
import torch
from util import preprocess, non_max_suppression, Annotator, Colors, process_mask, scale_coords, plot_masks, scale_masks, save_one_box
colors = Colors()

conf_thres = 0.25
iou_thres = 0.45
input_width = 1024
input_height = 1024
result_path = "./result"
image_path = "./data/images/20.png"
model_name = 'walls'
model_path = "./"
ONNX_MODEL = f"walls.onnx"
video_path = "test.mp4"
CLASSES = ['wall']
classes = None
agnostic_nms = False
max_det = 1000
save_crop = False
save_img = True
line_thickness = 3
hide_labels = False
hide_conf = False
names = 'walls'

sess = onnxruntime.InferenceSession(ONNX_MODEL)
isExist = os.path.exists(result_path)
if not isExist:
    os.makedirs(result_path)

orig = cv2.imread(image_path)   #read image
image_4c, image_3c = preprocess(orig, input_height, input_width)
outputs = sess.run(['output', 'onnx::Slice_533', 'onnx::Slice_642', 'onnx::Slice_748', '517'],{"images": image_4c.astype(np.float32)}) # (1, 3, input height, input width)

pred, proto = outputs[0], outputs[4]  #(1, 25200, 38) (1, 32, 160, 160)

pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

for i, det in enumerate(pred):  # per image
    p, im0 = image_path, orig.copy()

    p = Path(p)  # to Path
    save_path = str(f'._{i}')  # im.jpg
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    imc = im0.copy() if save_crop else im0  # for save_crop
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    if len(det):
        masks = process_mask(proto[i], det[:, 6:], det[:, :4], image_4c.shape[2:], upsample=True)  # HWC

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(image_4c.shape[2:], det[:, :4], im0.shape).round()

        # Mask plotting ----------------------------------------------------------------------------------------
        mcolors = [colors(int(cls), True) for cls in det[:, 5]]
        im_masks = plot_masks(image_4c[i], masks, mcolors)  # image with masks shape(imh,imw,3)
        annotator.im = scale_masks(image_4c.shape[2:], im_masks, im0.shape)  # scale to original h, w
        # Mask plotting ----------------------------------------------------------------------------------------

        # Write results
        orig_image = cv2.imread(image_path)
        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
            x_min, y_min, x_max, y_max = xyxy
            cv2.rectangle(orig_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Draw rectangle
            if save_img or save_crop:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file='.' / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        plt_name = 'Output/' + os.path.basename(image_path)
        cv2.imwrite(plt_name, orig_image)

    # Stream results
    im0 = annotator.result()

    # Save results (image with detections)
    if save_img:
        print('final image', im0.shape)
        cv2.imwrite('result/'+image_path.split('/')[-1], im0)
