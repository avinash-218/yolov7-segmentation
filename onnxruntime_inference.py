import cv2, os
import numpy as np
import onnxruntime
from util import *

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 640
result_path = "./result"
image_path = "./data/images/3.png"
model_name = 'walls'
model_path = "./"
ONNX_MODEL = f"walls.onnx"
video_path = "test.mp4"
CLASSES = ['wall']

sess = onnxruntime.InferenceSession(ONNX_MODEL)
isExist = os.path.exists(result_path)
if not isExist:
    os.makedirs(result_path)

image_3c = cv2.imread(image_path)
print('original shape', image_3c.shape)
image_4c, image_3c = preprocess(image_3c, input_height, input_width)
print('preprocessed shape', image_3c.shape, image_4c.shape)
outputs = sess.run(['output', 'onnx::Slice_533', 'onnx::Slice_642', 'onnx::Slice_748', '517'],{"images": image_4c.astype(np.float32)}) # (1, 3, input height, input width)
colorlist = gen_color(len(CLASSES))
outputs = [outputs[0], outputs[4]]
results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres) ##[box,mask,shape]
results = results[0]              ## batch=1
boxes, masks, shape = results
if isinstance(masks, np.ndarray):
    mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
    print('final images', mask_img.shape, vis_img.shape)
    print('--> Save inference result')
else:
    print("No segmentation result")


print("ONNX inference finish")
cv2.destroyAllWindows()


