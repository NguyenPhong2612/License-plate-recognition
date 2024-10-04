import torch 
import torch.nn as nn
from parseq.system import System
import yaml
import cv2
from PIL import Image
from wpodnet.lib_detection import load_model_wpod, detect_lp
import numpy as np
import gradio as gr 
from torchvision import transforms as T
import matplotlib.pyplot as plt
trans = T.Compose([
            T.Resize((224, 224), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'weights/best.ckpt'
config_path = 'parseq/config.yaml'
wpod_path = 'weights/wpod-net.h5'
wpod_net = load_model_wpod(wpod_path)

with open(config_path, 'r') as data:
    config = yaml.safe_load(data)
system = System(config)
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
system.load_state_dict(checkpoint['state_dict'])
system.to(device)

def predict(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    
    draw_image = image.copy()
    _, img_wapred, _, bounding_boxes = detect_lp(wpod_net, image, 0.5)
    if len(img_wapred) == 0:
        return "Can not detect license plate from image"
    else:
        system.eval()
        bounding_boxes = np.array(bounding_boxes).astype(int)
        for i in range(len(img_wapred)):
            img =  (img_wapred[i] * 255).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            image = trans(img).unsqueeze(0)
            with torch.no_grad():
                pred = system(image).softmax(-1)
            generated_text, _ = system.tokenizer.decode(pred)
            if len(generated_text[0]) >= 5:
                points = bounding_boxes[i]
                cv2.polylines(draw_image, [points], isClosed = True, color = (0, 255, 0), thickness = 2)
                position = (points[:, 0].min(), points[:, 1].min())
                cv2.putText(draw_image, generated_text[0], position, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color=(255, 255, 0), thickness = 2)
        return draw_image

interface = gr.Interface(
    fn = predict,
    inputs =[gr.components.Image()],
    outputs=[gr.components.Image()])
interface.launch(share = True, debug = True)
