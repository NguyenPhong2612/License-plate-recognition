import torch 
import torch.nn as nn
from parseq.system import System
import yaml
import cv2
from parseq.augmentation import trans
from PIL import Image
from wpodnet.lib_detection import load_model_wpod, detect_lp
import numpy as np
import gradio as gr 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'weights/parseq.ckpt'
config_path = 'parseq/config.yaml'
wpod_path = 'weights/wpod-net.h5'
wpod_net = load_model_wpod(wpod_path)

with open(config_path, 'r') as data:
    config = yaml.safe_load(data)
system = System(config)
checkpoint_path = 'weights/parseq.ckpt'
checkpoint = torch.load(checkpoint_path, map_location = 'cuda')
system.load_state_dict(checkpoint['state_dict'])
system.to(device)

def predict(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    _, img_wapred, _, _ = detect_lp(wpod_net, image, 0.5)
    if len(img_warped) == 0:
        return "Can not detect license plate from image"
    else:
        system.eval()
        pred_labels = []
        for i in range(len(img_warped)):
            img =  (img_wapred[i] * 255).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            image = trans(img).unsqueeze(0)
            with torch.no_grad():
                pred = system(image).softmax(-1)
            generated_text, _ = system.tokenizer.decode(pred)
            pred_labels.append(generated_text[0])
        return pred_labels

interface = gr.Interface(
    fn = predict,
    inputs =[gr.components.Image()],
    outputs=[gr.components.Textbox(label = "License plate", lines = 2)])
interface.launch(share = True, debug = True)
