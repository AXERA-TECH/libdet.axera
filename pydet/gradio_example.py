import os
import gradio as gr
from pydet import AXDet, ModelType
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
import argparse
import subprocess
import re

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_type', type=int)
    parser.add_argument('--image', type=str)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_kpt', type=int, default=0)
    args = parser.parse_args()

    # 初始化
    dev_type = AxDeviceType.unknown_device
    dev_id = -1
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
        dev_type = AxDeviceType.host_device
        dev_id = -1
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
        dev_type = AxDeviceType.axcl_device
        dev_id = 0
    else:
        raise Exception("No available device")

 
    det = AXDet(
        model_path=args.model,
        model_type=args.model_type,
        threshold=args.threshold,
        num_classes=args.num_classes,
        num_kpt=args.num_kpt,
        dev_type=dev_type,
        devid=dev_id,
    )
    
    def detect_image(img):
        results = det.detect(img)
        print(results)
        for obj in results:
            box = obj.box
            score = obj.score
            label = obj.label
            kpts = obj.kpts
            
            cv2.rectangle(img, box, (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {score:.2f}", (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if args.num_kpt > 0:
                for i in range(args.num_kpt):
                    cv2.circle(img, kpts[i], 15, (0, 0, 255), -1)
        return img


    # Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 Det Demo")

        with gr.Row():
            input_image = gr.Image(label="输入图像")
            output_image = gr.Image(label="输出图像")
        det_btn = gr.Button("Detect")


        det_btn.click(fn=detect_image, inputs=[input_image], outputs=[output_image])

    # 启动
    ips = get_all_local_ips()
    for ip in ips:
        print(f"* Running on local URL:  http://{ip}:7860")
    ip = "0.0.0.0"
    demo.launch(server_name=ip, server_port=7860)
    
    
    del det
    
    import atexit
    if devices_info['host']['available']:
        atexit.register(lambda: sys_deinit(AxDeviceType.host_device, -1))
    elif devices_info['devices']['count'] > 0:
        atexit.register(lambda: sys_deinit(AxDeviceType.axcl_device, 0))
    
    
