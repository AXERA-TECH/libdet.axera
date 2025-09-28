import os
from pydet import AXDet, ModelType
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_type', type=int)
    parser.add_argument('--image', type=str)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_kpt', type=int, default=0)
    parser.add_argument('--output', type=str, default='results.jpg')
    
    args = parser.parse_args()


    # 枚举设备
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
    
     # 加载图像
    img = cv2.imread(args.image)
    img_infer = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
     # 推理
    results = det.detect(img_infer)
    for obj in results:
        box = obj.box
        score = obj.score
        label = obj.label
        kpts = obj.kpts
        
        cv2.rectangle(img, box, (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {score:.2f}", (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if args.num_kpt > 0:
            for i in range(args.num_kpt):
                cv2.circle(img, kpts[i], 5, (0, 0, 255), -1)
    cv2.imwrite(args.output, img)
    del det

    if devices_info['host']['available']:
        sys_deinit(AxDeviceType.host_device, -1)
    elif devices_info['devices']['count'] > 0:
        sys_deinit(AxDeviceType.axcl_device, 0)