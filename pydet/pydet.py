import ctypes
import os
from typing import List, Tuple
import numpy as np
import platform
from pyaxdev import _lib, AxDeviceType, AxDevices, check_error

class ModelType(ctypes.c_int):
    ax_det_model_type_unknown = -1
    ax_det_model_type_yolov5 = 0
    ax_det_model_type_yolov8 = 1
    ax_det_model_type_yolov8_pose = 2
    ax_det_model_type_yolo11 = 3
    ax_det_model_type_yolo11_pose = 4

class DetInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
        ('devid', ctypes.c_char),
        ('model_type', ModelType),
        ('model_path', ctypes.c_char * 256),
        ('num_classes', ctypes.c_int),
        ('num_kpt', ctypes.c_int),
        ('threshold', ctypes.c_float),
        ('mean', ctypes.c_float * 3),
        ('std', ctypes.c_float * 3),
    ]

class DetImage(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('channels', ctypes.c_int),
        ('stride', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ]


class ObjectItem(ctypes.Structure):
    _fields_ = [
        ('box', ctypes.c_int * 4),
        ('kpts', ctypes.c_int * 2 * 32),
        ('num_kpt', ctypes.c_int),
        ('score', ctypes.c_float),
        ('label', ctypes.c_int),
    ]
    
class ObjectResult(ctypes.Structure):
    _fields_ = [
        ('objects', ObjectItem * 64),
        ('num_objs', ctypes.c_int),
    ]

_lib.ax_det_init.argtypes = [ctypes.POINTER(DetInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.ax_det_init.restype = ctypes.c_int

_lib.ax_det_deinit.argtypes = [ctypes.c_void_p]
_lib.ax_det_deinit.restype = ctypes.c_int

_lib.ax_det.argtypes = [ctypes.c_void_p, ctypes.POINTER(DetImage), ctypes.POINTER(ObjectResult)]
_lib.ax_det.restype = ctypes.c_int

class Object:
    def __init__(self, box: List[int], score: float, label: int, kpts: List[int] = []):
        self.box = box
        self.score = score
        self.label = label
        self.kpts = kpts

    def __repr__(self):
        return f"Object(box={self.box}, score={self.score:.2f}, label={self.label}, kpts={self.kpts})"
    
class AXDet:
    def __init__(self, model_path: str, model_type: ModelType, num_classes: int, 
                 num_kpt: int = 0, 
                 threshold: float= 0.25, 
                 mean: List[float] = [0,0,0], std: List[float] = [1,1,1],
                 dev_type: AxDeviceType = AxDeviceType.axcl_device,
                 devid: int = 0):
        self.handle = None
        self.init_info = DetInit()
        
        # 设置初始化参数
        self.init_info.dev_type = dev_type
        self.init_info.devid = devid
        
        # 设置路径
        self.init_info.model_type = model_type
        self.init_info.model_path = model_path.encode('utf-8')
        self.init_info.num_classes = num_classes
        self.init_info.num_kpt = num_kpt
        self.init_info.threshold = threshold
        for i in range(3):
            self.init_info.mean[i] = mean[i]
            self.init_info.std[i] = std[i]
        
        # 创建CLIP实例
        handle = ctypes.c_void_p()
        check_error(_lib.ax_det_init(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.ax_det_deinit(self.handle)

    def detect(self, image_data: np.ndarray):
      
        image = DetImage()
        image.data = ctypes.cast(image_data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))
        image.width = image_data.shape[1]
        image.height = image_data.shape[0]
        image.channels = image_data.shape[2]
        image.stride = image_data.shape[1] * image_data.shape[2]
        result = ObjectResult()
        check_error(_lib.ax_det(self.handle, ctypes.byref(image), ctypes.byref(result)))
        objects = []
        for i in range(result.num_objs):
            _obj = result.objects[i]
            
            obj = Object(
                box=[_obj.box[0], _obj.box[1], _obj.box[2], _obj.box[3]],
                score=_obj.score,
                label=_obj.label,
                kpts=[(_obj.kpts[j][0], _obj.kpts[j][1]) for j in range(_obj.num_kpt)],
            )
            objects.append(obj)
        return objects

