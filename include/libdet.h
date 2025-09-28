#ifndef __LIBDET_H__
#define __LIBDET_H__
#include "ax_devices.h"

#if defined(__cplusplus)
extern "C"
{
#endif
#define AX_DET_MAX_OBJ_NUM 64
#define AX_DET_MAX_KPT_NUM 32
    typedef enum
    {
        ax_det_errcode_failed = -1,
        ax_det_errcode_success = 0,
    } ax_det_errcode_e;

    typedef enum
    {
        ax_det_model_type_unknown = -1,
        ax_det_model_type_yolov5,
        ax_det_model_type_yolov8,
        ax_det_model_type_yolov8_pose,
        ax_det_model_type_yolo11,
        ax_det_model_type_yolo11_pose,
    } ax_det_model_type_e;

    typedef struct
    {
        int width;
        int height;
        int channels;
        int stride;
        void *data;
    } ax_det_img_t;

    typedef struct
    {
        struct
        {
            int x, y, w, h;
        } box;

        struct
        {
            int x, y;
        } kpts[AX_DET_MAX_KPT_NUM];
        int num_kpt;

        float score;
        int label;
    } ax_det_obj_t;

    typedef struct
    {
        ax_det_obj_t objects[AX_DET_MAX_OBJ_NUM];
        int num_objs;
    } ax_det_result_t;

    typedef struct
    {
        ax_devive_e dev_type; // Device type
        char devid;           // axcl device ID

        ax_det_model_type_e model_type;
        char model_path[256];

        int num_classes;
        int num_kpt; // for face/pose

        float threshold;

        // int anchors[18]; // for yolov5
        // int strides[3];

        float mean[3]; // for nchw float input model, not suggestion
        float std[3];  // for nchw float input model, not suggestion
    } ax_det_init_t;

    typedef void *ax_det_handle_t;

    int ax_det_init(ax_det_init_t *init, ax_det_handle_t *handle);
    int ax_det_deinit(ax_det_handle_t handle);

    int ax_det(ax_det_handle_t handle, ax_det_img_t *img, ax_det_result_t *result);

#if defined(__cplusplus)
}
#endif
#endif
