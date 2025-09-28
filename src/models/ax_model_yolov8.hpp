#pragma once
#include "ax_model_base.hpp"
#include "ax_det_common.hpp"

class ax_model_yolov8 : public ax_model_base
{
protected:
    int postprocess(ax_det_img_t *img, ax_det_result_t *result) override
    {
        std::vector<Object> proposals;
        for (int i = 0; i < 3; ++i)
        {
            auto feat_ptr = (float *)m_runner->get_output(i).pVirAddr;
            int32_t stride = strides[i];
            generate_proposals_yolov8_nhwc(stride, feat_ptr, threshold, proposals, input_w, input_h, num_classes);
        }
        std::vector<Object> objects_vec;
        get_out_bbox(proposals, objects_vec, threshold, input_h, input_w, img->height, img->width);

        result->num_objs = std::min((int)objects_vec.size(), AX_DET_MAX_OBJ_NUM);
        for (int i = 0; i < result->num_objs; i++)
        {
            result->objects[i].label = objects_vec[i].label;
            result->objects[i].score = objects_vec[i].prob;
            result->objects[i].box.x = objects_vec[i].rect.x;
            result->objects[i].box.y = objects_vec[i].rect.y;
            result->objects[i].box.w = objects_vec[i].rect.width;
            result->objects[i].box.h = objects_vec[i].rect.height;
        }
        return ax_det_errcode_success;
    }
};
typedef ax_model_yolov8 ax_model_yolo11;
REGISTER(ax_det_model_type_yolov8, ax_model_yolov8)
REGISTER(ax_det_model_type_yolo11, ax_model_yolo11)

class ax_model_yolov8pose : public ax_model_base
{
protected:
    int postprocess(ax_det_img_t *img, ax_det_result_t *result) override
    {
        std::vector<Object> proposals;
        for (int i = 0; i < 3; ++i)
        {
            // auto feat_ptr = (float *)m_runner->get_output(i).pVirAddr;
            auto feat_ptr = (float *)m_runner->get_output(i).pVirAddr;
            auto feat_kps_ptr = (float *)m_runner->get_output(i + 3).pVirAddr;
            int32_t stride = strides[i];
            // generate_proposals_yolov8_pose(stride, feat_ptr, threshold, proposals, input_w, input_h, num_classes);
            generate_proposals_yolov8_pose(stride, feat_ptr, feat_kps_ptr, threshold, proposals, input_w, input_h, num_kpt);
        }
        std::vector<Object> objects_vec;
        get_out_bbox(proposals, objects_vec, threshold, input_h, input_w, img->height, img->width);

        result->num_objs = std::min((int)objects_vec.size(), AX_DET_MAX_OBJ_NUM);
        for (int i = 0; i < result->num_objs; i++)
        {
            result->objects[i].label = objects_vec[i].label;
            result->objects[i].score = objects_vec[i].prob;
            result->objects[i].box.x = objects_vec[i].rect.x;
            result->objects[i].box.y = objects_vec[i].rect.y;
            result->objects[i].box.w = objects_vec[i].rect.width;
            result->objects[i].box.h = objects_vec[i].rect.height;
            result->objects[i].num_kpt = num_kpt;
            for (int j = 0; j < num_kpt; j++)
            {
                result->objects[i].kpts[j].x = objects_vec[i].kpt[j].x;
                result->objects[i].kpts[j].y = objects_vec[i].kpt[j].y;
            }
        }
        return ax_det_errcode_success;
    }
};
typedef ax_model_yolov8pose ax_model_yolo11pose;
REGISTER(ax_det_model_type_yolov8_pose, ax_model_yolov8pose)
REGISTER(ax_det_model_type_yolo11_pose, ax_model_yolo11pose)
