#pragma once
#include "ax_model_base.hpp"
#include "ax_det_common.hpp"

class ax_model_yolov5 : public ax_model_base
{
protected:
    int postprocess(ax_det_img_t *img, ax_det_result_t *result) override
    {
        float prob_threshold_unsigmoid = -1.0f * (float)std::log((1.0f / threshold) - 1.0f);
        std::vector<Object> proposals;
        for (int i = 0; i < 3; ++i)
        {
            auto feat_ptr = (float *)m_runner->get_output(i).pVirAddr;
            int32_t stride = strides[i];
            generate_proposals_yolov5(stride, feat_ptr, threshold, proposals, input_w, input_h, anchors.data(), prob_threshold_unsigmoid, num_classes);
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
REGISTER(ax_det_model_type_yolov5, ax_model_yolov5)