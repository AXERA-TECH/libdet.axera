#pragma once
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "libdet.h"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "object_register.hpp"

class ax_model_base
{
protected:
    ax_det_model_type_e model_type = ax_det_model_type_unknown;

    std::shared_ptr<ax_runner_base> m_runner = nullptr;
    cv::Mat m_image_input, m_image_input_letterbox;

    float threshold = 0.25f;
    float nms_threshold = 0.45f;
    int num_classes = 80;
    int num_kpt = 0;
    std::vector<float> anchors = {12, 16, 19, 36, 40, 28,
                                  36, 75, 76, 55, 72, 146,
                                  142, 110, 192, 243, 459, 401};
    std::vector<int> strides = {8, 16, 32};

    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> std = {1.0f, 1.0f, 1.0f};

    bool is_input_nchw = false;
    bool is_input_fp32 = false;
    int input_w = 640;
    int input_h = 640;

    virtual int preprocess(ax_det_img_t *img);
    virtual int postprocess(ax_det_img_t *img, ax_det_result_t *result) = 0;

    std::mutex m_mutex;

public:
    ax_model_base() = default;
    virtual ~ax_model_base() = default;

    virtual ax_det_errcode_e init(ax_det_init_t *init_info);
    virtual ax_det_errcode_e deinit();
    virtual ax_det_errcode_e inference(ax_det_img_t *img, ax_det_result_t *result)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        int ret = preprocess(img);
        if (ret != ax_det_errcode_success)
        {
            printf("preprocess failed\n");
            return ax_det_errcode_failed;
        }
        ret = m_runner->inference();
        if (ret != 0)
        {
            printf("inference failed\n");
            return ax_det_errcode_failed;
        }
        ret = postprocess(img, result);
        if (ret != ax_det_errcode_success)
        {
            printf("postprocess failed\n");
            return ax_det_errcode_failed;
        }
        return ax_det_errcode_success;
    }
};