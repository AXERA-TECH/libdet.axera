#include <vector>
#include <memory>

#include "ax_model_base.hpp"
#include "ax_det_common.hpp"
#include "utils/mmap.hpp"

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

int ax_model_base::preprocess(ax_det_img_t *image)
{
    cv::Mat cv_image(image->height, image->width, CV_8UC(image->channels), image->data, image->stride);
    switch (image->channels)
    {
    case 4:
        cv::cvtColor(cv_image, m_image_input, cv::COLOR_BGRA2BGR);
        break;
    case 1:
        cv::cvtColor(cv_image, m_image_input, cv::COLOR_GRAY2BGR);
        break;
    case 3:
        m_image_input = cv_image;
        break;
    default:
        ALOGE("only support channel 1,3,4 uint8 image");
        return ax_det_errcode_failed;
    }
    if (is_input_nchw && is_input_fp32)
    {
        ALOGI("nchw fp32");
        get_input_data_letterbox(m_image_input,
                                 m_image_input_letterbox.data,
                                 input_w,
                                 input_h);

        float *inputPtr = (float *)m_runner->get_input(0).pVirAddr;

        uchar *img_data = m_image_input_letterbox.data;

        int letterbox_cols = input_w;
        int letterbox_rows = input_h;
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = (float(img_data[in_index]) - mean[c]) * std[c];
                }
            }
        }
    }
    else
    {
        get_input_data_letterbox(m_image_input,
                                 (uint8_t *)m_runner->get_input(0).pVirAddr,
                                 input_w,
                                 input_h);
    }

    return ax_det_errcode_success;
}

ax_det_errcode_e ax_model_base::init(ax_det_init_t *init_info)
{
    threshold = init_info->threshold;
    num_classes = init_info->num_classes;
    num_kpt = init_info->num_kpt;
    model_type = init_info->model_type;

    ALOGI("model_type: %d, threshold: %5.2f, nms_threshold: %5.2f, num_classes: %d, num_kpt: %d",
          model_type, threshold, nms_threshold, num_classes, num_kpt);

    memcpy(mean.data(), init_info->mean, sizeof(float) * 3);
    memcpy(std.data(), init_info->std, sizeof(float) * 3);

    ALOGI("mean: %5.2f, %5.2f, %5.2f", init_info->mean[0], init_info->mean[1], init_info->mean[2]);
    ALOGI("std: %5.2f, %5.2f, %5.2f", init_info->std[0], init_info->std[1], init_info->std[2]);

    if (init_info->dev_type == ax_devive_e::host_device)
    {
        if (!get_ax_sys_loader().is_init() || !get_ax_engine_loader().is_init())
        {
            printf("axsys or axengine not init\n");
            return ax_det_errcode_failed;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        if (!getLoader().is_init())
        {
            printf("unsupport axcl\n");
            return ax_det_errcode_failed;
        }

        if (!axcl_Dev_IsInit(init_info->devid))
        {
            printf("axcl device %d not init\n", init_info->devid);
            return ax_det_errcode_failed;
        }
    }
    else
    {
        return ax_det_errcode_failed;
    }

    MMap image_mmap(init_info->model_path);

    if (init_info->dev_type == ax_devive_e::host_device)
    {

        m_runner = std::make_shared<ax_runner_ax650>();
        auto ret = m_runner->init(image_mmap.data(), image_mmap.size(), -1);
        if (ret != 0)
        {
            printf("ax650 init failed\n");
            return ax_det_errcode_failed;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        m_runner = std::make_shared<ax_runner_axcl>();
        auto ret = m_runner->init(image_mmap.data(), image_mmap.size(), init_info->devid);
        if (ret != 0)
        {
            printf("axcl init failed\n");
            return ax_det_errcode_failed;
        }
    }
    else
    {
        printf("unsupport dev type\n");
        return ax_det_errcode_failed;
    }

    is_input_nchw = m_runner->get_input(0).vShape[1] == 3;
    if (is_input_nchw)
    {
        input_w = m_runner->get_input(0).vShape[3];
        input_h = m_runner->get_input(0).vShape[2];
    }
    else
    {
        input_w = m_runner->get_input(0).vShape[2];
        input_h = m_runner->get_input(0).vShape[1];
    }
    m_image_input_letterbox = cv::Mat(input_h, input_w, CV_8UC3);
    ALOGI("input_w: %d, input_h: %d, is_input_nchw: %s", input_w, input_h, is_input_nchw ? "true" : "false");

    int nElements = 1;
    for (int i = 0; i < m_runner->get_input(0).vShape.size(); i++)
    {
        nElements *= m_runner->get_input(0).vShape[i];
    }
    is_input_fp32 = nElements * sizeof(float) == m_runner->get_input(0).nSize;
    ALOGI("nElements: %d, m_runner->get_input(0).nSize: %d, is_input_fp32: %s", nElements, m_runner->get_input(0).nSize, is_input_fp32 ? "true" : "false");

    return ax_det_errcode_success;
}

ax_det_errcode_e ax_model_base::deinit()
{
    if (m_runner)
    {
        m_runner->deinit();
    }
    return ax_det_errcode_success;
}
