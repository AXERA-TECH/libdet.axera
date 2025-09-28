#include "libdet.h"
#include "models/ax_model_yolov8.hpp"

struct ax_det_handle_internal_t
{
    std::shared_ptr<ax_model_base> m_model;
};

int ax_det_init(ax_det_init_t *init, ax_det_handle_t *handle)
{
    ax_det_handle_internal_t *internal = new ax_det_handle_internal_t();

    delete_fun destroy_fun = nullptr;
    ax_model_base *model = (ax_model_base *)OBJFactory::getInstance().getObjectByID(init->model_type, destroy_fun);
    if (!model)
    {
        printf("model not register\n");
        return ax_det_errcode_failed;
    }
    internal->m_model = std::shared_ptr<ax_model_base>(model, destroy_fun);

    int ret = internal->m_model->init(init);
    if (ret != 0)
    {
        printf("model init failed\n");
        return ax_det_errcode_failed;
    }

    *handle = internal;
    return 0;
}

int ax_det_deinit(ax_det_handle_t handle)
{
    ax_det_handle_internal_t *internal = (ax_det_handle_internal_t *)handle;
    if (internal->m_model)
    {
        internal->m_model->deinit();
    }
    delete internal;
    return 0;
}

int ax_det(ax_det_handle_t handle, ax_det_img_t *img, ax_det_result_t *result)
{
    ax_det_handle_internal_t *internal = (ax_det_handle_internal_t *)handle;
    if (!internal->m_model)
    {
        printf("model not init\n");
        return ax_det_errcode_failed;
    }

    int ret = internal->m_model->inference(img, result);
    if (ret != 0)
    {
        printf("model detect failed\n");
        return ax_det_errcode_failed;
    }

    return 0;
}
