#include "libdet.h"
#include "cmdline.hpp"
#include "timer.hpp"
#include <fstream>
#include <cstring>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }

    if (!ax_devices.host.available && ax_devices.devices.count == 0)
    {
        printf("no device available\n");
        return -1;
    }

    ax_det_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "model", true);
    parser.add<int>("model_type", 't', "model type", true);
    parser.add<std::string>("image", 'i', "image folder(jpg png etc....)", true);
    parser.add<float>("threshold", 0, "threshold", false, 0.25);
    parser.add<int>("num_classes", 'c', "num classes", false, 80);
    parser.add<int>("num_kpt", 0, "num kpt", false, 0);
    parser.add<std::string>("output", 'o', "output", false, "results.jpg");
    parser.parse_check(argc, argv);

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    init_info.num_classes = parser.get<int>("num_classes");
    init_info.num_kpt = parser.get<int>("num_kpt");

    init_info.model_type = ax_det_model_type_e(parser.get<int>("model_type"));

    sprintf(init_info.model_path, "%s", parser.get<std::string>("model").c_str());
    init_info.threshold = parser.get<float>("threshold");

    ax_det_handle_t handle;
    int ret = ax_det_init(&init_info, &handle);
    if (ret != ax_det_errcode_success)
    {
        printf("ax_det_init failed\n");
        return -1;
    }

    std::string image_src = parser.get<std::string>("image");
    cv::Mat src = cv::imread(image_src);
    if (src.empty())
    {
        printf("imread %s failed\n", image_src.c_str());
        return -1;
    }
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    ax_det_img_t img;
    img.data = src.data;
    img.width = src.cols;
    img.height = src.rows;
    img.channels = src.channels();
    img.stride = src.step;
    ax_det_result_t result;
    memset(&result, 0, sizeof(result));
    ret = ax_det(handle, &img, &result);
    if (ret != ax_det_errcode_success)
    {
        printf("ax_det failed\n");
        return -1;
    }
    printf("num_objs: %d\n", result.num_objs);

    cv::cvtColor(src, src, cv::COLOR_RGB2BGR);

    for (int i = 0; i < result.num_objs; i++)
    {
        ax_det_obj_t &obj = result.objects[i];
        cv::Rect rect(obj.box.x, obj.box.y, obj.box.w, obj.box.h);
        cv::rectangle(src, rect, cv::Scalar(0, 255, 0), 2);
        char label_info[128];
        sprintf(label_info, "%d %5.2f", obj.label, obj.score);
        cv::putText(src, label_info, cv::Point(obj.box.x, obj.box.y + 25), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);

        for (int j = 0; j < obj.num_kpt; j++)
        {
            cv::circle(src, cv::Point(obj.kpts[j].x, obj.kpts[j].y), 10, cv::Scalar(0, 255, 0), -1);
        }
    }

    cv::imwrite(parser.get<std::string>("output"), src);

    ax_det_deinit(handle);

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}