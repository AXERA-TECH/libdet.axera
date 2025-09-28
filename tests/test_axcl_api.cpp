#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"
#include "cmdline.hpp"
#include <fstream>

int main(int argc, char *argv[])
{
    auto ret = axclInit();
    if (ret != 0)
    {
        printf("axclInit failed\n");
        return -1;
    }

    axcl_Dev_Init(0);
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "model", true);
    parser.parse_check(argc, argv);

    ax_runner_axcl runner;
    std::ifstream file(parser.get<std::string>("model"), std::ios::binary);
    if (!file.is_open())
    {
        printf("open file failed\n");
        return -1;
    }
    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    runner.init(model_data.data(), model_data.size(), 0);

    runner.deinit();

    axcl_Dev_Exit(0);
    axclFinalize();
    return 0;
}