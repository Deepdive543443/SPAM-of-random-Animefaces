#include <iostream>
#include <random>
#include <stdlib.h>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "net.h"

static std::default_random_engine      generator;
static std::normal_distribution<float> distribution(0.0, 1.0);

static const float s_mean[3] = {-1.0f, -1.0f, -1.0f};
static const float s_norm[3] = {127.5f, 127.5f, 127.5f};

static ncnn::Mat randn_ncnn(int w, int h, int c)
{
    ncnn::Mat mat(w, h, c, (size_t)4);
    memset(mat.data, 0.f, w * h * c * 4);

    #pragma omp parallel for num_threads(3)
    for (int k = 0; k < c; k++) {
        float *c_ptr = mat.channel(k);
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                c_ptr[0] = distribution(generator);
                c_ptr++;
            }
        }
    }
    return mat;
}

int main(int argc, char **argv)
{
    int num_generate = argc > 1 ? atoi(argv[1]) : 1;
    printf("Hello TWNE\n");

    unsigned long int t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    generator.seed(t);

    ncnn::Option opt;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution    = true;
    opt.use_vulkan_compute       = true;

    ncnn::Net twne;
    twne.opt = opt;
    twne.load_param("../weight/TWNE_MINI.ncnn.param");
    twne.load_model("../weight/TWNE_MINI.ncnn.bin");

    std::system("mkdir images");
    for (int i = 0; i < num_generate; i++) {
        ncnn::Extractor ex    = twne.create_extractor();
        ncnn::Mat       noise = randn_ncnn(1, 1, 512);

        ex.input("in0", noise);
        ncnn::Mat outputs;
        ex.extract("out0", outputs);
        outputs.substract_mean_normalize(s_mean, s_norm);

        cv::Mat image(256, 256, CV_8UC3);
        outputs.to_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB);
        cv::imwrite("images/sample" + std::to_string(i + 1) + ".png", image);
        std::cout << "Write sample " << i + 1 << std::endl;
    }
    return 0;
}