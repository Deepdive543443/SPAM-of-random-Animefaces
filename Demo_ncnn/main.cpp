#include <iostream>
#include <random>
#include <stdlib.h>
// #include <time.h>
#include <chrono>
#include <thread>


#include "net.h"
// #include "simpleocv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


static std::default_random_engine generator;
static std::normal_distribution<float> distribution(0.0, 1.0);
// generator.seed(time(NULL));


void post_process_img(ncnn::Mat& img)
{
    const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
    const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
    img.substract_mean_normalize(_mean_, _norm_);
}


void randn_ncnn(ncnn::Mat &mat, int w, int h, int c)
{
    mat.create(w, h, c, (size_t) 4);

    memset(mat.data, 0.f, w * h * c * 4);

    #pragma omp parallel for num_threads(3)
    for (int k = 0; k < c; k++)
    {
        float *c_ptr = mat.channel(k);
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                c_ptr[0] = distribution(generator);
                c_ptr++;
            }
        } 
    }
}


int main()
{
    unsigned long int t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    printf("Hello TWNE\n");
    generator.seed(t);
    // std::mt19937 rng(generator());
    // srand(t);
    // std::srand(std::time(nullptr));

    ncnn::Option opt;
    // opt.lightmode = true;
    opt.num_threads = 4;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    // opt.use_int8_inference = true;
    opt.use_vulkan_compute = true;
    // opt.use_fp16_packed = true;
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    // opt.use_int8_storage = true;
    // opt.use_int8_arithmetic = true;
    // opt.use_packing_layout = true;
    // opt.use_shader_pack8 = false;
    // opt.use_image_storage = false;

    ncnn::Net twne;
    twne.opt = opt;
    twne.load_param("../weight/TWNE_MINI.ncnn.param");
    twne.load_model("../weight/TWNE_MINI.ncnn.bin");

    std::system("mkdir images");
    for (int i = 0; i < 200; i++)
    {
        ncnn::Extractor ex = twne.create_extractor();
        ncnn::Mat noise;
        randn_ncnn(noise, 1, 1, 512);

        ex.input("in0", noise);
        ncnn::Mat outputs;
        ex.extract("out0", outputs);
        post_process_img(outputs);

        cv::Mat image(256, 256, CV_8UC3);
        outputs.to_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB);

        cv::imwrite("images/sample" + std::to_string(i + 1) + ".png", image);
        std::cout << "Write sample " << i + 1 << std::endl; 
    }



    return 0;
}