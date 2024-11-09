#include <iostream>
#include <random>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "net.h"

static std::default_random_engine      generator;
static std::normal_distribution<float> distribution(0.0, 1.0);

static const float s_mean[3] = {  -1.0f,  -1.0f,  -1.0f };
static const float s_norm[3] = { 127.5f, 127.5f, 127.5f };

static const char  s_b64_tab[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                  'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                  'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
static std::string b64_encode(std::vector<uchar> &buf)
{
    std::string output("");

    int padding = buf.size() % 3;
    for (size_t i = 0; i < (buf.size() - 3 + padding); i += 3) {
        uint8_t char1 = ( buf[i]     & 0b11111100) >> 2;
        uint8_t char2 = ((buf[i]     & 0b00000011) << 4) | ((buf[i + 1] & 0b11110000) >> 4) & 0b111111;
        uint8_t char3 = ((buf[i + 1] & 0b00001111) << 2) | ((buf[i + 2] & 0b11000000) >> 6) & 0b111111;
        uint8_t char4 =   buf[i + 2] & 0b00111111;

        output += s_b64_tab[char1];
        output += s_b64_tab[char2];
        output += s_b64_tab[char3];
        output += s_b64_tab[char4];
    }

    switch (padding) {
        case 2:
            output += s_b64_tab[ buf[buf.size() - 1] >> 2];
            output += s_b64_tab[(buf[buf.size() - 1] & 0b11) << 4];
            output += "==";
            break;
        case 1:
            output += s_b64_tab[(  buf[buf.size() - 2] & 0b11111100) >> 2];
            output += s_b64_tab[(((buf[buf.size() - 2] & 0b00000011) << 4) | ((buf[buf.size() - 1] & 0b11110000) >> 4)) & 0b111111];
            output += s_b64_tab[(  buf[buf.size() - 1] & 0b00001111) << 2];
            output += "=";
        case 0:
            break;
    }

    return output;
}

static ncnn::Mat randn_ncnn(int w, int h, int c)
{
    ncnn::Mat mat(w, h, c, (size_t)4);
    memset(mat.data, 0.f, w * h * c * 4);

    #pragma omp parallel for num_threads(4)
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

    for (int i = 0; i < num_generate; i++) {
        std::string name = std::to_string(i);

        ncnn::Extractor ex    = twne.create_extractor();
        ncnn::Mat       noise = randn_ncnn(1, 1, 512);
        ncnn::Mat       outputs;

        ex.input("in0", noise);
        ex.extract("out0", outputs);
        outputs.substract_mean_normalize(s_mean, s_norm);

        cv::Mat image(256, 256, CV_8UC3);
        outputs.to_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB);
        cv::imwrite(name + ".png", image);

        std::vector<uchar> buf;
        cv::imencode(".png", image, buf);
        std::string b64_string = b64_encode(buf);
        FILE       *b64_file   = fopen((name + "b64").c_str(), "wb");
        if (b64_file) {
            fwrite(b64_string.c_str(), 1, b64_string.size(), b64_file);
            fclose(b64_file);
        }
        std::cout << b64_string << std::endl;
    }
    return 0;
}