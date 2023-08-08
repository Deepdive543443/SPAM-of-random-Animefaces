// Opencv lib
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

//NCNN lib
#include "net.h"

// Standard lib
#include <iostream>


// EdVince's implementation of randn in NCNN. Using more than 3 channels is not recommand at here
ncnn::Mat randn_mat(int weight, int height, int channels, int seed)
{
    cv::Mat cv_x(cv::Size(weight, height), CV_32FC(channels));
    cv::RNG rng(seed);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
    ncnn::Mat x_mat(weight, height, channels, (void*)cv_x.data);
    return x_mat.clone();
}


// NCNN's official print function
void pretty_print(const ncnn::Mat& m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");

    }
    printf("Matric shape: [%d, %d, %d]\n", m.c, m.h, m.w);
}


// Print opencv Mat's shape
void shape(cv::Mat& image)
{
    printf("Image's shape: [%d, %d, %d]\n", image.channels(), image.rows, image.cols);
    // std::cout << "Rows: " << rows << " columns: " << cols << " Channels: " << image.channels() << std::endl;
}

// Print ncnn Mat's shape
void shape(ncnn::Mat& m)
{
    printf("Matric shape: [%d, %d, %d]\n", m.c, m.h, m.w);
}

void post_process_img(ncnn::Mat& img)
{
    const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
    const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
    img.substract_mean_normalize(_mean_, _norm_);
}