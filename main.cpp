#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include "net.h"
#include "utils.h"


int main() 
{
    int num_img = 0;
    srand(time(NULL));
    // Load network
    ncnn::Net progan;
    progan.load_param("../../models/Progressive.ncnn.param");
    progan.load_model("../../models/Progressive.ncnn.bin");
    printf("Loaded");

    // Generate random noise
    while(true){
        ncnn::Mat noise = randn_mat(512, 1, 1, rand());
        noise = noise.reshape(1, 1, 512);
        shape(noise);

    
        // Forward
        ncnn::Extractor ex = progan.create_extractor();
        ex.input("in0", noise);
        ncnn::Mat outputs;
        ex.extract("out0", outputs);
        printf("Output's shape");
        shape(outputs);
        post_process_img(outputs);


        // Display the generated images
        cv::Mat cv_img = cv::Mat::zeros(outputs.w, outputs.h, CV_8UC3);
        outputs.to_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB);
        shape(cv_img);
        cv::imshow("Testing", cv_img);
        cv::waitKey(0);

        num_img++;
        
        if(num_img > 10){
            break;
        }
    }
    
    return 0;
}