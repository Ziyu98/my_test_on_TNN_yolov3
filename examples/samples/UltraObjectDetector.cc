#include "UltraObjectDetector.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <iostream>

UltraObjectDetector::UltraObjectDetector(float conf_threshold_, float nms_threshold_) {
    conf_threshold = conf_threshold_;
    nms_threshold = nms_threshold_;
}

UltraObjectDetector::~UltraObjectDetector() {}

int UltraObjectDetector::Detect(std::shared_ptr<TNN_NS::Mat> image_mat, int image_height, int image_width, std::vector<ObjectInfo> &object_list) {
    if (!image_mat || !image_mat->GetData()) {
        std::cout << "image is empty, please chekc!" << std::endl;
        return -1;
    }
    //image_h = image_height;
    //image_w = image_width;
#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
#endif

        // step 1. set input mat
        TNN_NS::MatConvertParam input_cvt_param;
        input_cvt_param.scale = {1.0 / 128, 1.0 / 128, 1.0 / 128, 0.0};
        input_cvt_param.bias  = {-127.0 / 128, -127.0 / 128, -127.0 / 128, 0.0};
        auto status = instance_->SetInputMat(image_mat, input_cvt_param);
        if (status != TNN_NS::TNN_OK) {
            LOGE("input_blob_convert.ConvertFromMatAsync Error: %s\n", status.description().c_str());
            return status;
        }

        // step 2. Forward
        status = instance_->ForwardAsync(nullptr);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.Forward Error: %s\n", status.description().c_str());
            return status;
        }
        // step 3. get output mat
        TNN_NS::MatConvertParam output_cvt_param;
        std::shared_ptr<TNN_NS::Mat> output_mat_res = nullptr;
        status = instance_->GetOutputMat(output_mat_res, output_cvt_param, "082_convolutional");
        if (status != TNN_NS::TNN_OK) {
            LOGE("GetOutputMat Error: %s\n", status.description().c_str());
            return status;
        }

#if TNN_SDK_ENABLE_BENCHMARK
        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        bench_result_.AddTime(elapsed);
#endif

        std::vector<ObjectInfo> bbox_collection;
        object_list.clear();
        ObjectInfo temp;
        temp.score = 0;
        temp.x1 = 0;
        temp.x2 = 0;
        temp.y1 = 0;
        temp.y2 = 0;
        object_list.push_back(temp);
        // GenerateBBox(bbox_collection, *(output_mat_res.get()), conf_threshold, nms_threshold);
#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    // Detection done
    return 0;
}

/*
 * Generating bbox from output blobs
 */
void UltraObjectDetector::GenerateBBox(std::vector<ObjectInfo> &bbox_collection, TNN_NS::Mat &res, float conf_threshold, float nms_threshold) {
    float *detect_res = (float *)res.GetData();
    std::vector<std::vector<float>> reshaped_res = {};
    std::vector<float> temp_line = {};
    for (int i = 0; i < 10647; i++) {
        for (int j = 0; j < 85; j++) {
            temp_line.push_back(detect_res[85 * i + j]);
        }
        reshaped_res.push_back(temp_line);
    }
    // converted_res = xywh2xyxy(reshaped_res);
    for (int i = 0; i < 4; i++) {
        ObjectInfo temp_info;
        temp_info.x1 = reshaped_res[i][0];
        temp_info.x2 = reshaped_res[i][1];
        temp_info.y1 = reshaped_res[i][2];
        temp_info.y2 = reshaped_res[i][3];
        temp_info.score = reshaped_res[i][4];
        bbox_collection.push_back(temp_info);
    }
}
