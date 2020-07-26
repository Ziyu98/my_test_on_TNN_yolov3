#ifndef UltraObjectDetector_hpp
#define UltraObjectDetector_hpp

#include "TNNSDKSample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

typedef struct ObjectInfo {
    float x1;
    float x2;
    float y1;
    float y2;
    float score;
} ObjectInfo;

class UltraObjectDetector : public TNN_NS::TNNSDKSample {
public:
    ~UltraObjectDetector();
    UltraObjectDetector(float conf_threshold_ = 0.4, float nms_threshold_ = 0.4);
    int Detect(std::shared_ptr<TNN_NS::Mat> image, int image_height, int image_width, std::vector<ObjectInfo> &object_list);
private:
    void GenerateBBox(std::vector<ObjectInfo> &bbox_collection, TNN_NS::Mat &res, float conf_threshold, float nms_threshold);
    //std::vector<std::vector<float>> UltraObjectDetector::xywh2xyxy(std::vector<std::vector<float>> reshaped_res);
private:
    float conf_threshold;
    float nms_threshold;
};

#endif /* UltraObjectDetector_hpp */