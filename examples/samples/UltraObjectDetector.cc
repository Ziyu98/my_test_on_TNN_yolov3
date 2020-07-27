#include "UltraObjectDetector.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <iostream>

UltraObjectDetector::UltraObjectDetector(int input_width, int input_height, float conf_threshold_, float nms_threshold_) {
    in_w = input_width;
    in_h = input_height;
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

void UltraObjectDetector::GetBoxes(TNN_NS::Mat &feature_map, std::vector<ObjectInfo> &output, int input_width, int input_height, float conf_threshold_, float nms_threshold_)
{
    // 
}

float UltraObjectDetector::Sigmoid(float x) {
    float exp_value;
    float return_value;
    exp_value = exp((double)-x);
    return_value = 1 / (1 + exp_value);
    return return_value;
}

int UltraObjectDetector::GetMaxClassScoreIndex(std::vector<float> scores) {
    float score = scores[0];
    int index = 0;
    for (int i = 1; i < scores.size(); i++) {
        if (scores[i] > score) {
            score = scores[i];
            index = i;
        }
    }
    return index;
}

void UltraObjectDetector::GetBBox(TNN_NS::Mat &feature_map, int tiles, std::vector<float> anchors, int input_width, int input_height, float conf_threshold_, std::vector<ObjectInfo> &output)
{
    // feature_map: e.g. float* --> 169*225, where tiles = 169
    float *fm_data = (float *)feature_map.GetData();
    std::vector<std::vector<float>> fm_reshaped = {};
    float tx, ty, tw, th, cf, bx, by, bw, bh, b_conf;
    std::vector<float> cp, b_scores;
    for (int i = 0; i < tiles; i++) {
        std::vector<float> temp = {};
        for (int j = 0; j < 255; j++){
            temp.push_back(fm_data[i*255 + j]);
        }
        fm_reshaped.push_back(temp);   // fm_reshaped: 169 * 255
    }
    int fm_dims = sqrt(tiles);
    for (int i = 0; i < 3; i++)   // anchor has 3 pairs
    {
        for (int cx = 0; cx < fm_dims; cx++){
            for (int cy = 0; cy < fm_dims; cy++){
                std::vector<float> temp = fm_reshaped[cx * fm_dims + cy];
                tx = temp[0 + 85 * i];
                ty = temp[1 + 85 * i];
                tw = temp[2 + 85 * i];
                th = temp[3 + 85 * i];
                cf = temp[4 + 85 * i];
                for(int j = 5; j < 85; j++) {
                    cp.push_back(temp[j + 85 * i]);
                }

                bx = (Sigmoid(tx) + cx) / fm_dims;
                by = (Sigmoid(ty) + cy) / fm_dims;
                bw = anchor[i * 2 + 0] * exp(tw) / input_width;
                bh = anchor[i * 2 + 1] * exp(th) / input_height;

                b_conf = Sigmoid(cf);
                for(auto cp_i : cp) {
                    b_scores.push_back(b_conf * Sigmoid(cp_i));
                }
                int b_class_index = GetMaxClassScoreIndex(b_scores);
                float b_class_score = b_scores[b_class_index];
                if b_class_score > conf_threshold_ {
                    ObjectInfo box;
                    memset(&box, 0, sizeof(box));
                    box.x1 = bx;
                    box.y1 = by;
                    box.x2 = bx + bw;
                    box.y2 = by + bh;
                    box.score = b_class_score;
                    output.push_back(box);
                }                
            }
        }
    }

}

std::vector<int> UltraObjectDetector::argsort(std::vector<float> scores) {
    std::vector<int> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&scores](int i1, int i2) {return scores[i1] < scores[i2];});
    return idx;
}
std::vector<int> UltraObjectDetector::RemoveLast(std::vector<int> & vec) {
    auto resultVec = vec;
    resultVec.erase(resultVec.end() - 1);
    return resultVec;
}

std::vector<float> UltraObjectDetector::Maximum(float & num, std::vector<float> & vec) {
    auto maxVec = vec;
    auto len = vec.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] < num)
            maxVec[idx] = num;
    return maxVec;
}

std::vector<float> UltraObjectDetector::CopyByIndexes(std::vector<float> & vec, std::vector<int> & idxs)
{
    std::vector<float> resultVec;
    for (auto & idx : idxs)
        resultVec.push_back(vec[idx]);
    return resultVec;
}

std::vector<float> UltraObjectDetector::Minimum(float & num, std::vector<float> & vec) {
    auto minVec = vec;
    auto len = vec.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] > num)
            minVec[idx] = num;
    return minVec;
}

std::vector<float> UltraObjectDetector::Subtract(std::vector<float> & vec1, std::vector<float> & vec2)
{
    std::vector<float> result;
    auto len = vec1.size();
  
    for (decltype(len) idx = 0; idx < len; ++idx)
    result.push_back(vec1[idx] - vec2[idx] + 1);
  
    return result;
}

std::vector<float> UltraObjectDetector::Divide(std::vector<float> & vec1, std::vector<float> & vec2)
{
    std::vector<float> resultVec;
    auto len = vec1.size();
  
    for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] / vec2[idx]);
  
    return resultVec;
}

std::vector<float> UltraObjectDetector::Multiply(std::vector<float> & vec1, std::vector<float> & vec2)
{
    std::vector<float> resultVec;
    auto len = vec1.size();
  
    for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] * vec2[idx]);
  
    return resultVec;
}

std::vector<int> UltraObjectDetector::WhereLarger(std::vector<float> & vec, float & threshold)
{
    std::vector<int> resultVec;
    auto len = vec.size();
  
    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] > threshold)
            resultVec.push_back(idx);
  
    return resultVec;
}

std::vector<int> UltraObjectDetector::RemoveByIndexes(std::vector<int> & vec, std::vector<int> & idxs)
{
    auto resultVec = vec;
    auto offset = 0;
  
    for (auto & idx : idxs) {
        resultVec.erase(resultVec.begin() + idx + offset);
        offset -= 1;
    }
  
    return resultVec;
}

// 非极大值抑制阈值筛选得到bbox
void UltraObjectDetector::donms(std::vector<ObjectInfo> &boxes, std::vector<ObjectInfo> &output, float nms_threshold_)
{
    std::vector<float> b_x1, b_y1, b_x2, b_y2, scores, areas;
    std::vector<int> order;
    for (auto box : boxes) {
        b_x1.push_back(box.x1);
        b_y1.push_back(box.y1);
        b_x2.push_back(box.x2);
        b_y2.push_back(box.y2);
        scores.push_back(box.score);
        areas.push_back((box.x2 - box.x1 + 1) * (box.y2 - box.y1 + 1));
    }
    idxs = argsort(scores);
    int last;
    int i;
    std::vector<int> pick;
    while (idxs.size() > 0) {
        last = idxs.size() - 1;
        i = idxs[last];
        pick.push_back(i);
        auto idxsWoLast = RemoveLast(idxs);
        auto xx1 = Maximum(b_x1[i], CopyByIndexes(b_x1, idxsWoLast));
        auto yy1 = Maximum(b_y1[i], CopyByIndexes(b_y1, idxsWoLast));
        auto xx2 = Minimum(b_x2[i], CopyByIndexes(b_x2, idxsWoLast));
        auto yy2 = Minimum(b_y2[i], CopyByIndexes(b_y2, idxsWoLast));

        auto w = Maximum(0, Subtract(xx2, xx1));
        auto h = Maximum(0, Subtract(yy2, yy1));
        auto overlap = Divide(Multiply(w, h), CopyByIndexes(areas, idxsWoLast));
        auto deleteIdxs = WhereLarger(overlap, nms_threshold_);
        deleteIdxs.push_back(last);
        idxs = RemoveByIndexes(idxs, deleteIdxs);
    }
    ObjectInfo object;
    for (auto idx : pick) {
        object.x1 = boxes[idx].x1;
        object.y1 = boxes[idx].y1;
        object.x2 = boxes[idx].x2;
        object.y2 = boxes[idx].y2;
        object.score = boxes[idx].score;
        output.push_back(object);
    }
}
