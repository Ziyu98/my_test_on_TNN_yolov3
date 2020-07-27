#include "objectdetector_jni.h"
#include "UltraObjectDetector.h"
#include "kannarotate.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<UltraObjectDetector> gDetector;
static int gComputeUnitType = 0;
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(init) (JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat confThreshold, jfloat nmsThreshold, jint computeUnitType)
{
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<UltraObjectDetector>(width, height, 0.4, 0.4);
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/yolov3.opt.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/yolov3.opt.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    TNN_NS::Status status;
    gComputeUnitType = computeUnitType;
    if (gComputeUnitType == 0) {
        gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsCPU, nchw);
    } else {
        gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsGPU, nchw);
    }
    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)(status));
        return -1;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 1;
    gDetector->SetBenchOption(bench_option);
    if (clsObjectInfo == NULL)
    {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ObjectDetector$ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsObjectInfo, "x1", "F");
        fidy1 = env->GetFieldID(clsObjectInfo, "y1", "F");
        fidx2 = env->GetFieldID(clsObjectInfo, "x2", "F");
        fidy2 = env->GetFieldID(clsObjectInfo, "y2", "F");
        fidscore = env->GetFieldID(clsObjectInfo, "score", "F");
    }
    return 0;
}

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(deinit) (JNIEnv *env, jobject thiz)
{
    gDetector = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height)
{
    jobjectArray objectInfoArray;
    int ret = -1;
    AndroidBitmapInfo sourceInfocolor;
    void* sourcePixelscolor;

    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        return 0;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return 0;
    }

    if (AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        return 0;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 10;
    gDetector->SetBenchOption(bench_option);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    auto asyncRefDetector = gDetector;
    std::vector<ObjectInfo> objectInfoList;
    TNN_NS::Status status = asyncRefDetector->Detect(input_mat, height, width, objectInfoList);
    AndroidBitmap_unlockPixels(env, imageSource);
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }
    LOGI("bench result: %s", asyncRefDetector->GetBenchResult().Description().c_str());
    char temp[128] = "";
    sprintf(temp, "device: %s \ntime:", (gComputeUnitType==0)?"arm":"gpu");
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("object info list size %d", objectInfoList.size());
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            LOGI("object[%d] %f %f %f %f score %f", i, objectInfoList[i].x1, objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2, objectInfoList[i].score);
            env->SetFloatField(objObjectInfo, fidx1, objectInfoList[i].x1);
            env->SetFloatField(objObjectInfo, fidy1, objectInfoList[i].y1);
            env->SetFloatField(objObjectInfo, fidx2, objectInfoList[i].x2);
            env->SetFloatField(objObjectInfo, fidy2, objectInfoList[i].y2);
            env->SetFloatField(objObjectInfo, fidscore, objectInfoList[i].score);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }
    return 0;
}