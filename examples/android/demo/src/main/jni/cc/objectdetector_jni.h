#ifndef ANDROID_OBJECTDETECTOR_JNI_H
#define ANDROID_OBJECTDETECTOR_JNI_H
#include <jni.h>
#define TNN_OBJECT_DETECTOR(sig) Java_com_tencent_tnn_demo_ObjectDetector_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat confThreshold, jfloat nmsThreshold, jint computUnitType);
JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_OBJECTDETECTOR_JNI_H