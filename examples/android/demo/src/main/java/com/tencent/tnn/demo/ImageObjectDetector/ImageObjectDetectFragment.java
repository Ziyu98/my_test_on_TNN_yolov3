package com.tencent.tnn.demo.ImageObjectDetector;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.ObjectDetector;
import com.tencent.tnn.demo.FaceDetector;
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.ImageClassify;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.CameraSetting;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;
import com.tencent.tnn.demo.common.sufaceHolder.DemoSurfaceHolder;

import java.util.ArrayList;
import java.io.IOException;

public class ImageObjectDetectFragment extends BaseFragment {
    private final static String TAG = ImageObjectDetectFragment.class.getSimpleName();
    private ObjectDetector mObjectDetector = new ObjectDetector();
    private static final String IMAGE = "tiger_cat.jpg";
    private static final int NET_H_INPUT = 416;
    private static final int NET_W_INPUT = 416;
    private Paint mPanit = new Paint();
    private ToggleButton mGPUSwitch;
    private Button mRunButton;
    private boolean mUseGPU = false;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
    }

    private String initModel()
    {
        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        //copy detect model to sdcard
        String[] modelPathsDetector = {
                "yolov3.opt.tnnmodel",
                "yolov3.opt.tnnproto",
        };

        for (int i = 0; i < modelPathsDetector.length; i++) {
            String modelFilePath = modelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "object_detector/"+modelFilePath, interModelFilePath);
        }
        return targetDir;
    }

    @Override
    public void onClick(View view) {
        int i = view.getId();
        if (i == R.id.back_rl) {
            clickBack();
        }
    }

    private void onSwitchGPU(boolean b)
    {
        mUseGPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
    }

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
        setView(R.layout.fragment_imageobjectdetector);
        setTitleGone();
        $$(R.id.back_rl);
        $$(R.id.gpu_switch);
        mGPUSwitch = $(R.id.gpu_switch);
        mGPUSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwitchGPU(b);
            }
        });
        mRunButton = $(R.id.run_button);
        mRunButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startDetect();
            }
        });
        final Bitmap originBitmap = FileUtils.readBitmapFromFile(getActivity().getAssets(), IMAGE);
        ImageView source = (ImageView)$(R.id.origin);
        source.setImageBitmap(originBitmap);
    }

    @Override
    public void openCamera() {

    }

    @Override
    public void startPreview(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void closeCamera() {

    }

    private void startDetect() {
        Bitmap originBitmap = FileUtils.readBitmapFromFile(getActivity().getAssets(), IMAGE);
        float scalew = originBitmap.getWidth()/(float)NET_W_INPUT;
        float scaleh = originBitmap.getHeight()/(float)NET_H_INPUT;
        Bitmap scaleBitmap = Bitmap.createScaledBitmap(originBitmap, NET_W_INPUT, NET_H_INPUT, false);
        ImageView source = (ImageView)$(R.id.origin);
        source.setImageBitmap(originBitmap);
        String modelPath = initModel();
        Log.d(TAG, "Init object detect " + modelPath);
        int result = mObjectDetector.init(modelPath, NET_W_INPUT, NET_H_INPUT, 0.4f, 0.4f, mUseGPU?1:0);
        if(result == 0) {
            Log.d(TAG, "detect from image");
            ObjectDetector.ObjectInfo[] objectInfoList = mObjectDetector.detectFromImage(scaleBitmap, NET_W_INPUT, NET_H_INPUT);
            Log.d(TAG, "detection from iamge result" + objectInfoList);
            int objectCount = 0;
            if (objectInfoList != null) {
                objectCount = objectInfoList.length;
            }
            if(objectInfoList != null && objectInfoList.length > 0) {
                Log.d(TAG, "detect object size" + objectInfoList.length);
            }
            String benchResult = "object count: " + objectCount + " " + Helper.getBenchResult();
            TextView result_view = (TextView)$(R.id.result);
            result_view.setText(benchResult);
        } else {
            Log.e(TAG, "failed to init model" + result);
        }
    }

    @Override
    public void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");

        getFocus();
    }

    @Override
    public void onPause() {
        Log.d(TAG, "onPause");
        super.onPause();
    }

    @Override
    public void onStop() {
        Log.i(TAG, "onStop");
        super.onStop();
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "onDestroy");
    }

    private void preview() {
        Log.i(TAG, "preview");

    }

    private void getFocus() {
        getView().setFocusableInTouchMode(true);
        getView().requestFocus();
        getView().setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View v, int keyCode, KeyEvent event) {
            if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_BACK) {
                clickBack();
                return true;
            }
            return false;
            }
        });
    }

}