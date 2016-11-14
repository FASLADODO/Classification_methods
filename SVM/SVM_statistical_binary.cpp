#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

void
saveWeight(Mat &M, string s){
    
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            float f = M.at<float>(i, j);
            fprintf(pOut, "%f", f);
            if(j == M.cols - 1) {fprintf(pOut, "\n");}
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}


void
readWeight(Mat M, string s){
    
    std::fstream my(s.c_str(),ios::in);
    for (int i = 0; i<M.rows; i++){
        for (int j = 0; j<M.cols; j++)
        {
            double f=0.0;
            if(my.is_open())
            {
                my>>f;
            }
            M.at<float>(i, j)=f;
        }
    }
}


int main(int, char**)
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    
    // Set up training data
    Mat trainingData = Mat::zeros(500,2, CV_32FC1);
    readWeight(trainingData, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//statistical_data//linearly separable//train.txt");

    int labels[500];
    for (int r = 0; r < 250; r++)
        labels[r] = 1;
    
    for (int r = 250; r < 500; r++)
        labels[r] = -1;
    
    Mat labelsMat(500, 1, CV_32SC1, labels);
    saveWeight(labelsMat, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//statistical_data//linearly separable//labels.txt");
    
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labelsMat);
    
    // Set up testing data
    Mat testingData = Mat::zeros(1,2, CV_32FC1);
    readWeight(testingData, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//statistical_data//linearly separable//test.txt");
    
    float response = svm->predict(testingData);
    cout << response;
     saveWeight(testingData, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//statistical_data//linearly separable//test_result.txt");

    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue (255,0,0);
    
    for (int i = 0; i < testingData.rows; ++i){
        for (int j = 0; j < testingData.cols; ++j){
    
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    }
    
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    
    
    // Show support vectors
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }
    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}
