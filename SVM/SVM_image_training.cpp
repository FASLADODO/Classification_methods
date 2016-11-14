#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;
using namespace cv;
using namespace cv::ml;


void
readWeight(Mat M, string s)
{
    std::fstream my(s.c_str(),ios::in);
    for (int i = 0; i<M.rows; i++){
        for (int j = 0; j<M.cols; j++)
        {
            float f=0.0;
            if(my.is_open())
            {
                my>>f;
            }
            M.at<float>(i, j)=f;
        }
    }
}

void
read_batch(string filename, vector<Mat> &vec, Mat &label,char separator = ','){
    ifstream file (filename.c_str(), ifstream::in);
    string line, path, classlabel;
    int number_of_images =0;
    if (file.is_open())
    {
        while (getline(file, line)) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);
            if(!path.empty() && !classlabel.empty()) {
                vec.push_back(imread(path, 0));
                label.at<double>(0, number_of_images) = (double)atoi(classlabel.c_str());
            }
            number_of_images++;
        }
    }
}


void
read_labels(string filename, Mat &label,char separator = ','){
    
    ifstream file (filename.c_str(), ifstream::in);
    string line, path, classlabel;
    int number_of_images =0;
    if (file.is_open())
    {
        while (getline(file, line)) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);
            if(!path.empty() && !classlabel.empty()) {
                label.at<int>(0, number_of_images) = (int)atoi(classlabel.c_str());
            }
            number_of_images++;
        }
    }
}

void
read_DB(Mat &X, Mat &Y){
    
    string filename;
    filename = "//Users//ykg2910//Documents//4th_year_projects//Assignment3//data.txt";
    Mat label1 = Mat::zeros(1, 2628, CV_32S);
    read_labels(filename, label1);
    label1.copyTo(Y);
    
    readWeight(X, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//output_feature.txt");
}

/////////////////////// MAIN  //////////////////////////
int
main(int, char**)
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // vector to hold the images
    Mat images = Mat::zeros(2628, 512, CV_32F);
    Mat label;
    read_DB(images, label);
    label = label.t();
    
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 1e-6));
    svm->train(images, ROW_SAMPLE, label);
    
    //Making predictions using trained SVM model
    Mat sampleMat = Mat::zeros(1, 512, CV_32F);
    readWeight(sampleMat, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//test_data.txt");
    float response = svm->predict(sampleMat);
    cout << response;
    
    
    
    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
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
