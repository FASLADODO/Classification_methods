// Abhishek Agrawal(ug201311001)
// Amit Jain(ug201310004)



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
saveWeight(Mat &M, string s){
    
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            int f = M.at<int>(i, j);
            fprintf(pOut, "%d", f);
            if(j == M.cols - 1) {fprintf(pOut, "\n");}
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}

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
//                cout << f << "--\n";
            }
            M.at<float>(i, j)=f;
        }
    }
}


/////////////////////// MAIN  //////////////////////////
int
main(int, char**)
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    
    //training data
    Mat train_data = Mat::zeros(1250, 512, CV_32FC1);
    readWeight(train_data, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//training_feature.txt");
    
    int train_labels[1250];
    for (int r = 0; r < 250; r++){
        train_labels[5*r] = 1;
        train_labels[5*r +1] = 2;
        train_labels[5*r +2] = 3;
        train_labels[5*r +3] = 4;
        train_labels[5*r +4] = 5;
    }
    Mat train_label(1250, 1, CV_32SC1, train_labels);
    saveWeight(train_label, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//training_labels.txt");
    

    
    //testing data
    Mat test_data = Mat::zeros(250, 512, CV_32FC1);
    readWeight(test_data, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//test_feature.txt");
    
    int test_labels[250];
    for (int r = 0; r < 50; r++){
        test_labels[5*r] = 1;
        test_labels[5*r +1] = 2;
        test_labels[5*r +2] = 3;
        test_labels[5*r +3] = 4;
        test_labels[5*r +4] = 5;
    }
    Mat test_label(250, 1, CV_32SC1, test_labels);
    saveWeight(test_label, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//test_labels.txt");
    
    
    
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 1e-6));
    svm->train(train_data, ROW_SAMPLE, train_label);
   
    
//    ParamGrid CvParamGrid_C(pow(2.0,-5), pow(2.0,15), pow(2.0,2));
//    ParamGrid CvParamGrid_gamma(pow(2.0,-15), pow(2.0,3), pow(2.0,2));
//    svm->trainAuto(train_data, labs, Mat(), Mat(), paramz,10, CvParamGrid_C, CvParamGrid_gamma, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
//    svm->get_params();
//    cout<<"gamma:"<<paramz.gamma<<endl;
//    cout<<"C:"<<paramz.C<<endl;
    
    
    //Making predictions using trained SVM model
    int response[250];
    float temp[512];
    for(int i = 0; i < 250; i++){
        for(int j = 0; j < 512; j++){
            temp[j] = test_data.at<float>(i,j);
        }
        Mat sampleMat(1, 512, CV_32FC1, &temp);
        saveWeight(sampleMat, "//Users//ykg2910//Documents//4th_year_projects//Assignment3//check.txt");
        response[i] = svm->predict(sampleMat);
    }
    Mat responses(250, 1, CV_32SC1, &response);
    
    Mat groundTruth = Mat::zeros(5, 250, CV_32SC1);
    for(int i=0; i<250; i++){
        groundTruth.at<int>(test_label.at<int>(i, 0), i) = 1.0;
    }
    
    
    //Finding error and accuracy
    Mat error = Mat::zeros(250, 1, CV_32SC1);
    error = test_label - responses;
    int count[6]  = {0,0,0,0,0,0};
    for(int i = 0; i < 250; i++){
        if(groundTruth.at<int>(responses.at<int>(i,0),i) == 1)
            count[responses.at<int>(i,0)]++;
    }

    waitKey(0);
    return 0;
}
