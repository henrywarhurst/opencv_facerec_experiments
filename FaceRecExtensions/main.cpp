/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "fisher_svm.h"

#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

using namespace cv;
using namespace cv::face;
using namespace std;

void train_fisherface_classifier(std::vector<Mat> &faces, std::vector<int> labels, Mat test_face);
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0);

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[])
{
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidennce threshold, call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    //model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
    //int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    //string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    //cout << result_message << endl;
    // Sometimes you'll need to get/set internal model data,
    // which isn't exposed by the public cv::FaceRecognizer.
    // Since each cv::FaceRecognizer is derived from a
    // cv::Algorithm, you can query the data.
    //
    // First we'll use it to set the threshold of the FaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    //model->set("threshold", 0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    //predictedLabel = model->predict(testSample);
    //cout << "Predicted class = " << predictedLabel << endl;
    // Here is how to get the eigenvalues of this Eigenfaces model:
    //Mat eigenvalues = model->getMat("eigenvalues");
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    //Mat W = model->getMat("eigenvectors");
    // From this we will display the (at most) first 10 Eigenfaces:
//    for (int i = 0; i < min(10, W.cols); i++) {
//        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
//        cout << msg << endl;
//        // get eigenvector #i
//        Mat ev = W.col(i).clone();
//        // Reshape to original size & normalize to [0...255] for imshow.
//        Mat grayscale = norm_0_255(ev.reshape(1, height));
//        // Show the image & apply a Jet colormap for better sensing.
//        Mat cgrayscale;
//        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
//        imshow(format("%d", i), cgrayscale);
//    }
    //waitKey(0);
    
    //train_fisherface_classifier(images, labels, testSample);
    FisherSvm face_recogniser;
    face_recogniser.train(images, labels, -1);
    int class_num = face_recogniser.predict(testSample);
    std::cout << class_num << std::endl;
    
    return 0;
}

void train_fisherface_classifier(std::vector<Mat> &faces, std::vector<int> labels, Mat testSample)
{
    // Check that the input data is the right shape
    int n_examples = (int) labels.size();
    if (n_examples != faces.size()) {
        std::cout << "Error: Different number of faces to labels!";
    }
    int data_format = CV_64FC1;
//    // Put the data into matrix form
//    int total = faces[0].rows*faces[0].cols;
//    Mat data(n_examples, total, data_format);
//    for (int i=0; i<n_examples; ++i) {
//        Mat xi = data.row(i);
//        faces[0].reshape(1, 1).convertTo(xi, data_format, 1, 0);
//    }
    Mat data = asRowMatrix(faces, data_format);
    // Get the number of classes
    std::set<int> label_set(labels.begin(), labels.end());
    int n_classes = (int) label_set.size();
    // Perform PCA, keeping (n - n_classes) eigenvalues
    int n_eigenvalues = n_examples - n_classes;
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, n_eigenvalues);
    // We keep n_classes eigenvectors here (we can use less if we want)
    Mat tmp = pca.project(data);
    
    //std::cout << tmp << std::endl;
    
    LDA lda(tmp, labels, n_classes);
    // Keep the mean for projecting new examples
    Mat mean = pca.mean.reshape(1, 1);
    mean.convertTo(mean, CV_64FC1);
    Mat output_eigenvectors;
    // Avoids type mismatch in gemm call
    Mat pca_eigenvectors;
    pca.eigenvectors.convertTo(pca_eigenvectors, data_format);
    // Do matrix multiplication
    gemm(pca_eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, output_eigenvectors, GEMM_1_T);
    // Project the original data
    std::vector<Mat> projections;
    for (int i=0; i<data.rows; ++i) {
        Mat p = LDA::subspaceProject(output_eigenvectors, mean, data.row(i));
        projections.push_back(p);
    }
    
    Mat q = LDA::subspaceProject(output_eigenvectors, mean, testSample.reshape(1, 1));
    
    //------------- SVM ----------------------------- //
    
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    // Train the SVC
    Mat projectionsProc = asRowMatrix(projections, CV_32FC1);
    Ptr<ml::TrainData> trainingData = ml::TrainData::create(projectionsProc, ml::SampleTypes::ROW_SAMPLE, labels);
    svm->train(trainingData);
    Mat result;
    Mat q_proc;
    q.convertTo(q_proc, CV_32FC1);
    svm->predict(q_proc, result);
    std::cout << result << std::endl;
    std::cout << result.at<float>(0,0) << std::endl;
    
    //------------- NN ------------------------------ //
//    // Find 1 nearest neighbour
//    double min_dist = DBL_MAX;
//    int min_class = -1;
//    for (size_t i=0; i<projections.size(); ++i) {
//        double dist = norm(projections[i], q, NORM_L2);
//        std::cout << dist << std::endl;
//        if (dist < min_dist) {
//            min_dist = dist;
//            min_class = labels[i];
//        }
//    }
//    std::cout << min_class << std::endl;
}


void svm_test(void)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    //svm->setGamma(3.0);
    //svm->setDegree(3);
    svm->setC(0.00001);
    float x_arr[2][6] = {{1.1, 2.403, 23.3, 41.23, 50, 83.4}, {0.8, 2.0, 25.4, 45.0, 55.1, 79.0}};
    int y_arr[6] = {1, 1, 1, 2, 2, 2};
    Mat X = Mat(6, 2, CV_32FC1, &x_arr);
    Mat y = Mat(6, 1, CV_32SC1, &y_arr);
    Ptr<ml::TrainData> trainingData = ml::TrainData::create(X, ml::SampleTypes::ROW_SAMPLE, y);
    svm->train(trainingData);
    float query_arr[2] = {40, 40};
    Mat query = Mat(1, 2, CV_32FC1, &query_arr);
    Mat result;
    svm->predict(query, result);
    std::cout << result << std::endl;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha, double beta) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}
