//
//  fisher_svm.cpp
//  FaceRecExtensions
//
//  Created by Henry Warhurst on 27/08/2015.
//  Copyright (c) 2015 Henry Warhurst. All rights reserved.
//

#include "fisher_svm.h"
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

FisherSvm::FisherSvm()
:_data_format_1(CV_64FC1), _data_format_2(CV_32FC1)
{
    _svm = ml::SVM::create();
    _svm->setType(ml::SVM::C_SVC);
    _svm->setKernel(ml::SVM::LINEAR);
}

void FisherSvm::train(const std::vector<Mat> &faces, const std::vector<int> &labels, int n_dimensions)
{
    // Check that the input data is the right shape
    int n_examples = (int) labels.size();
    if (n_examples != faces.size()) {
        std::cout << "Error: Different number of faces to labels!";
    }
    // Structure the data properly
    Mat data = asRowMatrix(faces, _data_format_1);
    // Get the unique number of classes
    std::set<int> label_set(labels.begin(), labels.end());
    int n_classes = (int) label_set.size();
    // Perform PCA, keeping (n - n_classes) eigenvalues
    int n_eigenvalues = n_examples - n_classes;
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, n_eigenvalues);
    // We keep n_classes eigenvectors here (we can use less if we want)
    Mat tmp = pca.project(data);
    // Perform Linear Discriminant Analysis
    LDA lda(tmp, labels, n_classes);
    // Keep the mean (used for projecting new examples to face space)
    _mean = pca.mean.reshape(1, 1);
    Mat pca_eigenvectors;
    pca.eigenvectors.convertTo(pca_eigenvectors, _data_format_1);
    // Matrix multiplication
    gemm(pca_eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    // Project the original data to face space
    std::vector<Mat> projections;
    for (int i=0; i<data.rows; ++i) {
        Mat p = LDA::subspaceProject(_eigenvectors, _mean, data.row(i));
        projections.push_back(p);
    }
    // Train the Support Vector Classifier
    Mat projectionsProc = asRowMatrix(projections, _data_format_2);
    Ptr<ml::TrainData> trainingData = ml::TrainData::create(projectionsProc, ml::SampleTypes::ROW_SAMPLE, labels);
    _svm->train(trainingData);
}

int FisherSvm::predict(const Mat &test_face)
{
    Mat q = LDA::subspaceProject(_eigenvectors, _mean, test_face.reshape(1,1));
    Mat result;
    Mat q_proc;
    q.convertTo(q_proc, _data_format_2);
    _svm->predict(q_proc, result);
    int class_num = (int) result.at<float>(0,0);
    return class_num;
}

Mat FisherSvm::asRowMatrix(const std::vector<Mat> &src, int rtype, double alpha, double beta)
{
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
