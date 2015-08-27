//
//  fisher_svm.h
//  FaceRecExtensions
//
//  Created by Henry Warhurst on 27/08/2015.
//  Copyright (c) 2015 Henry Warhurst. All rights reserved.
//

#ifndef __FaceRecExtensions__fisher_svm__
#define __FaceRecExtensions__fisher_svm__

// ------ Built-in Includes ------ //
#include <stdio.h>
#include <vector>

// ------- OpenCV Includes ------- //
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace cv;

class FisherSvm {
public:
    FisherSvm();
    // Each Mat in X is an image, n_dimensions is new dimensionality for LDA
    void train(const std::vector<Mat> &faces, const std::vector<int> &labels, int n_dimensions);
    // Predicts the class of a test image
    int predict(const Mat &test_face);
private:
    // ----- Functions ------- //
    static Mat asRowMatrix(const std::vector<Mat> &src, int rtype, double alpha = 1, double beta = 0);
    // ------- Data ---------- //
    Mat _eigenvectors;
    Mat _mean;
    Ptr<ml::SVM> _svm;
    int _data_format_1;
    int _data_format_2;
};

#endif /* defined(__FaceRecExtensions__fisher_svm__) */

