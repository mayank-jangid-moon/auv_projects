#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

cv::Mat redCompensate(const cv::Mat& img, int window) {
    float alpha = 1.0f;
    cv::Mat r, g, b;
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    r = channels[2];
    g = channels[1];
    b = channels[0];

    r.convertTo(r, CV_32F, 1.0 / 255.0);
    g.convertTo(g, CV_32F, 1.0 / 255.0);
    b.convertTo(b, CV_32F, 1.0 / 255.0);

    int height = img.rows, width = img.cols;
    int padsize = (window - 1) / 2;
    cv::Mat padr, padg;
    cv::copyMakeBorder(r, padr, padsize, padsize, padsize, padsize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(g, padg, padsize, padsize, padsize, padsize, cv::BORDER_REFLECT);

    cv::Mat ret = img.clone();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Mat slider = padr(cv::Rect(j, i, window, window));
            cv::Mat slideg = padg(cv::Rect(j, i, window, window));
            float r_mean = cv::mean(slider)[0];
            float g_mean = cv::mean(slideg)[0];
            float Irc = r.at<float>(i, j) + alpha * (g_mean - r_mean) * (1 - r.at<float>(i, j)) * g.at<float>(i, j);
            ret.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(Irc * 255);
        }
    }
    return ret;
}

cv::Mat gray_balance(const cv::Mat& image) {
    int L = 255;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar means = cv::mean(image);
    float Ravg = means[2], Gavg = means[1], Bavg = means[0];

    float Max = std::max({ Ravg, Gavg, Bavg });
    std::vector<float> ratio = { Max / Ravg, Max / Gavg, Max / Bavg };

    std::vector<float> satLevel = { 0.005f * ratio[0], 0.005f * ratio[1], 0.005f * ratio[2] };

    cv::Mat output = cv::Mat::zeros(image.size(), image.type());
    for (int ch = 0; ch < 3; ch++) {
        cv::Mat temp;
        channels[ch].convertTo(temp, CV_32F);
        temp = temp.reshape(1, 1);
        cv::Mat sorted;
        cv::sort(temp, sorted, cv::SORT_ASCENDING);
        int n = sorted.cols;
        float pmin = sorted.at<float>(static_cast<int>(satLevel[ch] * n));
        float pmax = sorted.at<float>(static_cast<int>((1 - satLevel[ch]) * n));
        channels[ch] = cv::max(cv::min(channels[ch], pmax), pmin);
        cv::normalize(channels[ch], channels[ch], 0, L, cv::NORM_MINMAX);
    }
    cv::merge(channels, output);
    return output;
}

cv::Mat gammaCorrection(const cv::Mat& img, float alpha, float gamma) {
    cv::Mat imgFloat;
    // Ensure the image is in float format with range [0, 1]
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    // Apply gamma correction
    cv::Mat output;
    cv::pow(imgFloat, gamma, output);

    // Scale by alpha and convert back to 8-bit
    output = alpha * output * 255.0;
    output.convertTo(output, CV_8U);

    return output;
}

int main() {
    cv::Mat org_img = cv::imread("test/ship.jpg");
    cv::Mat img = org_img.clone();
    cv::imshow("Input image", org_img);

    cv::Mat red_comp_img = redCompensate(img, 5);
    cv::imshow("Red Compensated", red_comp_img);

    cv::Mat wb_img = gray_balance(red_comp_img);
    cv::imshow("White Balanced Image", wb_img);

    float alpha = 1.0f;
    float gamma = 1.2f;
    cv::Mat gamma_crct_img = gammaCorrection(wb_img, alpha, gamma);
    cv::imshow("Gamma corrected White balance image", gamma_crct_img);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}