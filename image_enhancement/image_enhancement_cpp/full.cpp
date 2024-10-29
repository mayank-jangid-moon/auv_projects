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


cv::Mat hisStretching(const cv::Mat& img) {
    cv::Mat result;
    img.convertTo(result, CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(result, channels);
    for (int i = 0; i < 3; i++) {
        double minVal, maxVal;
        cv::minMaxLoc(channels[i], &minVal, &maxVal);
        channels[i] = (channels[i] - minVal) / (maxVal - minVal);
    }
    cv::merge(channels, result);
    return result;
}

cv::Mat sharp(const cv::Mat& img) {
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    cv::Mat GaussKernel = cv::getGaussianKernel(5, 3);
    GaussKernel = GaussKernel * GaussKernel.t();
    cv::Mat imBlur;
    cv::filter2D(imgFloat, imBlur, -1, GaussKernel);
    cv::Mat unSharpMask = imgFloat - imBlur;
    cv::Mat stretchIm = hisStretching(unSharpMask);
    cv::Mat result = (imgFloat + stretchIm) / 2;
    result.convertTo(result, CV_8U, 255);
    return result;
}

cv::Mat rgb_to_lab(const cv::Mat& rgb) {
    cv::Mat lab;
    cv::cvtColor(rgb, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Mat saliency_detection(const cv::Mat& img) {
    cv::Mat gfrgb;
    cv::GaussianBlur(img, gfrgb, cv::Size(3, 3), 3);
    cv::Mat lab;
    cv::cvtColor(gfrgb, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);
    cv::Scalar mean_lab = cv::mean(lab);
    cv::Mat result = (lab_channels[0] - mean_lab[0]).mul(lab_channels[0] - mean_lab[0]) +
        (lab_channels[1] - mean_lab[1]).mul(lab_channels[1] - mean_lab[1]) +
        (lab_channels[2] - mean_lab[2]).mul(lab_channels[2] - mean_lab[2]);
    return result;
}

cv::Mat Saturation_weight(const cv::Mat& img) {
    cv::Mat lab = rgb_to_lab(img);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    // Split BGR channels of the original image and convert to float type
    std::vector<cv::Mat> bgr_channels;
    cv::split(img, bgr_channels);
    for (auto& ch : bgr_channels) {
        ch.convertTo(ch, CV_32F);  // Convert each BGR channel to CV_32F
    }
    lab_channels[0].convertTo(lab_channels[0], CV_32F); // Convert LAB L channel to CV_32F

    // Ensure all matrices are of the same type and perform the operation
    cv::Mat result = (bgr_channels[0] - lab_channels[0]).mul(bgr_channels[0] - lab_channels[0]) +
        (bgr_channels[1] - lab_channels[0]).mul(bgr_channels[1] - lab_channels[0]) +
        (bgr_channels[2] - lab_channels[0]).mul(bgr_channels[2] - lab_channels[0]);

    // Divide by 3 for normalization and then take the square root
    result /= 3.0;
    cv::sqrt(result, result); // This will remain as CV_32F after sqrt

    // Convert back to CV_8U if needed for display or further processing
    result.convertTo(result, CV_8U);

    return result;
}




std::pair<cv::Mat, cv::Mat> norm_weight(const cv::Mat& w1, const cv::Mat& w2, const cv::Mat& w3,
    const cv::Mat& w4, const cv::Mat& w5, const cv::Mat& w6) {
    float K = 2.0f;
    float delta = 0.1f;
    cv::Mat nw1 = w1 + w2 + w3;
    cv::Mat nw2 = w4 + w5 + w6;
    cv::Mat w = nw1 + nw2;
    nw1 = (nw1 + delta) / (w + K * delta);
    nw2 = (nw2 + delta) / (w + K * delta);
    return std::make_pair(nw1, nw2);
}

std::vector<cv::Mat> gaussian_pyramid(const cv::Mat& img, int level) {
    std::vector<cv::Mat> out;
    cv::Mat temp = img.clone();
    out.push_back(temp);
    for (int i = 1; i < level; i++) {
        cv::pyrDown(temp, temp);
        out.push_back(temp);
    }
    return out;
}

std::vector<cv::Mat> laplacian_pyramid(const cv::Mat& img, int level) {
    std::vector<cv::Mat> out;
    cv::Mat temp = img.clone();
    for (int i = 0; i < level - 1; i++) {
        cv::Mat down, up;
        cv::pyrDown(temp, down);
        cv::pyrUp(down, up, temp.size());
        out.push_back(temp - up);
        temp = down;
    }
    out.push_back(temp);
    return out;
}

cv::Mat pyramid_reconstruct(const std::vector<cv::Mat>& pyramid) {
    cv::Mat result = pyramid.back();
    for (int i = pyramid.size() - 2; i >= 0; i--) {
        cv::pyrUp(result, result, pyramid[i].size());
        result += pyramid[i];
    }
    return result;
}

int main() {
    cv::Mat org_img = cv::imread("test/img1.jpg");
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

    cv::Mat sharpen_img = sharp(wb_img);
    cv::imshow("Sharpen White balance image", sharpen_img);

    cv::Mat gamma_img_lab = rgb_to_lab(gamma_crct_img);
    cv::Mat gamma_img_lab_1;
    cv::extractChannel(gamma_img_lab, gamma_img_lab_1, 0);
    gamma_img_lab_1 /= 255.0;

    cv::Mat sharpen_img_lab = rgb_to_lab(sharpen_img);
    cv::Mat sharpen_img_lab1;
    cv::extractChannel(sharpen_img_lab, sharpen_img_lab1, 0);
    sharpen_img_lab1 /= 255.0;

    cv::Mat laplacian = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::Mat WL1, WL2;
    cv::filter2D(gamma_img_lab_1, WL1, -1, laplacian, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(sharpen_img_lab1, WL2, -1, laplacian, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    cv::absdiff(WL1, cv::Mat::zeros(WL1.size(), WL1.type()), WL1);
    cv::absdiff(WL2, cv::Mat::zeros(WL2.size(), WL2.type()), WL2);

    cv::Mat WS1 = saliency_detection(gamma_crct_img);
    cv::Mat WS2 = saliency_detection(sharpen_img);

    cv::Mat WSat1 = Saturation_weight(gamma_crct_img);
    cv::Mat WSat2 = Saturation_weight(sharpen_img);

    auto [W1, W2] = norm_weight(WL1, WS1, WSat1, WL2, WS2, WSat2);

    int level = 3;

    std::vector<cv::Mat> Weight1 = gaussian_pyramid(W1, level);
    std::vector<cv::Mat> Weight2 = gaussian_pyramid(W2, level);

    std::vector<cv::Mat> channels1, channels2;
    cv::split(gamma_crct_img, channels1);
    cv::split(sharpen_img, channels2);

    std::vector<cv::Mat> r1 = laplacian_pyramid(channels1[2], level);
    std::vector<cv::Mat> g1 = laplacian_pyramid(channels1[1], level);
    std::vector<cv::Mat> b1 = laplacian_pyramid(channels1[0], level);

    std::vector<cv::Mat> r2 = laplacian_pyramid(channels2[2], level);
    std::vector<cv::Mat> g2 = laplacian_pyramid(channels2[1], level);
    std::vector<cv::Mat> b2 = laplacian_pyramid(channels2[0], level);

    std::vector<cv::Mat> R_r, G_g, B_b;
    for (int i = 0; i < level; i++) {
        R_r.push_back(Weight1[i].mul(r1[i]) + Weight2[i].mul(r2[i]));
        G_g.push_back(Weight1[i].mul(g1[i]) + Weight2[i].mul(g2[i]));
        B_b.push_back(Weight1[i].mul(b1[i]) + Weight2[i].mul(b2[i]));
    }

    cv::Mat R = pyramid_reconstruct(R_r);
    cv::Mat G = pyramid_reconstruct(G_g);
    cv::Mat B = pyramid_reconstruct(B_b);

    std::vector<cv::Mat> fusion_channels = { B, G, R };
    cv::Mat fusion;
    cv::merge(fusion_channels, fusion);

    cv::Mat final_result;
    fusion.convertTo(final_result, CV_8U);
    cv::imshow("Result [Fusion image]", final_result);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}


