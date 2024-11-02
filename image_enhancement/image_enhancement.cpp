#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <iostream>

using namespace std;

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

    cv::Mat integral_r, integral_g, integral_b;
    cv::integral(r, integral_r, CV_32F);
    cv::integral(g, integral_g, CV_32F);
    cv::integral(b, integral_b, CV_32F);

    cv::Mat ret = img.clone();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int x1 = std::max(j - window / 2, 0);
            int y1 = std::max(i - window / 2, 0);
            int x2 = std::min(j + window / 2, width - 1);
            int y2 = std::min(i + window / 2, height - 1);

            int area = (x2 - x1 + 1) * (y2 - y1 + 1);

            float r_mean = (integral_r.at<float>(y2 + 1, x2 + 1) - integral_r.at<float>(y1, x2 + 1) -
                integral_r.at<float>(y2 + 1, x1) + integral_r.at<float>(y1, x1)) / area;
            float g_mean = (integral_g.at<float>(y2 + 1, x2 + 1) - integral_g.at<float>(y1, x2 + 1) -
                integral_g.at<float>(y2 + 1, x1) + integral_g.at<float>(y1, x1)) / area;

            float lrc = r.at<float>(i, j) + alpha * (g_mean - r_mean) * (1 - r.at<float>(i, j)) * g.at<float>(i, j);
            ret.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(std::min(std::max(lrc * 255.0f, 0.0f), 255.0f));
        }
    }

    return ret;
}

cv::Mat gray_balance(const cv::Mat& image) {
    int L = 255;
    vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar means = cv::mean(image);
    float Ravg = means[2], Gavg = means[1], Bavg = means[0];

    float Max = max({ Ravg, Gavg, Bavg });
    vector<float> ratio = { Max / Ravg, Max / Gavg, Max / Bavg };
    vector<float> satLevel = { 0.005f * ratio[0], 0.005f * ratio[1], 0.005f * ratio[2] };

    cv::Mat output = image.clone();

#pragma omp parallel for
    for (int ch = 0; ch < 3; ch++) {
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::Mat hist;
        cv::calcHist(&channels[ch], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        vector<float> cumulative_hist(histSize, 0.0f);
        cumulative_hist[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++) {
            cumulative_hist[i] = cumulative_hist[i - 1] + hist.at<float>(i);
        }
        for (int i = 0; i < histSize; ++i) {
            cumulative_hist[i] /= cumulative_hist[histSize - 1];
        }

        int pmin_idx = 0, pmax_idx = histSize - 1;
        while (cumulative_hist[pmin_idx] < satLevel[ch] && pmin_idx < histSize - 1) pmin_idx++;
        while (cumulative_hist[pmax_idx] > (1 - satLevel[ch]) && pmax_idx > 0) pmax_idx--;

        float pmin = static_cast<float>(pmin_idx);
        float pmax = static_cast<float>(pmax_idx);

#pragma omp parallel for
        for (int i = 0; i < channels[ch].rows; ++i) {
            uchar* pixel = channels[ch].ptr<uchar>(i);
            for (int j = 0; j < channels[ch].cols; ++j) {
                float val = static_cast<float>(pixel[j]);
                val = std::min(std::max(val, pmin), pmax);
                output.at<cv::Vec3b>(i, j)[ch] = static_cast<uchar>((val - pmin) * L / (pmax - pmin));
            }
        }
    }
    return output;
}

cv::Mat gammaCorrection(const cv::Mat& img, float alpha, float gamma) {
    static cv::Mat lut(1, 256, CV_8U);
    static bool initialized = false;

    if (!initialized) {
#pragma omp parallel for
        for (int i = 0; i < 256; i++) {
            lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        initialized = true;
    }

    cv::Mat output;
    cv::LUT(img, lut, output);
    output.convertTo(output, CV_8U, alpha);
    return output;
}

int main() {
            auto start_total = chrono::high_resolution_clock::now();

    cv::Mat img = cv::imread("test/ship_1080.jpg");
    //cv::Mat img = org_img.clone();
    // cv::imshow("Input image", img);

            // auto start_red_comp = chrono::high_resolution_clock::now();
    cv::Mat red_comp_img = redCompensate(img, 1);
            // auto end_red_comp = chrono::high_resolution_clock::now();
            // chrono::duration<double, milli> elapsed_red_comp = end_red_comp - start_red_comp;
            // cout << "Red Compensation time: " << elapsed_red_comp.count() << " ms" << endl;

    //cv::imshow("Red Compensated", red_comp_img);

            // auto start_gray_balance = chrono::high_resolution_clock::now();
    cv::Mat wb_img = gray_balance(red_comp_img);
            // auto end_gray_balance = chrono::high_resolution_clock::now();
            // chrono::duration<double, milli> elapsed_gray_balance = end_gray_balance - start_gray_balance;
            // cout << "Gray Balance time: " << elapsed_gray_balance.count() << " ms" << endl;

    //cv::imshow("White Balanced Image", wb_img);

    float alpha = 1.0f;
    float gamma = 1.2f;

            // auto start_gamma_correction = chrono::high_resolution_clock::now();
    cv::Mat gamma_crct_img = gammaCorrection(wb_img, alpha, gamma);
            // auto end_gamma_correction = chrono::high_resolution_clock::now();
            // chrono::duration<double, milli> elapsed_gamma_correction = end_gamma_correction - start_gamma_correction;
            // cout << "Gamma Correction time: " << elapsed_gamma_correction.count() << " ms" << endl;

    // cv::imshow("Gamma corrected White balance image", gamma_crct_img);
    //cv::imwrite("output/1.jpg", gamma_crct_img);

        auto end_total = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed_total = end_total - start_total;
        cout << "\nTotal Elapsed time: " << elapsed_total.count() << " ms" << endl;

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}