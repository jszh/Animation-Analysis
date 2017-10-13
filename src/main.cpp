#include <opencv2/opencv.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <list>
#include <string>
#include <map>
#include <chrono>
#include <assert.h>

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>

using std::string;
using std::vector;
using std::list;
using std::to_string;

using namespace cv;
using cv::cuda::GpuMat;

const bool SHOW_IMG = false;
const bool SHOW_DETAILS = false;
const double TOLERANCE = 1.0;

void help() {
    std::cout << "usage: analysis [-i] [-r] [-a <min-area>] [-f <camera-fps>] [-d <display-fps>] [-s <start-frame>] <input> [<output>]\n";
}

class Cluster {
public:
    Cluster(double tolerance) {
        this->tolerance = tolerance;
    }
    double tolerance;
    vector<double> items;
    double minimum, maximum;

    bool can_add_item(double value) const {
        return items.size() == 0 || (value > minimum - tolerance && value < maximum + tolerance);
    }

    bool add_item(double value) {
        // no item in cluster yet; add this item
        if (items.size() == 0) {
            items.push_back(value);
            minimum = maximum = value;
            return true;
        } else {
            if (can_add_item(value)) {
                items.push_back(value);
                minimum = min(value, minimum);
                maximum = max(value, maximum);
                return true;
            }
            return false;
        }
    }

    void merge(Cluster& cluster) {
        items.insert( items.end(),
                      cluster.items.begin(),
                      cluster.items.end() );
        tolerance = max(tolerance, cluster.tolerance);
        minimum = min(minimum, cluster.minimum);
        maximum = max(maximum, cluster.maximum);
    }
};

list<Cluster> clusterize(vector<double>& data) {
    list<Cluster> clusters;
    for (auto value: data) {
        if (clusters.size() == 0) {
            Cluster cluster(TOLERANCE);
            cluster.add_item(value);
            clusters.push_back(cluster);
        } else {
            auto end = clusters.end();
            auto it1 = end;
            auto it2 = end;
            auto iter = clusters.begin();
            for ( ; iter != end; ++iter ) {
                if (iter->can_add_item(value)) {
                    if (it1 == end) {
                        it1 = iter;
                    } else {
                        it2 = iter;
                        break;
                    }
                }
            }

            if (it1 == end) {    // no suitable cluster exist
                Cluster cluster(TOLERANCE);
                cluster.add_item(value);
                clusters.push_back(cluster);
            } else if (it2 == end) { // one such cluster
                it1->add_item(value);
            } else {    // if two clusters can both add this item, merge these clusters
                it1->merge(*it2);
                it1->add_item(value);
                clusters.erase(it2);
            }
        }
    }

    clusters.sort([](const Cluster& a, const Cluster& b) {
            return a.items.size() > b.items.size();
        });

    for (auto& cluster: clusters)
        std::sort(cluster.items.begin(), cluster.items.end());
    return clusters;
}

vector<double> find_medians(list<Cluster>& clusters) {
    vector<double> medians;
    for (auto& cluster: clusters) {
        if (cluster.items.size() % 2 == 1) {
            medians.push_back(cluster.items[cluster.items.size()/2]);
        } else {
            double avg = (cluster.items[cluster.items.size()/2-1] + cluster.items[cluster.items.size()/2])/2;
            medians.push_back(avg);
        }
    }
    return medians;
}

vector<double> find_means(list<Cluster>& clusters) {
    vector<double> means;
    for (auto& cluster: clusters) {
        double mean = 0;
        for (auto value: cluster.items)
            mean += value;
        mean /= cluster.items.size();
        means.push_back(mean);
    }
    return means;
}

string print_translation(list<Cluster>& clusters, vector<double>& medians, vector<double>& means) {
    string ret = "";
    if (SHOW_DETAILS)
        std::cout << "-- ";
    int stat_it = 0;
    int pcount = 0;

    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
        if (it->items.size() >= 3) {
            if (SHOW_DETAILS)
                std::cout << "cnt=" << it->items.size()  << " " << it->minimum << " " << it->maximum << ",med=" << medians[stat_it] << ",avg=" << means[stat_it] << "; ";
            ret += "," + to_string(it->items.size()) + "," + to_string(it->minimum) + "," + to_string(it->maximum) + "," + to_string(medians[stat_it]) + "," + to_string(means[stat_it]);
            pcount++;
        }
        stat_it++;
        if (pcount >= 2) break;
    }

    if (SHOW_DETAILS)
        std::cout << "\n";

    if (pcount < 2) ret += ",,,,,";
    if (pcount < 1) ret += ",,,,,";
    return ret;
}

int main(int argc, char **argv) {

    string infile;
    string outfile;
    bool inverse_color = false;
    bool rotate = false;
    int min_area = 500;
    int cam_fps = 180;
    int disp_fps = 60;
    int start_frame = 0;
    int opt;

    // is the animation translation?
    bool translation = true;

    // parse the arguments
    while ((opt = getopt(argc, argv, "ira:f:d:s:")) != -1)
        switch(opt) {
            case 'i': inverse_color = true; break;
            case 'r': rotate = true; break;
            case 'a': min_area = atoi(optarg); break;
            case 'f': cam_fps = atoi(optarg); break;
            case 'd': disp_fps = atoi(optarg); break;
            case 's': start_frame = atoi(optarg); break;
            default: help(); exit(1);
        }


	std::ofstream ofs;

    // input / output files
    if (optind <= argc - 1) {
        infile = string(argv[optind]);
        if (optind <= argc - 2)
            outfile = string(argv[optind+1]);
        else
            outfile = infile.substr(0, infile.size()-4) + "_out.csv";
		ofs.open(outfile);
    } else {
        std::cerr << argv[0] << ": input file missing\n";
        help();
        exit(1);
    }

    if (cam_fps < disp_fps) {
        std::cerr << argv[0] << ": camera frame rate must be greater than or equal to display frame rate\n";
        exit(1);
    }

    double fr_mult = (double) cam_fps / (double) disp_fps;

    // set up video source
    auto video = VideoCapture(infile);

    // SURF descriptors
    auto surf = cuda::SURF_CUDA();

    vector<GpuMat> frames(fr_mult + 1);
    vector< GpuMat > keypoints(fr_mult + 1);
    vector<KeyPoint> kp1, kp2;
    GpuMat d_kp1, d_kp2;
    vector<GpuMat> descriptors(fr_mult + 1);
    vector<float> des1, des2;
    GpuMat d_des1, d_des2;

    // set video to start frame
    video.set(CAP_PROP_POS_MSEC, start_frame / (double)cam_fps * 1000);

    int iter = start_frame;
    GpuMat frame;
    Mat fr;
    int x1, x2, y1, y2;
    Mat img, img2;
	string stat = "index, h1_tx, h1_ty, h1_sx, h1_sy, h1_num, h2_tx, h2_ty, h2_sx, h2_sy, h2_num";
    if (translation) stat = "index, count_x1, min_x1, max_x1, med_x1, avg_x1, count_x2, min_x2, max_x2, med_x2, avg_x2, count_y1, min_y1, max_y1, med_y1, avg_y1, count_y2, min_y2, max_y2, med_y2, avg_y2";

    // while there are frames remaining
    auto start_time = std::chrono::steady_clock::now();
    while (video.read(fr)) {

        // rotate CW
        // https://stackoverflow.com/questions/15043152/rotate-opencv-matrix-by-90-180-270-degrees
        if (rotate) {
            transpose(fr, fr);
            flip(fr, fr, 1);
        }

        frame.upload(fr);

        // resize(frame, frame, Size(480, round(480.0/frame.cols*frame.rows)));

        // get grayscale image
        Mat gr(fr.rows, fr.cols, CV_8U);
        GpuMat gray(frame.rows, frame.cols, CV_8U);
        // inverse red and blue
        if (inverse_color) {
            cuda::cvtColor(frame, gray, COLOR_RGB2GRAY);
            cvtColor(fr, gr, COLOR_RGB2GRAY);
        } else {
            cuda::cvtColor(frame, gray, COLOR_BGR2GRAY);
            cvtColor(fr, gr, COLOR_BGR2GRAY);
        }

        if (img.empty()) {
            img = Mat::zeros(frame.rows, frame.cols*2, CV_8U);
            img2 = Mat::zeros(frame.rows, frame.cols*2, CV_8U);
        }

        iter++;

        // initialize first frames
        bool initd = true;
        for (auto& f: frames)
            if (f.empty()) {
                f = gray;
                initd = false;
                break;
            }
        if (!initd) continue;

		// output stat
		ofs << stat << std::endl;
        if (SHOW_DETAILS || iter % 100 == 0)
            std::cout << iter << std::endl;
		stat = to_string(iter);

        // compute difference between first & current frame
        GpuMat thresh(frame.rows, frame.cols, CV_8U);
        cuda::absdiff(gray, frames[0], thresh);

        for (size_t i = 0; i < frames.size(); ++i) {
            if (i < fr_mult) {
                frames[i] = frames[i+1];
                keypoints[i] = keypoints[i+1];
                descriptors[i] = descriptors[i+1];
            } else {
                frames[i] = gray;
                keypoints[i] = GpuMat();
                descriptors[i] = GpuMat();
            }
        }

        // threshold the difference
        cuda::threshold(thresh, thresh, 25, 255, THRESH_BINARY);
        Mat c_thresh(frame.rows, frame.cols, CV_8U);
        thresh.download(c_thresh);

        // dilate, find contours
        dilate(c_thresh, c_thresh, Mat(), Point(-1, -1), 2);
        vector<vector<Point> > contours;
        findContours(c_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        int cnt_count = 0;

        x1 = frame.cols;
        x2 = 0;
        y1 = frame.rows;
        y2 = 0;

        GpuMat mask(y1, x1, CV_8U, Scalar(0));

        for (auto contour: contours) {
            // ignore contour if too small
            if (contourArea(contour) < min_area) continue;

            // update outer bounding box
            Rect r = boundingRect(contour);
            if (x1 > r.x) x1 = r.x;
            if (x2 < r.x + r.width) x2 = r.x + r.width;
            if (y1 > r.y) y1 = r.y;
            if (y2 < r.y + r.height) y2 = r.y + r.height;

            // draw bounding box for contour
            rectangle(fr, r.tl(), r.br(), Vec3i(0, 255, 0));

            cnt_count++;
        }

        string empty_line = translation ? ",,,,,,,,,,,,,,,,,,," : ",,,,,,,,,";

        // found valid contours
        if (cnt_count >= 1 && (x2-x1)*(y2-y1) >= 3 * min_area) {
            rectangle(fr, Point(x1, y1), Point(x2, y2), Vec3i(0, 255, 0), 2);
            GpuMat roi(mask(Rect(x1, y1, x2-x1, y2-y1)));
            roi.setTo(255);
			//mask = thresh;

            // detect and compute features with SURF
            surf(gray, mask, d_kp1, d_des1);

            if (!keypoints[0].empty() && !descriptors[0].empty()) {
                d_kp2 = keypoints[0];
                d_des2 = descriptors[0];
            } else {
                surf(frames[0], mask, d_kp2, d_des2);
            }

            if (d_des1.empty() || d_des2.empty()) {
				stat += empty_line;
				continue;
			}

            // brute force matcher
            auto bf = cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
            vector< vector<DMatch> > matches;
            vector<DMatch> good;
            vector<double> dist_h, dist_v;
            bf->knnMatch(d_des1, d_des2, matches, 2);

            // ratio test
            for (auto vec: matches) {
                if (vec.size() == 2) {
                    if (1.5 * vec[0].distance < vec[1].distance)
                        good.push_back(vec[0]);
                }
            }

            std::vector<Point2f> dst_points, odst;
            std::vector<Point2f> src_points, osrc;

            // enough matches to calculate a transform
            if (good.size() > 5) {
                kp1.clear();
                kp2.clear();
                des1.clear();
                des2.clear();
                surf.downloadDescriptors(d_des1, des1);
                surf.downloadDescriptors(d_des2, des2);
                surf.downloadKeypoints(d_kp1, kp1);
                surf.downloadKeypoints(d_kp2, kp2);

                for (size_t i = 0; i < good.size(); ++i) {
                    // get the keypoints from the good matches
                    dst_points.push_back( kp1[ good[i].queryIdx ].pt );
                    src_points.push_back( kp2[ good[i].trainIdx ].pt );
                }

                assert(src_points.size() == dst_points.size());
                if (src_points.size() == 0) {
                    stat += empty_line;
                    continue;
				}

                if (translation) {
                    // find translational vector
                    for (auto m: good) {
                        double dx = kp1[m.queryIdx].pt.x - kp2[m.trainIdx].pt.x;
                        double dy = kp1[m.queryIdx].pt.y - kp2[m.trainIdx].pt.y;
                        dist_h.push_back(dx);
                        dist_v.push_back(dy);
                    }

                    // clusterize
                    auto clusters_v = clusterize(dist_v);
                    auto clusters_h = clusterize(dist_h);
                    auto medians_v = find_medians(clusters_v);
                    auto medians_h = find_medians(clusters_h);
                    auto means_v = find_means(clusters_v);
                    auto means_h = find_means(clusters_h);

                    // output translations
                    stat += print_translation(clusters_h, medians_h, means_h);
                    stat += print_translation(clusters_v, medians_v, means_v);

                } else {
                    // calculate homography using RANSAC
                    vector<char> rmask(good.size());    // matched points != 0
                    Mat H = findHomography(src_points, dst_points, CV_RANSAC, 0.5, rmask);
                    if (rmask.empty() || H.empty()) {
                        stat += empty_line;
                        continue;
                    }

                    int max_dist = 0;
                    int max_len = 0;

                    // calculate homography in outliers again
                    // classify matches according to their orientation
                    vector<size_t> indices;
                    int i = -1;
                    for (auto m: good) {
                        ++i;
                        if (rmask[i] == 0) {
                            odst.push_back( kp1[m.queryIdx].pt );
                            osrc.push_back( kp2[m.trainIdx].pt );
                            indices.push_back(i);
                            continue;
                        }
                    }

                    Mat fr0;
                    frames[0].download(fr0);
                    surf(gray, mask, keypoints[fr_mult], descriptors[fr_mult]);
                    try {
                        drawMatches(gr, kp1, fr0, kp2, good, img, Scalar::all(-1), Scalar::all(-1), rmask, 2);
                    } catch (Exception& e) {
                        std::cerr << iter << ": " << e.what() << std::endl;
                    }

                    // output translation
                    if (SHOW_DETAILS) {
                        std::cout << " trans 1: "<< H.at<double>(0, 2) << ", " <<  H.at<double>(1, 2) << "; ";
                        std::cout << " scale 1: "<< H.at<double>(0, 0) << ", " <<  H.at<double>(1, 1) << ", " << good.size()-osrc.size() << std::endl;
                    }
                    stat += ", " + std::to_string(H.at<double>(0, 2)) + ", " + std::to_string(H.at<double>(1, 2)) + ", " + std::to_string(H.at<double>(0, 0)) + ", " + std::to_string(H.at<double>(1, 1)) + ", " + std::to_string(good.size()-osrc.size());

                    if (osrc.size() > 5) {
                        vector<char> rmask2r(osrc.size());
                        Mat H2 = findHomography(osrc, odst, CV_RANSAC, 0.5, rmask2r);
                        if (!H2.empty()) {
                            vector<char> rmask2(good.size());
                            int count_match = 0;
                            for (size_t i = 0; i < rmask2r.size(); ++i) {
                                if (rmask2r[i] != 0) {
                                    rmask2[ indices[i] ] = rmask2r[i];
                                    count_match++;
                                }
                            }
                            try {
                                drawMatches(gr, kp1, fr0, kp2, good, img2, Scalar::all(-1), Scalar::all(-1), rmask2, 2);
                            } catch (Exception& e) {
                                std::cerr << iter << ": " << e.what() << std::endl;
                            }

                            if (SHOW_DETAILS) {
                                std::cout << " trans 2: "<< H2.at<double>(0, 2) << ", " <<  H2.at<double>(1, 2) << "; ";
                                std::cout << " scale 2: "<< H2.at<double>(0, 0) << ", " <<  H2.at<double>(1, 1) << ", " << count_match << std::endl;
                            }
                            stat += ", " + std::to_string(H2.at<double>(0, 2)) + ", " + std::to_string(H2.at<double>(1, 2)) + ", " + std::to_string(H2.at<double>(0, 0)) + ", " + std::to_string(H2.at<double>(1, 1)) + ", " + std::to_string(count_match);
                        }
                    } else {
                        stat += ",,,,,";
                    }
                }

            }

        } else {
			stat += empty_line;
		}

        if (SHOW_IMG) {
            resize(fr, fr, Size(360, round(360.0/fr.cols*fr.rows)));

            if (inverse_color)
                cvtColor(fr, fr, COLOR_RGB2BGR);
            imshow("Video", fr);

            if (!translation) {
                resize(img, img, Size(720, round(720.0/img.cols*img.rows)));
                resize(img2, img2, Size(720, round(720.0/img.cols*img.rows)));
                imshow("Matcher Result 1", img);
                imshow("Matcher Result 2", img2);
            }

            int key = waitKey(1);

            if (key == 'q') break;
            if (key == 'p') waitKey(0);
        }

    }

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = time - start_time;
    double duration = diff.count();
    std::cout << "time: " << duration << " s" << std::endl;

    video.release();
    destroyAllWindows();
	ofs.close();

    return 0;

}


