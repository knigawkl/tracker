#include "video.hpp"
#include "utils.hpp"


namespace video
{
    vidinfo get_video_info(const cv::VideoCapture& cap, int segment_size, const std::string& tmp_folder)
    {
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int video_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const int video_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        const int frame_cnt = video::get_trimmed_frame_cnt(cap, segment_size);
        const int segment_cnt = frame_cnt / segment_size;
        const std::string tmp_video = tmp_folder + "/input.mp4";
        vidinfo video_info = {
            .frame_cnt = frame_cnt,
            .segment_size = segment_size,
            .segment_cnt = segment_cnt,
            .width = video_w,
            .height = video_h,
            .fps = fps,
            .tmp_dir = tmp_folder,
            .tmp_video = tmp_video
        };
        return video_info;
    }

    int get_video_capture_frame_cnt(const cv::VideoCapture& cap)
    {
        return cap.get(cv::CAP_PROP_FRAME_COUNT);
    }

    int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment) 
    {
        int video_in_frame_cnt = get_video_capture_frame_cnt(cap);
        std::cout << "Input video frames count: " << video_in_frame_cnt << std::endl;
        // TODO: as for now it is a good enoguh solution to trim the video massively
        // in order to get such number of frames that SMCP may be applied
        // but later on it is probably a good idea to extend number of frames by multiplying the last frame
        // and then trimming the resulting video 
        //return video_in_frame_cnt / frames_in_segment * frames_in_segment;
        int trimmed_frame_cnt = video_in_frame_cnt / frames_in_segment * frames_in_segment;
        if (trimmed_frame_cnt % 2 == 0)
            trimmed_frame_cnt -= frames_in_segment;

        std::cout << "CUT " << trimmed_frame_cnt << std::endl;
        return trimmed_frame_cnt;
    }

    void trim_video(std::string video_in, std::string video_out, int frame_cnt) 
    {
        std::stringstream ss;
        ss << "ffmpeg -i " << video_in << " -vframes " << std::to_string(frame_cnt) 
        << " -acodec copy -vcodec copy " << video_out << " -y";
        std::string trim_command = ss.str();
        std::cout << "Executing: " << trim_command << std::endl;
        system(trim_command.c_str());
    }

    void prepare_tmp_video(const cv::VideoCapture& in_cap, const vidinfo& video_info, const std::string& in_video)
    {
        const int video_in_frame_cnt = get_video_capture_frame_cnt(in_cap);
        if (video_in_frame_cnt != video_info.frame_cnt)
        {
            std::string const trimmed_video = video_info.tmp_dir + "/trim.mp4";
            trim_video(in_video, trimmed_video, video_info.frame_cnt);
            std::cout << "Input video frames count cut to: " << video_info.frame_cnt << std::endl;
            utils::sys::mv(trimmed_video, video_info.tmp_video);
        }
        else
        {
            utils::sys::cp(in_video, video_info.tmp_video);
        }
    }

    void merge_frames(std::string out_video, const video::vidinfo& video_info)
    {
        std::stringstream ss;
        ss << "ffmpeg -framerate " << video_info.fps << " -i " << video_info.tmp_dir
           << "/img/frame%05d.jpeg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " << out_video << " -y";
        std::string merge_command = ss.str();
        std::cout << "Executing: " << merge_command << std::endl;
        system(merge_command.c_str());
    }
}
