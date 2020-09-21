#include <stdint.h>
#include <string>
#include <string_view>
#include <sstream>


void trim_video(std::string video_in, std::string video_out) {
    std::string frames_cnt = "10";	
    std::stringstream ss;
    ss << "ffmpeg -i " << video_in << " -vframes " << frames_cnt << " -acodec copy -vcodec copy " << video_out << " -y";
    std::string trim_command = ss.str();
    system(trim_command.c_str());
}

void create_net_cost_matrix() {

}

void save_input() {

}

int main() {
    // todo: move configuration to separate file
    constexpr uint8_t const segment_size = 50;
    constexpr uint8_t const segment_cnt = 18;
    std::string const input_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/input/pets3s.mp4";
    std::string const output_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/output/output.mp4";

    // this configuration may be local
    std::string const trimmed_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/tmp/tmp.mp4";

    trim_video(input_video, trimmed_video);

    return 0;
}
