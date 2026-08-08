#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::uint32_t kMagic = 0x4f4c4f4c; // "LOLO" on little-endian hosts.
constexpr std::uint16_t kProtocolVersion = 1;

enum class Command : std::uint16_t {
  Hello = 1,
  Reset = 2,
  Step = 3,
  Save = 4,
  Load = 5,
  Drop = 6,
  Close = 7,
};

#pragma pack(push, 1)
struct RequestHeader {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t command;
  std::uint32_t payload_size;
};

struct ResponseHeader {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t status;
  std::uint16_t command;
  std::uint16_t reserved;
  std::uint32_t payload_size;
};

struct FrameHeader {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

struct StepRequest {
  std::uint32_t button_mask;
  std::uint32_t frames;
};

struct HelloHeader {
  std::uint32_t name_length;
  std::uint32_t version_length;
  std::uint32_t base_width;
  std::uint32_t base_height;
  double fps;
};
#pragma pack(pop)

static_assert(sizeof(RequestHeader) == 12, "request protocol layout changed");
static_assert(sizeof(ResponseHeader) == 16, "response protocol layout changed");
static_assert(sizeof(HelloHeader) == 24, "hello protocol layout changed");

using retro_environment_t = bool (*)(unsigned, void *);
using retro_video_refresh_t = void (*)(const void *, unsigned, unsigned, std::size_t);
using retro_audio_sample_t = void (*)(std::int16_t, std::int16_t);
using retro_audio_sample_batch_t = std::size_t (*)(const std::int16_t *, std::size_t);
using retro_input_poll_t = void (*)();
using retro_input_state_t = std::int16_t (*)(unsigned, unsigned, unsigned, unsigned);

struct retro_game_info {
  const char *path;
  const void *data;
  std::size_t size;
  const char *meta;
};

struct retro_system_info {
  const char *library_name;
  const char *library_version;
  const char *valid_extensions;
  bool need_fullpath;
  bool block_extract;
};

struct retro_game_geometry {
  unsigned base_width;
  unsigned base_height;
  unsigned max_width;
  unsigned max_height;
  float aspect_ratio;
};

struct retro_system_timing {
  double fps;
  double sample_rate;
};

struct retro_system_av_info {
  retro_game_geometry geometry;
  retro_system_timing timing;
};

struct retro_variable {
  const char *key;
  const char *value;
};

constexpr unsigned kDeviceJoypad = 1;
constexpr unsigned kEnvGetOverscan = 2;
constexpr unsigned kEnvGetCanDupe = 3;
constexpr unsigned kEnvSetPerformanceLevel = 8;
constexpr unsigned kEnvGetSystemDirectory = 9;
constexpr unsigned kEnvSetPixelFormat = 10;
constexpr unsigned kEnvSetInputDescriptors = 11;
constexpr unsigned kEnvGetVariable = 15;
constexpr unsigned kEnvSetVariables = 16;
constexpr unsigned kEnvGetVariableUpdate = 17;
constexpr unsigned kEnvSetSupportNoGame = 18;
constexpr unsigned kEnvGetInputCapabilities = 24;
constexpr unsigned kEnvGetContentDirectory = 30;
constexpr unsigned kEnvGetSaveDirectory = 31;
constexpr unsigned kEnvSetControllerInfo = 35;
constexpr unsigned kEnvSetGeometry = 37;
constexpr unsigned kEnvGetLanguage = 39;
constexpr unsigned kEnvGetCoreOptionsVersion = 52;
constexpr unsigned kEnvSetCoreOptionsV2 = 67;
constexpr unsigned kEnvSetCoreOptionsV2Intl = 68;
constexpr unsigned kEnvSetSerializationQuirks = 87;

constexpr int kPixel0Rgb1555 = 0;
constexpr int kPixelXrgb8888 = 1;
constexpr int kPixelRgb565 = 2;

bool read_exact(int descriptor, void *destination, std::size_t size) {
  auto *bytes = static_cast<std::uint8_t *>(destination);
  std::size_t offset = 0;
  while (offset < size) {
    const auto count = ::read(descriptor, bytes + offset, size - offset);
    if (count == 0) {
      return false;
    }
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("failed to read protocol input");
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

void write_exact(int descriptor, const void *source, std::size_t size) {
  const auto *bytes = static_cast<const std::uint8_t *>(source);
  std::size_t offset = 0;
  while (offset < size) {
    const auto count = ::write(descriptor, bytes + offset, size - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("failed to write protocol output");
    }
    if (count == 0) {
      throw std::runtime_error("protocol output closed during write");
    }
    offset += static_cast<std::size_t>(count);
  }
}

template <typename T>
void append_value(std::vector<std::uint8_t> &destination, const T &value) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&value);
  destination.insert(destination.end(), bytes, bytes + sizeof(T));
}

void append_bytes(std::vector<std::uint8_t> &destination, const void *data, std::size_t size) {
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  destination.insert(destination.end(), bytes, bytes + size);
}

template <typename T>
T payload_value(const std::vector<std::uint8_t> &payload) {
  if (payload.size() != sizeof(T)) {
    throw std::runtime_error("invalid command payload size");
  }
  T result{};
  std::memcpy(&result, payload.data(), sizeof(T));
  return result;
}

struct Frame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> rgb;
};

struct Snapshot {
  std::vector<std::uint8_t> state;
  Frame frame;
};

class Host;
Host *g_host = nullptr;

class Host {
public:
  Host(std::string core_path, std::string rom_path)
      : core_path_(std::move(core_path)), rom_path_(std::move(rom_path)) {
    const auto slash = rom_path_.find_last_of('/');
    system_directory_ = slash == std::string::npos ? "." : rom_path_.substr(0, slash);
    save_directory_ = system_directory_;
    load_core();
    load_content();
  }

  ~Host() {
    if (game_loaded_ && retro_unload_game_) {
      retro_unload_game_();
    }
    if (initialized_ && retro_deinit_) {
      retro_deinit_();
    }
    if (library_) {
      ::dlclose(library_);
    }
    if (g_host == this) {
      g_host = nullptr;
    }
  }

  Host(const Host &) = delete;
  Host &operator=(const Host &) = delete;

  int run_protocol() {
    for (;;) {
      RequestHeader request{};
      if (!read_exact(STDIN_FILENO, &request, sizeof(request))) {
        return 0;
      }
      if (request.magic != kMagic || request.version != kProtocolVersion) {
        throw std::runtime_error("protocol header mismatch");
      }
      if (request.payload_size > 64 * 1024 * 1024) {
        throw std::runtime_error("protocol payload exceeds limit");
      }
      std::vector<std::uint8_t> payload(request.payload_size);
      if (!payload.empty() && !read_exact(STDIN_FILENO, payload.data(), payload.size())) {
        throw std::runtime_error("truncated protocol payload");
      }
      const auto command = static_cast<Command>(request.command);
      try {
        if (dispatch(command, payload)) {
          return 0;
        }
      } catch (const std::exception &error) {
        const std::string message = error.what();
        send_response(command, 1, std::vector<std::uint8_t>(message.begin(), message.end()));
      }
    }
  }

  bool environment(unsigned command, void *data) {
    switch (command) {
    case kEnvGetOverscan:
      *static_cast<bool *>(data) = false;
      return true;
    case kEnvGetCanDupe:
      *static_cast<bool *>(data) = true;
      return true;
    case kEnvSetPerformanceLevel:
    case kEnvSetInputDescriptors:
    case kEnvSetControllerInfo:
    case kEnvSetSupportNoGame:
    case kEnvSetVariables:
    case kEnvSetCoreOptionsV2:
    case kEnvSetCoreOptionsV2Intl:
    case kEnvSetSerializationQuirks:
      return true;
    case kEnvGetSystemDirectory:
    case kEnvGetContentDirectory:
      *static_cast<const char **>(data) = system_directory_.c_str();
      return true;
    case kEnvGetSaveDirectory:
      *static_cast<const char **>(data) = save_directory_.c_str();
      return true;
    case kEnvSetPixelFormat: {
      const int requested = *static_cast<const int *>(data);
      if (requested != kPixel0Rgb1555 && requested != kPixelXrgb8888 &&
          requested != kPixelRgb565) {
        return false;
      }
      pixel_format_ = requested;
      return true;
    }
    case kEnvGetVariable:
      static_cast<retro_variable *>(data)->value = nullptr;
      return false;
    case kEnvGetVariableUpdate:
      *static_cast<bool *>(data) = false;
      return true;
    case kEnvGetInputCapabilities:
      *static_cast<std::uint64_t *>(data) = std::uint64_t{1} << kDeviceJoypad;
      return true;
    case kEnvSetGeometry:
      if (data) {
        av_info_.geometry = *static_cast<const retro_game_geometry *>(data);
      }
      return true;
    case kEnvGetLanguage:
      *static_cast<unsigned *>(data) = 0;
      return true;
    case kEnvGetCoreOptionsVersion:
      *static_cast<unsigned *>(data) = 2;
      return true;
    default:
      return false;
    }
  }

  void video_refresh(const void *data, unsigned width, unsigned height, std::size_t pitch) {
    if (!data) {
      return;
    }
    current_frame_.width = width;
    current_frame_.height = height;
    current_frame_.rgb.resize(static_cast<std::size_t>(width) * height * 3);
    const auto *source = static_cast<const std::uint8_t *>(data);
    std::size_t destination = 0;
    for (unsigned y = 0; y < height; ++y) {
      const auto *row = source + static_cast<std::size_t>(y) * pitch;
      for (unsigned x = 0; x < width; ++x) {
        std::uint8_t red = 0;
        std::uint8_t green = 0;
        std::uint8_t blue = 0;
        if (pixel_format_ == kPixelXrgb8888) {
          const auto *pixel = row + static_cast<std::size_t>(x) * 4;
          blue = pixel[0];
          green = pixel[1];
          red = pixel[2];
        } else {
          const auto *pixel = row + static_cast<std::size_t>(x) * 2;
          const auto value = static_cast<std::uint16_t>(pixel[0] | (pixel[1] << 8));
          if (pixel_format_ == kPixelRgb565) {
            red = static_cast<std::uint8_t>(((value >> 11) & 0x1f) * 255 / 31);
            green = static_cast<std::uint8_t>(((value >> 5) & 0x3f) * 255 / 63);
            blue = static_cast<std::uint8_t>((value & 0x1f) * 255 / 31);
          } else {
            red = static_cast<std::uint8_t>(((value >> 10) & 0x1f) * 255 / 31);
            green = static_cast<std::uint8_t>(((value >> 5) & 0x1f) * 255 / 31);
            blue = static_cast<std::uint8_t>((value & 0x1f) * 255 / 31);
          }
        }
        current_frame_.rgb[destination++] = red;
        current_frame_.rgb[destination++] = green;
        current_frame_.rgb[destination++] = blue;
      }
    }
  }

  std::int16_t input_state(unsigned port, unsigned device, unsigned index,
                           unsigned button_id) const {
    if (port != 0 || device != kDeviceJoypad || index != 0 || button_id >= 32) {
      return 0;
    }
    return (button_mask_ & (std::uint32_t{1} << button_id)) != 0 ? 1 : 0;
  }

private:
  template <typename T> T symbol(const char *name) {
    ::dlerror();
    void *address = ::dlsym(library_, name);
    if (const char *error = ::dlerror()) {
      throw std::runtime_error(std::string("missing libretro symbol ") + name + ": " + error);
    }
    return reinterpret_cast<T>(address);
  }

  void load_core() {
    library_ = ::dlopen(core_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      throw std::runtime_error(std::string("unable to load core: ") + ::dlerror());
    }
    retro_set_environment_ = symbol<void (*)(retro_environment_t)>("retro_set_environment");
    retro_set_video_refresh_ = symbol<void (*)(retro_video_refresh_t)>("retro_set_video_refresh");
    retro_set_audio_sample_ = symbol<void (*)(retro_audio_sample_t)>("retro_set_audio_sample");
    retro_set_audio_sample_batch_ =
        symbol<void (*)(retro_audio_sample_batch_t)>("retro_set_audio_sample_batch");
    retro_set_input_poll_ = symbol<void (*)(retro_input_poll_t)>("retro_set_input_poll");
    retro_set_input_state_ = symbol<void (*)(retro_input_state_t)>("retro_set_input_state");
    retro_set_controller_port_device_ =
        symbol<void (*)(unsigned, unsigned)>("retro_set_controller_port_device");
    retro_init_ = symbol<void (*)()>("retro_init");
    retro_deinit_ = symbol<void (*)()>("retro_deinit");
    retro_reset_ = symbol<void (*)()>("retro_reset");
    retro_run_ = symbol<void (*)()>("retro_run");
    retro_get_system_info_ = symbol<void (*)(retro_system_info *)>("retro_get_system_info");
    retro_get_system_av_info_ =
        symbol<void (*)(retro_system_av_info *)>("retro_get_system_av_info");
    retro_load_game_ = symbol<bool (*)(const retro_game_info *)>("retro_load_game");
    retro_unload_game_ = symbol<void (*)()>("retro_unload_game");
    retro_serialize_size_ = symbol<std::size_t (*)()>("retro_serialize_size");
    retro_serialize_ = symbol<bool (*)(void *, std::size_t)>("retro_serialize");
    retro_unserialize_ = symbol<bool (*)(const void *, std::size_t)>("retro_unserialize");

    g_host = this;
    retro_set_environment_(&environment_callback);
    retro_set_video_refresh_(&video_callback);
    retro_set_audio_sample_(&audio_callback);
    retro_set_audio_sample_batch_(&audio_batch_callback);
    retro_set_input_poll_(&input_poll_callback);
    retro_set_input_state_(&input_state_callback);
    retro_init_();
    initialized_ = true;
    retro_set_controller_port_device_(0, kDeviceJoypad);
    retro_get_system_info_(&system_info_);
  }

  void load_content() {
    std::ifstream stream(rom_path_, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("unable to open ROM");
    }
    rom_bytes_ = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(stream), {});
    if (rom_bytes_.empty()) {
      throw std::runtime_error("ROM is empty");
    }
    retro_game_info game{rom_path_.c_str(), rom_bytes_.data(), rom_bytes_.size(), nullptr};
    if (!retro_load_game_(&game)) {
      throw std::runtime_error("libretro core rejected ROM");
    }
    game_loaded_ = true;
    retro_get_system_av_info_(&av_info_);
    run_frame();
    if (current_frame_.rgb.empty()) {
      throw std::runtime_error("libretro core did not produce a frame");
    }
  }

  bool dispatch(Command command, const std::vector<std::uint8_t> &payload) {
    switch (command) {
    case Command::Hello:
      require_empty(payload);
      send_response(command, 0, hello_payload());
      return false;
    case Command::Reset:
      require_empty(payload);
      button_mask_ = 0;
      retro_reset_();
      run_frame();
      send_response(command, 0, frame_payload());
      return false;
    case Command::Step: {
      const auto request = payload_value<StepRequest>(payload);
      if (request.frames == 0 || request.frames > 100000) {
        throw std::runtime_error("step frame count is outside allowed range");
      }
      button_mask_ = request.button_mask;
      for (std::uint32_t frame = 0; frame < request.frames; ++frame) {
        run_frame();
      }
      button_mask_ = 0;
      send_response(command, 0, frame_payload());
      return false;
    }
    case Command::Save: {
      require_empty(payload);
      const std::size_t size = retro_serialize_size_();
      if (size == 0) {
        throw std::runtime_error("core does not support save states");
      }
      Snapshot snapshot;
      snapshot.state.resize(size);
      snapshot.frame = current_frame_;
      if (!retro_serialize_(snapshot.state.data(), snapshot.state.size())) {
        throw std::runtime_error("core failed to serialize state");
      }
      const std::uint64_t handle = next_handle_++;
      snapshots_.emplace(handle, std::move(snapshot));
      std::vector<std::uint8_t> response;
      append_value(response, handle);
      send_response(command, 0, response);
      return false;
    }
    case Command::Load: {
      const auto handle = payload_value<std::uint64_t>(payload);
      const auto snapshot = snapshots_.find(handle);
      if (snapshot == snapshots_.end()) {
        throw std::runtime_error("unknown save-state handle");
      }
      if (!retro_unserialize_(snapshot->second.state.data(), snapshot->second.state.size())) {
        throw std::runtime_error("core rejected registered save state");
      }
      current_frame_ = snapshot->second.frame;
      button_mask_ = 0;
      send_response(command, 0, frame_payload());
      return false;
    }
    case Command::Drop: {
      const auto handle = payload_value<std::uint64_t>(payload);
      if (snapshots_.erase(handle) != 1) {
        throw std::runtime_error("unknown save-state handle");
      }
      send_response(command, 0, {});
      return false;
    }
    case Command::Close:
      require_empty(payload);
      send_response(command, 0, {});
      return true;
    default:
      throw std::runtime_error("unknown protocol command");
    }
  }

  static void require_empty(const std::vector<std::uint8_t> &payload) {
    if (!payload.empty()) {
      throw std::runtime_error("command requires an empty payload");
    }
  }

  std::vector<std::uint8_t> hello_payload() const {
    const std::string name = system_info_.library_name ? system_info_.library_name : "unknown";
    const std::string version =
        system_info_.library_version ? system_info_.library_version : "unknown";
    HelloHeader header{static_cast<std::uint32_t>(name.size()),
                       static_cast<std::uint32_t>(version.size()), av_info_.geometry.base_width,
                       av_info_.geometry.base_height, av_info_.timing.fps};
    std::vector<std::uint8_t> payload;
    append_value(payload, header);
    append_bytes(payload, name.data(), name.size());
    append_bytes(payload, version.data(), version.size());
    return payload;
  }

  std::vector<std::uint8_t> frame_payload() const {
    if (current_frame_.rgb.empty()) {
      throw std::runtime_error("no framebuffer is available");
    }
    const FrameHeader header{current_frame_.width, current_frame_.height, 3};
    std::vector<std::uint8_t> payload;
    payload.reserve(sizeof(header) + current_frame_.rgb.size());
    append_value(payload, header);
    append_bytes(payload, current_frame_.rgb.data(), current_frame_.rgb.size());
    return payload;
  }

  static void send_response(Command command, std::uint16_t status,
                            const std::vector<std::uint8_t> &payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("response exceeds protocol size limit");
    }
    const ResponseHeader response{kMagic, kProtocolVersion, status,
                                  static_cast<std::uint16_t>(command), 0,
                                  static_cast<std::uint32_t>(payload.size())};
    write_exact(STDOUT_FILENO, &response, sizeof(response));
    if (!payload.empty()) {
      write_exact(STDOUT_FILENO, payload.data(), payload.size());
    }
  }

  void run_frame() { retro_run_(); }

  static bool environment_callback(unsigned command, void *data) {
    return g_host && g_host->environment(command, data);
  }
  static void video_callback(const void *data, unsigned width, unsigned height,
                             std::size_t pitch) {
    if (g_host) {
      g_host->video_refresh(data, width, height, pitch);
    }
  }
  static void audio_callback(std::int16_t, std::int16_t) {}
  static std::size_t audio_batch_callback(const std::int16_t *, std::size_t frames) {
    return frames;
  }
  static void input_poll_callback() {}
  static std::int16_t input_state_callback(unsigned port, unsigned device, unsigned index,
                                           unsigned button_id) {
    return g_host ? g_host->input_state(port, device, index, button_id) : 0;
  }

  std::string core_path_;
  std::string rom_path_;
  std::string system_directory_;
  std::string save_directory_;
  void *library_ = nullptr;
  bool initialized_ = false;
  bool game_loaded_ = false;
  int pixel_format_ = kPixel0Rgb1555;
  std::uint32_t button_mask_ = 0;
  std::uint64_t next_handle_ = 1;
  std::vector<std::uint8_t> rom_bytes_;
  Frame current_frame_;
  retro_system_info system_info_{};
  retro_system_av_info av_info_{};
  std::unordered_map<std::uint64_t, Snapshot> snapshots_;

  void (*retro_set_environment_)(retro_environment_t) = nullptr;
  void (*retro_set_video_refresh_)(retro_video_refresh_t) = nullptr;
  void (*retro_set_audio_sample_)(retro_audio_sample_t) = nullptr;
  void (*retro_set_audio_sample_batch_)(retro_audio_sample_batch_t) = nullptr;
  void (*retro_set_input_poll_)(retro_input_poll_t) = nullptr;
  void (*retro_set_input_state_)(retro_input_state_t) = nullptr;
  void (*retro_set_controller_port_device_)(unsigned, unsigned) = nullptr;
  void (*retro_init_)() = nullptr;
  void (*retro_deinit_)() = nullptr;
  void (*retro_reset_)() = nullptr;
  void (*retro_run_)() = nullptr;
  void (*retro_get_system_info_)(retro_system_info *) = nullptr;
  void (*retro_get_system_av_info_)(retro_system_av_info *) = nullptr;
  bool (*retro_load_game_)(const retro_game_info *) = nullptr;
  void (*retro_unload_game_)() = nullptr;
  std::size_t (*retro_serialize_size_)() = nullptr;
  bool (*retro_serialize_)(void *, std::size_t) = nullptr;
  bool (*retro_unserialize_)(const void *, std::size_t) = nullptr;
};

std::pair<std::string, std::string> parse_arguments(int argc, char **argv) {
  std::string core;
  std::string rom;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--core" && index + 1 < argc) {
      core = argv[++index];
    } else if (argument == "--rom" && index + 1 < argc) {
      rom = argv[++index];
    } else {
      throw std::runtime_error("usage: lolo-libretro-host --core PATH --rom PATH");
    }
  }
  if (core.empty() || rom.empty()) {
    throw std::runtime_error("usage: lolo-libretro-host --core PATH --rom PATH");
  }
  return {core, rom};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto [core, rom] = parse_arguments(argc, argv);
    Host host(core, rom);
    return host.run_protocol();
  } catch (const std::exception &error) {
    std::cerr << "lolo-libretro-host: " << error.what() << '\n';
    return 1;
  }
}
