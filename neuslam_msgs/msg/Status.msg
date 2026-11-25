uint8 INITIALIZING=0
uint8 TRACKING=1
uint8 LOST=2

std_msgs/Header header
uint8 state
string mode

uint64 frame_id
uint64 keyframe_id
bool is_keyframe
uint32 num_keyframes
uint32 num_map_points
uint32 num_loop_closures

# Output rate, latency from image stamp to publication, and time inside NeuSLAM
float32 rate_hz
float32 latency_ms
float32 pipeline_ms
uint32 frames_dropped

# True when no TF from base_link to the camera existed and base_link was placed at the left camera
bool base_link_assumed
