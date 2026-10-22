//! glm5_next/cuda/multi.py TP control: shared Flash Next records (Ash Hart, TensorFold; b12x RoCEnante).
//! Leader/Follower accept any lanes.backend.Backend, including cuda_lanes.Lanes(E).
const impl = @import("flashnext").mirror;
pub const Leader = impl.Leader;
pub const Follower = impl.Follower;
pub const Op = impl.Op;
pub const Records = impl.Records;
pub const Frames = impl.Frames;
pub const Broadcast = impl.Broadcast;
pub const Agreement = impl.Agreement;
pub const Watch = impl.Watch;
pub const Coll = impl.Coll;
pub const Channel = impl.Channel;
pub const Call = impl.Call;
pub const WindowCall = impl.WindowCall;
pub const DraftCall = impl.DraftCall;
pub const StreamSpec = impl.StreamSpec;
