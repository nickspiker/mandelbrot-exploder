// Scaled f32 perturbation loop — the compute twin of `run_f32` in perturbation.rs.
// One invocation per pixel; the host resumes a pixel by resetting `status` to 0.
// `State` must match `DeltaState` field for field (120 bytes).

struct Params {
    s: f32,          // zoom magnitude S
    ref_last: u32,   // index of the reference orbit's last entry
    periodic: u32,   // 1: Z[ref_last] == 0 == Z[0], indices wrap
    count: u32,
};

struct State {
    dz: vec2<f32>,
    dcs: vec2<f32>,
    orbit: vec2<f32>,
    coord_acc: vec2<f32>,
    flame_u: vec2<f32>,
    flame_w: vec2<f32>,
    z_out: vec2<f32>,
    z_min: f32,
    dist_sum: f32,
    energy_u: f32,
    energy_w: f32,
    probe_acc: f32,
    m: u32,
    iter: u32,
    limit: u32,
    status: u32,
    probe_n: u32,
    probe_start: u32,
    probe_next: u32,
    probe_windows: u32,
    probe_blocked: u32,
    period: u32,
    pad: u32,
};

const STATUS_RUNNING: u32 = 0u;
const STATUS_ESCAPED: u32 = 1u;
const STATUS_CONFIRMED: u32 = 2u;
const STATUS_DEAD: u32 = 3u;
const PROBE_WINDOWS: u32 = 8u;
const PROBE_SATURATE: f32 = 1e20;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> zref: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> states: array<State>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn probe_arm(st: ptr<function, State>, n: u32, start: u32) {
    (*st).probe_n = n;
    (*st).probe_start = start;
    (*st).probe_next = start + n;
    (*st).probe_acc = 1.0;
    (*st).probe_windows = 0u;
}

fn probe_failed(st: ptr<function, State>) {
    if ((*st).probe_blocked != 0u) {
        let n = (*st).probe_blocked;
        (*st).probe_blocked = 0u;
        let start = (((*st).iter + n - 1u) / n) * n;
        probe_arm(st, n, start);
    }
}

// Window end: true confirms the candidate.
fn probe_window_end(st: ptr<function, State>) -> bool {
    if ((*st).probe_acc < 1.0) {
        (*st).probe_n = 0u;
        return true;
    }
    (*st).probe_windows += 1u;
    if ((*st).probe_windows < PROBE_WINDOWS) {
        (*st).probe_start = (*st).iter;
        (*st).probe_next = (*st).iter + (*st).probe_n;
        (*st).probe_acc = 1.0;
        return false;
    }
    // Windows exhausted: confirm anyway (see run_f32).
    (*st).probe_n = 0u;
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    var st = states[i];
    if (st.status != STATUS_RUNNING) {
        return;
    }
    let s = params.s;
    let last = params.ref_last;
    let periodic = params.periodic != 0u;
    let dcs = st.dcs;

    loop {
        if (st.iter >= st.limit) {
            break;
        }
        let zm = zref[st.m];
        let z = zm + st.dz * s;
        let z_mag_sq = dot(z, z);
        // NaN and inf both fail this; nothing pre-escape is legitimately above 4.
        if (!(z_mag_sq < 1e30)) {
            st.status = STATUS_DEAD;
            break;
        }
        if (z_mag_sq > 4.0) {
            st.z_out = z;
            st.status = STATUS_ESCAPED;
            break;
        }
        if (st.probe_n != 0u && st.iter == st.probe_next) {
            let n = st.probe_n;
            if (probe_window_end(&st)) {
                st.period = n;
                st.status = STATUS_CONFIRMED;
                break;
            }
        }
        if (st.probe_n != 0u && st.iter >= st.probe_start) {
            st.probe_acc = min(st.probe_acc * 4.0 * z_mag_sq, PROBE_SATURATE);
        }
        let it = f32(st.iter);
        let iter_sq = it * it;
        let z_mag = sqrt(z_mag_sq);
        st.energy_u += 1.0;
        st.flame_u += z / z_mag;
        st.energy_w += iter_sq;
        st.flame_w += z * (iter_sq / z_mag);
        let w = iter_sq / z_mag_sq;
        st.dist_sum += w;
        st.coord_acc += z * w;

        if (st.z_min > z_mag_sq) {
            let ratio = z_mag_sq / st.z_min;
            st.z_min = z_mag_sq;
            if (ratio < 0.25) {
                if (st.probe_n != 0u && st.iter % st.probe_n == 0u) {
                    st.probe_blocked = st.iter;
                } else {
                    st.probe_blocked = 0u;
                    probe_arm(&st, st.iter, st.iter);
                }
            }
        }

        st.orbit = cmul(st.orbit, z) * 2.0 + vec2<f32>(s, 0.0);
        st.dz = cmul(zm + z, st.dz) + dcs;
        st.m += 1u;
        st.iter += 1u;
        if (st.m >= last) {
            if (periodic && st.m == last) {
                st.m = 0u;
            } else {
                let zn = zref[min(last, st.m)] + st.dz * s;
                st.dz = zn / s;
                st.m = 0u;
            }
        } else {
            let t = st.dz * s;
            let zn = zref[st.m] + t;
            if (dot(zn, zn) < dot(t, t)) {
                st.dz = zn / s;
                st.m = 0u;
            }
        }
    }
    if (st.iter >= st.limit && st.status == STATUS_RUNNING) {
        st.status = STATUS_RUNNING;
    }
    states[i] = st;
}
