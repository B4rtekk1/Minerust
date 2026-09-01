struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    _screen_size:   vec2<f32>,
    _water_level:   f32,
    reflection_mode: f32,
    moon_position:  vec3<f32>,
    _pad1_moon:     f32,
    moon_intensity: f32,
    wind_dir_x:     f32,
    wind_dir_z:     f32,
    wind_speed:     f32,
    rain_factor:    f32,
    sky_visibility: f32,
    menu_blur:      f32,
    _pad_uniforms:  f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// ---------------------------------------------------------------------------
// Stałe
// ---------------------------------------------------------------------------
const TAU:                  f32 = 6.28318530718;
const PI:                   f32 = 3.14159265359;
const MIE_G:                f32 = 0.78;   // asymetria fazy Mie
const STAR_DENSITY:         f32 = 420.0;  // gęstość pola gwiazd
const MOON_DISK_THRESHOLD:  f32 = 0.9985; // próg kąta dysku księżyca
const MOON_DISK_HALF_ANGLE: f32 = 0.0045; // połowa kąta dysku księżyca [rad]

// ---------------------------------------------------------------------------
// Struktury
// ---------------------------------------------------------------------------
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_pos:    vec2<f32>,
    @location(1) world_dir:  vec3<f32>,   // kierunek widoku interpolowany z VS
};

// Zestaw parametrów atmosferycznych obliczanych raz w fs_sky
// i przekazywanych do funkcji pomocniczych.
struct AtmosParams {
    sun_dir:  vec3<f32>,
    moon_dir: vec3<f32>,
    day:      f32,
    dusk:     f32,
    night:    f32,
    sun_h:    f32,
};

// ---------------------------------------------------------------------------
// Vertex shader — kierunek widoku obliczany tu raz, nie w każdym fragmencie
// ---------------------------------------------------------------------------
@vertex
fn vs_sky(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.xy, 0.9999, 1.0);
    out.ndc_pos       = model.position.xy;

    let clip    = vec4<f32>(model.position.xy, 1.0, 1.0);
    let world_h = uniforms.inv_view_proj * clip;
    out.world_dir = world_h.xyz / world_h.w - uniforms.camera_pos;
    return out;
}

// ---------------------------------------------------------------------------
// Hash / noise
// ---------------------------------------------------------------------------
fn hash11(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise. Unlike a sum of visible sine bands, this keeps cloud
// silhouettes irregular and avoids the repeating "wavy" pattern at the horizon.
fn base_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    let a = hash21(cell);
    let b = hash21(cell + vec2<f32>(1.0, 0.0));
    let c = hash21(cell + vec2<f32>(0.0, 1.0));
    let d = hash21(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fraktalny fbm — 5 oktaw; znacznie organiczniejszy wygląd chmur
fn fbm(p: vec2<f32>) -> f32 {
    var val  = 0.0;
    var amp  = 0.5;
    var freq = 1.0;
    var pp   = p;
    for (var i = 0; i < 5; i++) {
        val  += amp * base_noise(pp * freq);
        amp  *= 0.5;
        freq *= 2.07;
        pp   += vec2<f32>(0.31, 0.17); // offsety by uniknąć symetrii
    }
    return val;
}

// ---------------------------------------------------------------------------
// Fazy rozpraszania
// ---------------------------------------------------------------------------
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2    = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 0.001), 1.5);
    return 1.5 * (1.0 - g2) * (1.0 + cos_theta * cos_theta) / ((2.0 + g2) * denom);
}

// ---------------------------------------------------------------------------
// Gradient atmosferyczny
// ---------------------------------------------------------------------------
fn atmospheric_gradient(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    let h = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);

    let zenith_day   = vec3<f32>(0.18, 0.44, 0.88);
    let horizon_day  = vec3<f32>(0.66, 0.82, 0.98);
    let zenith_night = vec3<f32>(0.002, 0.005, 0.015);
    let horizon_night= vec3<f32>(0.010, 0.014, 0.034);
    let dusk_low     = vec3<f32>(0.96, 0.40, 0.18);
    let dusk_high    = vec3<f32>(0.24, 0.10, 0.28);

    var sky = mix(horizon_day,   zenith_day,   pow(h, 0.78)) * ap.day;
    sky    += mix(horizon_night, zenith_night, pow(h, 0.72)) * ap.night;
    sky    += mix(dusk_low,      dusk_high,    pow(h, 1.1))  * ap.dusk * 0.52;

    let horizon_band = 1.0 - smoothstep(0.0, 0.32, abs(view_dir.y));
    sky += vec3<f32>(0.16, 0.22, 0.30) * horizon_band * (0.55 * ap.day + 0.12 * ap.dusk);

    return sky;
}

// ---------------------------------------------------------------------------
// Rozpraszanie atmosferyczne
// ---------------------------------------------------------------------------
fn atmospheric_scatter(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    let cos_theta = dot(view_dir, ap.sun_dir);
    let mu        = clamp(cos_theta, -1.0, 1.0);
    let rayleigh  = rayleigh_phase(mu);
    let mie       = mie_phase(mu, MIE_G);

    let view_height = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
    let altitude    = pow(view_height, 0.6);
    let horizon     = 1.0 - smoothstep(0.0, 0.42, abs(view_dir.y));

    let rayleigh_color = vec3<f32>(0.14, 0.28, 0.62) * rayleigh;
    let mie_color      = vec3<f32>(1.0, 0.72, 0.42)  * mie;
    let dusk_tint      = mix(vec3<f32>(1.0, 0.40, 0.16), vec3<f32>(0.60, 0.28, 0.50), altitude);

    var scatter  = rayleigh_color * (0.45 * ap.day + 0.12 * ap.dusk);
    scatter     += mie_color      * (0.30 * ap.day + 0.45 * ap.dusk) * horizon;
    scatter     += dusk_tint      * ap.dusk * horizon * 0.20;
    scatter     += vec3<f32>(0.010, 0.012, 0.020) * ap.night * altitude;

    return scatter;
}

// ---------------------------------------------------------------------------
// Słońce  (pow zastąpione mnożeniami — ~5× szybciej na GPU)
// ---------------------------------------------------------------------------
fn sun_glow(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    // Rozszerzony zakres fade — miększe przejście przy wschodzie/zachodzie
    let above = smoothstep(-0.12, 0.04, ap.sun_h);
    if above <= 0.001 { return vec3<f32>(0.0); }

    let cos_s = clamp(dot(view_dir, ap.sun_dir), 0.0, 1.0);

    // pow(cos_s, 160) przez mnożenia sekwencyjne
    let c2   = cos_s  * cos_s;
    let c4   = c2     * c2;
    let c8   = c4     * c4;
    let c16  = c8     * c8;
    let c32  = c16    * c16;
    let c64  = c32    * c32;
    let c128 = c64    * c64;
    let core = c128   * c32;                       // ≈ cos^160

    // pow(cos_s, 36) = c32 * c4
    let inner = c32 * c4;

    // pow(cos_s, 8) = c8
    let outer = c8;

    // pow(cos_s, 2.5) — tu pow() jest nieunikniony (niecałkowity wykładnik)
    let halo = pow(cos_s, 2.5) * 0.03;

    let horizon_boost = 1.0 - smoothstep(-0.12, 0.42, ap.sun_h);
    let warm = mix(vec3<f32>(1.0, 0.82, 0.54), vec3<f32>(1.0, 0.97, 0.93),
                   smoothstep(0.12, 0.75, ap.sun_h));

    return warm * (core * 1.8 + inner * 0.52 + outer * 0.12 + halo * 0.8)
        * pow(above, 0.6) * (0.65 + 0.45 * horizon_boost);
}

// ---------------------------------------------------------------------------
// Chmury z fbm (organiczniejszy wygląd) + stopniowe ściemnianie nocą
// ---------------------------------------------------------------------------
fn cloud_layer(view_dir: vec3<f32>, ap: AtmosParams, time: f32) -> vec3<f32> {
    let cloud_vis = clamp(ap.day * 1.4 + ap.dusk * 0.8 + ap.night * 0.12, 0.0, 1.0);
    if cloud_vis <= 0.001 { return vec3<f32>(0.0); }

    let band = pow(clamp(1.0 - abs(view_dir.y) * 1.18, 0.0, 1.0), 1.7);
    if band <= 0.001 { return vec3<f32>(0.0); }

    let lon   = atan2(view_dir.z, view_dir.x) / TAU + 0.5;
    let lat   = view_dir.y * 0.5 + 0.5;
    let p     = vec2<f32>(lon * 5.2 + time * 0.0025, lat * 2.2);
    let drift = vec2<f32>(time * 0.008, time * 0.003);
    let n       = fbm(p + drift * 0.18);
    let n2      = fbm(p * 1.8 - drift * 0.8);

    let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
    let cov_thresh = mix(0.72, 0.48, rain);
    let coverage = smoothstep(cov_thresh, cov_thresh + 0.28, n * 0.70 + n2 * 0.30);
    let wisps    = smoothstep(0.52, 0.78, fbm(p * 2.9 + drift * 1.3));
    let layer    = coverage * band * band;
    if layer <= 0.001 { return vec3<f32>(0.0); }

    let sun_light    = pow(max(dot(view_dir, ap.sun_dir), 0.0), 12.0) * 0.30;
    let horizon_warm = vec3<f32>(1.0, 0.74, 0.50);
    let cloud_day    = vec3<f32>(0.90, 0.93, 0.97);
    let cloud_dim = vec3<f32>(0.48, 0.54, 0.65);
    let cloud_night  = vec3<f32>(0.06, 0.08, 0.14);
    let cloud_storm  = vec3<f32>(0.22, 0.26, 0.31);
    let dusk_tint    = mix(vec3<f32>(1.0, 0.64, 0.42), vec3<f32>(0.78, 0.36, 0.48), lat);

    var tint  = mix(cloud_dim, cloud_day, ap.day);
    tint      = mix(tint, cloud_night, ap.night * 0.85);
    tint      = mix(tint, dusk_tint,  ap.dusk * 0.75);
    tint      = mix(tint, cloud_storm, rain * 0.88);
    tint     += horizon_warm * (1.0 - smoothstep(0.0, 0.20, abs(view_dir.y))) * ap.dusk * 0.16;

    return tint * layer * cloud_vis
        * (0.24 + 0.34 * ap.day + sun_light * 0.20) * mix(1.0, 1.32, rain)
        * (0.82 + wisps * 0.18);
}

// ---------------------------------------------------------------------------
// Mgła horyzontalna
// ---------------------------------------------------------------------------
fn horizon_haze(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    let band = pow(1.0 - smoothstep(0.0, 0.36, abs(view_dir.y)), 1.4);
    let warm  = vec3<f32>(0.98, 0.58, 0.30);
    let cool  = vec3<f32>(0.20, 0.32, 0.48);
    return mix(cool, warm, ap.dusk) * band * (0.10 + 0.05 * ap.day);
}

// ---------------------------------------------------------------------------
// Pole gwiazd z migotaniem
// ---------------------------------------------------------------------------
fn star_field(view_dir: vec3<f32>, time: f32, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 { return vec3<f32>(0.0); }

    let p = floor(view_dir * STAR_DENSITY);
    let h = hash21(p.xz + vec2<f32>(p.y, p.y * 0.37));
    if h < 0.978 { return vec3<f32>(0.0); }

    let twinkle    = 0.75 + 0.25 * sin(time * mix(1.5, 4.0, hash11(h * 73.0)) + h * TAU);
    let horizon    = smoothstep(0.02, 0.16, view_dir.y);
    let brightness = smoothstep(0.978, 0.9995, h) * twinkle * horizon * night_factor;

    let warm_star = vec3<f32>(1.00, 0.90, 0.74);
    let cool_star = vec3<f32>(0.78, 0.88, 1.00);
    let tint      = mix(warm_star, cool_star, hash11(h * 91.7));

    return tint * brightness * 1.25;
}

// ---------------------------------------------------------------------------
// Smugi meteorów (shooting stars) — animowane w czasie
// ---------------------------------------------------------------------------
fn shooting_stars(view_dir: vec3<f32>, time: f32, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 { return vec3<f32>(0.0); }

    let horizon = smoothstep(0.02, 0.14, view_dir.y);
    if horizon < 0.001 { return vec3<f32>(0.0); }

    var result = vec3<f32>(0.0);

    // 3 niezależne meteory z różnymi fazami
    for (var i = 0u; i < 3u; i++) {
        let fi     = f32(i);
        let period = 7.0 + fi * 4.3;                    // okresy: 7, 11.3, 15.6 s
        let t      = fract((time * 0.2 + fi * 0.37) / period);
        if t > 0.18 { continue; }                        // smuga widoczna przez 18% okresu

        let fade = smoothstep(0.0, 0.04, t) * smoothstep(0.18, 0.10, t);
        if fade < 0.001 { continue; }

        // Losowy kierunek startu i trajektoria dla każdego meteoru
        let seed   = fi * 13.7;
        let start  = normalize(vec3<f32>(
            sin(seed * 1.3) * 0.8, 0.3 + hash11(seed) * 0.5, cos(seed * 2.1) * 0.8));
        let trail  = normalize(vec3<f32>(sin(seed * 0.7), -0.4, cos(seed * 1.9)));

        // Pozycja głowicy meteoru wzdłuż trajektorii
        let head_pos = normalize(start + trail * t * 3.5);
        let cos_h    = dot(view_dir, head_pos);

        // Smuga: punkty wzdłuż trail za głowicą
        var streak = 0.0;
        for (var s = 0u; s < 8u; s++) {
            let sf      = f32(s) / 7.0;
            let seg_pos = normalize(start + trail * (t - sf * 0.06) * 3.5);
            let cos_seg = dot(view_dir, seg_pos);
            streak     += pow(max(cos_seg, 0.0), 800.0) * (1.0 - sf * 0.85);
        }

        result += vec3<f32>(0.95, 0.92, 0.85) * streak * fade * night_factor * horizon * 2.5;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Aurora borealis — pionowe kurtyny przy głębokiej nocy
// ---------------------------------------------------------------------------
fn aurora(view_dir: vec3<f32>, time: f32, night_factor: f32) -> vec3<f32> {
    // Pojawia się tylko przy mocnej nocy i dość wysoko nad horyzontem
    let strength = smoothstep(0.65, 0.95, night_factor);
    if strength < 0.001 { return vec3<f32>(0.0); }

    // Aurora żyje tylko powyżej horyzontu
    let h = clamp(view_dir.y, 0.0, 1.0);
    if h < 0.02 { return vec3<f32>(0.0); }

    let lon = atan2(view_dir.z, view_dir.x) / TAU; // 0..1

    // Pionowe kurtyny — kilka harmonicznych w osi lon
    let wave1 = sin(lon * 8.0  + time * 0.18) * 0.5 + 0.5;
    let wave2 = sin(lon * 13.0 - time * 0.11) * 0.5 + 0.5;
    let wave3 = sin(lon * 5.0  + time * 0.07 + 1.4) * 0.5 + 0.5;
    let curtain = wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2;

    // Pionowa obwiednia: wąski pas na średnich wysokościach
    let height_band = smoothstep(0.0, 0.12, h) * smoothstep(0.72, 0.30, h);

    // Wewnętrzne migotanie
    let shimmer = sin(lon * 42.0 + time * 1.3) * 0.5 + 0.5;
    let intensity = curtain * height_band * (0.6 + shimmer * 0.4);

    if intensity < 0.001 { return vec3<f32>(0.0); }

    // Gradient koloru: zielono-teal u dołu → fioletowy u góry
    let green  = vec3<f32>(0.10, 0.72, 0.38);
    let purple = vec3<f32>(0.42, 0.12, 0.68);
    let teal   = vec3<f32>(0.05, 0.60, 0.72);
    let hue    = mix(mix(green, teal, wave2), purple, h * 0.6);

    return hue * intensity * strength * 0.45;
}

// ---------------------------------------------------------------------------
// Dysk księżyca z detalem powierzchni i bezpiecznym tangent-frame
// ---------------------------------------------------------------------------
fn moon_disk(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    if ap.night < 0.01 { return vec3<f32>(0.0); }

    let cos_m = dot(view_dir, ap.moon_dir);

    if cos_m < MOON_DISK_THRESHOLD {
        let halo = pow(max(cos_m, 0.0), 64.0) * 0.03
                 + pow(max(cos_m, 0.0), 12.0) * 0.008;
        return vec3<f32>(0.52, 0.58, 0.76) * halo * ap.night;
    }

    // Bezpieczny tangent-frame — unika degeneracji gdy moon_dir ≈ up
    let up_ref = select(vec3<f32>(0.0, 1.0, 0.0),
                        vec3<f32>(1.0, 0.0, 0.0),
                        abs(ap.moon_dir.y) > 0.99);
    let right  = normalize(cross(ap.moon_dir, up_ref));
    let up     = normalize(cross(right, ap.moon_dir));

    let disk_uv = vec2<f32>(
        dot(view_dir - ap.moon_dir, right),
        dot(view_dir - ap.moon_dir, up)
    ) / MOON_DISK_HALF_ANGLE;

    let disk_r2 = dot(disk_uv, disk_uv);
    if disk_r2 > 1.0 { return vec3<f32>(0.0); }

    let limb           = sqrt(max(0.0, 1.0 - disk_r2));
    let limb_darkening = mix(0.58, 1.0, limb);
    let disk_mask      = 1.0 - smoothstep(0.95, 1.0, disk_r2);

    // Subtelny detail powierzchni — kratery przez hash noise
    let crater_noise = hash21(disk_uv * 8.0)  * 0.04
                     + hash21(disk_uv * 19.0) * 0.02
                     - 0.03;
    let surface_detail = 1.0 + crater_noise;

    let surface_tint = mix(
        vec3<f32>(0.92, 0.88, 0.74),
        vec3<f32>(0.85, 0.91, 0.98),
        clamp(ap.moon_dir.y * 2.0, 0.0, 1.0),
    );

    let halo = vec3<f32>(0.52, 0.58, 0.76)
        * (pow(max(cos_m, 0.0), 20.0) * 0.014
         + pow(max(cos_m, 0.0), 72.0) * 0.045);

    return (surface_tint * limb_darkening * surface_detail * disk_mask + halo) * ap.night;
}

fn rainbow(view_dir: vec3<f32>, ap: AtmosParams) -> vec3<f32> {
    let rain_vis = smoothstep(0.04, 0.42, uniforms.rain_factor)
        * (1.0 - smoothstep(0.72, 1.0, uniforms.rain_factor));
    let vis = smoothstep(0.02, 0.18, ap.sun_h)
        * (1.0 - smoothstep(0.35, 0.55, ap.sun_h)) * rain_vis;
    if vis < 0.001 { return vec3<f32>(0.0); }

    let anti_sun  = normalize(vec3<f32>(-ap.sun_dir.x, ap.sun_dir.y, -ap.sun_dir.z));
    let cos_angle = clamp(dot(view_dir, anti_sun), -1.0, 1.0);
    let angle_deg = degrees(acos(cos_angle));   // 0..180°

    let arc       = 1.0 - smoothstep(0.0, 1.0, abs(angle_deg - 42.0) / 2.0);
    if arc < 0.001 { return vec3<f32>(0.0); }

    let arc_r = 1.0 - smoothstep(0.0, 1.0, abs(angle_deg - 42.5) / 1.4);
    let arc_g = 1.0 - smoothstep(0.0, 1.0, abs(angle_deg - 42.0) / 1.4);
    let arc_b = 1.0 - smoothstep(0.0, 1.0, abs(angle_deg - 41.5) / 1.4);

    let height_mask = smoothstep(-0.05, 0.15, view_dir.y);

    return vec3<f32>(arc_r, arc_g, arc_b) * arc * height_mask * vis * 0.28;
}

fn underwater_color(ndc_pos: vec2<f32>, view_dir: vec3<f32>, time: f32) -> vec4<f32> {
    let uv       = ndc_pos * 0.5 + 0.5;
    let ripples  = sin(uv.x * 12.0 + time * 1.8) * sin(uv.y * 8.5 - time * 1.2);
    let caustics = ripples * 0.03 + sin((uv.x + uv.y) * 16.0 + time * 0.7) * 0.015;

    let depth_fog  = mix(vec3<f32>(0.012, 0.075, 0.22), vec3<f32>(0.05, 0.20, 0.36), uv.y);
    let up         = max(0.0, view_dir.y);
    let light_rays = up * up * (sin(uv.x * 18.0 + time * 2.0) * 0.5 + 0.5) * 0.10;
    let tint       = vec3<f32>(0.18, 0.42, 0.58) + vec3<f32>(0.08, 0.12, 0.14) * caustics;

    return vec4<f32>((depth_fog + tint + light_rays) * (1.0 + caustics), 1.0);
}

@fragment
fn fs_sky(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(in.world_dir);

    if uniforms.is_underwater > 0.5 {
        return underwater_color(in.ndc_pos, view_dir, uniforms.time);
    }

    var ap: AtmosParams;
    ap.sun_dir  = normalize(uniforms.sun_position);
    ap.moon_dir = normalize(uniforms.moon_position);
    ap.sun_h    = ap.sun_dir.y;
    ap.day      = smoothstep(-0.02, 0.22,  ap.sun_h);
    ap.dusk     = 1.0 - smoothstep(0.02, 0.34, abs(ap.sun_h));
    ap.night    = clamp(-ap.sun_h * 4.0 - 0.05, 0.0, 1.0);

    let open_sky = clamp(uniforms.sky_visibility, 0.0, 1.0)
        * select(0.0, 1.0, uniforms.camera_pos.y >= 0.0);
    let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
    let celestial_visibility = open_sky * mix(1.0, 0.08, rain);
    let night_sky = ap.night * celestial_visibility;

    var sky = atmospheric_gradient(view_dir, ap);
    sky    += atmospheric_scatter(view_dir, ap);
    sky    += horizon_haze(view_dir, ap);
    // Overcast weather keeps a brighter horizon and a darker zenith, matching
    // the terrain fog while retaining enough contrast to read cloud layers.
    let overcast_h = pow(clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0), 0.72);
    let overcast = mix(vec3<f32>(0.43, 0.47, 0.51), vec3<f32>(0.15, 0.18, 0.23), overcast_h);
    sky = mix(sky, overcast, rain * 0.74);

    sky    += sun_glow(view_dir, ap) * celestial_visibility;
    sky    += cloud_layer(view_dir, ap, uniforms.time) * open_sky;
    sky    += star_field(view_dir, uniforms.time, night_sky);
    sky    += shooting_stars(view_dir, uniforms.time, night_sky);
    sky    += aurora(view_dir, uniforms.time, night_sky);
    sky    += moon_disk(view_dir, ap) * celestial_visibility;
    sky    += rainbow(view_dir, ap) * open_sky;

    if ap.dusk > 0.01 {
        let warm_band = pow(max(0.0, 1.0 - abs(view_dir.y)), 2.6) * ap.dusk;
        sky += vec3<f32>(1.0, 0.40, 0.16) * warm_band * 0.08;
    }

    sky *= vec3<f32>(0.98, 0.99, 1.02);
    sky  = clamp(sky, vec3<f32>(0.0), vec3<f32>(1.8));

    return vec4<f32>(sky, 1.0);
}
