// ─────────────────────────────────────────────────────────────────────────────
// Terrain spline system
// ─────────────────────────────────────────────────────────────────────────────

/// A single (input, output) control point on a [`TerrainSpline`].
///
/// Points must be stored in strictly ascending `input` order for [`TerrainSpline::sample`]
/// to produce correct results.
#[derive(Clone, Copy)]
struct SplinePoint {
    /// The noise value at which this control point is defined, typically in `[-1, 1]`.
    input: f64,
    /// The world-space output value (e.g., block height) at this control point.
    output: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// TerrainSpline
// ─────────────────────────────────────────────────────────────────────────────

/// A piecewise Catmull-Rom spline that maps raw noise values to terrain
/// height or modifier outputs.
///
/// The spline is defined by an ordered list of `(input, output)` control
/// points and evaluated by [`sample`].  Between control points the curve is
/// interpolated with the Catmull-Rom formula, giving smooth, continuous
/// derivatives across segment boundaries.  Outside the defined range the
/// curve is clamped to the first or last output value (no extrapolation).
///
/// # Pre-built splines
///
/// Three named constructors are provided for the chunk generator:
///
/// | Spline | Input source | Output meaning |
/// |---|---|---|
/// | [`continental`] | `noise_continents` | Base block height for land/ocean. |
/// | [`erosion`] | `noise_erosion` | Multiplier applied to local terrain amplitude. |
/// | [`peaks_valleys`] | `noise_pv` | Height offset added to mountain peaks. |
///
/// # Usage
/// ```rust
/// let spline = TerrainSpline::continental();
/// let height = spline.sample(continent_noise); // → world-space Y
/// ```
pub struct TerrainSpline {
    /// Control points in ascending `input` order.
    points: Vec<SplinePoint>,
}

impl TerrainSpline {
    /// Constructs a spline from a slice of `(input, output)` pairs.
    ///
    /// Pairs should be provided in **ascending input order**.  No sorting is
    /// performed internally; out-of-order points will produce incorrect
    /// interpolation results.
    ///
    /// # Parameters
    /// - `pairs` – Ordered control points as `(noise_value, output_value)` tuples.
    pub fn new(pairs: &[(f64, f64)]) -> Self {
        let points = pairs
            .iter()
            .map(|(i, o)| SplinePoint {
                input:  *i,
                output: *o,
            })
            .collect();
        Self { points }
    }

    /// Evaluates the spline at `t` using Catmull-Rom interpolation.
    ///
    /// # Algorithm
    ///
    /// 1. **Boundary clamping** – values at or beyond the first/last input are
    ///    returned as the corresponding endpoint output with no interpolation.
    /// 2. **Segment search** – walks the control-point list to find the segment
    ///    `[p1, p2]` that contains `t` (i.e., `p1.input ≤ t < p2.input`).
    /// 3. **Ghost points** – Catmull-Rom requires four points `(p0, p1, p2, p3)`.
    ///    When the segment is at either end of the list the missing neighbor is
    ///    duplicated from the nearest available point, effectively producing a
    ///    zero-slope tangent at the boundary.
    /// 4. **`segment_t`** – the parameter `t` is rescaled to `[0, 1]` within
    ///    the current segment before being passed to [`catmull_rom`].
    ///
    /// # Parameters
    /// - `t` – The noise value to evaluate, typically in `[-1.0, 1.0]`.
    ///
    /// # Returns
    /// The interpolated output value at `t`, clamped at the endpoints.
    pub fn sample(&self, t: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if self.points.len() == 1 {
            return self.points[0].output;
        }

        let first = &self.points[0];
        let last  = &self.points[self.points.len() - 1];

        // Clamp: do not extrapolate beyond the defined range.
        if t <= first.input {
            return first.output;
        }
        if t >= last.input {
            return last.output;
        }

        // Find the segment index `i` such that `points[i].input ≤ t < points[i+1].input`.
        let mut i = 0;
        while i < self.points.len() - 1 && self.points[i + 1].input < t {
            i += 1;
        }

        // Catmull-Rom ghost-point handling: duplicate endpoint when the segment
        // is at the beginning or end of the control-point list.
        let p0 = if i > 0 {
            &self.points[i - 1]
        } else {
            &self.points[i] // duplicate p1 as p0 → zero incoming tangent
        };
        let p1 = &self.points[i];
        let p2 = &self.points[i + 1];
        let p3 = if i + 2 < self.points.len() {
            &self.points[i + 2]
        } else {
            &self.points[i + 1] // duplicate p2 as p3 → zero outgoing tangent
        };

        // Rescale `t` from the segment's input range to [0, 1].
        let segment_t = (t - p1.input) / (p2.input - p1.input);

        catmull_rom(p0.output, p1.output, p2.output, p3.output, segment_t)
    }

    // ── Pre-built named splines ───────────────────────────────────────────── //

    /// Returns the **continental** spline, which converts `noise_continents`
    /// output into a base block height.
    ///
    /// The curve rises steeply from the ocean floor (~25 blocks) through the
    /// coast (~58 blocks) to the lowlands (~68 blocks) and continues climbing
    /// to extreme mountain peaks at 200 blocks for the highest continental
    /// values.  These heights are the *starting point* before erosion and
    /// peaks-and-valleys offsets are applied.
    ///
    /// | Input  | Output (blocks) | Description        |
    /// |--------|-----------------|--------------------|
    /// | −1.05  | 25              | Deep ocean floor   |
    /// | −0.50  | 40              | Open ocean         |
    /// | −0.20  | 58              | Coast / Beach      |
    /// | −0.10  | 62              | Shore              |
    /// |  0.00  | 68              | Lowlands           |
    /// |  0.20  | 76              | Plains             |
    /// |  0.40  | 90              | Hills              |
    /// |  0.60  | 120             | Highlands          |
    /// |  0.80  | 160             | Mountains          |
    /// |  1.00  | 200             | Extreme mountains  |
    pub fn continental() -> Self {
        Self::new(&[
            (-1.05, 25.0),  // Deep ocean floor
            (-0.5,  40.0),  // Ocean
            (-0.2,  58.0),  // Coast/Beach
            (-0.1,  62.0),  // Shore
            (0.0,   68.0),  // Lowlands
            (0.2,   76.0),  // Plains
            (0.4,   90.0),  // Hills
            (0.6,  120.0),  // Highlands
            (0.8,  160.0),  // Mountains
            (1.0,  200.0),  // Extreme mountains
        ])
    }

    /// Returns the **erosion** spline, which converts `noise_erosion` output
    /// into a terrain amplitude multiplier.
    ///
    /// The multiplier is applied to local terrain noise before adding it to the
    /// base height, so high erosion values (close to 1.0) produce flatter
    /// terrain and low values (close to −1.0) produce deep canyons and sharp
    /// relief.
    ///
    /// | Input  | Multiplier | Description              |
    /// |--------|------------|--------------------------|
    /// | −1.00  | 1.5        | Very rough (deep canyons)|
    /// | −0.50  | 1.2        | Rough                    |
    /// |  0.00  | 1.0        | Normal                   |
    /// |  0.50  | 0.6        | Smooth                   |
    /// |  1.00  | 0.3        | Very smooth (flat plains)|
    pub fn erosion() -> Self {
        Self::new(&[
            (-1.0, 1.5), // Very rough (deep canyons)
            (-0.5, 1.2), // Rough
            (0.0,  1.0), // Normal
            (0.5,  0.6), // Smooth
            (1.0,  0.3), // Very smooth (flat plains)
        ])
    }

    /// Returns the **peaks-and-valleys** spline, which converts `noise_pv`
    /// output into a signed height offset.
    ///
    /// Negative outputs push terrain down into valleys; positive outputs push
    /// it up into peaks.  The steep climb between 0.5 and 1.0 (+55 blocks)
    /// concentrates sharp mountain summits in a narrow noise band, while the
    /// valley side has a gentler gradient.
    ///
    /// Only the positive half (peaks) is used in the mountain biome calculation
    /// (`pv_offset.max(0.0) * 0.6`), so the spline doubles as a one-sided
    /// peak amplifier when sampled that way.
    ///
    /// | Input  | Offset (blocks) | Description   |
    /// |--------|-----------------|---------------|
    /// | −1.00  | −40             | Deep valley   |
    /// | −0.50  | −15             | Shallow valley|
    /// |  0.00  |   0             | Flat          |
    /// |  0.50  |  25             | Hill          |
    /// |  1.00  |  80             | Sharp peak    |
    pub fn peaks_valleys() -> Self {
        Self::new(&[
            (-1.0, -40.0), // Deep valley
            (-0.5, -15.0), // Shallow valley
            (0.0,    0.0), // Flat
            (0.5,   25.0), // Hill
            (1.0,   80.0), // Sharp peak
        ])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Catmull-Rom interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates the Catmull-Rom cubic interpolant at parameter `t ∈ [0, 1]`.
///
/// Given four collinear control values `(p0, p1, p2, p3)`, the curve passes
/// through `p1` (at `t = 0`) and `p2` (at `t = 1`).  `p0` and `p3` act as
/// phantom points that define the tangent directions at the endpoints:
///
/// ```text
/// tangent at p1 = (p2 - p0) / 2
/// tangent at p2 = (p3 - p1) / 2
/// ```
///
/// The standard uniform Catmull-Rom matrix formulation used here is:
///
/// ```text
/// f(t) = 0.5 × [1, t, t², t³] × ⎡  0   2   0   0 ⎤ × [p0, p1, p2, p3]ᵀ
///                                 ⎢ -1   0   1   0 ⎥
///                                 ⎢  2  -5   4  -1 ⎥
///                                 ⎣ -1   3  -3   1 ⎦
/// ```
///
/// Expanding gives the form used in the implementation:
/// ```text
/// 0.5 × (2·p1 + (-p0 + p2)·t + (2·p0 - 5·p1 + 4·p2 - p3)·t²
///              + (-p0 + 3·p1 - 3·p2 + p3)·t³)
/// ```
///
/// # Parameters
/// - `p0` – Control value before the segment start (used for tangent only).
/// - `p1` – Value at `t = 0` (segment start; curve passes through this).
/// - `p2` – Value at `t = 1` (segment end; curve passes through this).
/// - `p3` – Control value after the segment end (used for tangent only).
/// - `t`  – Interpolation parameter in `[0, 1]`.
///
/// # Returns
/// Interpolated value between `p1` and `p2` with smooth tangents.
fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Linearly interpolates between two height values.
///
/// At `blend_factor = 0.0` the result is `h1`; at `blend_factor = 1.0` the
/// result is `h2`.  Intermediate values produce a weighted average.
///
/// # Parameters
/// - `h1`           – Height at the start biome / region.
/// - `h2`           – Height at the end biome / region.
/// - `blend_factor` – Weight in `[0, 1]`; typically derived from
///                    [`biome_blend_weight`].
///
/// # Returns
/// `h1 × (1 − blend_factor) + h2 × blend_factor`
#[allow(dead_code)]
pub fn blend_heights(h1: f64, h2: f64, blend_factor: f64) -> f64 {
    h1 * (1.0 - blend_factor) + h2 * blend_factor
}

/// Computes a `[0, 1]` blend weight from how far a noise value is from a
/// threshold boundary.
///
/// Returns `0.0` exactly at the threshold (fully blended / transition center)
/// and `1.0` when the noise value is at least `blend_width` away from the
/// threshold (fully committed to one side).
///
/// This function is useful for softening hard biome edges: values well inside
/// a biome region get weight `1.0` (use the local biome height unmodified),
/// while values near the boundary get a fractional weight that can feed into
/// [`blend_heights`] to cross-fade with the neighboring biome.
///
/// # Parameters
/// - `noise_value`  – The raw noise sample at the query position.
/// - `threshold`    – The noise value at which the biome boundary lies.
/// - `blend_width`  – The half-width of the transition zone.  Noise values
///                    within `threshold ± blend_width` receive a weight < 1.
///
/// # Returns
/// `|noise_value − threshold| / blend_width`, clamped to `[0, 1]`.
///
/// # Example
/// ```rust
/// // 20% through the transition zone → weight = 0.2
/// let w = biome_blend_weight(0.12, 0.10, 0.10); // dist = 0.02, width = 0.10
/// ```
#[allow(dead_code)]
pub fn biome_blend_weight(noise_value: f64, threshold: f64, blend_width: f64) -> f64 {
    let dist_from_edge = (noise_value - threshold).abs();
    if dist_from_edge >= blend_width {
        1.0
    } else {
        dist_from_edge / blend_width
    }
}