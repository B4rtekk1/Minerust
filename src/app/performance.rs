/// Returns CPU time consumed by this process, in milliseconds.
///
/// Unlike `Instant`, this excludes time spent waiting for the GPU, the window
/// system, or the scheduler.  It includes both user-mode and kernel-mode work
/// performed by every thread in the process.
#[cfg(target_os = "windows")]
pub fn process_cpu_time_ms() -> Option<f64> {
    use std::mem::MaybeUninit;
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    unsafe {
        let mut creation = MaybeUninit::<FILETIME>::zeroed();
        let mut exit = MaybeUninit::<FILETIME>::zeroed();
        let mut kernel = MaybeUninit::<FILETIME>::zeroed();
        let mut user = MaybeUninit::<FILETIME>::zeroed();

        if GetProcessTimes(
            GetCurrentProcess(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        ) == 0
        {
            return None;
        }

        let filetime_to_100ns =
            |time: FILETIME| u64::from(time.dwLowDateTime) | (u64::from(time.dwHighDateTime) << 32);
        let cpu_100ns = filetime_to_100ns(kernel.assume_init())
            .saturating_add(filetime_to_100ns(user.assume_init()));
        Some(cpu_100ns as f64 / 10_000.0)
    }
}

/// Returns CPU time consumed by the calling thread, in milliseconds.
///
/// Unlike process accounting this excludes concurrent chunk- and mesh-worker
/// activity. It also excludes time blocked on the GPU, swap chain, window
/// system, or scheduler.
#[cfg(target_os = "windows")]
pub fn current_thread_cpu_time_ms() -> Option<f64> {
    use std::mem::MaybeUninit;

    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};

    unsafe {
        let mut creation = MaybeUninit::<FILETIME>::zeroed();
        let mut exit = MaybeUninit::<FILETIME>::zeroed();
        let mut kernel = MaybeUninit::<FILETIME>::zeroed();
        let mut user = MaybeUninit::<FILETIME>::zeroed();

        if GetThreadTimes(
            GetCurrentThread(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        ) == 0
        {
            return None;
        }

        #[inline]
        fn filetime_to_100ns(time: FILETIME) -> u64 {
            u64::from(time.dwLowDateTime) | (u64::from(time.dwHighDateTime) << 32)
        }

        let cpu_100ns = filetime_to_100ns(kernel.assume_init())
            .saturating_add(filetime_to_100ns(user.assume_init()));
        Some(cpu_100ns as f64 / 10_000.0)
    }
}

/// Platforms without a process CPU-time API use the elapsed wall-clock time.
/// Keeping this fallback preserves the HUD on non-Windows builds.
#[cfg(not(target_os = "windows"))]
pub fn process_cpu_time_ms() -> Option<f64> {
    None
}

#[cfg(not(target_os = "windows"))]
pub fn current_thread_cpu_time_ms() -> Option<f64> {
    None
}

/// Samples OS CPU accounting over a sufficiently long interval.  Per-frame
/// deltas are deliberately avoided: Windows accounts CPU time in coarse ticks,
/// which makes individual high-FPS frames commonly read as 0.00 ms.
pub struct CpuUsageAverager {
    interval_start: std::time::Instant,
    main_cpu_start: Option<f64>,
    process_cpu_start: Option<f64>,
    frames: u32,
}

impl CpuUsageAverager {
    pub fn new() -> Self {
        Self {
            interval_start: std::time::Instant::now(),
            main_cpu_start: current_thread_cpu_time_ms(),
            process_cpu_start: process_cpu_time_ms(),
            frames: 0,
        }
    }

    /// Records one presented frame. Returns main-thread and whole-process
    /// CPU averages in ms/frame every 500 ms.
    pub fn record_frame(&mut self) -> Option<(f32, f32)> {
        self.frames += 1;
        if self.interval_start.elapsed().as_secs_f32() < 0.5 {
            return None;
        }

        let main_end = current_thread_cpu_time_ms();
        let process_end = process_cpu_time_ms();
        let frames = self.frames.max(1) as f64;
        let main_avg = match (self.main_cpu_start, main_end) {
            (Some(start), Some(end)) => ((end - start).max(0.0) / frames) as f32,
            _ => 0.0,
        };
        let process_avg = match (self.process_cpu_start, process_end) {
            (Some(start), Some(end)) => ((end - start).max(0.0) / frames) as f32,
            _ => 0.0,
        };

        self.interval_start = std::time::Instant::now();
        self.main_cpu_start = main_end;
        self.process_cpu_start = process_end;
        self.frames = 0;
        Some((main_avg, process_avg))
    }
}
