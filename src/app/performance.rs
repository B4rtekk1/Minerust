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

/// Platforms without a process CPU-time API use the elapsed wall-clock time.
/// Keeping this fallback preserves the HUD on non-Windows builds.
#[cfg(not(target_os = "windows"))]
pub fn process_cpu_time_ms() -> Option<f64> {
    None
}
