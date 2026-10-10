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

/// Includes CPU time of all threads, including generation and mesh workers.
#[cfg(target_os = "linux")]
pub fn process_cpu_time_ms() -> Option<f64> {
    let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock_gettime initializes the valid output pointer on success.
    // Only read the timespec after checking its return value.
    if unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
        return None;
    }
    let time = unsafe { time.assume_init() };
    Some(time.tv_sec as f64 * 1000.0 + time.tv_nsec as f64 / 1_000_000.0)
}

/// Other platforms retain the elapsed wall-clock fallback in the HUD.
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub fn process_cpu_time_ms() -> Option<f64> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn process_clock_accounts_for_cpu_work() {
        let before = process_cpu_time_ms().expect("process CPU clock unavailable");
        let mut value = 1u64;
        for i in 0..100_000u64 {
            value = std::hint::black_box(value.wrapping_mul(31).wrapping_add(i));
        }
        std::hint::black_box(value);
        let after = process_cpu_time_ms().unwrap();
        assert!(after.is_finite() && after > before);
    }
}
