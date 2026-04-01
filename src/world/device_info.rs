use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

pub struct DeviceInfo {
    pub cpu_cores: usize,
    pub cpu_physical_cores: Option<usize>,
    pub cpu_name: String,
    pub cpu_brand: String,
    pub total_memory_gb: u64,
}

impl DeviceInfo {
    pub fn collect() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );

        let cpus = sys.cpus();

        let cpu_cores = cpus.len();

        let cpu_physical_cores = System::physical_core_count();

        let cpu_name = cpus
            .first()
            .map(|c| c.name().to_string())
            .unwrap_or_default();

        let cpu_brand = cpus
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_default();

        let total_memory_gb = sys.total_memory() / 1_073_741_824;

        Self {
            cpu_cores,
            cpu_physical_cores,
            cpu_name,
            cpu_brand,
            total_memory_gb,
        }
    }
}
