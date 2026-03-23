use wgpu;
use windows::core::factory;
use crate::logger::{log, LogLevel};
#[cfg(all(target_os = "windows", feature = "dx12"))]
use windows::core::Interface;
use windows::Win32::Graphics::Dxgi;

/// Upper bounds on the number of subchunks, vertices, and indices that may be
/// resident in GPU memory simultaneously.
///
/// Computed at startup by [`GpuMemoryInfo::calculate_buffer_limits`] based on
/// detected VRAM, or constructed with [`BufferLimits::default`] for a
/// conservative 1.5 GB baseline when memory detection is unavailable.
#[derive(Debug, Clone, Copy)]
pub struct BufferLimits {
    /// Maximum number of sub-chunk mesh objects that may be allocated.
    /// Derived as the minimum of `max_vertices / 1000` and
    /// `max_indices / 2500`, capped at 1 000 000.
    pub max_subchunks: usize,
    /// Maximum total vertices across all resident chunk meshes.
    /// Sized to consume ~60% of the usable VRAM budget at 56 bytes per vertex.
    pub max_vertices: usize,
    /// Maximum total indices across all resident chunk meshes.
    /// Sized to consume ~15% of the usable VRAM budget at 4 bytes per index.
    pub max_indices: usize,
}

impl Default for BufferLimits {
    /// Returns conservative limits sized for a ~1.5 GB VRAM budget:
    /// - 100 000 sub-chunks
    /// - 4 000 000 vertices
    /// - 10 000 000 indices
    fn default() -> Self {
        Self {
            max_subchunks: 100_000,
            max_vertices: 4_000_000,
            max_indices: 10_000_000,
        }
    }
}

/// VRAM capacity information retrieved from the GPU at startup.
///
/// Used by [`GpuMemoryInfo::calculate_buffer_limits`] to size world-geometry
/// buffers relative to the actual hardware available. Obtained via
/// [`GpuMemoryInfo::detect`].
pub struct GpuMemoryInfo {
    /// Total dedicated VRAM reported by the GPU, in bytes.
    pub total_vram_bytes: u64,
    /// VRAM currently available for new allocations, in bytes.
    /// On Vulkan this equals `total_vram_bytes`; on DX12 it reflects the
    /// OS budget minus current usage; on Metal it is the recommended working
    /// set size minus currently allocated bytes.
    pub available_vram_bytes: u64,
    /// The wgpu backend the adapter is using.
    pub backend: wgpu::Backend,
}

impl GpuMemoryInfo {
    /// Queries the GPU for VRAM capacity using the backend-specific detection
    /// path appropriate for `adapter`.
    ///
    /// Detection is dispatched on the active backend:
    ///
    /// | Backend | Feature gate            | Detection method                        |
    /// |---------|-------------------------|-----------------------------------------|
    /// | Vulkan  | `vulkan`                | `ash` — iterates device-local heaps     |
    /// | DX12    | `dx12` + Windows only   | DXGI `IDXGIAdapter1` / `IDXGIAdapter3`  |
    /// | Metal   | `metal` + macOS only    | `metal::Device` working-set query       |
    /// | Other   | —                       | Panics (unsupported backend)            |
    ///
    /// When the appropriate feature flag is disabled or detection fails, each
    /// path falls back to [`GpuMemoryInfo::fallback_memory`] (1.5 GB).
    ///
    /// # Panics
    ///
    /// Panics if the adapter reports a backend other than Vulkan, DX12, or
    /// Metal, as these are the only supported graphics backends.
    pub fn detect(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let backend = info.backend;

        log(LogLevel::Info, &format!("Detected GPU adapter: {} ({:?})", info.name, backend));

        let (total_vram_bytes, available_vram_bytes) = match backend {
            wgpu::Backend::Vulkan => Self::get_vulkan_memory(&info.name),
            wgpu::Backend::Dx12 => Self::get_dx12_memory(),
            wgpu::Backend::Metal => Self::get_metal_memory(),
            _ => {
                panic!("Unknown backend");
            }
        };

        Self {
            total_vram_bytes,
            available_vram_bytes,
            backend,
        }
    }

    /// Detects VRAM via Vulkan when the `vulkan` feature is enabled.
    ///
    /// Creates a temporary `ash` Vulkan instance, enumerates physical devices,
    /// selects the one matching `adapter_name`, and sums all device-local
    /// memory heaps. Falls back to [`GpuMemoryInfo::fallback_memory`] if the
    /// feature is disabled or initialisation fails.
    fn get_vulkan_memory(adapter_name: &str) -> (u64, u64) {
        #[cfg(feature = "vulkan")]
        {
            match VulkanMemoryDetector::new(adapter_name) {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory(),
            }
        }
        #[cfg(not(feature = "vulkan"))]
        Self::fallback_memory()
    }

    /// Detects VRAM via DirectX 12 when the `dx12` feature is enabled on
    /// Windows.
    ///
    /// Uses DXGI to query `DedicatedVideoMemory` from the first enumerated
    /// adapter. If `IDXGIAdapter3` is available, also queries the current OS
    /// budget and usage to compute a more accurate available figure. Falls
    /// back to [`GpuMemoryInfo::fallback_memory`] if the feature/platform
    /// guard is not satisfied or initialisation fails.
    fn get_dx12_memory() -> (u64, u64) {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            match Dx12MemoryDetector::new() {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory(),
            }
        }
        #[cfg(not(any(feature = "dx12", target_os = "windows")))]
        Self::fallback_memory()
    }

    /// Detects VRAM via Metal when the `metal` feature is enabled on macOS.
    ///
    /// Queries `recommendedMaxWorkingSetSize` and `currentAllocatedSize` from
    /// the system default Metal device. Falls back to
    /// [`GpuMemoryInfo::fallback_memory`] if the feature/platform guard is
    /// not satisfied or initialisation fails.
    fn get_metal_memory() -> (u64, u64) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match MetalMemoryDetector::new() {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory(),
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        Self::fallback_memory()
    }

    /// Returns a conservative `(total, available)` fallback of **1.5 GB each**,
    /// used when backend-specific detection is disabled or fails.
    fn fallback_memory() -> (u64, u64) {
        let fallback = 1.5 * 1024f32 * 1024f32; // 1.5 GB in bytes
        (fallback as u64, fallback as u64)
    }

    /// Derives [`BufferLimits`] from the detected total VRAM.
    ///
    /// Only 40% of total VRAM is considered usable to leave headroom for
    /// textures, uniforms, and OS/driver overhead. That budget is then
    /// partitioned:
    ///
    /// | Allocation     | Share  | Per-element size |
    /// |----------------|--------|------------------|
    /// | Vertex buffer  | 60 %   | 56 bytes         |
    /// | Index buffer   | 15 %   | 4 bytes          |
    /// | (reserved)     | 25 %   | —                |
    ///
    /// `max_subchunks` is capped at 1 000 000 and derived conservatively as
    /// `min(max_vertices / 1000, max_indices / 2500)`.
    pub fn calculate_buffer_limits(&self) -> BufferLimits {
        // Use only 40% of total VRAM to leave room for textures and OS overhead.
        let usable_vram = (self.total_vram_bytes as f64 * 0.4) as u64;

        const VERTEX_SIZE: u64 = 56; // bytes per Vertex struct
        const INDEX_SIZE: u64 = 4;   // bytes per u32 index

        let vertex_budget = (usable_vram as f64 * 0.6) as u64;
        let index_budget = (usable_vram as f64 * 0.15) as u64;

        let max_vertices = (vertex_budget / VERTEX_SIZE) as usize;
        let max_indices = (index_budget / INDEX_SIZE) as usize;

        // Estimate max sub-chunks conservatively from both limits, hard-capped
        // at 1 000 000 to avoid pathological allocations on high-VRAM hardware.
        let max_subchunks = (max_vertices / 1000)
            .min(max_indices / 2500)
            .min(1_000_000);

        BufferLimits {
            max_subchunks,
            max_vertices,
            max_indices,
        }
    }
}

/// Detects Vulkan device-local VRAM by creating a temporary `ash` instance.
///
/// Compiled only when the `vulkan` feature is enabled. The instance is
/// destroyed immediately after the heap query to avoid conflicting with the
/// wgpu Vulkan instance created later.
#[cfg(feature = "vulkan")]
struct VulkanMemoryDetector {
    /// Sum of all `DEVICE_LOCAL` Vulkan memory heap sizes, in bytes.
    total_vram: u64,
}

#[cfg(feature = "vulkan")]
impl VulkanMemoryDetector {
    /// Creates a minimal Vulkan instance, selects the physical device whose
    /// name matches `adapter_name`, and sums its device-local memory heaps.
    ///
    /// Falls back to the first enumerated device if no name match is found,
    /// which handles cases where the wgpu and Vulkan name strings differ
    /// slightly.
    ///
    /// # Errors
    ///
    /// Returns a `String` error if the Vulkan entry point cannot be loaded,
    /// the instance cannot be created, or no physical devices are found.
    fn new(adapter_name: &str) -> Result<Self, String> {
        use ::ash::{vk, Entry};

        let total_vram = unsafe {
            let entry = Entry::load().map_err(|e| e.to_string())?;
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 2, 0),
                ..Default::default()
            };

            let crate_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            };

            let instance = entry
                .create_instance(&crate_info, None)
                .map_err(|e| e.to_string())?;

            let devices = instance
                .enumerate_physical_devices()
                .map_err(|e| e.to_string())?;
            let mut physical_device = devices
                .first()
                .ok_or("No physical device found")?
                .to_owned();

            // Prefer the device whose name matches the wgpu adapter name.
            for device in &devices {
                let props = instance.get_physical_device_properties(*device);
                let name = unsafe {
                    std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .to_string()
                };
                if name == adapter_name {
                    physical_device = *device;
                    break;
                }
            }

            let mem_props = instance.get_physical_device_memory_properties(physical_device);

            // Sum only device-local heaps; shared/system heaps are excluded.
            let mut total = 0u64;
            log(LogLevel::Info, "Vulkan memory heaps:");
            for i in 0..mem_props.memory_heap_count {
                let heap = mem_props.memory_heaps[i as usize];
                let is_device_local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                log(LogLevel::Info, &format!(
                     "  Heap {}: {} MB (device_local: {})",
                    i,
                    heap.size / 1024 / 1024,
                    is_device_local
                ));

                if is_device_local {
                    total += heap.size;
                }
            }

            // Destroy the temporary instance before wgpu creates its own.
            instance.destroy_instance(None);
            total
        };
        Ok(Self { total_vram })
    }

    /// Returns `(total_vram, total_vram)`.
    ///
    /// Vulkan does not expose a live "available" figure without additional
    /// extensions, so both values are the same total device-local size.
    fn get_memory(&self) -> (u64, u64) {
        (self.total_vram, self.total_vram)
    }
}

/// Detects DX12 VRAM via DXGI on Windows.
///
/// Compiled only when the `dx12` feature is enabled and the target OS is
/// Windows. Uses `IDXGIAdapter1` for total VRAM and, if available,
/// `IDXGIAdapter3::QueryVideoMemoryInfo` for a live available figure.
#[cfg(all(target_os = "windows", feature = "dx12"))]
struct Dx12MemoryDetector {
    /// Total dedicated video memory reported by `DXGI_ADAPTER_DESC1`, in bytes.
    total_vram: u64,
    /// Available VRAM derived from the DXGI budget minus current usage, in
    /// bytes. Equals `total_vram` when `IDXGIAdapter3` is unavailable.
    available_vram: u64,
}

#[cfg(all(target_os = "windows", feature = "dx12"))]
impl Dx12MemoryDetector {
    /// Creates a DXGI factory, enumerates the first adapter, and queries its
    /// memory figures.
    ///
    /// Attempts to cast to `IDXGIAdapter3` for a live budget query. If the
    /// cast fails (e.g. older DXGI version), `available_vram` is set to
    /// `total_vram` as a conservative estimate.
    ///
    /// # Errors
    ///
    /// Returns a formatted `String` if the DXGI factory cannot be created,
    /// no adapters are found, or the adapter description cannot be retrieved.
    fn new() -> Result<Self, String> {
        use windows::Win32::Graphics::Dxgi::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC1;

        let (total, available) = unsafe {
            let factory: IDXGIFactory4 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                    .map_err(|e| format!("Failed to create DXGI factory: {:?}", e))?;

            let adapter: IDXGIAdapter1 = factory
                .EnumAdapters1(0)
                .map_err(|e| format!("Failed to enumerate adapters: {:?}", e))?;

            let desc = adapter
                .GetDesc1()
                .map_err(|e| format!("Failed to get adapter description: {:?}", e))?;

            let total = desc.DedicatedVideoMemory as u64;

            log(LogLevel::Info, "DirectX 12 adapter info:");
            let name_len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..name_len]);
            log(LogLevel::Info, &format!("  Name: {}", name));
            log(LogLevel::Info, &format!("  Dedicated video memory: {} MB", total / 1024 / 1024));
            log(LogLevel::Info, &format!("  Dedicated system memory: {} MB", desc.DedicatedSystemMemory as u64 / 1024 / 1024));
            log(LogLevel::Info, &format!("  Shared system memory: {} MB", desc.SharedSystemMemory as u64 / 1024 / 1024));

            // IDXGIAdapter3 exposes live budget/usage — use it when available.
            let available = match adapter.cast::<IDXGIAdapter3>() {
                Ok(adapter3) => {
                    let mut mem_info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                    match adapter3.QueryVideoMemoryInfo(
                        0,
                        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                        &mut mem_info,
                    ) {
                        Ok(_) => {
                            log(LogLevel::Info, &format!("  Budget: {} MB", mem_info.Budget / 1024 / 1024));
                            log(LogLevel::Info, &format!("  Current usage: {} MB", mem_info.CurrentUsage / 1024 / 1024));
                            log(Level::Info, &format!("  Available for reservation: {} MB", mem_info.AvailableForReservation / 1024 / 1024));
                            log(LogLevel::Info, &format!("  Reserved: {} MB", mem_info.CurrentReservation / 1024 / 1024));
                            // Saturating sub guards against transient over-budget states.
                            mem_info.Budget.saturating_sub(mem_info.CurrentUsage)
                        }
                        Err(e) => {
                            log(LogLevel::Warning, &format!("  Failed to query video memory info, using total VRAM as available: "), e);
                            total
                        }
                    }
                }
                Err(_) => {
                    log(LogLevel::Warning, "  IDXGIAdapter3 not available, using total VRAM as available");
                    total
                }
            };

            (total, available)
        };

        Ok(Self {
            total_vram: total,
            available_vram: available,
        })
    }

    /// Returns `(total_vram, available_vram)` as queried from DXGI.
    fn get_memory(&self) -> (u64, u64) {
        (self.total_vram, self.available_vram)
    }
}

/// Detects Metal VRAM on macOS using the system default Metal device.
///
/// Compiled only when the `metal` feature is enabled and the target OS is
/// macOS. Uses `recommendedMaxWorkingSetSize` as the total figure and
/// subtracts `currentAllocatedSize` to derive available bytes.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalMemoryDetector {
    /// `recommendedMaxWorkingSetSize` from the default Metal device, in bytes.
    recommended: u64,
    /// `recommendedMaxWorkingSetSize` minus `currentAllocatedSize`, in bytes.
    available: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalMemoryDetector {
    /// Obtains the system default Metal device and queries its working-set
    /// size and current allocation.
    ///
    /// # Errors
    ///
    /// Returns a `String` error if no system default Metal device is found
    /// (e.g. running on a system without Metal support).
    fn new() -> Result<Self, String> {
        use metal::Device;

        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let recommended = device.recommended_max_working_set_size();
        let allocated = device.current_allocated_size();
        // Saturating sub handles the edge case where allocated briefly exceeds recommended.
        let available = recommended.saturating_sub(allocated);

        Ok(Self {
            recommended,
            available,
        })
    }

    /// Returns `(recommended_max_working_set_size, available_bytes)`.
    fn get_memory(&self) -> (u64, u64) {
        (self.recommended, self.available)
    }
}