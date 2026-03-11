use wgpu;
use windows::core::factory;
#[cfg(all(target_os = "windows", feature = "dx12"))]
use windows::core::Interface;
use windows::Win32::Graphics::Dxgi;

#[derive(Debug, Clone, Copy)]
pub struct BufferLimits {
    pub max_subchunks: usize,
    pub max_vertices: usize,
    pub max_indices: usize,
}

impl Default for BufferLimits {
    fn default() -> Self {
        Self {
            max_subchunks: 100_000,
            max_vertices: 4_000_000,
            max_indices: 10_000_000,
        }
    }
}
pub struct GpuMemoryInfo {
    pub total_vram_bytes: u64,
    pub available_vram_bytes: u64,
    pub backend: wgpu::Backend,
}

impl GpuMemoryInfo{
    pub fn detect(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let backend = info.backend;

        tracing::info!("GPU backend {:?}", backend);

        let (total_vram_bytes, available_vram_bytes) = match backend {
            wgpu::Backend::Vulkan => {
                Self::get_vulkan_memory(&info.name)
            }
            wgpu::Backend::Dx12 => {
                Self::get_dx12_memory()
            }
            wgpu::Backend::Metal => {
                Self::get_metal_memory()
            }

            _ => {
                panic!("Unknown backend");
            }
        };

        Self { total_vram_bytes, available_vram_bytes, backend }
    }

    fn get_vulkan_memory(adapter_name: &str) -> (u64, u64) {
        #[cfg(feature = "vulkan")]
        {
            match VulkanMemoryDetector::new(adapter_name) {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory()
            }
        }
        #[cfg(not(feature = "vulkan"))]
        Self::fallback_memory()
    }

    fn get_dx12_memory() -> (u64, u64) {
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        {
            match Dx12MemoryDetector::new() {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory()
            }
        }

        #[cfg(not(any(feature = "dx12", target_os = "windows")))]
        Self::fallback_memory()
    }

    fn get_metal_memory() -> (u64, u64) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match MetalMemoryDetector::new() {
                Ok(detector) => detector.get_memory(),
                Err(_) => Self::fallback_memory()
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        Self::fallback_memory()
    }

    fn fallback_memory() -> (u64, u64) {
        let fallback = 1.5 * 1024f32 * 1024f32; // 1.5GB
        (fallback as u64, fallback as u64)
    }

    pub fn calculate_buffer_limits(&self) -> BufferLimits {
        let usable_vram = (self.total_vram_bytes as f64 * 0.4) as u64;

        const VERTEX_SIZE: u64 = 56;
        const INDEX_SIZE: u64 = 4;

        let vertex_budget = (usable_vram as f64 * 0.6) as u64;
        let index_budget = (usable_vram as f64 * 0.15) as u64;

        let max_vertices = (vertex_budget / VERTEX_SIZE) as usize;
        let max_indices = (index_budget / INDEX_SIZE) as usize;

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

#[cfg(feature = "vulkan")]
struct VulkanMemoryDetector {
    total_vram: u64,
}

#[cfg(feature = "vulkan")]
impl VulkanMemoryDetector {
    fn new(adapter_name: &str) -> Result<Self, String> {
        use::ash::{vk, Entry};

        let total_vram = unsafe {
            let entry = Entry::load().map_err(|e| e.to_string())?;
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 2, 0),
                ..Default::default()
            };

            let crate_info = vk::InstanceCreateInfo{
                p_application_info: &app_info,
                ..Default::default()
            };

            let instance = entry.create_instance(&crate_info, None).map_err(|e| e.to_string())?;

            let devices = instance.enumerate_physical_devices().map_err(|e| e.to_string())?;
            let mut physical_device = devices
                .first()
                .ok_or("No physical device found")?
                .to_owned();

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

            let mut total = 0u64;
            tracing::info!("Vulkan memory heaps:");
            for i in 0..mem_props.memory_heap_count {
                let heap = mem_props.memory_heaps[i as usize];
                let is_device_local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                tracing::info!(
                    "  Heap {}: {} MB (device_local: {})",
                    i,
                    heap.size / 1024 / 1024,
                    is_device_local
                );

                if is_device_local {
                    total += heap.size;
                }
            }

            instance.destroy_instance(None);
            total
        };
        Ok(Self {total_vram})
    }

    fn get_memory(&self) -> (u64, u64) {
        (self.total_vram, self.total_vram)
    }
}

#[cfg(all(target_os = "windows", feature = "dx12"))]
struct Dx12MemoryDetector {
    total_vram: u64,
    available_vram: u64,
}

#[cfg(all(target_os = "windows", feature = "dx12"))]
impl Dx12MemoryDetector {
    fn new() -> Result<Self, String> {
        use windows::Win32::Graphics::Dxgi::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC1;

        let (total, available) = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| format!("Failed to create DXGI factory: {:?}", e))?;

            let adapter: IDXGIAdapter1 = factory.EnumAdapters1(0)
                .map_err(|e| format!("Failed to enumerate adapters: {:?}", e))?;

            let desc = adapter.GetDesc1()
                .map_err(|e| format!("Failed to get adapter description: {:?}", e))?;

            let total = desc.DedicatedVideoMemory as u64;

            tracing::info!("DirectX 12 adapter info:");

            let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..name_len]);
            tracing::info!("  Name: {}", name);
            tracing::info!("  Dedicated video memory: {} MB", total / 1024 / 1024);
            tracing::info!("  Dedicated system memory: {} MB", desc.DedicatedSystemMemory as u64 / 1024 / 1024);
            tracing::info!("  Shared system memory: {} MB", desc.SharedSystemMemory as u64 / 1024 / 1024);

            let available = match adapter.cast::<IDXGIAdapter3>() {
                Ok(adapter3) => {
                    let mut mem_info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                    match adapter3.QueryVideoMemoryInfo(
                        0,
                        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                        &mut mem_info,
                    ) {
                        Ok(_) => {
                            tracing::info!("  Budget: {} MB", mem_info.Budget / 1024 / 1024);
                            tracing::info!("  Current usage: {} MB", mem_info.CurrentUsage / 1024 / 1024);
                            tracing::info!("  Available for reservation: {} MB", mem_info.AvailableForReservation / 1024 / 1024);
                            tracing::info!("  Current reservation: {} MB", mem_info.CurrentReservation / 1024 / 1024);

                            mem_info.Budget.saturating_sub(mem_info.CurrentUsage)
                        }
                        Err(e) => {
                            tracing::warn!("  Failed to query video memory info: {:?}", e);
                            total
                        }
                    }
                }
                Err(_) => {
                    tracing::info!("  IDXGIAdapter3 not available, using total VRAM as available");
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

    fn get_memory(&self) -> (u64, u64) {
        (self.total_vram, self.available_vram)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalMemoryDetector {
    recommended: u64,
    available: u64
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalMemoryDetector {
    fn new() -> Result<Self, String> {
        use metal::Device;

        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let recommended = device.recommended_max_working_set_size();
        let allocated = device.current_allocated_size();
        let available = recommended.saturating_sub(allocated);

        Ok(Self { recommended, available })
    }

    fn get_memory(&self) -> (u64, u64) {
        (self.recommended, self.available)
    }
}

