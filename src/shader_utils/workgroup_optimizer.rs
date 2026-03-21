/*use std::collections::HashMap;
use wgpu::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub &'static str);

pub struct WorkgroupOptimizer {
    cache: HashMap<ShaderId, u32>,
    device: Device,
    adapter: Adapter,
}

impl WorkgroupOptimizer {
    pub async fn new(device: Device, adapter: Adapter) -> Self {
        let mut opt = Self {
            cache: HashMap::new(),
            device,
            adapter,
        };
        opt.load_cache().await;
        opt
    }

}
 */