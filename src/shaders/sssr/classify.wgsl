@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var material_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> rays: array<u32>;
@group(0) @binding(4) var<storage, read_write> ray_count: atomic<u32>;
@group(0) @binding(5) var raw: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let size=textureDimensions(normal_tex); if any(id.xy>=size){return;} let p=vec2<i32>(id.xy);
 let m=textureLoad(material_tex,p,0); textureStore(raw,p,vec4<f32>(0.0));
 if(m.y < .5 || u.reflection_mode < .5){return;}
 // Keep one ray per water pixel while validating the raw intersection pass.
 // Variable-rate classification needs reconstruction in the temporal stage.
 let i=atomicAdd(&ray_count,1u); rays[i]=id.y*size.x+id.x;
}
