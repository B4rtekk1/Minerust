@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst); if any(id.xy >= size) { return; }
  let ss = vec2<i32>(textureDimensions(src)) - 1; let p = vec2<i32>(id.xy) * 2;
  let d = min(min(textureLoad(src, clamp(p, vec2(0), ss), 0).r, textureLoad(src, clamp(p+vec2(1,0),vec2(0),ss),0).r), min(textureLoad(src,clamp(p+vec2(0,1),vec2(0),ss),0).r,textureLoad(src,clamp(p+vec2(1,1),vec2(0),ss),0).r));
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(d));
}
