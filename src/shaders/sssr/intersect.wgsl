@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var hierarchy: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var material_tex: texture_2d<f32>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var blue_noise: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> rays: array<u32>;
@group(0) @binding(8) var raw: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<storage, read_write> ray_count: atomic<u32>;
fn sample_ggx(n: vec3<f32>, v: vec3<f32>, rough: f32, rnd: vec2<f32>) -> vec3<f32> {
 let a=max(.002,rough*rough); let phi=6.2831853*rnd.x; let cos_t=sqrt((1.0-rnd.y)/(1.0+(a*a-1.0)*rnd.y)); let sin_t=sqrt(max(0.,1.-cos_t*cos_t));
 let up=select(vec3<f32>(0.,1.,0.),vec3<f32>(1.,0.,0.),abs(n.y)>.98); let t=normalize(cross(up,n)); let b=cross(n,t); let h=normalize(t*cos(phi)*sin_t+b*sin(phi)*sin_t+n*cos_t);
 return normalize(reflect(-v,h));
}

// `clip_origin + clip_direction * t` is the ray in homogeneous clip space.
// Solve its perspective divide against the next screen-space cell boundary,
// rather than advancing by an arbitrary world-space distance.
fn boundary_t(clip_origin: vec4<f32>, clip_direction: vec4<f32>, ndc: f32, axis: u32) -> f32 {
 let numerator=ndc*clip_origin.w-clip_origin[axis];
 let denominator=clip_direction[axis]-ndc*clip_direction.w;
 if(abs(denominator)<0.000001){return 1e30;}
 return numerator/denominator;
}

fn next_cell_t(q: vec3<f32>, t: f32, clip_origin: vec4<f32>, clip_direction: vec4<f32>, dim: vec2<u32>, ray_screen_direction: vec2<f32>) -> f32 {
 let cell=min(vec2<u32>(q.xy*vec2<f32>(dim)),dim-vec2<u32>(1u));
 let inv_dim=1.0/vec2<f32>(dim);
 let boundary_x=select(f32(cell.x)*inv_dim.x,f32(cell.x+1u)*inv_dim.x,ray_screen_direction.x>=0.0);
 let boundary_y=select(f32(cell.y)*inv_dim.y,f32(cell.y+1u)*inv_dim.y,ray_screen_direction.y>=0.0);
 let tx=boundary_t(clip_origin,clip_direction,boundary_x*2.0-1.0,0u);
 // Screen UV has inverted Y relative to NDC.
 let ty=boundary_t(clip_origin,clip_direction,1.0-boundary_y*2.0,1u);
 let minimum_t=t+max(0.0001,abs(t)*0.00001);
 var result=1e30;
 if(tx>minimum_t){result=min(result,tx);}
 if(ty>minimum_t){result=min(result,ty);}
 return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 if (id.x >= atomicLoad(&ray_count)) { return; }
 let size=textureDimensions(normal_tex); let index=rays[id.x]; let p=vec2<u32>(index%size.x,index/size.x); if(p.y>=size.y){return;} let ip=vec2<i32>(p);
 let m=textureLoad(material_tex,ip,0); if(m.y<.5){return;} let uv=(vec2<f32>(p)+.5)/vec2<f32>(size); let origin=reconstruct_world(uv,textureLoad(depth_tex,ip,0).r,u.inv_view_proj);
 let n=normalize(textureLoad(normal_tex,ip,0).xyz*2.0-1.0); let v=normalize(u.camera_pos-origin); let bn=textureLoad(blue_noise,vec2<i32>(p & vec2<u32>(127u)),0).xy; let r=fract(bn+hash2(p,u.frame_index)); let dir=sample_ggx(n,v,m.x,r);
 let clip_origin=u.view_proj*vec4<f32>(origin,1.0);
 let clip_direction=u.view_proj*vec4<f32>(dir,0.0);
 var t=.12; var hit=false; var hit_uv=vec2<f32>(0.); let max_mip=i32(textureNumLevels(hierarchy)-1u); var mip=max_mip;
 for(var step=0;step<48;step++){
  let q=project(origin+dir*t,u.view_proj);
  if(any(q.xy<vec2(0.002))||any(q.xy>vec2(.998))||q.z<=0.||q.z>=1.){break;}
  let dim=textureDimensions(hierarchy,mip);
  let hp=clamp(vec2<i32>(q.xy*vec2<f32>(dim)),vec2(0),vec2<i32>(dim)-1);
  let d=textureLoad(hierarchy,hp,mip).r;
  let thickness=.0015+t*.00014;
  if(q.z>=d-thickness && mip>0){mip-=1;continue;}
  if(mip==0){let delta=q.z-d;if(delta>=0.&&delta<=thickness){hit=true;hit_uv=q.xy;break;}}
  let clip=clip_origin+clip_direction*t;
  // Sign of d(clip.xy / clip.w) / dt. This avoids a heuristic probe step.
  let ray_screen_direction=vec2<f32>(
   clip_direction.x*clip.w-clip.x*clip_direction.w,
   -(clip_direction.y*clip.w-clip.y*clip_direction.w)
  );
  let next_t=next_cell_t(q,t,clip_origin,clip_direction,dim,ray_screen_direction);
  if(next_t>=1e29){break;}
  t=next_t+max(0.0001,abs(next_t)*0.00001);
  mip=min(mip+1,max_mip);
 }
 if(hit){let c=textureLoad(scene,vec2<i32>(hit_uv*vec2<f32>(size)),0).rgb;let edge=min(min(hit_uv.x,1.-hit_uv.x),min(hit_uv.y,1.-hit_uv.y));textureStore(raw,ip,vec4<f32>(c,smoothstep(.01,.10,edge)*(1.-smoothstep(35.,58.,t))));}
}
