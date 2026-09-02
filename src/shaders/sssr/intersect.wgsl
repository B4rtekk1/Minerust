@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var hierarchy: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var material_tex: texture_2d<f32>;
@group(0) @binding(5) var depth_tex: texture_2d<f32>;
@group(0) @binding(6) var blue_noise: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> rays: array<u32>;
@group(0) @binding(8) var raw: texture_storage_2d<rgba16float, write>;
fn sample_ggx(n: vec3<f32>, v: vec3<f32>, rough: f32, rnd: vec2<f32>) -> vec3<f32> {
 let a=max(.002,rough*rough); let phi=6.2831853*rnd.x; let cos_t=sqrt((1.0-rnd.y)/(1.0+(a*a-1.0)*rnd.y)); let sin_t=sqrt(max(0.,1.-cos_t*cos_t));
 let up=select(vec3<f32>(0.,1.,0.),vec3<f32>(1.,0.,0.),abs(n.y)>.98); let t=normalize(cross(up,n)); let b=cross(n,t); let h=normalize(t*cos(phi)*sin_t+b*sin(phi)*sin_t+n*cos_t);
 return normalize(reflect(-v,h));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let size=textureDimensions(normal_tex); let index=rays[id.x]; let p=vec2<u32>(index%size.x,index/size.x); if(p.y>=size.y){return;} let ip=vec2<i32>(p);
 let m=textureLoad(material_tex,ip,0); if(m.y<.5){return;} let uv=(vec2<f32>(p)+.5)/vec2<f32>(size); let origin=reconstruct_world(uv,textureLoad(depth_tex,ip,0).r,u.inv_view_proj);
 let n=oct_decode(textureLoad(normal_tex,ip,0).xy); let v=normalize(u.camera_pos-origin); let bn=textureLoad(blue_noise,vec2<i32>(p & vec2<u32>(127u)),0).xy; let r=fract(bn+hash2(p,u.frame_index)); let dir=sample_ggx(n,v,m.x,r);
 var t=.12; var hit=false; var hit_uv=vec2<f32>(0.); var mip=i32(textureNumLevels(hierarchy)-1u);
 for(var step=0;step<48;step++){ let q=project(origin+dir*t,u.view_proj); if(any(q.xy<vec2(0.002))||any(q.xy>vec2(.998))||q.z<=0.||q.z>=1.){break;} let dim=vec2<i32>(textureDimensions(hierarchy,mip)); let hp=clamp(vec2<i32>(q.xy*vec2<f32>(dim)),vec2(0),dim-1); let d=textureLoad(hierarchy,hp,mip).r; let thickness=.0015+t*.00014; if(q.z>=d-thickness){if(mip==0){hit=true;hit_uv=q.xy;break;} mip-=1;}else{t+=max(.16,t*.08)*exp2(f32(mip)*.25);mip=min(mip+1,i32(textureNumLevels(hierarchy)-1u));}}
 if(hit){let c=textureLoad(scene,vec2<i32>(hit_uv*vec2<f32>(size)),0).rgb;let edge=min(min(hit_uv.x,1.-hit_uv.x),min(hit_uv.y,1.-hit_uv.y));textureStore(raw,ip,vec4<f32>(c,smoothstep(.01,.10,edge)*(1.-smoothstep(35.,58.,t))));}
}
