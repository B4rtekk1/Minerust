use rustc_hash::FxHashMap;

use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::core::vertex::Vertex;
use crate::render::mesh::{add_greedy_quad, add_quad};
use crate::world::generator::ChunkGenerator;

pub struct World {
    pub chunks: FxHashMap<(i32, i32), Chunk>,
    /// Chunk coords of the player the last time chunk cleanup was run.
    /// Cleanup only re-runs when the player moves to a different chunk.
    last_cleanup_cx: i32,
    last_cleanup_cz: i32,
    pub seed: u32,
    /// Single source of truth for all terrain generation.
    generator: ChunkGenerator,
}

impl World {
    pub fn new() -> Self {
        Self::new_with_seed(2137) // TODO: Randomize seed, for now it's fixed, easier to debug
    }

    pub fn new_with_seed(seed: u32) -> Self {
        let generator = ChunkGenerator::new(seed);

        let mut world = World {
            chunks: FxHashMap::default(),
            last_cleanup_cx: i32::MIN,
            last_cleanup_cz: i32::MIN,
            seed,
            generator,
        };

        let spawn_cx = 0;
        let spawn_cz = 0;
        // Ensure initial visible area matches configured render distance.
        // Additional chunks are still generated asynchronously while exploring.
        let initial_radius = RENDER_DISTANCE;
        for cx in (spawn_cx - initial_radius)..=(spawn_cx + initial_radius) {
            for cz in (spawn_cz - initial_radius)..=(spawn_cz + initial_radius) {
                // Use the stored generator for consistent terrain with the async loader
                if !world.chunks.contains_key(&(cx, cz)) {
                    let chunk = world.generator.generate_chunk(cx, cz);
                    world.chunks.insert((cx, cz), chunk);
                }
            }
        }

        world
    }


    pub fn ensure_chunk_generated(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        self.generate_chunk(cx, cz);
    }

    pub fn update_chunks_around_player(&mut self, player_x: f32, player_z: f32) -> Vec<(i32, i32)> {
        let player_cx = (player_x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (player_z / CHUNK_SIZE as f32).floor() as i32;

        // Only run the O(N) scan when the player actually moves to a new chunk.
        // This avoids iterating every loaded chunk entry on each chunk-arrival event.
        if player_cx == self.last_cleanup_cx && player_cz == self.last_cleanup_cz {
            return Vec::new();
        }
        self.last_cleanup_cx = player_cx;
        self.last_cleanup_cz = player_cz;

        let chunks_to_remove: Vec<(i32, i32)> = self
            .chunks
            .keys()
            .filter(|(cx, cz)| {
                let dx = (*cx - player_cx).abs();
                let dz = (*cz - player_cz).abs();
                dx > CHUNK_UNLOAD_DISTANCE || dz > CHUNK_UNLOAD_DISTANCE
            })
            .cloned()
            .collect();

        for key in &chunks_to_remove {
            self.chunks.remove(key);
        }

        chunks_to_remove
    }

    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        self.generator.get_biome(x, z)
    }

    pub fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        self.generator.get_terrain_height_pub(x, z)
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.generator.is_cave_entrance_pub(x, z, surface_height)
    }

    /// Delegate chunk generation to the cached `ChunkGenerator`.
    fn generate_chunk(&mut self, cx: i32, cz: i32) {
        let chunk = self.generator.generate_chunk(cx, cz);
        self.chunks.insert((cx, cz), chunk);
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= WORLD_HEIGHT {
            return BlockType::Air;
        }
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
            chunk.get_block(lx, y, lz)
        } else {
            BlockType::Air
        }
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set_block(lx, y, lz, block);
        }
    }

    pub fn set_block_player(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set_block(lx, y, lz, block);
            chunk.player_modified = true;
        }
    }

    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_block(x, y, z).is_solid()
    }

    pub fn is_subchunk_occluded(&self, cx: i32, cz: i32, sy: i32) -> bool {
        // A subchunk is occluded if it's fully opaque and all 6 adjacent subchunks are also fully opaque.
        // This is a conservative but very fast check.
        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
            if !chunk.subchunks[sy as usize].is_fully_opaque {
                return false;
            }

            // Check +Y and -Y (within same chunk)
            if sy > 0 && !chunk.subchunks[(sy - 1) as usize].is_fully_opaque {
                return false;
            }
            if sy < NUM_SUBCHUNKS - 1 && !chunk.subchunks[(sy + 1) as usize].is_fully_opaque
            {
                return false;
            }
            // Borders of world height are not occluded
            if sy == 0 || sy == NUM_SUBCHUNKS - 1 {
                return false;
            }

            // Check X and Z neighbors
            let neighbors = [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)];
            for (ncx, ncz) in neighbors {
                if let Some(nchunk) = self.chunks.get(&(ncx, ncz)) {
                    if !nchunk.subchunks[sy as usize].is_fully_opaque {
                        return false;
                    }
                } else {
                    // If neighbor chunk is not loaded, we can't be sure it's occluded
                    return false;
                }
            }

            return true;
        }
        false
    }

    pub fn find_spawn_point(&self) -> (f32, f32, f32) {
        for radius in 0..50 {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let x = dx;
                    let z = dz;
                    let height = self.get_terrain_height(x, z);
                    let biome = self.get_biome(x, z);

                    if height >= SEA_LEVEL
                        && !matches!(biome, Biome::Ocean | Biome::River | Biome::Lake)
                    {
                        return (x as f32 + 0.3, (height + 1) as f32, z as f32 + 0.5);
                    }
                }
            }
        }
        (0.5, 80.0, 0.5)
    }

    pub fn build_subchunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> ((Vec<Vertex>, Vec<u32>), (Vec<Vertex>, Vec<u32>)) {
        // Larger initial capacity reduces reallocations for dense subchunks
        let mut vertices = Vec::with_capacity(4096);
        let mut indices = Vec::with_capacity(2048);
        let mut water_vertices = Vec::with_capacity(1024);
        let mut water_indices = Vec::with_capacity(512);

        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;

        const PAD: usize = 1;
        const S: usize = CHUNK_SIZE as usize + PAD * 2; // 18
        const SH: usize = SUBCHUNK_HEIGHT as usize + PAD * 2; // 18

        let mut block_cache = [BlockType::Air; S * SH * S];

        // Fetch one block from any neighbouring chunk (HashMap access, done once per cell)
        let fetch = |wx: i32, wy: i32, wz: i32| -> BlockType {
            if wy < 0 || wy >= WORLD_HEIGHT {
                return BlockType::Air;
            }
            let cx = wx.div_euclid(CHUNK_SIZE);
            let cz = wz.div_euclid(CHUNK_SIZE);
            let lx = wx.rem_euclid(CHUNK_SIZE);
            let lz = wz.rem_euclid(CHUNK_SIZE);
            if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                chunk.get_block(lx, wy, lz)
            } else {
                // If neighbor chunk is missing, assume Water below sea level to prevent ugly side faces
                if wy < SEA_LEVEL {
                    BlockType::Water
                } else {
                    BlockType::Air
                }
            }
        };

        for px in 0..S as i32 {
            for py in 0..SH as i32 {
                for pz in 0..S as i32 {
                    let wx = base_x + px - PAD as i32;
                    let wy = base_y + py - PAD as i32;
                    let wz = base_z + pz - PAD as i32;
                    block_cache[(px as usize) * SH * S + (py as usize) * S + (pz as usize)] =
                        fetch(wx, wy, wz);
                }
            }
        }

        // Fast inline accessor: lx,ly,lz in [-1, CHUNK_SIZE] (i.e. padded coords)
        let get_block_fast = |lx: i32, ly: i32, lz: i32| -> BlockType {
            let px = (lx + PAD as i32) as usize;
            let py = (ly + PAD as i32) as usize;
            let pz = (lz + PAD as i32) as usize;
            block_cache[px * SH * S + py * S + pz]
        };

        // World-coord version used in a few places below
        let get_block_world = |wx: i32, wy: i32, wz: i32| -> BlockType {
            get_block_fast(wx - base_x, wy - base_y, wz - base_z)
        };

        let mut biome_map: [[Option<Biome>; CHUNK_SIZE as usize]; CHUNK_SIZE as usize] =
            [[None; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];

        // ============= GREEDY MESHING IMPLEMENTATION =============
        //
        // Instead of creating one quad per visible block face, we merge adjacent faces
        // of the same type into larger quads. This reduces vertex count by 50-90%.
        //
        // Algorithm:
        // 1. First pass: Handle special blocks (stairs) with naive approach
        // 2. For each face direction, for each slice perpendicular to that direction:
        //    a. Build a 2D mask of visible faces with their attributes
        //    b. Greedily merge adjacent faces with matching attributes
        //    c. Emit merged quads with tiled UVs

        // Face attributes for greedy merge comparison
        #[derive(Clone, Copy, PartialEq)]
        struct FaceAttrs {
            block: BlockType,
            color: [u8; 3], // Quantized color for comparison
            tex_index: u8,
            is_active: bool,
        }

        impl Default for FaceAttrs {
            fn default() -> Self {
                FaceAttrs {
                    block: BlockType::Air,
                    color: [0, 0, 0],
                    tex_index: 0,
                    is_active: false,
                }
            }
        }

        // Helper to quantize color for comparison (avoids floating point issues)
        // Quantize to 64 levels per channel for balance between merging and quality
        let quantize_color = |c: [f32; 3]| -> [u8; 3] {
            [
                ((c[0] * 255.0) as u8) & 0xFC, // 6 bits = 64 levels
                ((c[1] * 255.0) as u8) & 0xFC,
                ((c[2] * 255.0) as u8) & 0xFC,
            ]
        };

        // First pass: Handle special blocks (WoodStairs, etc.) with naive approach
        for lx in 0..CHUNK_SIZE {
            for ly in 0..SUBCHUNK_HEIGHT {
                for lz in 0..CHUNK_SIZE {
                    let y = base_y + ly;
                    let world_x = base_x + lx;
                    let world_z = base_z + lz;
                    let block = get_block_world(world_x, y, world_z);

                    if block == BlockType::Air {
                        continue;
                    }

                    let is_water = block == BlockType::Water;
                    let (target_verts, target_inds) = if is_water {
                        (&mut water_vertices, &mut water_indices)
                    } else {
                        (&mut vertices, &mut indices)
                    };

                    // WoodStairs uses naive approach (complex geometry)
                    if block == BlockType::WoodStairs {
                        let x = world_x as f32;
                        let y_f = y as f32;
                        let z = world_z as f32;
                        let color = block.color();
                        let tex_top = block.tex_top();
                        let tex_side = block.tex_side();
                        let r = block.roughness();
                        let m = block.metallic();

                        let neighbors = [
                            get_block_world(world_x - 1, y, world_z),
                            get_block_world(world_x + 1, y, world_z),
                            get_block_world(world_x, y - 1, world_z),
                            get_block_world(world_x, y + 1, world_z),
                            get_block_world(world_x, y, world_z - 1),
                            get_block_world(world_x, y, world_z + 1),
                        ];

                        // Bottom Face (-Y)
                        if block.should_render_face_against(neighbors[2]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z + 1.0],
                                [x, y_f, z],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f, z + 1.0],
                                [0.0, -1.0, 0.0],
                                color,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Top of bottom slab
                        add_quad(
                            target_verts,
                            target_inds,
                            [x, y_f + 0.5, z],
                            [x, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z],
                            [0.0, 1.0, 0.0],
                            color,
                            tex_top,
                            r,
                            m,
                        );
                        // Top of top slab
                        if block.should_render_face_against(neighbors[3]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f + 1.0, z + 0.5],
                                [x, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [0.0, 1.0, 0.0],
                                color,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Front face (-Z)
                        if block.should_render_face_against(neighbors[4]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f, z],
                                [x, y_f, z],
                                [x, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z],
                                [0.0, 0.0, -1.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Riser
                        add_quad(
                            target_verts,
                            target_inds,
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x, y_f + 0.5, z + 0.5],
                            [x, y_f + 1.0, z + 0.5],
                            [x + 1.0, y_f + 1.0, z + 0.5],
                            [0.0, 0.0, -1.0],
                            color,
                            tex_side,
                            r,
                            m,
                        );
                        // Back face (+Z)
                        if block.should_render_face_against(neighbors[5]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z + 1.0],
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [0.0, 0.0, 1.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Left face (-X)
                        if block.should_render_face_against(neighbors[0]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z],
                                [x, y_f, z + 1.0],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 0.5, z],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f + 0.5, z + 0.5],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 0.5],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Right face (+X)
                        if block.should_render_face_against(neighbors[1]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [x + 1.0, y_f + 0.5, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        continue;
                    }
                }
            }
        }

        // Second pass: Greedy meshing for regular solid blocks
        // Process each of 6 face directions separately

        // Face direction: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
        for face_dir in 0..6 {
            // Determine slice dimensions based on face direction
            // For X faces: slices are in YZ plane, iterate over X
            // For Y faces: slices are in XZ plane, iterate over Y
            // For Z faces: slices are in XY plane, iterate over Z

            let (slice_count, dim1_size, dim2_size): (i32, i32, i32) = match face_dir {
                0 | 1 => (CHUNK_SIZE, SUBCHUNK_HEIGHT, CHUNK_SIZE), // X faces: YZ slices
                2 | 3 => (SUBCHUNK_HEIGHT, CHUNK_SIZE, CHUNK_SIZE), // Y faces: XZ slices
                4 | 5 => (CHUNK_SIZE, CHUNK_SIZE, SUBCHUNK_HEIGHT), // Z faces: XY slices
                _ => unreachable!(),
            };

            // For each slice perpendicular to face direction
            for slice in 0..slice_count {
                // Build mask of visible faces in this slice
                let mut mask: Vec<FaceAttrs> =
                    vec![FaceAttrs::default(); (dim1_size * dim2_size) as usize];

                for d1 in 0..dim1_size {
                    for d2 in 0..dim2_size {
                        // Convert slice coordinates to world coordinates
                        let (lx, ly, lz): (i32, i32, i32) = match face_dir {
                            0 | 1 => (slice, d1, d2),
                            2 | 3 => (d1, slice, d2),
                            4 | 5 => (d1, d2, slice),
                            _ => unreachable!(),
                        };

                        let y = base_y + ly;
                        let world_x = base_x + lx;
                        let world_z = base_z + lz;
                        let block = get_block_world(world_x, y, world_z);

                        // Water uses naive approach even in the greedy pass to prevent gaps
                        // caused by vertex displacement on large quads. We still draw it via IndirectManager.
                        if block == BlockType::Water {
                            let neighbors = [
                                get_block_world(world_x - 1, y, world_z),
                                get_block_world(world_x + 1, y, world_z),
                                get_block_world(world_x, y - 1, world_z),
                                get_block_world(world_x, y + 1, world_z),
                                get_block_world(world_x, y, world_z - 1),
                                get_block_world(world_x, y, world_z + 1),
                            ];

                            if block.should_render_face_against(neighbors[face_dir as usize]) {
                                let x = world_x as f32;
                                let y_f = y as f32;
                                let z = world_z as f32;
                                let color = block.color();
                                let tex = block.tex_top();
                                let r = block.roughness();
                                let m = block.metallic();

                                match face_dir {
                                    0 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x, y_f, z],
                                        [x, y_f, z + 1.0],
                                        [x, y_f + 1.0, z + 1.0],
                                        [x, y_f + 1.0, z],
                                        [-1.0, 0.0, 0.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    1 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x + 1.0, y_f, z + 1.0],
                                        [x + 1.0, y_f, z],
                                        [x + 1.0, y_f + 1.0, z],
                                        [x + 1.0, y_f + 1.0, z + 1.0],
                                        [1.0, 0.0, 0.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    2 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x, y_f, z + 1.0],
                                        [x, y_f, z],
                                        [x + 1.0, y_f, z],
                                        [x + 1.0, y_f, z + 1.0],
                                        [0.0, -1.0, 0.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    3 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x, y_f + 1.0, z],
                                        [x, y_f + 1.0, z + 1.0],
                                        [x + 1.0, y_f + 1.0, z + 1.0],
                                        [x + 1.0, y_f + 1.0, z],
                                        [0.0, 1.0, 0.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    4 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x + 1.0, y_f, z],
                                        [x, y_f, z],
                                        [x, y_f + 1.0, z],
                                        [x + 1.0, y_f + 1.0, z],
                                        [0.0, 0.0, -1.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    5 => add_quad(
                                        &mut water_vertices,
                                        &mut water_indices,
                                        [x, y_f, z + 1.0],
                                        [x + 1.0, y_f, z + 1.0],
                                        [x + 1.0, y_f + 1.0, z + 1.0],
                                        [x, y_f + 1.0, z + 1.0],
                                        [0.0, 0.0, 1.0],
                                        color,
                                        tex,
                                        r,
                                        m,
                                    ),
                                    _ => {}
                                }
                            }
                            continue;
                        }

                        // Skip air and other special blocks
                        if block == BlockType::Air || block == BlockType::WoodStairs {
                            continue;
                        }

                        // Get neighbor in face direction
                        let (nx, ny, nz) = match face_dir {
                            0 => (world_x - 1, y, world_z),
                            1 => (world_x + 1, y, world_z),
                            2 => (world_x, y - 1, world_z),
                            3 => (world_x, y + 1, world_z),
                            4 => (world_x, y, world_z - 1),
                            5 => (world_x, y, world_z + 1),
                            _ => unreachable!(),
                        };
                        let neighbor = get_block_world(nx, ny, nz);

                        if !block.should_render_face_against(neighbor) {
                            continue;
                        }

                        // Compute color (with biome for grass/leaves)
                        let needs_biome = block == BlockType::Grass || block == BlockType::Leaves;
                        let biome = if needs_biome {
                            let lx_idx = lx as usize;
                            let lz_idx = lz as usize;
                            if biome_map[lx_idx][lz_idx].is_none() {
                                biome_map[lx_idx][lz_idx] = Some(self.get_biome(world_x, world_z));
                            }
                            biome_map[lx_idx][lz_idx]
                        } else {
                            None
                        };

                        let color = match face_dir {
                            2 => block.bottom_color(),
                            3 => {
                                if block == BlockType::Grass {
                                    biome.unwrap().grass_color()
                                } else {
                                    block.top_color()
                                }
                            }
                            _ => {
                                if block == BlockType::Grass {
                                    block.color()
                                } else if block == BlockType::Leaves {
                                    biome.unwrap().leaves_color()
                                } else {
                                    block.color()
                                }
                            }
                        };

                        let tex_index = match face_dir {
                            2 => block.tex_bottom(),
                            3 => block.tex_top(),
                            _ => block.tex_side(),
                        };

                        let idx = (d1 * dim2_size + d2) as usize;
                        mask[idx] = FaceAttrs {
                            block,
                            color: quantize_color(color),
                            tex_index: tex_index as u8,
                            is_active: true,
                        };
                    }
                }

                // Greedy merge: scan the mask and merge adjacent faces
                for d1 in 0..dim1_size {
                    let mut d2 = 0;
                    while d2 < dim2_size {
                        let idx = (d1 * dim2_size + d2) as usize;
                        let face = mask[idx];

                        if !face.is_active {
                            d2 += 1;
                            continue;
                        }

                        // Found an active face, try to extend it
                        let mut width = 1i32;
                        while d2 + width < dim2_size {
                            let next_idx = (d1 * dim2_size + d2 + width) as usize;
                            if mask[next_idx] == face {
                                width += 1;
                            } else {
                                break;
                            }
                        }

                        // Try to extend height (d1 direction)
                        let mut height = 1i32;
                        'height_loop: while d1 + height < dim1_size {
                            for w in 0..width {
                                let check_idx = ((d1 + height) * dim2_size + d2 + w) as usize;
                                if mask[check_idx] != face {
                                    break 'height_loop;
                                }
                            }
                            height += 1;
                        }

                        // Mark merged faces as inactive
                        for h in 0..height {
                            for w in 0..width {
                                let clear_idx = ((d1 + h) * dim2_size + d2 + w) as usize;
                                mask[clear_idx].is_active = false;
                            }
                        }

                        // Emit the merged quad
                        let _block = face.block;
                        // Water is no longer here, but we keep the target selection for safety/future use
                        let (target_verts, target_inds) = (&mut vertices, &mut indices);

                        let color = [
                            face.color[0] as f32 / 255.0,
                            face.color[1] as f32 / 255.0,
                            face.color[2] as f32 / 255.0,
                        ];
                        let tex_index = face.tex_index as f32;
                        let roughness = 1.0;
                        let metallic = 0.0;

                        // Convert slice + d1/d2 + width/height to world coordinates
                        let (x0, y0, z0, x1, y1, z1) = match face_dir {
                            0 => {
                                // -X face
                                let x = (base_x + slice) as f32;
                                let y0 = (base_y + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                let y1 = y0 + height as f32;
                                let z1 = z0 + width as f32;
                                (x, y0, z0, x, y1, z1)
                            }
                            1 => {
                                // +X face
                                let x = (base_x + slice + 1) as f32;
                                let y0 = (base_y + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                let y1 = y0 + height as f32;
                                let z1 = z0 + width as f32;
                                (x, y0, z0, x, y1, z1)
                            }
                            2 => {
                                // -Y face
                                let y = (base_y + slice) as f32;
                                let x0 = (base_x + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                let x1 = x0 + height as f32;
                                let z1 = z0 + width as f32;
                                (x0, y, z0, x1, y, z1)
                            }
                            3 => {
                                // +Y face
                                let y = (base_y + slice + 1) as f32;
                                let x0 = (base_x + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                let x1 = x0 + height as f32;
                                let z1 = z0 + width as f32;
                                (x0, y, z0, x1, y, z1)
                            }
                            4 => {
                                // -Z face
                                let z = (base_z + slice) as f32;
                                let x0 = (base_x + d1) as f32;
                                let y0 = (base_y + d2) as f32;
                                let x1 = x0 + height as f32;
                                let y1 = y0 + width as f32;
                                (x0, y0, z, x1, y1, z)
                            }
                            5 => {
                                // +Z face
                                let z = (base_z + slice + 1) as f32;
                                let x0 = (base_x + d1) as f32;
                                let y0 = (base_y + d2) as f32;
                                let x1 = x0 + height as f32;
                                let y1 = y0 + width as f32;
                                (x0, y0, z, x1, y1, z)
                            }
                            _ => unreachable!(),
                        };

                        // Emit quad with proper vertex order and tiled UVs
                        match face_dir {
                            0 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z0],
                                [x0, y0, z1],
                                [x0, y1, z1],
                                [x0, y1, z0],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            1 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x1, y0, z1],
                                [x1, y0, z0],
                                [x1, y1, z0],
                                [x1, y1, z1],
                                [1.0, 0.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            2 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z1],
                                [x0, y0, z0],
                                [x1, y0, z0],
                                [x1, y0, z1],
                                [0.0, -1.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            3 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y1, z0],
                                [x0, y1, z1],
                                [x1, y1, z1],
                                [x1, y1, z0],
                                [0.0, 1.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            4 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x1, y0, z0],
                                [x0, y0, z0],
                                [x0, y1, z0],
                                [x1, y1, z0],
                                [0.0, 0.0, -1.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                height as f32,
                                width as f32,
                            ),
                            5 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z1],
                                [x1, y0, z1],
                                [x1, y1, z1],
                                [x0, y1, z1],
                                [0.0, 0.0, 1.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                height as f32,
                                width as f32,
                            ),
                            _ => {}
                        }

                        d2 += width;
                    }
                }
            }
        }

        ((vertices, indices), (water_vertices, water_indices))
    }
}
