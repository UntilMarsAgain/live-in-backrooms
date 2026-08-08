//! 客户端资产装配：把共享解析结果注册进 GPU 资产管理器并组装视图场景。
//!
//! [`load_scene`] 是 glTF 场景入口：
//! - 纯解析在 [`lbr_shared::asset`]（`parse_glb` / `GlbAssets`），本模块只做
//!   注册与组装；
//! - 网格/贴图按 File 条目注册（数据本体在内存层，句柄可重载）；
//! - 场景树（[`Scene`]）按 glTF 节点层级构建，材质作为渲染信息放进
//!   [`ClientScene`]（共享树不携带材质）。

use std::collections::HashMap;

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use gltf::scene::Transform as GltfTransform;

use crate::render::AssetManager;
use crate::scene::ClientScene;
use lbr_shared::asset::parse_glb;
use lbr_shared::core::asset::Handle;
use lbr_shared::core::game_path::GamePath;
use lbr_shared::core::material::Material;
use lbr_shared::core::mesh::Mesh;
use lbr_shared::core::texture::Texture;
use lbr_shared::core::transform::Transform;
use lbr_shared::scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};

/// 从 glTF 文件加载场景（按游戏路径从合并资源空间读取）：
/// 网格/贴图资产注册进 `assets`，返回带层级的客户端视图场景。
pub fn load_scene(path: &GamePath, assets: &mut AssetManager) -> Result<ClientScene> {
    // 先从合并资源空间读字节（借用 assets.space() 在 read 后结束），
    // 之后可安全地 &mut assets 注册资产。
    let bytes = assets.space().read(path)?;
    let (document, _buffers, _images, glb) =
        parse_glb(&bytes).with_context(|| format!("解析 glTF 失败：{path}"))?;
    let scene = document.default_scene().context("glTF 文件没有默认场景")?;

    // 1. 解析结果进内存层 + 注册 File 条目（mesh / texture 各按数组索引，
    //    与 [`lbr_shared::asset::GlbLoader`] 的 entries 顺序一致，可重载）。
    let mesh_handles: Vec<Handle<Mesh>> = glb
        .meshes
        .iter()
        .enumerate()
        .map(|(i, _)| assets.meshes_mut().register_file(path.clone(), Box::new(i as u32)))
        .collect();
    let texture_handles: Vec<Handle<Texture>> = glb
        .textures
        .iter()
        .enumerate()
        .map(|(i, _)| {
            assets
                .textures_mut()
                .register_file(path.clone(), Box::new(i as u32))
        })
        .collect();
    assets.memory_insert(path.clone(), Box::new(glb));

    // 2. document 网格 → primitive 句柄列表（mesh_handles 是全局 primitive 顺序）。
    let mut mesh_keys: HashMap<usize, Vec<Handle<Mesh>>> = HashMap::new();
    let mut offset = 0;
    for mesh in document.meshes() {
        let count = mesh.primitives().count();
        mesh_keys.insert(mesh.index(), mesh_handles[offset..offset + count].to_vec());
        offset += count;
    }

    // 3. 场景树构建：句柄已注册，这里只负责层级与材质（材质进视图场景）。
    let mut loader = Loader {
        mesh_keys,
        texture_keys: texture_handles,
        materials: HashMap::new(),
    };
    let mut out = Scene::new();
    for node in scene.nodes() {
        loader.load_node(&mut out, node, None)?;
    }

    let mut view = ClientScene::from_scene(out);
    for (key, material) in loader.materials {
        view.set_material(key, material);
    }
    Ok(view)
}

/// 加载过程中的临时状态。
struct Loader {
    /// glTF 网格索引 → 已注册的句柄列表（每个 primitive 一个）。
    mesh_keys: HashMap<usize, Vec<Handle<Mesh>>>,
    /// glTF 图片索引 → 已注册的贴图句柄（File 条目，索引即 image 下标）。
    texture_keys: Vec<Handle<Texture>>,
    /// 网格节点 → 材质（组装时收集，最后放进 ClientScene）。
    materials: HashMap<ObjectKey, Material>,
}

impl Loader {
    /// 递归加载 glTF 节点：每个节点一个容器物体，网格与子节点挂在它下面。
    fn load_node(
        &mut self,
        scene: &mut Scene,
        node: gltf::scene::Node<'_>,
        parent: Option<ObjectKey>,
    ) -> Result<()> {
        let (translation, rotation, scale) = match node.transform() {
            GltfTransform::Decomposed {
                translation,
                rotation,
                scale,
            } => (
                Vec3::from(translation),
                Quat::from_array(rotation),
                Vec3::from(scale),
            ),
            GltfTransform::Matrix { matrix } => {
                let (scale, rotation, translation) =
                    Mat4::from_cols_array_2d(&matrix).to_scale_rotation_translation();
                (translation, rotation, scale)
            }
        };
        let children: Vec<_> = node.children().collect();
        let mesh_materials = match node.mesh() {
            Some(mesh) => Some(self.register_mesh(&mesh)?),
            None => None,
        };
        // 既没有网格也没有子节点的空节点：跳过。
        // TODO: 拼图接口需要此类节点，届时需关闭剪枝
        if mesh_materials.is_none() && children.is_empty() {
            return Ok(());
        }
        // 容器物体：承载局部变换，无网格；primitive 与子节点都挂在它下面。
        let container = match parent {
            Some(p) => scene
                .attach(
                    p,
                    SceneObject::new(
                        SceneObjectKind::Empty,
                        Transform::new(translation, rotation, scale),
                    ),
                )
                .expect("父节点必然存活"),
            None => scene.add_object(SceneObject::new(
                SceneObjectKind::Empty,
                Transform::new(translation, rotation, scale),
            )),
        };
        // 每个 primitive 一个可绘制物体（单位局部变换，位置已由容器决定）；
        // 材质收集进 `materials`，由 ClientScene 持有。
        if let Some(meshes) = mesh_materials {
            for (key, material) in meshes {
                let child = scene
                    .attach(
                        container,
                        SceneObject::new(SceneObjectKind::Mesh(key), Transform::IDENTITY),
                    )
                    .expect("容器节点必然存活");
                self.materials.insert(child, material);
            }
        }
        for child in children {
            self.load_node(scene, child, Some(container))?;
        }
        Ok(())
    }

    /// 取一个 glTF 网格各 primitive 的 (句柄, 材质) 列表（句柄在 load_scene 预注册）。
    fn register_mesh(&self, mesh: &gltf::Mesh<'_>) -> Result<Vec<(Handle<Mesh>, Material)>> {
        let keys = self
            .mesh_keys
            .get(&mesh.index())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(keys.len());
        for (key, primitive) in keys.iter().zip(mesh.primitives()) {
            out.push((*key, self.material_from_primitive(&primitive)?));
        }
        Ok(out)
    }

    /// 读取 primitive 的材质：基础色因子 + 基础色贴图。
    fn material_from_primitive(&self, primitive: &gltf::Primitive<'_>) -> Result<Material> {
        // gltf::Material 总是存在（未指定时是默认材质，因子为 1）。
        let gltf_material = primitive.material();
        let pbr = gltf_material.pbr_metallic_roughness();
        // 贴图已在 load_scene 预注册为 File 条目，这里按 image 索引直接取句柄。
        // 基础色/金属度粗糙度是 `Info`，法线是 `NormalTexture`，分开处理。
        let tex_of_info =
            |info: Option<gltf::texture::Info>| info.map(|i| self.texture_keys[i.texture().source().index()]);
        let tex_of_normal = |info: Option<gltf::material::NormalTexture>| {
            info.map(|i| self.texture_keys[i.texture().source().index()])
        };
        Ok(Material {
            base_color: pbr.base_color_factor(),
            base_color_texture: tex_of_info(pbr.base_color_texture()),
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: tex_of_info(pbr.metallic_roughness_texture()),
            normal_texture: tex_of_normal(gltf_material.normal_texture()),
        })
    }
}

#[cfg(test)]
mod tests;
