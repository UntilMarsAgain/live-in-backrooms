//! 资产库核心测试。

use crate::engine::core::gc::GcPolicy;

use super::*;
use crate::engine::core::data::texture::Texture;
use crate::engine::core::resource::MergedResourceSpace;

fn manager() -> AssetManager {
    AssetManager::new(MergedResourceSpace::new(std::env::temp_dir()))
}

/// 测试用文件重载器：模拟"完整重解析"，按 extra 返回对应数据。
fn u32_reloader(
    _space: &MergedResourceSpace,
    _type_id: TypeId,
    extra: &dyn Any,
) -> anyhow::Result<Box<dyn Any + Send + Sync>> {
    let index = extra
        .downcast_ref::<u32>()
        .ok_or_else(|| anyhow::anyhow!("extra 类型不符"))?;
    Ok(Box::new(10 + *index))
}

/// 临时合并资源空间：在系统临时目录建 `test/{file}` 并写入内容。
fn temp_manager_with(tag: &str, file: &str, content: &[u8]) -> (AssetManager, GamePath) {
    let dir = std::env::temp_dir().join(format!("asset-async-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let ns = dir.join("test");
    std::fs::create_dir_all(&ns).expect("创建测试目录");
    std::fs::write(ns.join(file), content).expect("写测试文件");
    let space = MergedResourceSpace::new(dir);
    let path: GamePath = format!("test:{file}").parse().expect("合法路径");
    (AssetManager::new(space), path)
}

/// 测试用异步文件加载器：scan 读结构（忽略内容）、parse 产出一份数据。
/// `fail_parse` 模拟"扫描成功但解析失败"；`multi_type` 额外产出 String 条目。
#[derive(Clone)]
struct FakeFileLoader {
    fail_parse: bool,
    multi_type: bool,
}

impl FileLoader for FakeFileLoader {
    fn scan(
        &self,
        _bytes: &[u8],
    ) -> anyhow::Result<Vec<(TypeId, Vec<Box<dyn Any + Send + Sync>>)>> {
        let mut out = vec![(
            TypeId::of::<u32>(),
            vec![Box::new(0u32) as Box<dyn Any + Send + Sync>],
        )];
        if self.multi_type {
            out.push((
                TypeId::of::<String>(),
                vec![Box::new(0u32) as Box<dyn Any + Send + Sync>],
            ));
        }
        Ok(out)
    }

    fn parse(&self, _bytes: &[u8]) -> anyhow::Result<Vec<(TypeId, Vec<LoadedEntry>)>> {
        // 慢一点，让"Loading"状态可观察。
        std::thread::sleep(Duration::from_millis(50));
        if self.fail_parse {
            anyhow::bail!("模拟解析失败");
        }
        let mut out = vec![(
            TypeId::of::<u32>(),
            vec![(
                Box::new(7u32) as Box<dyn Any + Send + Sync>,
                Box::new(0u32) as Box<dyn Any + Send + Sync>,
            )],
        )];
        if self.multi_type {
            out.push((
                TypeId::of::<String>(),
                vec![(
                    Box::new("hi".to_string()) as Box<dyn Any + Send + Sync>,
                    Box::new(0u32) as Box<dyn Any + Send + Sync>,
                )],
            ));
        }
        Ok(out)
    }

    fn extra_eq(&self, a: &dyn Any, b: &dyn Any) -> bool {
        match (a.downcast_ref::<u32>(), b.downcast_ref::<u32>()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }
}

#[test]
fn register_and_get_roundtrip() {
    let mut assets = manager();
    let handle = assets.register(Mesh::triangle());
    assert!(assets.is_valid(handle));
    assert!(matches!(
        assets.data_source(handle),
        Some(EntryData::Inline(_))
    ));
    assert_eq!(assets.state(handle), Some(AssetState::Resident));
    assert!(
        assets.get::<Mesh>(handle).is_some(),
        "内联数据应能 downcast 取回"
    );
}

/// 世代句柄：卸载后旧句柄失效；复用槽位不会误用旧句柄。
#[test]
fn removed_handle_stays_invalid_across_slot_reuse() {
    let mut assets = manager();
    let a = assets.register(Texture::white());
    let removed = assets.remove(a);
    assert!(removed.is_some());
    assert!(!assets.is_valid(a));
    assert!(assets.data_source(a).is_none());

    // 槽位复用由 slotmap 管理：旧键世代不匹配，永远失效。
    let b = assets.register(Texture::checkerboard(2, 1));
    assert_ne!(b.key(), a.key(), "新句柄应是不同键（世代不同）");
    assert!(!assets.is_valid(a));
    assert!(assets.is_valid(b));
}

/// pin/unpin 是引用计数：多次 pin 需同样多次 unpin 才回到 Resident；
/// 对失效句柄操作返回 false。
#[test]
fn pin_unpin_transitions_state() {
    let mut assets = manager();
    let handle = assets.register(Mesh::quad());
    assert!(assets.pin(handle));
    assert_eq!(assets.state(handle), Some(AssetState::Pinned));
    assert!(assets.pinned(handle));
    // 第二次 pin：计数 +1；只 unpin 一次仍是 Pinned。
    assert!(assets.pin(handle));
    assert!(assets.unpin(handle));
    assert_eq!(
        assets.state(handle),
        Some(AssetState::Pinned),
        "计数未归零仍驻留"
    );
    assert!(assets.pinned(handle));
    // 计数归零 → Resident。
    assert!(assets.unpin(handle));
    assert!(!assets.pinned(handle));
    assert_eq!(assets.state(handle), Some(AssetState::Resident));

    let stale = Handle::<Mesh> {
        key: DefaultKey::default(),
        _marker: PhantomData,
    };
    assert!(!assets.pin(stale));
    assert!(!assets.unpin(stale));
}

/// RAII 守卫：pin_guard 构造时 pin，离开作用域自动 unpin。
#[test]
fn pin_guard_releases_on_drop() {
    let mut assets = manager();
    let handle = assets.register(Mesh::cube());
    // 构造时 pin（计数 1）；守卫持有独占借用，作用域内不能再借 assets。
    {
        let guard = assets.pin_guard(handle).expect("句柄有效");
        assert_eq!(guard.handle(), handle);
        // 作用域结束自动 unpin。
    }
    assert!(!assets.pinned(handle));
    assert_eq!(assets.state(handle), Some(AssetState::Resident));

    // 失效句柄：pin_guard 返回 None。
    let stale = Handle::<Mesh> {
        key: DefaultKey::default(),
        _marker: PhantomData,
    };
    assert!(assets.pin_guard(stale).is_none());
}

/// 按类型遍历：只产出该类型的存活句柄。
#[test]
fn iter_of_filters_by_type() {
    let mut assets = manager();
    let tex_a = assets.register(Texture::white());
    let tex_b = assets.register(Texture::checkerboard(2, 1));
    let _mesh = assets.register(Mesh::cube());

    let tex_handles: Vec<_> = assets.iter_of::<Texture>().collect();
    assert_eq!(tex_handles.len(), 2);
    assert!(tex_handles.contains(&tex_a));
    assert!(tex_handles.contains(&tex_b));
    assert_eq!(assets.iter_of::<Mesh>().count(), 1);
}

/// 资源类型参数在编译期隔离：Handle<Mesh> 不能传给纹理注册表。
#[test]
fn handle_types_are_distinct() {
    let mesh: Handle<Mesh> = Handle {
        key: DefaultKey::default(),
        _marker: PhantomData,
    };
    let texture: Handle<Texture> = Handle {
        key: DefaultKey::default(),
        _marker: PhantomData,
    };
    // 不同 T 的 Handle 不是同一个类型（编译器保证），此处仅作存在性说明。
    let _ = (mesh, texture);
}

/// B1.2 关联计数：重载器只在最后一条存活条目被 remove 时释放。
#[test]
fn file_reloader_freed_when_last_entry_removed() {
    let mut assets = manager();
    let path: GamePath = "test:file.glb".parse().expect("合法路径");
    assets.set_file_reloader(path.clone(), u32_reloader);
    let a = assets.register_file::<u32>(path.clone(), Box::new(0u32), 7u32);
    let b = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);
    assert_eq!(assets.file_refs.get(&path), Some(&2));

    // 只移除一条：引用计数仍 >0，重载器保留。
    assert!(assets.remove(a).is_some());
    assert!(!assets.reloaders.is_empty());

    // 最后一条被移除：重载器释放（无需 gc）。
    assert!(assets.remove(b).is_some());
    assert!(assets.file_refs.is_empty());
    assert!(assets.reloaders.is_empty());
}

/// 反向索引：路径 → 句柄；remove 时同步删除；规范化路径是同一个键。
#[test]
fn file_entries_index_tracks_and_sync_removes() {
    let mut assets = manager();
    let path: GamePath = "test:a//b.glb".parse().unwrap(); // 规范化 → a/b.glb
    assert_eq!(path.path(), "a/b.glb");
    let a = assets.register_file::<u32>(path.clone(), Box::new(0u32), 7u32);
    let b = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);

    let handles = assets.loaded_handles_of::<u32>(&path);
    assert_eq!(handles.len(), 2);
    assert!(handles.contains(&a));
    assert!(handles.contains(&b));

    // remove 同步删除索引条目。
    assets.remove(a);
    assert_eq!(assets.loaded_handles_of::<u32>(&path), vec![b]);
    assets.remove(b);
    assert!(assets.loaded_handles_of::<u32>(&path).is_empty());
    assert!(assets.file_entries.is_empty());
}

/// 文件条目：数据本体在槽位（单一存储点），来源可见。
#[test]
fn file_entry_holds_own_data_and_source_visible() {
    let mut assets = manager();
    let path: GamePath = "test:file.glb".parse().expect("合法路径");
    let handle = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);
    assert_eq!(assets.get(handle), Some(&8u32), "文件条目数据在槽位");
    assert_eq!(assets.source_of(handle), Some(&path));
    assert!(matches!(
        assets.data_source(handle),
        Some(EntryData::File { source, .. }) if source == &path
    ));
}

/// 文件条目封装：内存层缺失时 `get` 自动经重载器重读（无需外部 ensure）。
#[test]
fn file_entry_get_auto_reloads_from_disk() {
    let mut assets = manager();
    let path: GamePath = "test:data.bin".parse().expect("合法路径");
    assets.set_file_reloader(path.clone(), u32_reloader);

    // 数据在槽位：get 直接返回。
    let handle = assets.register_file::<u32>(path.clone(), Box::new(2u32), 99u32);
    assert_eq!(assets.get(handle), Some(&99u32));

    // 卸载后数据丢弃 → get 经重载器完整重解析（按 extra 取回）。
    assets.unload_memory(&path);
    assert_eq!(assets.state(handle), Some(AssetState::DiskOnly));
    assert!(assets.get_cached(handle).is_none());
    assert_eq!(assets.get(handle), Some(&12u32), "重读后返回新数据");
    assert_eq!(assets.state(handle), Some(AssetState::Resident));
}

/// 智能 gc（按最近使用窗口）：释放非 Pinned 且超窗未使用的文件数据，
/// Pinned 与最近使用过的保留。
#[test]
fn gc_evicts_unpinned_file_data() {
    let mut assets = manager();
    let path: GamePath = "test:data.bin".parse().expect("合法路径");
    assets.set_file_reloader(path.clone(), u32_reloader);
    let pinned = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
    let loose = assets.register_file::<u32>(path.clone(), Box::new(1u32), 2u32);
    assets.pin(pinned);

    assets.gc(&GcPolicy::default()); // 窗口 0：只保留 Pinned 与"此刻"使用的。
                                     // Pinned 保留数据；非 Pinned 逐出，但 get 能重读。
    assert!(assets.get_cached(pinned).is_some());
    assert!(assets.get_cached(loose).is_none());
    assert_eq!(assets.get(loose), Some(&11u32));
}

/// 最近使用保护：get 过的条目在窗口内不被 gc 逐出。
#[test]
fn gc_keeps_recently_used_entries() {
    let mut assets = manager();
    let path: GamePath = "test:data.bin".parse().expect("合法路径");
    assets.set_file_reloader(path.clone(), u32_reloader);
    let old = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
    let fresh = assets.register_file::<u32>(path.clone(), Box::new(1u32), 2u32);
    assets.get(fresh); // 使用 fresh，把它标记为最近使用。

    assets.gc(&GcPolicy::default());
    assert!(assets.get_cached(fresh).is_some(), "最近使用的应保留");
    assert!(assets.get_cached(old).is_none(), "未使用的应逐出");
}

/// 异步加载（文件级）：立即注册占位句柄（Loading），get 阻塞等待填充完成。
#[test]
fn load_file_async_parses_off_thread_and_get_waits() {
    let (mut assets, path) = temp_manager_with("wait", "async.bin", b"x");
    assets
        .load_file_async(
            FakeFileLoader {
                fail_parse: false,
                multi_type: false,
            },
            path.clone(),
        )
        .expect("scan 应成功");
    let handles = assets.loaded_handles_of::<u32>(&path);
    assert_eq!(handles.len(), 1);
    assert_eq!(assets.handle_state(handles[0]), HandleState::Loading);
    assert!(assets.status().in_flight >= 1);

    // get 强制等待：阻塞到后台填充完成。
    assert_eq!(*assets.get(handles[0]).unwrap(), 7);
    assert_eq!(assets.handle_state(handles[0]), HandleState::Ready);
    assert_eq!(assets.status().in_flight, 0);
}

/// FileLoader 一次 parse 产出多种类型（1:N）：各类型占位句柄都被填充。
#[test]
fn load_file_async_produces_multiple_types() {
    let (mut assets, path) = temp_manager_with("multi", "multi.bin", b"x");
    assets
        .load_file_async(
            FakeFileLoader {
                fail_parse: false,
                multi_type: true,
            },
            path.clone(),
        )
        .expect("scan 应成功");
    let numbers = assets.loaded_handles_of::<u32>(&path);
    let strings = assets.loaded_handles_of::<String>(&path);
    assert_eq!(numbers.len(), 1);
    assert_eq!(strings.len(), 1);
    assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
    assert_eq!(assets.get(strings[0]).unwrap().as_str(), "hi");
}

/// 同步加载（文件级、多类型）：一次 parse 立即注册全部类型条目（无占位阶段）。
#[test]
fn load_file_registers_all_types_immediately() {
    let (mut assets, path) = temp_manager_with("sync", "sync.bin", b"x");
    assets
        .load_file(
            FakeFileLoader {
                fail_parse: false,
                multi_type: true,
            },
            path.clone(),
        )
        .expect("同步加载应成功");
    let numbers = assets.loaded_handles_of::<u32>(&path);
    let strings = assets.loaded_handles_of::<String>(&path);
    assert_eq!(numbers.len(), 1);
    assert_eq!(strings.len(), 1);
    // 同步路径没有占位阶段：注册完就是 Ready。
    assert_eq!(assets.handle_state(numbers[0]), HandleState::Ready);
    assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
    assert_eq!(assets.get(strings[0]).unwrap().as_str(), "hi");
    assert_eq!(assets.status().in_flight, 0);

    // 逐出后 get 经重载器自动重读（与异步入口共用同一重载器逻辑）。
    assets.unload_memory(&path);
    assert_eq!(assets.handle_state(numbers[0]), HandleState::DiskOnly);
    assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
}

/// 同文件二次 `load_file_async`：不 scan、不 parse，复用已有句柄。
#[test]
fn load_file_async_dedupes_same_path() {
    let (mut assets, path) = temp_manager_with("dedup", "dedup.bin", b"x");
    for _ in 0..2 {
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: false,
                },
                path.clone(),
            )
            .expect("scan 应成功");
    }
    // 等第一次完成，然后再次调用：仍复用同一个句柄，不新增条目。
    let handles = assets.loaded_handles_of::<u32>(&path);
    assert_eq!(handles.len(), 1);
    assert_eq!(*assets.get(handles[0]).unwrap(), 7);
    assets
        .load_file_async(
            FakeFileLoader {
                fail_parse: false,
                multi_type: false,
            },
            path.clone(),
        )
        .expect("scan 应成功");
    assert_eq!(assets.loaded_handles_of::<u32>(&path).len(), 1);
    assert_eq!(assets.iter_of::<u32>().count(), 1);
}

/// 解析失败：占位句柄全部移除（句柄失效），引用计数与反向索引清理干净。
#[test]
fn load_file_async_failure_removes_placeholders() {
    let (mut assets, path) = temp_manager_with("fail", "fail.bin", b"x");
    assets
        .load_file_async(
            FakeFileLoader {
                fail_parse: true,
                multi_type: false,
            },
            path.clone(),
        )
        .expect("scan 应成功");
    let handles = assets.loaded_handles_of::<u32>(&path);
    assert_eq!(handles.len(), 1);
    assert!(assets.status().in_flight >= 1);

    // get 阻塞到失败清理完成（占位移除 → 返回 None），比轮询确定。
    assert!(assets.get(handles[0]).is_none(), "失败后 get 应返回 None");
    assert_eq!(assets.status().in_flight, 0);
    assert!(assets.loaded_handles_of::<u32>(&path).is_empty());
    assert_eq!(assets.handle_state(handles[0]), HandleState::Invalid);
    assert!(assets.file_refs.is_empty(), "引用计数应随占位移除清零");
}

/// 异步加载的数据逐出（DiskOnly）后，get 经重载器完整重解析并自动取回。
#[test]
fn load_file_async_data_reloads_after_eviction() {
    let (mut assets, path) = temp_manager_with("reload", "reload.bin", b"x");
    assets
        .load_file_async(
            FakeFileLoader {
                fail_parse: false,
                multi_type: false,
            },
            path.clone(),
        )
        .expect("scan 应成功");
    let handle = assets.loaded_handles_of::<u32>(&path)[0];
    assert_eq!(*assets.get(handle).unwrap(), 7);

    assets.unload_memory(&path);
    assert_eq!(assets.handle_state(handle), HandleState::DiskOnly);
    assert_eq!(*assets.get(handle).unwrap(), 7, "重读后取回新数据");
    assert_eq!(assets.handle_state(handle), HandleState::Ready);
}

/// 状态查询：句柄状态（Ready/DiskOnly/Invalid）与库状态。
#[test]
fn handle_state_and_status_queries() {
    let mut assets = manager();
    let h = assets.register(5u32);
    assert_eq!(assets.handle_state(h), HandleState::Ready);

    let path: GamePath = "test:data.bin".parse().expect("合法路径");
    assets.set_file_reloader(path.clone(), u32_reloader);
    let f = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
    assets.unload_memory(&path);
    assert_eq!(assets.handle_state(f), HandleState::DiskOnly);

    let status = assets.status();
    assert_eq!(status.entries, 2);
    assert_eq!(status.ready, 1);
    assert_eq!(status.disk_only, 1);
    assert_eq!(status.in_flight, 0);

    assets.remove(h);
    assert_eq!(assets.handle_state(h), HandleState::Invalid);
}
