//! 文件加载与重载：`AssetManager` 的同步/异步文件入口与占位填充。
#![allow(dead_code)]

use super::*;

impl AssetManager {
    /// 注册一个**类型擦除**的占位文件条目（[`FileLoader`] 异步加载用：
    /// 调用方只有 `TypeId`，没有静态类型 `T`）。占位句柄由调用方经
    /// `loaded_handles_of::<T>` 取回。
    fn register_pending_erased(
        &mut self,
        source: GamePath,
        type_id: TypeId,
        extra: Box<dyn Any + Send + Sync>,
    ) -> DefaultKey {
        *self.file_refs.entry(source.clone()).or_insert(0) += 1;
        let key = self.register_with_source_erased(
            AssetState::Loading,
            type_id,
            EntryData::File {
                source: source.clone(),
                extra,
                data: None,
            },
        );
        self.file_entries.entry(source).or_default().push(key);
        key
    }

    /// 注册一个文件的"重载器"（每个来源一次）：内存层缺失时用它重新解析。
    pub fn set_file_reloader(
        &mut self,
        source: GamePath,
        reload: impl Fn(&MergedResourceSpace, TypeId, &dyn Any) -> anyhow::Result<Box<dyn Any + Send + Sync>>
            + Send
            + Sync
            + 'static,
    ) {
        self.reloaders.insert(source, Box::new(reload));
    }

    /// 文件级重载器：数据逐出（DiskOnly）后**完整重解析**文件，按（类型, extra）
    /// 找回对应条目。同步/异步入口共用，保证所有加载方式的重载行为一致。
    fn file_reloader_for<L>(
        loader: L,
        source: GamePath,
    ) -> Box<
        dyn Fn(&MergedResourceSpace, TypeId, &dyn Any) -> anyhow::Result<Box<dyn Any + Send + Sync>>
            + Send
            + Sync,
    >
    where
        L: FileLoader + Clone,
    {
        Box::new(
            move |space: &MergedResourceSpace, type_id: TypeId, extra: &dyn Any| {
                let bytes = space.read(&source)?;
                let parsed = loader.parse(&bytes)?;
                for (tid, entries) in parsed {
                    if tid != type_id {
                        continue;
                    }
                    for (data, entry_extra) in entries {
                        if loader.extra_eq(extra, &*entry_extra) {
                            return Ok(data);
                        }
                    }
                }
                anyhow::bail!("重载时找不到对应条目：{source}")
            },
        )
    }

    // ---- 文件加载（同步 / 异步） ----

    /// **同步加载（文件级、多类型）**：主线程读文件并**完整解析一次**，立即把
    /// 该文件所有类型的条目注册进槽位（数据直接驻留内存，无占位阶段）。
    ///
    /// - 同文件已有条目（任意类型）→ 直接返回，不解析，复用已有句柄；
    /// - 与 [`Self::load_file_async`] 共用同一重载器逻辑（数据逐出后重放
    ///   [`FileLoader`] 完整重解析）；
    /// - 调用方用 `loaded_handles_of::<T>(&path)` 取各类型句柄。
    pub fn load_file<L>(&mut self, loader: L, path: GamePath) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        let bytes = self.space.read(&path)?;
        let parsed = loader.parse(&bytes)?;
        self.register_parsed_file(loader, path, parsed)
    }

    /// 注册一份**已解析**的文件条目（全部类型）并配置重载器；同文件已有条目
    /// 则跳过。同步/异步加载与 `load_scene`（资产层需要 glTF document 的
    /// 特殊入口）共用此逻辑，保证重载器一致。
    pub(crate) fn register_parsed_file<L>(
        &mut self,
        loader: L,
        path: GamePath,
        parsed: FileLoadResult,
    ) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        // 去重：同路径已有条目（任意类型）→ 复用，不再注册。
        if self.file_entries.contains_key(&path) {
            return Ok(());
        }
        if parsed.is_empty() {
            anyhow::bail!("文件没有可注册的条目：{path}");
        }
        // 条目数据直接进槽位（单一存储点，Resident）。
        for (type_id, entries) in parsed {
            for (data, extra) in entries {
                self.register_file_erased(path.clone(), type_id, extra, data);
            }
        }
        // 配置重载器：数据逐出后完整重解析，按（类型, extra）找回对应条目。
        self.set_file_reloader(path.clone(), Self::file_reloader_for(loader, path));
        Ok(())
    }

    /// **异步加载（文件级、多类型）**：立即注册每类型占位句柄（`Loading`），
    /// 后台线程**完整解析一次**产出该文件所有类型的条目后填充；`get` 遇到
    /// Loading 会阻塞等待。
    ///
    /// - 调用方先拿占位句柄（`loaded_handles_of::<T>(&path)`），随后 `get` 阻塞；
    /// - 同文件已有条目（任意类型，含加载中）→ 直接返回，不 scan、不 parse，
    ///   复用已有句柄；
    /// - 一个文件解析一次产出全部类型（如 glb 的 Mesh + Texture），没有
    ///   per-type 双解析；
    /// - 加载失败：占位句柄全部移除（调用方持有的句柄随之失效）；
    /// - 数据逐出（DiskOnly）后的重读经重载器完整重解析（重放 [`FileLoader`]）。
    pub fn load_file_async<L>(&mut self, loader: L, path: GamePath) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        // 去重：同路径已有条目（任意类型）→ 复用，不 scan、不 parse。
        if self.file_entries.contains_key(&path) {
            return Ok(());
        }
        // 1. 主线程读文件 + 轻量结构扫描（只读结构，不解析缓冲区）。
        let bytes = self.space.read(&path)?;
        let scanned = loader.scan(&bytes)?;
        if scanned.is_empty() {
            anyhow::bail!("文件没有可注册的条目：{path}");
        }
        // 2. 按类型注册占位句柄（各类型独立 extra 列表，顺序与 parse 对应）。
        for (type_id, extras) in scanned {
            for extra in extras {
                self.register_pending_erased(path.clone(), type_id, extra);
            }
        }
        // 3. 配置重载器（与同步入口共用同一份逻辑）。
        self.set_file_reloader(
            path.clone(),
            Self::file_reloader_for(loader.clone(), path.clone()),
        );
        // 4. 后台完整解析（一次产出所有类型）→ 回主线程 `pump` 填充占位。
        self.in_flight.insert(path.clone());
        let tx = self.load_tx.clone();
        let cond = self.load_cond.clone();
        std::thread::spawn(move || {
            let result = loader.parse(&bytes);
            let _ = tx.send((path, result));
            cond.1.notify_all();
        });
        Ok(())
    }

    /// 消费后台完成的结果：成功则填充占位句柄的数据，失败则移除占位句柄。
    pub(crate) fn pump(&mut self) {
        while let Ok((path, result)) = self.load_rx.try_recv() {
            self.in_flight.remove(&path);
            let keys: Vec<DefaultKey> = self
                .file_entries
                .get(&path)
                .map(|handles| handles.iter().copied().collect())
                .unwrap_or_default();
            match result {
                Ok(batches) => {
                    // 各类型条目与该类型占位句柄按注册顺序一一对应
                    // （scan 的注册顺序 = parse 的产出顺序）。
                    let mut filled = HashSet::new();
                    for (type_id, entries) in batches {
                        let typed_keys: Vec<DefaultKey> = keys
                            .iter()
                            .copied()
                            .filter(|key| {
                                self.slots
                                    .get(*key)
                                    .is_some_and(|slot| slot.type_id == type_id)
                            })
                            .collect();
                        for (key, (data, _extra)) in typed_keys.into_iter().zip(entries) {
                            if let Some(slot) = self.slots.get_mut(key) {
                                if let Some(EntryData::File {
                                    data: slot_data, ..
                                }) = &mut slot.data
                                {
                                    *slot_data = Some(data);
                                }
                                slot.state = AssetState::Resident;
                            }
                            filled.insert(key);
                        }
                    }
                    // 保险：scan 注册了但 parse 没产出的占位（加载器 bug）按失败清理，
                    // 避免 `get` 永久阻塞。
                    for key in keys {
                        if !filled.contains(&key)
                            && self
                                .slots
                                .get(key)
                                .is_some_and(|slot| slot.state == AssetState::Loading)
                        {
                            self.remove_key(key);
                        }
                    }
                }
                Err(_) => {
                    // 加载失败：该文件的占位句柄全部移除（调用方句柄随之失效）。
                    for key in keys {
                        self.remove_key(key);
                    }
                }
            }
        }
    }
}
