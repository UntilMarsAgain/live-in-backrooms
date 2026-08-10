//! 环境数据测试：HDR/LDR 解码、立方体贴图/辐照度/预过滤/BRDF LUT 转换验证。

use super::*;

/// 仓库内如有测试 HDR（assets 目录不入库），验证解码结果合理。
#[test]
fn decodes_repo_hdr_if_present() {
    let path = Path::new("test/test.hdr");
    if !path.is_file() {
        return;
    }
    let env = Environment::from_hdr_file(path).expect("HDR 应能解码");
    assert!(env.width > 0 && env.height > 0);
    assert_eq!(env.rgb.len(), (env.width * env.height) as usize);
    // HDRI 不该全黑：至少有一个非零像素。
    assert!(
        env.rgb
            .iter()
            .any(|p| p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0)
    );
}

/// LDR 解码：sRGB 中灰（128）应线性化为约 0.216，且曝光可放大。
#[test]
fn from_ldr_image_linearizes_and_applies_exposure() {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        2,
        2,
        image::Rgb([128, 128, 128]),
    ));
    let env = Environment::from_ldr_image(&img, 1.0).expect("LDR 应能构造");
    assert_eq!((env.width, env.height), (2, 2));
    // 128/255 ≈ 0.502，sRGB → linear ≈ 0.216。
    let linear = env.rgb[0][0];
    assert!((linear - 0.216).abs() < 0.01, "线性化偏差过大：{linear}");

    let env2 = Environment::from_ldr_image(&img, 2.0).expect("LDR 应能构造");
    assert!(
        (env2.rgb[0][0] - 2.0 * linear).abs() < 1e-5,
        "曝光应线性放大"
    );
}

/// 自动识别：`from_bytes` 按内容识别（PNG → LDR、HDR → HDR）；
/// `from_file` 按后缀识别（.png → LDR、.hdr → HDR）。
#[test]
fn from_bytes_auto_detects_format() {
    // PNG：用 image 编码器生成一张 2×1 的字节。
    let png =
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(2, 1, image::Rgb([255, 0, 0])));
    let mut png_bytes = Vec::new();
    png.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .expect("PNG 编码应成功");
    let env = Environment::from_bytes(&png_bytes).expect("PNG 自动识别应成功");
    // LDR 路径：红色 255 → 线性 1.0。
    assert!((env.rgb[0][0] - 1.0).abs() < 1e-4, "PNG 红色应线性化为 1.0");

    // from_file 按后缀：临时 .png 文件走 LDR。
    let tmp = std::env::temp_dir().join("env_from_file_test.png");
    std::fs::write(&tmp, &png_bytes).expect("写临时 PNG");
    let env = Environment::from_file(&tmp).expect(".png 后缀应走 LDR");
    assert!((env.rgb[0][0] - 1.0).abs() < 1e-4);
    let _ = std::fs::remove_file(&tmp);

    // HDR：仓库内如有测试 HDR，验证 from_file 按 .hdr 后缀走 HDR 路径。
    let path = Path::new("test/test.hdr");
    if path.is_file() {
        let env = Environment::from_file(path).expect("HDR 自动识别应成功");
        let max = env
            .rgb
            .iter()
            .fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
        assert!(max > 1.0, "HDR 应有 >1 的高动态范围值（区别于 LDR）");
    }
}
