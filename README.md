
# OpenUtau DiffSinger Lunai (Ma0shu Fork)

> 基于 [keirokeer/OpenUtau-DiffSinger-Lunai](https://github.com/keirokeer/OpenUtau-DiffSinger-Lunai) 的自用修改版本，用于解决个人使用中的一些痛点问题

---

## ✨ 修改内容

### 📝 歌词相关

* 新增 **歌词快捷编辑功能**（详见下方说明）
* 优化歌词编辑窗口：

  * 取消强制前台限制
  * 应用歌词时才进行校验，减少卡顿
* 引入 [HubertFA](https://github.com/wolfgitpr/HubertFA) 模型：

  * 支持歌词自动填入（稳定性一般）
  * 需从 **Release** 下载模型并作为 Dependency 加载

---

### 🎛 Expression 相关

* 新增 **表情曲线批量修改** 功能
  * 位于菜单栏「批量编辑」下
  * ! 需要曲线本身有初始数据，可以随便在空白区画点曲线

* 渲染器建议优化：

  * 自动隐藏「当前渲染器不支持」的表情
  * 打开 PianoRoll 时自动获取建议表情

* 表情列表增强：

  * 提供 **中文名称显示**

---

### ⚙️ 其他改动

* 移除多实例限制

  ! 注意：

  * 可能导致渲染错位等异常
  * **正式调教建议使用单窗口**

---

### 🐞 Bug 修复

* 修复 Lunai 在 SingerHub 加载失败时弹窗报错

  * 在国内网络环境下此情况视为正常

---

## 🧩 歌词快捷编辑说明

功能入口：
👉 位于菜单栏 `?` 按钮旁的 **「词」按钮**

### 操作逻辑

* **普通点击音符**

  * 将该音符歌词替换为 `+~`
  * 原歌词不会删除，而是整体向后顺延

* **Alt + 点击音符**

  * 在点击位置分割音符
  * 与普通分割不同：

    * 新音符不会变为 `+`
    * 后续歌词会自动向前补位

* **拖动音符边界**

  * 若为两个音符之间的边界：

    * 自动同时调整相邻音符长度
    * 行为相当于 **Alt + 拖动**

---

## 📦 下载

👉 Windows x64 版本成品及HubertFA可在 **Release 页面**直接下载

---

## 🛠 编译

```powershell
$ver=(git describe --tags --match '[0-9]*' --abbrev=0).Trim();
$sha=(git rev-parse --short HEAD).Trim();
$env:APPVEYOR_BUILD_VERSION=$ver;
dotnet publish OpenUtau/OpenUtau.csproj -c Release -r win-x64 --self-contained true -o bin/win-x64 /p:Version=$ver /p:FileVersion=$ver /p:AssemblyVersion=$ver /p:InformationalVersion="$ver+$sha"
```

---

## 🍒 更多

> Feel free to cherry-pick.

---

## 📖 原项目README

# OpenUtau

OpenUtau is a free, open-source editor made for the UTAU community.

[![Build](https://img.shields.io/github/actions/workflow/status/stakira/OpenUtau/build.yml?style=for-the-badge)](https://github.com/stakira/OpenUtau/actions/workflows/build.yml)
[![Discord](https://img.shields.io/discord/551606189386104834?style=for-the-badge&label=discord&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/UfpMnqMmEM)
[![QQ Qroup](https://img.shields.io/badge/QQ-485658015-blue?style=for-the-badge)](https://qm.qq.com/cgi-bin/qm/qr?k=8EtEpehB1a-nfTNAnngTVqX3o9xoIxmT&jump_from=webapi)
[![Trello](https://img.shields.io/badge/trello-go-blue?style=for-the-badge&logo=trello)](https://trello.com/b/93ANoCIV/openutau)

## Getting started

[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=windows-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-win-x64.zip)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=windows-x86&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-win-x86.zip)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=macos-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-osx-x64.dmg)</br>
[![Download](https://img.shields.io/static/v1?style=for-the-badge&logo=github&label=download&message=linux-x64&labelColor=FF347C&color=4ea6ea)](https://github.com/stakira/OpenUtau/releases/latest/download/OpenUtau-linux-x64.tar.gz)

It is **strongly recommended** that you read these Github wiki pages before using the software.
- [Getting-Started](https://github.com/stakira/OpenUtau/wiki/Getting-Started)
- [Resamplers](https://github.com/stakira/OpenUtau/wiki/Resamplers-and-Wavtools)
- [Phonemizers](https://github.com/stakira/OpenUtau/wiki/Phonemizers)
- [FAQ](https://github.com/stakira/OpenUtau/wiki/FAQ)

- [中文使用说明](https://opensynth.miraheze.org/wiki/OpenUTAU/%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)

## How to contribute

Tried OpenUtau and not satisfied? Don't just walk away! You can help:
- Report issues on our [Discord server](https://discord.gg/UfpMnqMmEM) or Github.
- Suggest features on Discord or Github.
- Add or update translations for your language on [Crowdin](https://crowdin.com/project/oxygen-dioxideopenutau).

Know how to code? Got an idea for an improvement? Don't keep it to yourself!
- Contribute fixes via pull requests.
- Check out the development roadmap on [Trello](https://trello.com/b/93ANoCIV/openutau) and discuss it on Discord.

## Plugin development

Want to contribute plugins to help other users? Check out our API documentation:
- [Editing Macros API Document](OpenUtau.Core/Editing/README.md)
- [Phonemizers API Document](OpenUtau.Core/Api/README.md)

## Main features

Navigate the interface naturally and fluently using the mouse and scroll wheel. Keyboard shortcuts are also available.

![Editor](Misc/GIFs/editor.gif)

Easily create songs and covers using the feature-rich MIDI editor.

![Editor](Misc/GIFs/editor2.gif)

Create expressive vibratos with the easy-to-use vibrato editor.

![Vibrato](Misc/GIFs/vibrato.gif)

Pre-rendering and built-in resamplers let you quickly preview your work.

![Playback](Misc/GIFs/playback.gif)

See the [Getting-Started Wiki page](https://github.com/stakira/OpenUtau/wiki/Getting-Started) for more!

## All features
- Modern user experience.
- Easy navigation using the mouse and keyboard.
- Feature-rich MIDI editor.
  - Support for importing VSQX (Vocaloid 4) tracks.
- Selective backward compatibility with UTAU.
  - OpenUtau aims to solve problems with fewer steps. It is not designed to replicate UTAU features exactly.
- Extensible real-time phonetic editing.
  - Includes phonemizers for different phonetic systems (VCV, CVVC, Arpasing, etc.) in many different languages (English, Japanese, Chinese, Korean, Russian and more).
- Expressions replace the standard UTAU "flags" for tuning.
  - The built-in WORLDLINE-R resampler supports curve tuning, similar to many vocal synth editors.
- Internationalisation, including UI translation and file system encoding support.
  - Unlike UTAU, there is no need to change your system locale to use OpenUtau.
- Smooth preview/rendering experience.
  - Pre-rendering allows OpenUtau to render vocals before playback, saving time during editing and tuning.
- Supports ENUNU AI singers. See the [ENUNU wiki page](https://github.com/stakira/OpenUtau/wiki/ENUNU-NNSVS-Support) for more info.
- Easy-to-use plugin system.
- Versatile resampling engine interface.
  - Compatible with most UTAU resamplers.
- Runs on Windows (32/64 bit), macOS, and Linux.

### What it doesn't do
- While OpenUtau can do very minimal mixing, it will not replace your digital audio workstation of choice.
- OpenUtau does not aim for Vocaloid compatibility, except for some limited features.
