# Mac-First Local Voice-to-Summary Vertical Slice (MVP)

## Goal Description

为 “LocalMind Audio” 产品构建最小的、完全端侧的 macOS 端到端流水线：导入一个本地音频文件（`.m4a` / `.wav` / `.mp3`），用本地 Whisper 模型转写，再用本地 LLM 配合工作流 Prompt 模板（如会议纪要 / 待办事项）对转写文本进行分析，并产出符合 schema 的结构化摘要——将音频元数据、转写文本、摘要及完整溯源信息持久化到本地存储，且**运行时零网络访问**。

该切片刻意做成 LocalMind Audio PRD 中最薄的主干。它在引入任何多设备、流式或精度增强工作之前，先在 Mac 上验证核心价值闭环（导入 → 转写 → 分析 → 持久化）。为了在投入原生应用之前先消解激进的资源指标风险，实现分阶段推进：先做 Python 可行性 / 基准测试 CLI（M0–M1），再做结构化分析 + 持久化层（M2），最后做 Swift/SwiftUI 原生封装（M3）。

以下能力明确**不在本切片范围内**，推迟到后续产品阶段：多设备算力中继（iPhone/Watch、AirDrop/LAN）、流式 / 实时转写、基于 VAD 的静音剔除、热词增强、说话人分离、降噪。但 Mac 端应保持前向兼容，接受无状态的文件摄入，以便后续中继路径无需返工即可接入。

塑造本计划的已核实生态事实：
- 整个 monorepo 中**没有 MLX Whisper STT，也没有 whisper.cpp**；`mlx-vlm` 的音频模块是 VLM 编码器，而非转写。因此 STT 是真正的**外部依赖**——Python 阶段用 `mlx-whisper`（pip），M3 评估 Swift 原生方案（`whisper.cpp` / `whisper.spm`）。
- `llama.cpp` 的 Swift 绑定（`LlamaContext` actor，配合拉取式 `completion_loop()` token 流）是最终 Swift 应用的 LLM 路径。
- 规格中的 “E-cores + ANE” 性能说法对 MLX/llama.cpp 栈（GPU/CPU 路径）**不可靠**；本计划不假设使用 ANE。
- 代码库中尚无 AVFoundation 音频 I/O；音频解码确属全新工作（Python 用 `ffmpeg`/`librosa`；Swift 用 AVFoundation）。

已确认的用户决策（仍开放的条目见 `## Pending User Decisions`）：
- 最低支持硬件：任意配备 **16 GB** 统一内存的 Apple 芯片 Mac。
- 规格中的 Mac 预算（峰值 `<6 GB`、`RTF < 0.08`）属于**需测量并报告的努力目标**，而非通过 / 不通过的硬性门槛。
- STT 采用**质量分级**：whisper `small`/`medium` 是必须可用的基线；`large-v3` 是在内存允许时的基准 / 质量目标。

## Acceptance Criteria

遵循 TDD 理念，每条标准都包含正向与负向测试以实现确定性验证。由于资源预算已确认为**努力目标**，性能类 AC 验证的是指标被*测量并报告*，而非达到具体数值（见 AC-6）。

- AC-1: 流水线在无任何网络访问的情况下导入并解码本地 `.m4a`、`.wav`、`.mp3` 文件为归一化 PCM 表示。
  - Positive Tests (expected to PASS):
    - 导入一个有效的 16 kHz `.wav` 测试样本，得到样本数符合预期的解码后单声道 PCM 缓冲。
    - 导入一个立体声 44.1 kHz `.m4a` 测试样本，正确重采样 / 下混为单声道 16 kHz 缓冲。
    - 导入一个有效的 `.mp3` 测试样本成功，并报告其真实时长。
  - Negative Tests (expected to FAIL):
    - 导入零字节或被截断 / 损坏的音频文件时，以清晰的解码错误被拒绝，而非崩溃或静默返回空缓冲。
    - 导入不支持的容器格式（如未启用时的 `.flac`）时，以明确的 “unsupported format” 错误被拒绝。

- AC-2: 流水线使用分级 STT 模型选择，为约 10、30、60 分钟的基准测试样本产出带时间戳的转写片段。
  - Positive Tests (expected to PASS):
    - 用基线模型（`small`/`medium`）转写 10/30/60 分钟样本，得到有序片段，每个片段均带 `start`/`end` 时间戳且文本非空。
    - 片段时间戳单调非递减，且不超过音频时长。
    - 所选模型层级及其标识被记录到本次运行的溯源信息中。
  - Negative Tests (expected to FAIL):
    - 产出扁平、无时间信息的转写（无逐片段 `start`/`end`）时被片段校验器拒绝。
    - 请求一个未预置（无本地权重）的模型层级时，以 “model not available” 错误快速失败，而非尝试下载。
  - AC-2.1: 长音频转写采用分块策略，不会一次性把整段解码音频对所有模型加载。
    - Positive: 60 分钟样本通过有界分块完成转写；峰值音频缓冲内存不会在超出配置分块窗口后随整文件线性增长。
    - Negative: 在 60 分钟样本上禁用分块的配置，被标记为长文件路径不支持。

- AC-3: 流水线产出符合带版本 JSON schema 的结构化摘要，并对非法模型输出有明确的修复 / 失败策略。
  - Positive Tests (expected to PASS):
    - 正常转写产出可通过 schema 校验的 JSON，包含 `decisions`、`action_items`（每项含可空的 `owner` 与 `due_date` 字段）、`open_questions` 等部分。
    - 每条摘要论断都包含引用，指向一个或多个转写片段 ID（或时间戳）。
    - 当模型首次输出格式有误时，一次有界的修复尝试产出合法 JSON，且该修复被记录到溯源信息中。
  - Negative Tests (expected to FAIL):
    - 缺少必填 schema 字段的输出被校验器拒绝（而非静默接受）。
    - 修复尝试耗尽后，本次运行持久化一个明确的 `summary_failed` 工件（含原始输出与错误），而非产出空白或臆造的摘要。

- AC-4: 所有工件通过单一规范化持久化边界，连同完整溯源信息持久化到本地存储。
  - Positive Tests (expected to PASS):
    - 一次成功运行后，存储中包含相互关联的 `audio_asset`、有序的 `transcript_segment` 记录、`summary_artifact`、`inference_run` 以及 `model_manifest` 引用。
    - `inference_run` 记录 Prompt 模板哈希、模型标识 / 哈希、解码参数、分块参数以及各阶段指标。
    - 在新进程中重新打开存储，可无损读回完整运行记录。
  - Negative Tests (expected to FAIL):
    - 未关联转写与音频资产即持久化的摘要，被引用完整性检查拒绝。
    - 两个都自称规范化的存储层（如临时 JSON 与存储不一致）被一致性检查捕获。

- AC-5: 在一次完整的转写并摘要运行中，运行时执行零网络访问。
  - Positive Tests (expected to PASS):
    - 在出站网络被阻断（离线 / 拒绝出站）的情况下，完整运行成功完成。
    - 在模型已本地预置时，运行期间不发生 DNS 解析、HTTP(S) 请求或模型仓库拉取（由出站监控 / 断网测试框架验证）。
  - Negative Tests (expected to FAIL):
    - 运行时尝试解析或从远程仓库拉取模型的运行被检测到并使断网测试失败。
    - 运行期间任何遥测、崩溃上报或更新检查的网络调用使断网测试失败。

- AC-6: 每次运行测量并报告各阶段性能与资源指标（目标为努力目标，而非门槛）。
  - Positive Tests (expected to PASS):
    - 每次运行分别报告解码、STT、LLM 分析、持久化的耗时，以及相对音频时长计算的整体 RTF。
    - 峰值内存的报告明确说明测量方法，并区分 CPU 常驻内存与 Metal/GPU 分配。
    - 在 16 GB 参考 Mac 上为 10/30/60 分钟样本产出机器可读的基准报告，将实测值与努力目标 `<6 GB` / `RTF < 0.08` 对比。
  - Negative Tests (expected to FAIL):
    - 仅报告单一端到端时间、无各阶段拆分的运行，被基准测试框架拒绝。
    - 未说明测量对象（CPU RSS 还是 GPU）的峰值内存数据，被报告校验器拒绝。

- AC-7: 在任何 SwiftUI 封装开始之前，存在一份稳定的命令行契约并由测试覆盖。
  - Positive Tests (expected to PASS):
    - CLI 暴露有文档的子命令（如 `transcribe`、`summarize`、`analyze`、`benchmark`），读取输入并按既定 schema 发出 JSON / JSONL 进度事件。
    - CLI 契约由断言 JSON 输出形态的测试覆盖，独立于最终 UI。
  - Negative Tests (expected to FAIL):
    - 未提升版本号即对 CLI 输出 schema 做破坏性变更，被契约测试捕获。
    - 绕过既定 CLI/IPC 契约的 SwiftUI 封装工作，被标记为超出当前里程碑范围。

- AC-8: 长时任务支持进度报告、取消以及临时文件清理。
  - Positive Tests (expected to PASS):
    - 60 分钟样本在转写与分析期间发出增量进度事件。
    - 运行中途取消会及时停止后续计算并移除中间临时文件。
  - Negative Tests (expected to FAIL):
    - 取消后遗留孤立临时文件的运行，使清理测试失败。
    - 长文件路径无任何进度信号的运行被拒绝。

- AC-9: 模型离线预置并带完整性校验，且在模型缺失时运行时行为确定。
  - Positive Tests (expected to PASS):
    - 有文档的预置步骤将 Whisper 与 LLM 权重放入已知本地目录；首次使用前由校验和 / 清单校验确认完整性。
    - 权重存在时，运行仅使用本地文件。
  - Negative Tests (expected to FAIL):
    - 校验和与清单不匹配的权重文件，在推理开始前被拒绝。
    - 权重缺失时，运行以明确的 “model not provisioned” 提示快速失败，而非尝试网络下载。

## Path Boundaries

路径边界定义了实现质量与选择的可接受范围。

### Upper Bound (Maximum Acceptable Scope)
一个分阶段实现，交付：(M0) 带校验和清单的离线模型预置步骤，以及固定的样本 / 基准测试框架；(M1) 一个 Python `mlx-whisper` + LLM 基准测试 CLI，具备稳定的 JSON/JSONL 契约，在 10/30/60 分钟样本上测量并报告各阶段指标，并以经验方式判定顺序加载与并发加载模型；(M2) 分块 map-reduce 摘要，产出带片段引用、符合 schema 的 JSON，配合严格的修复 / 失败策略，连同完整溯源持久化到规范化本地存储；以及 (M3) 一个 Swift/SwiftUI macOS 应用，封装已验证的契约，通过 `llama.cpp` `LlamaContext` 运行 LLM，经由 SwiftData 在规范化存储之上持久化，强制 App Sandbox + 无网络授权，并验证零网络保证。所有 AC 均由正向与负向测试覆盖。这在不引入被推迟特性的前提下完成 Mac 优先的价值闭环。

### Lower Bound (Minimum Acceptable Scope)
一个 Python CLI（M0–M2），可导入 `.wav`/`.m4a`/`.mp3`，用基线 whisper 层级将基准样本转写为带时间戳的片段，产出符合 schema 的 JSON 摘要（至少含 decisions / action_items / open_questions 以及可用的修复 / 失败路径），将音频元数据 + 转写 + 摘要 + 溯源持久化到单一规范化本地存储，完全离线运行并通过断网测试，并产出各阶段基准报告。这在不含原生 Swift 应用（M3 可推迟）的情况下满足 AC-1 至 AC-9，足以验证可行性。

### Allowed Choices
- 可使用：在 M0–M2 用 Python 配合 `mlx-whisper`（和 / 或 `whisper.cpp`）做 STT；用 MLX-LM 或 `llama.cpp` 做本地 LLM；Python 中用 `ffmpeg`/`librosa`/`soundfile` 做音频解码；在 M0–M2 用 SQLite 和 / 或 JSON 工件文件作为规范化存储；M3 用 SwiftUI + Swift 6 + `llama.cpp` `LlamaContext` + SwiftData（作为规范化存储之上的应用门面）+ AVFoundation；带明确 schema 版本控制的 JSON-schema 校验器。
- 可使用：在**原型阶段**无条件使用带文档 IPC 契约的 Python 推理 worker 子进程；其是否也可随 M3 应用一并交付属于 DEC-1（开放）。
- 不可使用：任何云端 STT/LLM API、运行时远程模型仓库拉取、遥测、崩溃上报或更新检查的网络调用；将 ANE 作为 MLX/llama.cpp 路径的假定加速器；本切片中的热词、说话人分离、VAD、降噪、流式或多设备传输（均推迟）。

> **Note on Deterministic Designs**: 草案固定了若干选择（100% 端侧、零网络、本地沙盒存储、Apple 生态栈）。这些约束被视为硬性且不可协商。在草案留有工程余地之处（模型层级、持久化引擎、原型运行时），上述边界描述了可接受范围，开放条目则上提至 `## Pending User Decisions`。

## Feasibility Hints and Suggestions

> **Note**: 本节仅供参考与理解。这些是概念性建议，而非强制要求。

### Conceptual Approach

```
provision (M0):  fetch+verify weights -> local model dir + checksum manifest
                 assemble fixed audio fixtures (10/30/60 min) + expected-output examples

run (M1..M2):
  decode(audioFile) -> pcm16k_mono                 # ffmpeg/soundfile; record duration
  segments = []
  for chunk in chunk_audio(pcm, window):           # bounded memory; AC-2.1
      segments += whisper_transcribe(chunk, tier)  # mlx-whisper; timestamps per segment
  validate_segments(segments)                      # AC-2

  partials = []
  for group in chunk_transcript(segments, ctx):    # map step; respects LLM context limit
      partials += llm(prompt_template, group)       # local LLM
  summary = llm_reduce(prompt_template, partials)   # reduce step -> structured draft
  summary = validate_or_repair(summary, schema)     # bounded repairs; else summary_failed (AC-3)

  persist(audio_asset, segments, summary, inference_run, model_manifest)   # canonical store (AC-4)
  emit(benchmark_report{per-stage timings, RTF, peak_mem cpu/gpu})         # AC-6

wrap (M3): SwiftUI app -> same CLI/IPC contract -> llama.cpp LlamaContext + SwiftData facade
                       -> App Sandbox + no-network entitlements -> verify AC-5
```

- 在 M1 中以经验方式判定模型是顺序常驻还是并发常驻：在 16 GB 上同时加载 Whisper 与 7B LLM 可能超出努力目标预算；在 STT 与 LLM 阶段之间顺序加载 / 卸载，是先做基准测试的安全默认。
- 让 CLI 的 JSONL 进度事件形态贴合最终的 Swift IPC 契约，使 M3 的接线变得机械化。
- 把 Prompt 模板、schema 与分块参数视为带版本的输入，记录在每个 `inference_run` 中以保证可复现。

### Relevant References
- `Agent4Kernel/llama.cpp/examples/llama.swiftui/` — `LlamaContext` actor，拉取式 `completion_loop()` token 流；M3 Swift 应用的 LLM 路径。
- `Agent4Kernel/mlx-vlm/mlx_vlm/generate.py` — MLX-LM 加载器 + 采样工具（LLM 分析参考；注意这些不是 Whisper STT）。
- `Agent4Kernel/mlx-vlm/mlx_vlm/models/gemma4/audio.py` — MLX 音频编码器梅尔频谱先例（VLM 编码器，非转写流水线；仅供参考）。
- `Agent4Kernel/Agent-Signal-Bar/` — 仅 macOS 的 Swift 6 SwiftUI 应用骨架，`Package.swift` 目标布局、`FileManager` I/O 与状态管理模式，可用于 M3 封装。
- `Agent4Kernel/Agent-Signal-Bar/Sources/AgentSignalLightCore/SignalStateStore.swift` — 基于文件的状态存储 + POSIX 锁模式，适用于规范化本地存储与前向兼容的无状态摄入。
- `Agent4Kernel/Codex-Quota-Viewer/Sources/CodexQuotaViewer/SafeSwitchBackup.swift` 与 `VaultAccountRecordWriter.swift` — CryptoKit 完整性哈希、原子写入、POSIX `0o600/0o700` 加固；可作为 AC-9 清单校验及未来静态加密的参考。
- `Agent4Kernel/oh-my-pi/packages/coding-agent/src/stt/stt-controller.ts` — STT 状态机（idle → recording → transcribing），含临时文件清理与中止处理；AC-8 参考。
- `Agent4Kernel/oh-my-pi/scripts/macos-entitlements.plist` — 强化运行时授权先例，用于 M3 无网络沙盒。
- `Agent4Kernel/vllm/vllm/multimodal/audio.py` — `split_audio()` 低能量分块边界检测；AC-2.1 分块策略参考。
- 外部（不在仓库内）：`mlx-whisper` pip 包 — M0–M2 的关键 STT 依赖。

## Dependencies and Sequence

### Milestones

1. M0 — Provisioning & Harness：建立离线、可复现的基础。
   - Phase A: 离线模型预置 — 本地模型目录、校验和 / 清单校验、有文档的一次性权重获取（在应用之外）（AC-9）。
   - Phase B: 样本与基准测试框架 — 固定的 10/30/60 分钟音频样本、期望转写容差与示例摘要，以及机器可读的基准报告格式（支撑 AC-2、AC-6）。

2. M1 — Benchmark CLI：验证可行性并锁定契约。
   - Step 1: 将 `.wav`/`.m4a`/`.mp3` 解码为归一化 PCM 的音频解码（AC-1）。
   - Step 2: 带分块的分级 `mlx-whisper` 转写，产出带时间戳片段（AC-2、AC-2.1）。
   - Step 3: 带子命令 + 进度事件的稳定 CLI JSON/JSONL 契约；各阶段指标、RTF 与 CPU/GPU 峰值内存报告；以经验方式判定模型顺序 / 并发常驻（AC-6、AC-7、AC-8）。

3. M2 — Structured Analysis & Persistence：把转写转化为可信工件。
   - Step 1: 尊重上下文上限的分块 map-reduce LLM 摘要（AC-3）。
   - Step 2: 带片段引用的带版本 JSON schema + 有界修复 / 失败策略（AC-3）。
   - Step 3: 带溯源与引用完整性检查的规范化本地存储（AC-4）。
   - Step 4: 覆盖完整运行的断网验证框架（AC-5）。

4. M3 — Swift/SwiftUI Native Wrap：交付原生 Mac 应用（按 Lower Bound 可推迟）。
   - Step 1: 在 M1 CLI/IPC 契约之上的 SwiftUI 应用骨架 + 文件选择器 + 进度 UI（AC-7、AC-8）。
   - Step 2: 经由 `llama.cpp` `LlamaContext` 的 LLM；经由 Swift 原生 `whisper.cpp`/`whisper.spm` 或捆绑的 Python worker 的 STT（受 DEC-1 约束）。
   - Step 3: 规范化存储之上的 SwiftData 门面（AC-4）；App Sandbox + 无网络授权；验证零网络（AC-5）。

依赖说明（相对关系，非时间）：M1 依赖 M0（权重 + 样本）。M2 依赖 M1（转写片段 + CLI 契约）。M3 依赖稳定的 M1 契约与 M2 的 schema / 存储，以及 DEC-1 的落定。断网框架（AC-5）可在 M2（CLI）与 M3（应用）间复用。

## Task Breakdown

每个任务必须恰好带一个路由标签：
- `coding`: 由 Claude 实现
- `analyze`: 经由 Codex 执行（`/humanize:ask-codex`）

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 实现离线模型预置：本地模型目录 + 校验和 / 清单校验；记录一次性权重获取文档 | AC-9 | coding | - |
| task2 | 组装固定的 10/30/60 分钟音频样本、期望输出示例与基准报告格式 | AC-2, AC-6 | coding | - |
| task3 | 实现音频解码（`.wav`/`.m4a`/`.mp3` → 归一化 PCM），含损坏 / 不支持格式的拒绝 | AC-1 | coding | task1 |
| task4 | 实现带有界分块的分级 `mlx-whisper` 转写与带时间戳片段校验 | AC-2, AC-2.1 | coding | task1, task3 |
| task5 | 在 16 GB 参考 Mac 上对模型顺序常驻与并发常驻做基准测试；给出默认建议 | AC-6 | analyze | task4 |
| task6 | 定义并实现稳定的 CLI JSON/JSONL 契约（子命令 + 进度事件）及契约测试 | AC-7, AC-8 | coding | task4 |
| task7 | 实现各阶段指标 + RTF + CPU/GPU 峰值内存报告与机器可读基准报告 | AC-6 | coding | task6 |
| task8 | 设计带版本的摘要 JSON schema（decisions、action_items{owner,due_date}、open_questions、引用） | AC-3 | analyze | - |
| task9 | 实现尊重上下文上限的分块 map-reduce LLM 摘要 | AC-3 | coding | task6, task8 |
| task10 | 实现 schema 校验 + 有界修复 / 失败策略，耗尽时持久化 `summary_failed` 工件 | AC-3 | coding | task8, task9 |
| task11 | 实现规范化本地存储（实体 + 溯源）及引用完整性检查 | AC-4 | coding | task6 |
| task12 | 实现断网验证框架并对整条流水线运行 | AC-5 | coding | task6, task11 |
| task13 | 实现长文件进度、取消与临时文件清理 | AC-8 | coding | task6 |
| task14 | M3：在 CLI/IPC 契约之上的 SwiftUI 骨架；`llama.cpp` LlamaContext LLM；SwiftData 门面；App Sandbox + 无网络授权（受 DEC-1 约束） | AC-4, AC-5, AC-7, AC-8 | coding | task6, task10, task11, task12 |

## Claude-Codex Deliberation

### Agreements
- Mac 优先的垂直切片范围正确；推迟流式、VAD、热词、说话人分离与多设备中继是恰当的 MVP 边界。
- 在任何 Swift 打包之前先做基准优先的 Python CLI（M1），是在真实硬件上验证最高风险假设（内存、RTF）的正确方式。
- 带回指转写片段 ID 引用的、带版本的结构化摘要 schema，是一个强有力且可测试的产品锚点。
- 进度、取消、临时文件清理、各阶段计时与峰值内存报告属于 MVP，因为长达一小时的本地推理即便正确也会让人感觉像坏了。
- 应从 MLX/llama.cpp 路径的性能说法中移除 ANE；AVFoundation 音频 I/O 确属全新工作。

### Resolved Disagreements
- STT 复用 vs 外部依赖：草案暗示复用 `mlx-vlm` 的梅尔频谱代码做 Whisper。代码勘探表明那些是 VLM 编码器、无转写流水线。结论：STT 是明确的外部依赖（`mlx-whisper`/`whisper.cpp`）；计划中的引用已据此更正。
- 持久化分层（Codex 指出 SwiftData 与 SQLite/JSON 冲突）：结论——单一**规范化**存储（SQLite + JSON 工件）在 M0–M2 全程权威；M3 中的 SwiftData 是同一规范化 schema 之上的应用门面，而非第二事实来源。
- 模型预置 + 零网络完整性（Codex 要求一等公民式预置）：结论——新增带校验和清单的 M0 预置（AC-9）与显式断网运行时框架（AC-5）；一次性权重获取在应用之外完成。
- 峰值内存歧义：结论——AC-6 要求说明测量方法并将 CPU RSS 与 Metal/GPU 分配区分开。
- JSON 修复 / 失败策略不明确：结论——AC-3 固定有界修复次数、schema 版本控制以及耗尽时持久化 `summary_failed` 工件。
- large-v3 默认 vs 紧预算下的可行性：结论——用户选择质量分级 STT（基线 `small`/`medium`，`large-v3` 作目标），消除了预算冲突。

### Convergence Status
- 已完成 Codex 首轮分析（Phase 3）与一轮收敛回合（Phase 5）；所有 Codex `REQUIRED_CHANGES` 要么已纳入计划，要么上提为用户决策。
- Final Status: `partially_converged` — Claude/Codex 的技术分歧已解决；残留条目为真正的产品决策，已转入 `## Pending User Decisions`（尤其是 DEC-1，用户选择推迟到 M1 基准测试之后再定）。

## Pending User Decisions

- DEC-1: Python 是否可随 M3 macOS 应用一并交付，还是仅限原型？
  - Claude Position: 交付纯 Swift 的 M3 应用（LLM 用 `llama.cpp`，STT 用 Swift 原生 `whisper.cpp`/`whisper.spm`），以规避 macOS 打包 / 沙盒 / 签名 / 离线依赖的陡坡；把 Python 限制在 M0–M2。
  - Codex Position: 可以推迟，但若 Python 可交付，计划必须为捆绑 worker 增加打包、沙盒、IPC、崩溃恢复与离线依赖处理。
  - Tradeoff Summary: 纯 Swift 更便于分发与沙盒，但需要 Swift 原生 STT 集成；捆绑 Python worker 上线更快，但增加可观的打包 / 沙盒风险。
  - Decision Status: `PENDING` — 用户选择在 M1 基准测试展示 Swift 原生路径能否达标后再定。触发点：M1 结束时复审。

- DEC-2: 模型权重如何预置——随应用捆绑、用户导入，还是由独立离线步骤准备？
  - Claude Position: 采用由校验和清单校验的用户导入 / 离线准备的本地模型目录（AC-9），保持应用二进制小巧，并将网络严格限定为应用之外的一次性手动预下载。
  - Codex Position: 无论采用何种方式，预置都必须是一等公民——权重位置、导入流程、校验和 / 清单校验、许可证追踪、量化格式、磁盘占用与模型缺失时的启动行为都必须明确。
  - Tradeoff Summary: 捆绑对用户最简单但会撑大应用并使许可 / 更新复杂化；用户导入 / 离线准备保持应用精简且许可清晰，但增加一个安装步骤。
  - Decision Status: `PENDING`（建议：用户导入 / 离线准备目录）。

- DEC-3: MVP 是否需要静态加密，还是推迟、暂以沙盒 + 文件系统权限兜底？
  - Claude Position: M1–M2 推迟静态加密；依靠 App Sandbox + POSIX `0o600/0o700` 加固与仅本地存储，并在可分发的 M3 再考虑 CryptoKit 加密。
  - Codex Position: 对语音转写而言这可能重要；它需要密钥管理决策，且不应作为未界定特性藏在 M3 内。
  - Tradeoff Summary: 推迟会加快可行性切片，但在原型期会把敏感转写以未加密形式落盘；现在就要求则在核心闭环验证之前引入密钥管理范围。
  - Decision Status: `PENDING`（建议：推迟到 M3，期间以沙盒 + 权限兜底）。

- DEC-4: MVP 最看重哪种产出——原始转写准确度、待办事项抽取、会议纪要，还是隐私证明？
  - Claude Position: 以会议纪要 + 待办事项抽取作为主要摘要工件，并将断网隐私证明（AC-5）作为同等重要的产品承诺。
  - Codex Position: N/A - open question（Codex 将其列为需要用户产品优先级的决策）。
  - Tradeoff Summary: 所选优先级会塑造 Prompt 模板设计以及哪条 AC 被视为头号成功指标；对规格中的会议 / 头脑风暴场景而言，纪要 + 待办事项是最可演示的价值。
  - Decision Status: `PENDING`（建议：会议纪要 + 待办事项）。

## Implementation Notes

### Code Style Requirements
- 实现代码与注释**不得**包含计划专用术语，如 “AC-”、“Milestone”、“Step”、“Phase” 或类似的工作流标记。
- 这些术语仅用于计划文档，不应进入最终代码库。
- 代码中改用描述性、贴合领域的命名（如 `transcribe_segments`、`summary_schema`、`benchmark_report`、`model_manifest`）。
- 在代码中遵守硬性产品约束：运行时路径上无网络客户端；所有存储位于本地沙盒之下；模型加载代码中不内置 ANE 假设。

--- Original Design Draft Start ---

# Mac-First Local Voice-To-Summary Vertical Slice

## Original Idea

# 全本地 AI 工作语音助手 (LocalMind Audio) 产品需求与技术规格说明书 (PRD & Tech Spec)

## 文档元信息
- **产品名称**: LocalMind Audio (暂定)
- **当前版本**: v1.0 (纯端侧架构)
- **部署策略**: 100% 本地化 (iOS / macOS)
- **核心特性**: 硬件级隐私、零网络依赖

---

## 1. 产品概述与核心愿景
在日常办公、会议及头脑风暴场景中，高效记录与总结工作内容是核心痛点。然而，现有的云端语音转文本（STT）和大模型（LLM）工具往往面临严重的数据隐私泄露风险，尤其是涉及商业机密或公司敏感讨论时。此外，高昂的订阅费和网络依赖也限制了使用场景。

**LocalMind Audio** 的核心愿景是建立一套**完全运行于用户本地苹果生态（Mac/iPhone/Watch）**的智能语音转文字与内容分析系统。通过压榨端侧 Apple Silicon (ANE/GPU) 的硬件算力，实现高度精准、极低延迟且绝对安全的本地数据闭环。首期产品聚焦于 iPhone 和 Mac 的联动部署，最大程度贴合用户工作发生在电脑和手机上的现实情况。

---

## 2. 跨设备架构设计与算力分工
初始版本放弃复杂的云服务器架构，采用苹果生态内的“多端联动、本地自主”方案。根据设备算力与使用习惯，进行如下分工：

| 目标设备 | 核心职责 | 技术特征与算力利用 |
| :--- | :--- | :--- |
| **Apple Watch** | 便携式录音入口、快捷一键触发、短语音备忘录采集。 | AVFoundation 后台录音，压缩为低码率 AAC 格式存储。 |
| **iPhone (iOS)** | 移动办公场景下的音频采集、中等长度音频实时/近实时转写、移动端轻量级摘要分析。 | 调用 CoreML 优化版的 Whisper 模型，利用苹果神经网络引擎 (ANE) 硬件加速。 |
| **Mac (macOS)** | 高性能处理中心、长音频高精度转写（Large模型）、深度的上下文关联分析与复盘看板。 | 利用 MLX 框架或 Llama.cpp，充分压榨统一内存与 M系列芯片 GPU/ANE 算力。 |

---

## 3. 核心功能模块规格 (Functional Specification)

### 3.1 音频采集与智能预处理模块
- **一键隐藏录音**：支持 iOS 小组件、Mac 菜单栏挂载以及 Apple Watch 复杂功能（Complication）一键唤醒录音，支持后台运行与息屏录制。
- **本地端点检测 (VAD)**：集成轻量级端侧 VAD（如基于 CoreML 的 Silero VAD 改编版）。在录音过程中自动剔除长时间的静音环境噪音，降低后续 STT 推理的计算量，规避模型幻觉。
- **智能降噪与人声增强**：针对 Apple Watch 垂手录音或 iPhone 桌面放置拾音的低信噪比场景，使用 CoreAudio 内建的音频过滤器进行低通及人声频段增强。

### 3.2 本地高性能语音转文本 (Local STT)
- **多级模型策略**：
  - **iPhone 端**：默认搭载 *Whisper-Base/Small-CoreML* 量化版，满足日常快速速记与低延迟转写需求，内存占用控制在 500MB 以内。
  - **Mac 端**：默认搭载 *Whisper-Large-v3* 经 Apple Silicon / MLX 推理加速优化版（支持 INT4/INT8 量化），专攻大型会议、专业技术名词的高精度转写。
- **专用词表外挂 (Hot-words Boosting)**：允许用户在本地配置“专业术语/人名/公司名”本地词表，在转写时进行偏置注入，大幅提升中英夹杂及垂直技术领域的转写准确率。
- **本地说话人分离 (Speaker Diarization)**：Mac 端提供本地声纹聚类功能，支持区分不同发言人（如：发言人A、发言人B），并支持本地手动标注真实姓名。

### 3.3 本地上下文智能分析 (Local LLM Analysis)
- **端侧大模型矩阵**：
  - **iPhone 端**：接入 3B 参数规模模型（如 Qwen-2.5-3B-Instruct 转化为 CoreML 格式），执行即时的待办事项（Action Items）提取。
  - **Mac 端**：接入 7B-8B 参数规模模型（如 Qwen-2.5-7B-Instruct / Llama-3-8B 优化版），支持全篇复杂长文本的结构化总结。
- **预设工作流 Prompt 模板**：内建“会议纪要模式”、“灵感头脑风暴提取”、“技术方案复盘”、“口水话过滤精简”等高实用性本地 Prompt。

> 🔒 **数据安全与隐私底线规格**
> 所有音频原始文件、转写文本、分析摘要数据，**一律存储在用户本地的沙盒 (Sandbox) 与 CoreData 数据库中**。绝不向任何外部服务器发送请求。设备间同步严格基于苹果原生加密的 AirDrop 或局域网点对点通信（或用户自主选择的本地加密数据包同步）。

---

## 4. 核心技术架构与推理优化选型 (Tech Stack)

### 4.1 核心技术栈选择
- **前端界面与生态逻辑**: SwiftUI (一套代码通过多端适配部署于 watchOS, iOS, macOS)。
- **数据存储**: SwiftData / CoreData，本地音频采用高压缩率且利于流式流转的 M4A (AAC 编码) 格式。
- **STT 推理引擎**: 
  - iOS: `whisper.spm` (基于 whisper.cpp 的 Swift 封装) 或 Apple 原生 CoreML 编译模型。
  - macOS: **MLX Framework**（苹果官方针对 Apple Silicon 优化的机器学习框架），利用统一内存架构实现极限并发推理。
- **LLM 推理引擎**: 
  - iOS: `LLM.swift` / CoreML-Execution。
  - macOS: **MLX-LM** 或 `Llama.cpp` 的 Swift 绑定，确保最大化释放 GPU 和 ANE 性能。

### 4.2 性能与资源约束指标 (Resource Budgets)
为确保工具在手机和电脑后台运行时不引发系统杀进程或严重发热，设定如下严格的资源指标：

| 性能指标 | iPhone 运行指标约束 (以 iPhone 15 Pro 及以上为例) | Mac 运行指标约束 (以 M系列 统一内存芯片为例) |
| :--- | :--- | :--- |
| **峰值内存占用** | < 1.2 GB (VAD + Whisper-Base + 3B LLM 分阶段唤醒) | < 6.0 GB (Whisper-Large-v3-4bit + 7B LLM-4bit 并发) |
| **推理速度比 (RTF)** | < 0.25 (即 60 分钟音频转写小于 15 分钟) | < 0.08 (即 60 分钟音频高精度转写与分析在 5 分钟内完成) |
| **电池/功耗控制** | 录音状态下每小时耗电 < 3%；STT 推理时允许短时发热，但需实施计算降频控温。 | 完全在后台静默运行，利用低功耗核心 (E-cores) 与 ANE 协同，不干扰主流工作软件运行。 |

---

## 5. 阶段性实施路线图 (Roadmap)

### Phase 1: Mac 端最小可行性产品 (MVP) 验证
优先在 macOS 上打通核心逻辑。利用 Mac 强劲的统一内存支撑全尺寸 Whisper-Large 与 7B 大模型的本地运行。开发出导入音频文件->本地转写->本地 Prompt 总结的完整闭环，验证本地词表注入（Hot-words）和说话人分离算法的准确度率。

### Phase 2: iPhone 移动端迁移与双端联动
引入 CoreML 框架，将模型进行 4-bit 量化裁剪后移植至 iOS。实现 iPhone 现场录音、本地轻量转写。建立局域网/AirDrop 本地传输机制，允许用户在手机上录音，一键“投喂”给 Mac 进行更深度的超长文本 LLM 矩阵分析。

### Phase 3: Apple Watch 接入与交互极致优化
开发 watchOS 独立客户端，主打“抬腕即录”的极简交互。打通 Watch 采集、iPhone 动态接收并充当本地算力中继的流动链路。优化底层算子计算，结合最新的端侧 Attention 加速库设计，将本地语音智能的延迟压低到极致。

## Primary Direction: Mac-First Vertical Slice

### Rationale

Builds the single smallest end-to-end pipeline (import audio file → MLX Whisper transcribe → local LLM prompt summary) on macOS only, deferring all multi-device complexity to prove the core value loop fastest.

### Approach Summary

Build a macOS-native application (a CLI + minimal SwiftUI shell) that implements the tightest end-to-end loop of the LocalMind Audio product: import a local audio file → run local Whisper transcription on the MLX framework → apply a local LLM prompt template → emit a structured summary, persisting both transcript and summary to a local SwiftData/CoreData store.

Core mechanism and affected components:
- **Audio import**: load/decode local M4A/WAV/MP3 files via `AVAudioFile` / `AVAudioEngine`.
- **Local STT pipeline**: MLX-based Whisper-Large-v3 inference on Apple Silicon, reusing the mel-spectrogram preprocessing already demonstrated in the sibling `mlx-vlm` audio encoder; ported to Swift or invoked via a Python subprocess for the prototype.
- **Local LLM analysis**: a 7B–8B quantized model (Qwen-2.5-7B / Llama-3-8B, INT4) via MLX-LM or llama.cpp Swift bindings, applying a single prompt template (e.g. meeting minutes / action-item extraction) to start.
- **Output & persistence**: write transcript + summary to a local SwiftData/CoreData DB and export as Markdown/JSON.
- **Scaffolding**: a macOS menubar + SwiftUI app shell, file picker, and progress UI, deferred behind a CLI-first prototype to validate latency/memory before wiring UI.

Recommended sequencing: stand up a Python CLI prototype (orchestrating `mlx-vlm` Whisper + MLX-LM) to validate end-to-end RTF and peak memory against the spec budgets on the target M-series Mac, then wrap the validated loop in Swift/SwiftUI. This contains the MLX↔Swift binding risk to a later, lower-stakes step.

### Objective Evidence

- `Agent4Kernel/mlx-vlm/mlx_vlm/models/gemma4/audio.py` + `audio_feature_extractor.py` — Conformer-based audio encoder already ported to MLX; viable mel-spectrogram + inference pipeline on Apple Silicon.
- `Agent4Kernel/mlx-vlm/mlx_vlm/generate.py` (lines 11–48) — production-grade MLX-LM loader + sampling utilities; MLX handles unified-memory + GPU/ANE scheduling natively.
- `Agent4Kernel/llama.cpp/examples/llama.swiftui/` (`LlamaState.swift`) — SwiftUI LLM loader/inference example with model download, documents-directory management, and async `complete()`; validates llama.cpp Swift bindings on macOS.
- `Agent4Kernel/Agent-Signal-Bar/` — full macOS menubar + SwiftUI app (Swift 6, macOS 14+) demonstrating packaging, `FileManager` file I/O, and `@StateObject`/`ObservableObject` state management directly transferable to the app shell.
- `Agent4Kernel/KernelOwl/experiments/host_wrapper.mm` — Objective-C++ Metal compute pipeline pattern, validating the approach of wrapping GPU/ANE operators for Apple Silicon.
- `Agent4Kernel/oh-my-pi/packages/coding-agent/src/stt/stt-controller.ts` — STT state-machine pattern (idle → recording → transcribing) with dependency resolution, temp-file cleanup, and abort handling — a reference for the pipeline controller.
- `SoundMind/spec.md` §5 Phase 1 — the spec itself prioritizes exactly this macOS "import → transcribe → summarize" loop as the first deliverable.

### Known Risks

- **MLX-to-Swift binding gap**: MLX is Python-first; Whisper inference may require a subprocess call or hand-porting mel-spectrogram math to Swift `Accelerate`. No Swift-native MLX binding found in the ecosystem.
- **Quantization & memory**: Whisper-Large-v3 (~1.5B) + 7B LLM (even INT4) ≈ 3.5 GB+ unified memory on load; must validate against the spec's <6 GB Mac budget on actual M1+ hardware (OOM risk on base M1/VMs).
- **RTF target is tight**: spec demands RTF <0.08 (60 min audio in 5 min); achievable on M3+ but requires careful batching/streaming, not a naive sequential pipeline.
- **Scope-creep pressure**: VAD, hot-words boosting, and diarization are tempting but should be deferred past the first slice to keep the surface minimal.
- **Forward-compat**: the Phase-2 iPhone→Mac relay implies the Mac side should accept stateless file ingestion now, even though relay itself is out of scope.

## Alternative Directions Considered

### Alt-1: Cross-Device Compute Relay
- Gist: Treat the distributed orchestration layer as the central problem — a device-capability registry (memory, RTF ceiling, available models) advertised over Bonjour/mDNS or BLE, a request-response relay envelope wrapping audio/transcript buffers with processing hints, and routing so that Watch captures → iPhone does light Whisper-Base → Mac does deep 7B/8B analysis. Introduces a `DeviceOrchestrator` layer plus an extension of the file-based state-store pattern to track durable, acked relay sessions.
- Objective Evidence:
  - `Agent4Kernel/Agent-Signal-Bar/Sources/AgentSignalLightCore/SignalStateStore.swift` — file-based JSON state with POSIX `flock` locks; maps to relay session tracking.
  - `Agent4Kernel/oh-my-pi/python/omp-rpc/` — typed RPC over stdio + event hooks; protocol/event-streaming model portable to Swift (Combine / Network.framework).
  - `Agent4Kernel/llama.cpp/examples/llama.swiftui/` (`LlamaState.swift`) — local model loading + streaming generation, ready to wrap with a remote-inference proxy.
  - `SoundMind/spec.md` lines 82–83 — Phase 2 explicitly names the LAN/AirDrop "record on phone, feed to Mac" relay.
  - Survey of sibling apps shows zero existing MultipeerConnectivity / Network.framework / Bonjour usage — a real, purpose-built infrastructure gap.
- Why not primary: It is a Phase-2 concern whose value is meaningless until the single-device transcribe→summarize loop exists and is proven, and it carries the highest networking/sandbox/battery risk for no MVP payoff.

### Alt-2: Streaming Real-Time Engine
- Gist: Center the design on the low-latency live path — streaming VAD-gated, chunked Whisper inference with partial-result emission. A streaming mel-spectrogram frontend (incremental STFT emitting frames per hop), a CoreML Silero-style VAD that strips silence (~40–60% compute reduction), chunked encoder inference with a KV-cache held across a speech segment and reset on silence boundaries, and Metal flash-attention kernels to keep per-chunk latency sub-100ms.
- Objective Evidence:
  - `Agent4Kernel/llama.cpp/tools/mtmd/mtmd-audio.h` (lines 83–113) + `mtmd-audio.cpp` (lines 630–730) — a complete stateful frame-by-frame ISTFT processor (`process_frame()`/`flush()`) invertible for streaming STFT encoding.
  - `Agent4Kernel/llama.cpp/tools/mtmd/mtmd-audio.cpp` (lines 553–578) — existing 3000-frame mel chunking loop supporting incremental, non-blocking processing.
  - `Agent4Kernel/KerSor/attempts/flash-attn-decode-kv4096/attempt/kernel.metal` (and `kv512` variant) — production KV-cache-aware Metal attention kernels for streaming tokens.
  - `mtmd-audio.h` `mtmd_audio_preprocessor` virtual interface (lines 52–78) — streaming variants subclass without touching batch logic.
- Why not primary: Streaming forward-STFT and a VAD wrapper are net-new kernel work (~1.5–2k LOC) with no direct repo precedent, an orthogonal time-domain concern the batch MVP does not need to prove value.

### Alt-3: Verifiable-Privacy Spine
- Gist: Treat the absolute no-network guarantee as a verifiable architectural property, not an implicit byproduct of on-device inference — minimal App Sandbox/hardened-runtime entitlements that deny egress, a runtime auditing subsystem that intercepts `URLSession` attempts and writes a signed append-only audit trail, CryptoKit file-level encryption with POSIX hardening and manifest integrity checks for audio/transcripts/summaries, and an explicit user-approved encrypted sync boundary (AirDrop / MultipeerConnectivity).
- Objective Evidence:
  - `Agent4Kernel/Codex-Quota-Viewer/Sources/CodexQuotaViewer/SafeSwitchBackup.swift` — CryptoKit SHA256 integrity (line 196), atomic writes, permission management, manifest versioning.
  - `Agent4Kernel/Codex-Quota-Viewer/Sources/CodexQuotaViewer/VaultAccountRecordWriter.swift` — POSIX 0o600/0o700 permission hardening.
  - `Agent4Kernel/Agent-Signal-Bar/Tests/AgentSignalLightCoreTests/AgentSignalLightCoreTests.swift` (line 98) — already tests a `"network":"restricted"` permission profile.
  - `Agent4Kernel/oh-my-pi/scripts/macos-entitlements.plist` + `oh-my-humanize/scripts/macos-entitlements.plist` — hardened-runtime entitlements precedent.
- Why not primary: It hardens and proves a loop that does not yet exist; the privacy spine is highest-value as a cross-cutting layer applied once there is a functioning data flow to constrain, and its crypto/audit overhead competes with the strict RTF budget.

### Alt-4: Domain Accuracy Layer
- Gist: Make the transcription-quality differentiators the primary value — hot-word bias injection (a local vocabulary of technical terms / names injected as decode-time logit bias for mixed Chinese-English accuracy) plus local speaker diarization (x-vector speaker embeddings + unsupervised clustering into Speaker A/B with manual name labelling), treating the base STT engine as a commodity.
- Objective Evidence:
  - `Agent4Kernel/vllm/vllm/config/speech_to_text.py` (lines 38–42) — existing `hotwords` field in `SpeechToTextParams`.
  - `Agent4Kernel/KernelOwl/.venv/.../wav2vec2_with_lm/processing_wav2vec2_with_lm.py` (lines 268–276, 315–318, 404–406) — reference hotword-weighted beam-search decoding (`hotword_weight`).
  - `Agent4Kernel/KernelOwl/.venv/.../unispeech_sat/modeling_unispeech_sat.py` — `UniSpeechSatForXVector` speaker-embedding pattern for voiceprints.
  - `Agent4Kernel/vllm/vllm/multimodal/audio.py` (lines 294–349) — `split_audio()` low-energy chunk-boundary detection, infrastructure for diarization segmentation.
  - `Agent4Kernel/KernelOwl/owl/experience_builder/clustering.py` — reusable clustering grouping pattern for speaker clusters.
- Why not primary: These features sit downstream of a working transcription loop and add stacked model memory (x-vector + Whisper-Large + LLM) plus tuning-sensitive accuracy validation — refinements best layered onto, not substituted for, the core slice.

### Alt-5: Workflow Intelligence Layer
- Gist: Treat the post-transcription LLM application surface as the primary product — a prompt-template registry (meeting-minutes, action-items, brainstorm, tech-review, filler-removal), schema-validated structured output captured via terminating tools into SwiftData, and a lightweight, token-budgeted workflow runtime driving the transcript → analysis → structured output → dashboard pipeline.
- Objective Evidence:
  - `Agent4Kernel/KernelOwl/owl/prompts/templates/` (~40 j2 templates) + `owl/prompts/rendering.py` — Jinja2 template rendering with context stacking.
  - `Agent4Kernel/pi-dynamic-workflows/src/structured-output.ts` — TypeBox schema validation with tool-termination semantics.
  - `Agent4Kernel/KernelOwl/owl/prompts/system_prompt/template_plan.py` — `PromptTemplatePlan` overlay/composition pattern.
  - `Agent4Kernel/pi-dynamic-workflows/src/workflow.ts` — `runWorkflow()` phase tracking, token budgets, early termination.
  - `SoundMind/spec.md` §3.3 — preset workflow prompt templates named as a requirement.
- Why not primary: The template/structured-output precedent is all Python/TypeScript with no Swift implementation, so it requires the most novel Apple-side adaptation, and it is only meaningful once a transcript exists to feed it.

## Synthesis Notes

The Mac-First slice is deliberately the thinnest spine, and each alternative is a layer that grafts cleanly onto it once the core loop runs. If retrospective output quality matters most, fold Alt-5's prompt-template registry and schema-validated structured output into the LLM-analysis step instead of a single hardcoded prompt. If the spec's privacy promise must be demonstrable from day one, adopt Alt-3's entitlement denial + CryptoKit-encrypted local store and append-only audit trail as the persistence layer of the slice rather than a bolt-on. If transcription accuracy on technical/mixed-language audio is the make-or-break, pull Alt-4's hot-word logit-bias injection into the Whisper decode step early (diarization can wait). Alt-2's streaming engine and Alt-1's cross-device relay are the natural Phase-2/3 evolutions: keep the Mac side accepting stateless file ingestion and emitting partial-result callbacks so that the relay and streaming paths can attach later without reworking the core.

--- Original Design Draft End ---
