import SwiftUI

struct YueyRuntimePanel: View {
    @ObservedObject var controller: YueyController
    @ObservedObject var manager: ServiceManager
    let runtime: ServiceRuntime
    let showModels: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("yuey · native GGML").font(.headline)
                Spacer()
                let installed = NativeRuntimeReceipt.read(for: runtime.service)
                Text(installed.map { "\($0.version) installed" } ?? "runtime not installed")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if runtime.service.nativeRuntime?.version == "v0.2.1" {
                Text("This older runtime can fail on very short MIDI scores. Update the Yuey runtime before release testing.")
                    .font(.caption).foregroundStyle(.orange)
            }
            if runtime.isBootstrapping {
                Button("cancel runtime install") { manager.cancelNativeRuntimeInstall(serviceID: runtime.id) }
            }
            HStack {
                Picker("backend", selection: $controller.settings.device) {
                    Text("automatic").tag("auto")
                    if YueySettings.automaticDevice == "mtl" { Text("Metal").tag("mtl") }
                    Text("CPU").tag("cpu")
                }.frame(maxWidth: 240)
                Picker("generation tier", selection: $controller.settings.encoding) {
                    Text("automatic").tag("auto")
                    ForEach(controller.installedTiers, id: \.self) { tier in Text(tier).tag(tier) }
                    if !controller.installedTiers.contains(controller.settings.encoding), controller.settings.encoding != "auto" {
                        Text("\(controller.settings.encoding) (not installed)").tag(controller.settings.encoding)
                    }
                }.frame(maxWidth: 280)
                Button("models…", action: showModels)
            }
            HStack {
                Picker("instrumental", selection: $controller.settings.instrumentalMethod) {
                    Text("transfer melody to instruments").tag("transfer")
                    Text("remove melody (backing only)").tag("rest")
                }.frame(maxWidth: 400)
                Toggle("instrumental adapter", isOn: $controller.settings.useInstrumentalAdapter)
            }
            HStack {
                Text("natural duration cap (seconds)").font(.caption)
                TextField("96", value: $controller.settings.naturalMaxSeconds, format: .number)
                    .frame(width: 60)
                Button(runtime.isRunning ? "save and restart Yuey" : "save settings") {
                    controller.saveSettings(manager: manager)
                }.disabled(runtime.isBootstrapping || controller.isBusy)
                Spacer()
            }
            Text("\(YueySettings.automaticDevice == "mtl" ? "Metal on Apple Silicon" : "CPU on Intel") · localhost:8007 · models always unload after jobs")
                .font(.caption).foregroundStyle(.secondary)
            if let encoding = YueyModelCatalog.chooseEncoding(installed: controller.installedTiers,
                chosen: controller.settings.encoding,
                recommended: NativeRuntimeReceipt.read(for: runtime.service)?.recommendedEncoding ?? "Q4_K_M") {
                Text("next launch: \(encoding) · natural duration cap is 30–600 seconds; requests may lower it")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !controller.message.isEmpty {
                Text(controller.message).font(.caption).textSelection(.enabled)
            }
        }
        .onAppear { controller.refresh() }
        .onChange(of: runtime.bootstrapState) { _, _ in controller.refresh() }
    }
}

struct YueyModelsView: View {
    @ObservedObject var controller: YueyController
    @State private var pendingRemoval: [YueyModelFile] = []
    @State private var showRemovalConfirmation = false
    private var blocked: Bool { controller.isBusy || !controller.canMutate() }
    private func size(_ files: [YueyModelFile]) -> String {
        ByteCountFormatter.string(fromByteCount: files.reduce(0) { $0 + $1.bytes }, countStyle: .file)
    }
    private func confirmRemoval(_ files: [YueyModelFile]) {
        pendingRemoval = files
        showRemovalConfirmation = true
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Yuey models").font(.headline)
                Spacer()
                Link("model license", destination: URL(string: "https://huggingface.co/\(YueyModelCatalog.repository)#license")!)
                    .font(.caption)
            }
            Text("Download the shared files and one generation tier. No Python environment or Hugging Face token is needed. Stop Yuey before downloading or removing files.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("download recommended setup (\(controller.installedTiers.first ?? controller.recommendedTier))") {
                    controller.download(tier: controller.installedTiers.first ?? controller.recommendedTier, includeShared: true)
                }.disabled(blocked)
                Button("refresh") { controller.refresh() }.disabled(controller.isBusy)
            }
            Text("generation tiers").font(.headline)
            ForEach(Array(zip(YueyModelCatalog.tiers, YueyModelCatalog.generation)), id: \.0) { tier, file in
                HStack {
                    VStack(alignment: .leading) {
                        Text("\(tier) · \(size([file]))")
                        Text(tier == "Q4_K_M" ? "compact · 8 GB GPU recommended" : tier == "Q8_0" ? "high quality · 12 GB GPU recommended" : "full quality · 16 GB GPU recommended")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(controller.installedTiers.contains(tier) ? "downloaded" : "not downloaded")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(controller.installedTiers.contains(tier) ? "verify" : "download") { controller.download(tier: tier) }
                        .disabled(blocked)
                    if controller.filesOnDisk.contains(file.name) {
                        Button("remove…") { confirmRemoval([file]) }.disabled(blocked)
                    }
                }.padding(8).background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
            }
            Divider()
            HStack {
                Text("shared components · \(size(YueyModelCatalog.shared))").font(.headline)
                Spacer()
                Button(controller.sharedReady ? "verify shared files" : "download shared files") {
                    controller.download(includeShared: true)
                }.disabled(blocked)
                if YueyModelCatalog.shared.contains(where: { controller.filesOnDisk.contains($0.name) }) {
                    Button("remove…") { confirmRemoval(YueyModelCatalog.shared) }.disabled(blocked)
                }
            }
            Text("VAE, tokenizer, transcription model, semantic tokenizer and two built-in adapters. Downloaded once, shared by all tiers.")
                .font(.caption).foregroundStyle(.secondary)
            Text(controller.sharedReady ? "all shared files downloaded" : "shared files incomplete")
                .font(.caption)
            if controller.isBusy {
                HStack { ProgressView(value: controller.progress); Button("cancel") { controller.cancel() } }
            }
            if !controller.message.isEmpty { Text(controller.message).font(.caption).textSelection(.enabled) }
            if let directory = controller.directory {
                Text(directory.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Text("Downloads are revision-pinned and SHA-256 verified before installation.").font(.caption).foregroundStyle(.secondary)
        }
        .onAppear { controller.refresh() }
        .alert("Move Yuey model files to Trash?", isPresented: $showRemovalConfirmation) {
            Button("cancel", role: .cancel) { pendingRemoval = [] }
            Button("move to Trash", role: .destructive) { controller.remove(pendingRemoval); pendingRemoval = [] }
        } message: {
            Text("\(pendingRemoval.count) file(s), up to \(size(pendingRemoval)). Yuey will need these downloaded again to use the affected features. Files remain recoverable in Trash.")
        }
    }
}
