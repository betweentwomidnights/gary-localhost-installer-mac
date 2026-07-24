import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct SA3LoraTrainingSheet: View {
    @ObservedObject var trainer: SA3LoraTrainingManager
    @ObservedObject var autolabeler: SA3AutolabelManager

    let serviceIsRunning: Bool
    let environmentReady: Bool
    let tokenConfigured: Bool
    let careyServiceIsRunning: Bool
    let careyEnvironmentReady: Bool
    let careyTrainingIsActive: Bool
    let onSuggestMetadata: (String) async throws -> SA3MetadataSuggestion
    let onStartAutolabel: (String, SA3PromptStyle) -> Void
    let onStart: (SA3LoraTrainingRequest) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var datasetPath = ""
    @State private var triggerText = ""
    @State private var steps = "2000"
    @State private var rank = "16"
    @State private var adapterType = "dora"
    @State private var layerScope = "all-projections"
    @State private var cropSeconds = "47"
    @State private var learningRate = "1e-4"
    @State private var saveEvery = "500"
    @State private var loudnessFixEnabled = false
    @State private var targetLatentRMS = "0.90"
    @State private var formError: String?
    @State private var isPromptEditorPresented = false

    private let adapterOptions = [
        ("lora", "lora"),
        ("dora", "dora"),
        ("bora", "bora"),
        ("lora-xs", "lora xs"),
        ("dora-rows-xs", "dora rows xs"),
        ("dora-cols-xs", "dora columns xs"),
        ("bora-xs", "bora xs"),
    ]
    // The official-specialized DiT is still selectable through the trainer CLI
    // (--dit-engine) and remains covered by tests. It is not surfaced in the UI:
    // the 2,000-step Billie A/B found no quality difference against the generic
    // engine while running ~29% slower and ~3.75 GB heavier.
    private let ditEngine = "gary-generic"
    // Full-track training likewise stays available through the CLI
    // (--full-tracks). It is not surfaced here because the 285 s window costs
    // ~83 s/step and peaks at ~40 GiB on this class of machine.
    private let fullTracks = false
    /// Upper bound for the UI crop field. See the note in `startTraining()`.
    private static let maxCropSeconds: Double = 95
    private let layerScopeOptions = [
        ("all-projections", "all projections (228 layers)"),
        ("attention-feedforward", "attention + feed-forward (168 layers)"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("train sa3 lora with mlx")
                        .font(.title2.weight(.semibold))
                    Text("train an audio folder, then blend the adapter in gary.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if trainer.state?.isActive == true {
                    ProgressView()
                        .controlSize(.small)
                }
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("close")
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)
            }
            .padding(20)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    warnings
                    trainingForm
                    actionRow
                    jobStatus
                    trainingLog
                }
                .padding(20)
            }

            Divider()
            HStack {
                Text("powered by the vendored underfit-compatible mlx training path.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("close") { dismiss() }
            }
            .padding(14)
        }
        .frame(minWidth: 760, idealWidth: 860, minHeight: 680, idealHeight: 780)
        .onAppear { trainer.refresh() }
        .sheet(isPresented: $isPromptEditorPresented) {
            SA3DatasetPromptEditorSheet(
                datasetPath: datasetPath,
                autolabeler: autolabeler,
                sa3EnvironmentReady: environmentReady,
                careyServiceIsRunning: careyServiceIsRunning,
                careyEnvironmentReady: careyEnvironmentReady,
                careyTrainingIsActive: careyTrainingIsActive,
                onSuggestMetadata: onSuggestMetadata,
                onStartAutolabel: onStartAutolabel
            )
        }
    }

    @ViewBuilder
    private var warnings: some View {
        if serviceIsRunning {
            warning(
                "stop sa3 before training. inference and training both need the model in unified memory.",
                color: .orange
            )
        }
        if !environmentReady {
            warning("build the sa3 environment before training.", color: .orange)
        }
        if !tokenConfigured {
            warning("save a hugging face token in the sa3 panel before training.", color: .orange)
        }
        if let error = formError ?? trainer.launchError {
            warning(error, color: .red)
        }
    }

    private var trainingForm: some View {
        GroupBox("training setup") {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                GridRow(alignment: .top) {
                    Text("model contract")
                    Text(
                        "Trains against Stable Audio 3 medium-base. The exported "
                            + "adapter is applied to Stable Audio 3 medium during inference."
                    )
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                }

                GridRow {
                    Text("lora name")
                    TextField("bell-arpeggio", text: $name)
                        .textFieldStyle(.roundedBorder)
                }

                GridRow {
                    Text("audio folder")
                    HStack {
                        TextField("choose a folder of audio files", text: $datasetPath)
                            .textFieldStyle(.roundedBorder)
                        Button("choose...") { chooseDatasetFolder() }
                        Button("edit prompts") {
                            isPromptEditorPresented = true
                        }
                        .disabled(
                            datasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        )
                    }
                }

                GridRow {
                    Text("shared trigger word")
                    VStack(alignment: .leading, spacing: 4) {
                        TextField("optional — e.g. my-trigger", text: $triggerText)
                            .textFieldStyle(.roundedBorder)
                        Text(
                            "Prepended to every caption during training (“trigger, caption”). Leave blank to train on captions alone. It is not written to sidecars or dice prompts."
                        )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                GridRow {
                    Text("adapter")
                    Picker("adapter", selection: $adapterType) {
                        ForEach(adapterOptions, id: \.0) { option in
                            Text(option.1).tag(option.0)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 240, alignment: .leading)
                    .garyPickerAccent()
                }

                GridRow(alignment: .top) {
                    Text("layer scope")
                    VStack(alignment: .leading, spacing: 4) {
                        Picker("layer scope", selection: $layerScope) {
                            ForEach(layerScopeOptions, id: \.0) { option in
                                Text(option.1).tag(option.0)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 280, alignment: .leading)
                        .garyPickerAccent()
                        Text(
                            "Which DiT projections get adapters. All projections is "
                                + "Gary's default. Attention + feed-forward matches the "
                                + "official trainer's default, dropping the embedding "
                                + "and local-conditioning projections for a smaller, "
                                + "faster adapter. The seconds conditioner is adapted "
                                + "either way."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                GridRow {
                    Text("core settings")
                    HStack(spacing: 14) {
                        numericField("steps", text: $steps, width: 82)
                        numericField("rank", text: $rank, width: 72)
                    }
                }

                GridRow(alignment: .top) {
                    Text("training window")
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(alignment: .top, spacing: 10) {
                            numericField(
                                "random crop seconds",
                                text: $cropSeconds,
                                width: 96
                            )
                            Text(
                                "Samples a new offset on every step. Shorter windows train "
                                    + "faster and use less memory but do not always begin at "
                                    + "the song's start."
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.top, 18)
                        }
                    }
                }

                GridRow {
                    Text("optimization")
                    HStack(spacing: 14) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("learning rate")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("1e-4", text: $learningRate)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 104)
                            Text("decimal: \(learningRateDecimal)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(learningRateDecimal == "invalid" ? .red : .secondary)
                        }
                        numericField("save every", text: $saveEvery, width: 82)
                    }
                }

                GridRow(alignment: .top) {
                    Text("loudness")
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle(isOn: $loudnessFixEnabled) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("experimental loudness fix")
                                Text(
                                    "normalizes each track's encoded latent RMS; "
                                        + "pre-encoding will take longer."
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                        }
                        .garyCheckboxStyle()

                        if loudnessFixEnabled {
                            HStack(alignment: .top, spacing: 10) {
                                numericField(
                                    "target latent RMS",
                                    text: $targetLatentRMS,
                                    width: 92
                                )
                                Text(
                                    "0.90 matches base-model loudness. "
                                        + "Lower is quieter; higher is hotter."
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.top, 18)
                            }
                        }
                    }
                }

            }
            .padding(.top, 6)
        }
    }

    private var actionRow: some View {
        HStack {
            Button {
                startTraining()
            } label: {
                Label(
                    trainer.isLaunching ? "launching..." : "start training",
                    systemImage: "waveform.badge.plus"
                )
            }
            .garyPrimaryButtonStyle()
            .disabled(!canStart)

            Button("cancel training", role: .destructive) {
                trainer.cancel()
            }
            .disabled(trainer.state?.isActive != true)

            Button("refresh") {
                trainer.refresh()
            }

            Spacer()

            if let runDir = trainer.state?.runDir,
               trainer.state?.finalCheckpointPath == nil {
                Button {
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: runDir)
                } label: {
                    Label("reveal run", systemImage: "folder")
                }
            }
        }
    }

    @ViewBuilder
    private var jobStatus: some View {
        if let state = trainer.state {
            GroupBox("current job") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(state.name ?? "sa3 lora")
                            .font(.headline)
                        Text(state.status ?? "unknown")
                            .font(.caption.weight(.bold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(statusColor(state.status).opacity(0.16))
                            .foregroundStyle(statusColor(state.status))
                            .clipShape(Capsule())
                        Spacer()
                        if let adapter = state.adapterType {
                            Text(adapter)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }

                    if let maxSteps = state.maxSteps, maxSteps > 0 {
                        let current = min(state.currentStep ?? 0, maxSteps)
                        ProgressView(value: Double(current), total: Double(maxSteps))
                        Text("\(current) / \(maxSteps) steps")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    if let message = state.message {
                        Text(message)
                    }
                    if let error = state.error, !error.isEmpty {
                        Text(error)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                    }

                    if let checkpoint = state.finalCheckpointPath,
                       FileManager.default.fileExists(atPath: checkpoint) {
                        HStack {
                            Text(checkpoint)
                                .font(.caption.monospaced())
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .textSelection(.enabled)
                            Spacer()
                            Button {
                                let runDir = state.runDir
                                    ?? URL(fileURLWithPath: checkpoint)
                                        .deletingLastPathComponent().path
                                NSWorkspace.shared.selectFile(
                                    nil,
                                    inFileViewerRootedAtPath: runDir
                                )
                            } label: {
                                Label("reveal run", systemImage: "folder")
                            }
                        }
                    }
                }
                .padding(.top, 6)
            }
        }
    }

    @ViewBuilder
    private var trainingLog: some View {
        if trainer.state != nil || !trainer.logText.isEmpty {
            GroupBox("training log") {
                ScrollView {
                    Text(trainer.logText.isEmpty ? "(waiting for output)" : trainer.logText)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(minHeight: 180, maxHeight: 280)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 5))
                .padding(.top, 6)
            }
        }
    }

    private var canStart: Bool {
        !serviceIsRunning
            && environmentReady
            && tokenConfigured
            && !trainer.isLaunching
            && trainer.state?.isActive != true
            && !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !datasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (!loudnessFixEnabled
                || targetLatentRMSValue.map { (0.5...1.3).contains($0) } == true)
    }

    private var learningRateDecimal: String {
        guard let value = Double(learningRate.trimmingCharacters(in: .whitespacesAndNewlines)),
              value.isFinite,
              value > 0 else {
            return "invalid"
        }
        return String(format: "%.12f", value)
            .replacingOccurrences(of: #"0+$"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression)
    }

    private var targetLatentRMSValue: Double? {
        guard let value = Double(
            targetLatentRMS.trimmingCharacters(in: .whitespacesAndNewlines)
        ), value.isFinite else {
            return nil
        }
        return value
    }

    private func numericField(
        _ label: String,
        text: Binding<String>,
        width: CGFloat
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    private func warning(_ text: String, color: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(color)
            Text(text)
                .font(.callout)
            Spacer()
        }
        .padding(10)
        .background(color.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 7))
    }

    private func chooseDatasetFolder() {
        let panel = NSOpenPanel()
        panel.title = "choose sa3 training audio folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false

        guard panel.runModal() == .OK, let url = panel.url else { return }
        datasetPath = url.path
        if name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            name = suggestedName(for: url.lastPathComponent)
        }
        formError = nil
        trainer.clearError()
    }

    private func startTraining() {
        let parsedTargetLatentRMS = targetLatentRMSValue ?? 0.90
        let requestedCropSeconds: Double? = Double(cropSeconds)
        guard let parsedSteps = Int(steps),
              let parsedRank = Int(rank),
              let parsedCropSeconds = requestedCropSeconds,
              let parsedLearningRate = Double(learningRate),
              parsedLearningRate.isFinite,
              parsedLearningRate > 0,
              let parsedSaveEvery = Int(saveEvery),
              !loudnessFixEnabled
                || (0.5...1.3).contains(parsedTargetLatentRMS) else {
            formError = "check the numeric training settings."
            return
        }
        // Measured on a base M4 Air: 95 s of audio (1,024 latents) runs about
        // 4.3 s/step at ~16 GiB peak, while 190 s jumps to ~29 s/step at ~29 GiB
        // and starts swapping. Keep the UI on the measured-good side of that
        // cliff; --crop-seconds and --full-tracks remain uncapped in the CLI.
        guard parsedCropSeconds <= Self.maxCropSeconds else {
            formError =
                "crop seconds is capped at \(Int(Self.maxCropSeconds)) on this "
                + "platform; longer windows are far slower and can exhaust memory."
            return
        }
        formError = nil
        trainer.clearError()
        onStart(
            SA3LoraTrainingRequest(
                name: name,
                datasetPath: datasetPath,
                triggerText: triggerText,
                steps: parsedSteps,
                rank: parsedRank,
                adapterType: adapterType,
                ditEngine: ditEngine,
                layerScope: layerScope,
                fullTracks: fullTracks,
                cropSeconds: parsedCropSeconds,
                learningRate: parsedLearningRate,
                saveEvery: parsedSaveEvery,
                loudnessFixEnabled: loudnessFixEnabled,
                targetLatentRMS: parsedTargetLatentRMS
            )
        )
    }

    private func suggestedName(for value: String) -> String {
        value.lowercased()
            .replacingOccurrences(
                of: #"[^a-z0-9_-]+"#,
                with: "-",
                options: .regularExpression
            )
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(64)
            .description
    }

    private func statusColor(_ status: String?) -> Color {
        switch status {
        case "completed":
            return .green
        case "failed":
            return .red
        case "cancelled":
            return .secondary
        case "cancelling":
            return .orange
        default:
            return .blue
        }
    }
}
