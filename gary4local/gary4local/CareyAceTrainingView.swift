import AppKit
import SwiftUI

struct CareyAceTrainingSheet: View {
    @ObservedObject var trainer: CareyAceTrainingManager

    let serviceIsRunning: Bool
    let environmentReady: Bool
    let onStart: (CareyAceTrainingRequest) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var datasetPath = ""
    @State private var trigger = ""
    @State private var model = "base"
    @State private var timestepMu = "-0.4"
    @State private var captionLmModel = "acestep-5Hz-lm-1.7B"
    @State private var overwriteCaptions = false
    @State private var bpmKeySanityCheck = true
    @State private var adapterType = "dora"
    @State private var moduleProfile = "balanced"
    @State private var rank = "64"
    @State private var epochs = "150"
    @State private var maxSteps = "0"
    @State private var saveEvery = "25"
    @State private var saveBestAfter = "25"
    @State private var batchSize = "1"
    @State private var gradientAccumulation = "1"
    @State private var learningRate = "3e-4"
    @State private var lossWeighting = "min_snr"
    @State private var snrGamma = "5"
    @State private var maxDuration = "240"
    @State private var genreRatio = "20"
    @State private var preprocessDevice = "auto"
    @State private var preprocessPrecision = "fp32"
    @State private var dtype = "bf16"
    @State private var formError: String?
    @State private var sidecarEditorPresented = false
    @State private var advancedSettingsExpanded = false
    @State private var isTrainingLogPinnedToBottom = true
    @State private var trainingLogScrollRequestID = 0
    @State private var pendingTrainingLogReveal = false

    private let fixedWeightDecay = 0.01
    private let fixedCfgRatio = 0.15
    private let fixedAnalysisDuration = 0.0
    private let captionLmBackend = "mlx"
    private let trainingLogSectionID = "carey-ace-training-log"

    private let captionLmModels = [
        "acestep-5Hz-lm-0.6B",
        "acestep-5Hz-lm-1.7B",
        "acestep-5Hz-lm-4B",
    ]

    var body: some View {
        ScrollViewReader { proxy in
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("train ace-step lora with mlx")
                            .font(.title2.weight(.semibold))
                        Text("prepare ACE sidecars, train LoRA or DoRA, then register it in Carey.")
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
                        setupForm
                        actionRow
                        jobStatus
                        trainingLog
                    }
                    .padding(20)
                }

                Divider()
                HStack {
                    Text("uses Carey's two-pass ACE preprocessor, then the native MLX adapter trainer.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("close") { dismiss() }
                }
                .padding(14)
            }
            .frame(minWidth: 780, idealWidth: 900, minHeight: 720, idealHeight: 840)
            .onAppear { trainer.refresh() }
            .onChange(of: trainer.isLaunching, initial: false) { _, _ in
                revealTrainingLogIfNeeded(with: proxy)
            }
            .onChange(of: trainer.state?.status, initial: false) { _, _ in
                revealTrainingLogIfNeeded(with: proxy)
            }
            .onChange(of: trainer.logText, initial: false) { _, _ in
                revealTrainingLogIfNeeded(with: proxy)
            }
            .sheet(isPresented: $sidecarEditorPresented) {
                CareyAceSidecarEditorSheet(datasetPath: datasetPath)
            }
        }
    }

    @ViewBuilder
    private var warnings: some View {
        if serviceIsRunning {
            warning(
                "stop carey before captioning or training. inference, the captioner, and training all need ACE models in unified memory.",
                color: .orange
            )
        }
        if !environmentReady {
            warning("build the carey environment before training.", color: .orange)
        }
        if model == "xl-base" {
            warning(
                "XL-base training requires substantially more memory; the runtime preflight will verify measured headroom before the first batch.",
                color: .orange
            )
        }
        if let advisory = vramAdvisory {
            warning(advisory, color: .orange)
        }
        if let error = formError ?? trainer.launchError {
            warning(error, color: .red)
        }
    }

    private var setupForm: some View {
        VStack(alignment: .leading, spacing: 14) {
            GroupBox("dataset") {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                    GridRow {
                        Text("lora name")
                        TextField("billie-style", text: $name)
                            .textFieldStyle(.roundedBorder)
                    }
                    GridRow {
                        Text("audio folder")
                        HStack {
                            TextField("choose a folder with audio; caption / prepare writes .txt sidecars", text: $datasetPath)
                                .textFieldStyle(.roundedBorder)
                            Button("choose...") { chooseDatasetFolder() }
                            Button("edit prompts / sidecars") {
                                sidecarEditorPresented = true
                            }
                            .disabled(datasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                    GridRow {
                        Text("custom trigger")
                        TextField("optional shared trigger, such as billie", text: $trigger)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                .padding(.top, 6)
            }

            GroupBox("captioner") {
                VStack(alignment: .leading, spacing: 12) {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                        GridRow {
                            Text("LM model")
                            Picker("LM model", selection: $captionLmModel) {
                                Text("0.6B · not recommended").tag("acestep-5Hz-lm-0.6B")
                                Text("1.7B · good quality").tag("acestep-5Hz-lm-1.7B")
                                Text("4B · best quality, large memory").tag("acestep-5Hz-lm-4B")
                            }
                            .labelsHidden()
                            .frame(width: 250)
                            .garyPickerAccent()
                        }
                        GridRow {
                            Text("backend")
                            Text("MLX")
                                .font(.caption.monospaced().weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                        GridRow {
                            Text("sidecars")
                            VStack(alignment: .leading, spacing: 4) {
                                Toggle("overwrite existing sidecars", isOn: $overwriteCaptions)
                                    .garyCheckboxStyle()
                                Text("use this only when you want to replace human edits.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        GridRow {
                            Text("metadata")
                            VStack(alignment: .leading, spacing: 4) {
                                Toggle("BPM/key sanity check", isOn: $bpmKeySanityCheck)
                                    .garyCheckboxStyle()
                                Text("local tempo and chroma estimates correct obvious LM mistakes; ambiguous values stay editable.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    HStack(alignment: .center, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("caption and prepare dataset")
                                .font(.subheadline.weight(.semibold))
                            Text("captions missing sidecars with understand_music, applies optional BPM/key checks, then writes editable dataset metadata for review.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            startTraining(prepareOnly: true)
                        } label: {
                            Label(
                                trainer.isLaunching ? "launching..." : "caption / prepare",
                                systemImage: "doc.badge.gearshape"
                            )
                        }
                        .garyPrimaryButtonStyle()
                        .disabled(!canPrepare)
                    }
                }
                .padding(.top, 6)
            }

            GroupBox("training") {
                VStack(alignment: .leading, spacing: 14) {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                        GridRow {
                            Text("base model")
                            Picker("base model", selection: $model) {
                                Text("ace-step-v15-base").tag("base")
                                Text("ace-step-v15-xl-base").tag("xl-base")
                            }
                            .labelsHidden()
                            .frame(width: 210)
                            .garyPickerAccent()
                        }
                        GridRow {
                            Text("adapter type")
                            VStack(alignment: .leading, spacing: 4) {
                                Picker("adapter type", selection: $adapterType) {
                                    Text("DoRA").tag("dora")
                                    Text("LoRA").tag("lora")
                                }
                                .labelsHidden()
                                .frame(width: 120)
                                .garyPickerAccent()
                                Text("DoRA is recommended; LoRA is the lighter, simpler option.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        GridRow {
                            Text("epochs")
                            VStack(alignment: .leading, spacing: 4) {
                                TextField("epochs", text: $epochs)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 74)
                                Text("one epoch is one pass over every track.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        GridRow {
                            Text("learning rate")
                            VStack(alignment: .leading, spacing: 4) {
                                TextField("learning rate", text: $learningRate)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 110)
                                Text("1e-4 trains lightly; 3e-4 trains hard. Current: \(learningRateDescription).")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        GridRow {
                            Text("max track seconds")
                            VStack(alignment: .leading, spacing: 4) {
                                TextField("max track seconds", text: $maxDuration)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 88)
                                Text("longer tracks require more memory.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Button {
                            withAnimation(.easeInOut(duration: 0.15)) {
                                advancedSettingsExpanded.toggle()
                            }
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: advancedSettingsExpanded ? "chevron.down" : "chevron.right")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .frame(width: 12)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("advanced / power user settings")
                                        .font(.subheadline.weight(.semibold))
                                    Text(advancedSummary)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                            .padding(.vertical, 4)
                        }
                        .buttonStyle(.plain)

                        if advancedSettingsExpanded {
                        Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                            GridRow {
                                Text("timestep μ")
                                VStack(alignment: .leading, spacing: 4) {
                                    Picker("timestep μ", selection: $timestepMu) {
                                        Text("default · μ -0.4").tag("-0.4")
                                        Text("experimental vocal · μ 0.0").tag("0.0")
                                    }
                                    .labelsHidden()
                                    .frame(width: 220)
                                    .garyPickerAccent()
                                    Text("advanced training schedule only; try 0.0 for datasets with vocals.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            GridRow {
                                Text("adapter coverage")
                                VStack(alignment: .leading, spacing: 4) {
                                    Picker("adapter coverage", selection: $moduleProfile) {
                                        Text("balanced attention + MLP").tag("balanced")
                                        Text("attention only · legacy").tag("attention")
                                    }
                                    .labelsHidden()
                                    .frame(width: 240)
                                    .garyPickerAccent()
                                    Text("recommended; distributes rank across attention and feed-forward projections.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            GridRow {
                                Text("rank")
                                VStack(alignment: .leading, spacing: 4) {
                                    TextField("rank", text: $rank)
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 74)
                                    Text(moduleProfile == "balanced" ? "reference rank; projection families scale around it." : "uniform attention rank.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            GridRow {
                                Text("save every")
                                TextField("save every", text: $saveEvery)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 74)
                            }
                            GridRow {
                                Text("batch size")
                                TextField("batch size", text: $batchSize)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 74)
                            }
                            GridRow {
                                Text("grad accum")
                                TextField("grad accum", text: $gradientAccumulation)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 74)
                            }
                            GridRow {
                                Text("loss weighting")
                                Picker("loss weighting", selection: $lossWeighting) {
                                    Text("Min-SNR (recommended)").tag("min_snr")
                                    Text("none (flat MSE)").tag("none")
                                }
                                .labelsHidden()
                                .frame(width: 190)
                                .garyPickerAccent()
                            }
                            GridRow {
                                Text("Min-SNR gamma")
                                VStack(alignment: .leading, spacing: 4) {
                                    TextField("Min-SNR gamma", text: $snrGamma)
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 74)
                                        .disabled(lossWeighting == "none")
                                    Text("5.0 is the reference setting.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            GridRow {
                                Text("genre prompt ratio")
                                VStack(alignment: .leading, spacing: 4) {
                                    TextField("genre prompt ratio", text: $genreRatio)
                                        .textFieldStyle(.roundedBorder)
                                        .frame(width: 74)
                                    Text("Each epoch, \(genreRatio)% of tracks use genre instead of caption.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding(.top, 8)
                        }
                    }

                    Text("Use caption / prepare first, review the sidecars, then train.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 4) {
                        Text("To learn more about LoRA training or use a more advanced setup, try")
                        Button("Side-Step") {
                            openSideStep()
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.blue)
                        .underline()
                        .help("Open the Side-Step repository in your browser")
                        Text("by koda-dernet")
                        Text(".")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(.top, 6)
            }
        }
    }

    private var actionRow: some View {
        HStack {
            Button {
                startTraining(prepareOnly: false)
            } label: {
                Label(
                    trainer.isLaunching ? "launching..." : "train \(adapterType == "dora" ? "DoRA" : "LoRA")",
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
                        Text(state.name ?? "ace-step lora")
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
                            Text([adapter, state.moduleProfile].compactMap { $0 }.joined(separator: " / "))
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
                    } else if let maxEpochs = state.maxEpochs, maxEpochs > 0 {
                        let current = min(state.currentEpoch ?? 0, maxEpochs)
                        ProgressView(value: Double(current), total: Double(maxEpochs))
                        Text("\(current) / \(maxEpochs) epochs")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    if let loss = state.currentLoss {
                        Text("loss \(String(format: "%.6f", loss))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    if let totalFiles = state.totalFiles, totalFiles > 0 {
                        let current = min(state.currentFile ?? 0, totalFiles)
                        Text("\(current) / \(totalFiles) files")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    if let captionedCount = state.captionedCount {
                        Text("\(captionedCount) captioned")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if let captionLmModel = state.captionLmModel {
                        let backend = state.captionLmBackend?.uppercased()
                        let captionerText = backend.map {
                            "captioner \(captionLmModel) / \($0)"
                        } ?? "captioner \(captionLmModel)"
                        Text(captionerText)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    if let sampleCount = state.sampleCount {
                        Text("\(sampleCount) samples")
                            .font(.caption)
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
                                NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: runDir)
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
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("training log")
                            .font(.headline)
                        Spacer()
                        if !isTrainingLogPinnedToBottom {
                            Button {
                                isTrainingLogPinnedToBottom = true
                                trainingLogScrollRequestID += 1
                            } label: {
                                Label("scroll to bottom", systemImage: "arrow.down.circle")
                                    .font(.caption)
                            }
                            .buttonStyle(.borderless)
                        }
                    }

                    TrainingLogTextView(
                        text: trainer.logText,
                        placeholder: "(waiting for output)",
                        isPinnedToBottom: isTrainingLogPinnedToBottom,
                        scrollRequestID: trainingLogScrollRequestID,
                        onPinnedToBottomChanged: { isTrainingLogPinnedToBottom = $0 }
                    )
                    .frame(minHeight: 180, maxHeight: 280)
                    .background(Color(NSColor.textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                }
                .padding(.top, 6)
            }
            .id(trainingLogSectionID)
        }
    }

    private var canPrepare: Bool {
        environmentReady
            && !serviceIsRunning
            && !trainer.isLaunching
            && trainer.state?.isActive != true
            && !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !datasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && parsedRequest(prepareOnly: true) != nil
    }

    private var canStart: Bool {
        canPrepare && parsedRequest(prepareOnly: false) != nil
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

    private func decimalField(
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
        panel.title = "choose ace-step training audio folder"
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

    private func startTraining(prepareOnly: Bool) {
        guard let request = parsedRequest(prepareOnly: prepareOnly) else {
            formError = "check the numeric training settings."
            return
        }
        formError = nil
        trainer.clearError()
        if prepareOnly {
            pendingTrainingLogReveal = true
            isTrainingLogPinnedToBottom = true
            trainingLogScrollRequestID += 1
        }
        onStart(request)
    }

    private func revealTrainingLogIfNeeded(with proxy: ScrollViewProxy) {
        guard pendingTrainingLogReveal else { return }
        guard trainer.state != nil || !trainer.logText.isEmpty else { return }
        pendingTrainingLogReveal = false
        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.2)) {
                proxy.scrollTo(trainingLogSectionID, anchor: .top)
            }
        }
    }

    private var learningRateDescription: String {
        guard let value = Double(learningRate.trimmingCharacters(in: .whitespacesAndNewlines)),
              value.isFinite,
              value > 0 else {
            return "invalid"
        }
        return String(format: "%.12f", value)
            .replacingOccurrences(of: #"0+$"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression)
    }

    private var advancedSummary: String {
        let lossLabel = lossWeighting == "min_snr" ? "Min-SNR" : "flat MSE"
        let profileLabel = moduleProfile == "balanced" ? "balanced" : "attention only"
        return "\(profileLabel) · rank \(rank) · \(lossLabel)"
    }

    private var vramAdvisory: String? {
        let parsedRank = Int(rank) ?? 0
        let parsedMaxDuration = Double(maxDuration) ?? 0
        if model == "xl-base" {
            return nil
        }
        if moduleProfile == "balanced" && parsedRank >= 128 && parsedMaxDuration > 240 {
            return "High-memory combination: balanced rank 128+ with tracks over 240 seconds. The runtime preflight may stop this run if measured headroom is unsafe."
        }
        if moduleProfile == "balanced" && parsedRank >= 128 {
            return "Balanced rank 128+ is an advanced memory configuration. The runtime preflight will measure real post-offload headroom."
        }
        if parsedMaxDuration > 240 {
            return "Tracks over 240 seconds increase activation memory. The runtime preflight will verify headroom before training."
        }
        return nil
    }

    private func openSideStep() {
        guard let url = URL(string: "https://github.com/koda-dernet/Side-Step") else { return }
        NSWorkspace.shared.open(url)
    }

    private func parsedRequest(prepareOnly: Bool) -> CareyAceTrainingRequest? {
        guard let parsedRank = Int(rank),
              let parsedEpochs = Int(epochs),
              let parsedMaxSteps = Int(maxSteps),
              let parsedSaveEvery = Int(saveEvery),
              let parsedSaveBestAfter = Int(saveBestAfter),
              let parsedBatchSize = Int(batchSize),
              let parsedGradientAccumulation = Int(gradientAccumulation),
              let parsedLearningRate = Double(learningRate),
              let parsedTimestepMu = Double(timestepMu),
              let parsedSnrGamma = Double(snrGamma),
              let parsedMaxDuration = Double(maxDuration),
              let parsedGenreRatio = Int(genreRatio),
              parsedRank > 0,
              parsedEpochs > 0,
              parsedMaxSteps >= 0,
              parsedSaveEvery > 0,
              parsedSaveBestAfter > 0,
              parsedBatchSize > 0,
              parsedGradientAccumulation > 0,
              parsedLearningRate.isFinite,
              parsedLearningRate > 0,
              parsedTimestepMu.isFinite,
              parsedSnrGamma > 0,
              parsedMaxDuration > 0,
              (0...100).contains(parsedGenreRatio) else {
            return nil
        }
        let parsedAlpha = parsedRank * 2

        return CareyAceTrainingRequest(
            name: name,
            datasetPath: datasetPath,
            trigger: trigger,
            model: model,
            caption: prepareOnly ? "understand_music" : "skip",
            captionLmModel: captionLmModel,
            captionLmBackend: captionLmBackend,
            overwriteCaptions: overwriteCaptions,
            bpmAnalysis: bpmKeySanityCheck,
            keyAnalysis: bpmKeySanityCheck,
            analysisDuration: fixedAnalysisDuration,
            rank: parsedRank,
            alpha: parsedAlpha,
            adapterType: adapterType,
            moduleProfile: moduleProfile,
            timestepMu: parsedTimestepMu,
            epochs: parsedEpochs,
            maxSteps: parsedMaxSteps,
            saveEvery: parsedSaveEvery,
            saveBestAfter: parsedSaveBestAfter,
            batchSize: parsedBatchSize,
            gradientAccumulation: parsedGradientAccumulation,
            learningRate: parsedLearningRate,
            weightDecay: fixedWeightDecay,
            cfgRatio: fixedCfgRatio,
            lossWeighting: lossWeighting,
            snrGamma: parsedSnrGamma,
            maxDuration: parsedMaxDuration,
            genreRatio: parsedGenreRatio,
            preprocessDevice: preprocessDevice,
            preprocessPrecision: preprocessPrecision,
            dtype: dtype,
            prepareOnly: prepareOnly
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

private struct TrainingLogTextView: NSViewRepresentable {
    let text: String
    let placeholder: String
    let isPinnedToBottom: Bool
    let scrollRequestID: Int
    let onPinnedToBottomChanged: (Bool) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPinnedToBottomChanged: onPinnedToBottomChanged)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = false
        textView.usesFontPanel = false
        textView.usesFindPanel = true
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.drawsBackground = false
        textView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.textContainerInset = NSSize(width: 8, height: 8)
        textView.minSize = .zero
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = true
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )

        scrollView.documentView = textView
        context.coordinator.attach(scrollView: scrollView, textView: textView)
        context.coordinator.applyDisplayedText(
            displayedText,
            in: scrollView,
            forceScrollToBottom: isPinnedToBottom,
            scrollRequestID: scrollRequestID
        )
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.onPinnedToBottomChanged = onPinnedToBottomChanged
        context.coordinator.applyDisplayedText(
            displayedText,
            in: scrollView,
            forceScrollToBottom: isPinnedToBottom,
            scrollRequestID: scrollRequestID
        )
        context.coordinator.reportPinnedStateIfNeeded(for: scrollView, force: true)
    }

    private var displayedText: String {
        text.isEmpty ? placeholder : text
    }

    final class Coordinator: NSObject {
        var onPinnedToBottomChanged: (Bool) -> Void
        private weak var textView: NSTextView?
        private var boundsObserver: NSObjectProtocol?
        private var lastKnownPinnedToBottom = true
        private var lastScrollRequestID = 0

        init(onPinnedToBottomChanged: @escaping (Bool) -> Void) {
            self.onPinnedToBottomChanged = onPinnedToBottomChanged
        }

        deinit {
            if let boundsObserver {
                NotificationCenter.default.removeObserver(boundsObserver)
            }
        }

        func attach(scrollView: NSScrollView, textView: NSTextView) {
            self.textView = textView
            scrollView.contentView.postsBoundsChangedNotifications = true

            if let boundsObserver {
                NotificationCenter.default.removeObserver(boundsObserver)
            }

            boundsObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification,
                object: scrollView.contentView,
                queue: .main
            ) { [weak self, weak scrollView] _ in
                guard let self, let scrollView else { return }
                self.reportPinnedStateIfNeeded(for: scrollView)
            }
        }

        func applyDisplayedText(
            _ nextText: String,
            in scrollView: NSScrollView,
            forceScrollToBottom: Bool,
            scrollRequestID: Int
        ) {
            guard let textView else { return }
            let hasScrollRequest = scrollRequestID != lastScrollRequestID
            lastScrollRequestID = scrollRequestID

            guard textView.string != nextText else {
                if forceScrollToBottom || hasScrollRequest {
                    scrollToBottom(scrollView)
                }
                return
            }

            let wasPinnedToBottom = isPinnedToBottom(scrollView)
            let previousY = scrollView.contentView.bounds.origin.y

            textView.string = nextText
            if let textContainer = textView.textContainer {
                textView.layoutManager?.ensureLayout(for: textContainer)
            }

            if forceScrollToBottom || hasScrollRequest || wasPinnedToBottom {
                scrollToBottom(scrollView)
            } else {
                restoreScrollPosition(previousY, in: scrollView)
            }
        }

        func reportPinnedStateIfNeeded(for scrollView: NSScrollView, force: Bool = false) {
            let pinned = isPinnedToBottom(scrollView)
            guard force || pinned != lastKnownPinnedToBottom else { return }
            lastKnownPinnedToBottom = pinned
            onPinnedToBottomChanged(pinned)
        }

        private func restoreScrollPosition(_ previousY: CGFloat, in scrollView: NSScrollView) {
            guard let documentHeight = scrollView.documentView?.bounds.height else { return }
            let viewportHeight = scrollView.contentView.bounds.height
            let maxY = max(0, documentHeight - viewportHeight)
            let newY = min(previousY, maxY)
            scrollView.contentView.scroll(to: NSPoint(x: 0, y: newY))
            scrollView.reflectScrolledClipView(scrollView.contentView)
        }

        private func scrollToBottom(_ scrollView: NSScrollView) {
            guard let documentHeight = scrollView.documentView?.bounds.height else { return }
            let viewportHeight = scrollView.contentView.bounds.height
            let bottomY = max(0, documentHeight - viewportHeight)
            scrollView.contentView.scroll(to: NSPoint(x: 0, y: bottomY))
            scrollView.reflectScrolledClipView(scrollView.contentView)
        }

        private func isPinnedToBottom(_ scrollView: NSScrollView) -> Bool {
            guard let documentHeight = scrollView.documentView?.bounds.height else {
                return true
            }
            let visibleMaxY = scrollView.contentView.bounds.maxY
            return documentHeight - visibleMaxY <= 24
        }
    }
}

private struct CareyAceSidecarFields: Equatable {
    var caption = ""
    var genre = ""
    var bpm = ""
    var bpmSource = ""
    var lmBpm = ""
    var localBpm = ""
    var filenameBpm = ""
    var keyscale = ""
    var keySource = ""
    var lmKeyscale = ""
    var localKeyscale = ""
    var timesignature = ""
    var language = ""
    var isInstrumental = false
    var customTag = ""
    var lyrics = ""
}

private struct CareyAceSidecarEntry: Identifiable, Equatable {
    let id: String
    let audioPath: String
    let audioName: String
    let sidecarPath: String
    var exists: Bool
    var fields: CareyAceSidecarFields
}

private struct CareyAceSidecarEditorSheet: View {
    let datasetPath: String

    @Environment(\.dismiss) private var dismiss
    @State private var entries: [CareyAceSidecarEntry] = []
    @State private var originalEntries: [CareyAceSidecarEntry] = []
    @State private var selectedID: String?
    @State private var message: String?
    @State private var error: String?

    private var selectedIndex: Int? {
        guard let selectedID else { return nil }
        return entries.firstIndex { $0.id == selectedID }
    }

    private var changedCount: Int {
        entries.indices.filter { index in
            guard originalEntries.indices.contains(index) else { return true }
            return entries[index].fields != originalEntries[index].fields
        }.count
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("edit ACE sidecars")
                        .font(.title3.weight(.semibold))
                    Text(datasetPath)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
                Spacer()
                Button("reload") { load() }
                Button("close") { dismiss() }
            }
            .padding(16)

            Divider()

            if let error {
                warning(error, color: .red)
                    .padding([.horizontal, .top], 16)
            } else if let message {
                warning(message, color: .green)
                    .padding([.horizontal, .top], 16)
            }

            HStack(spacing: 0) {
                trackList
                    .frame(width: 250)
                Divider()
                editorPane
            }
            .frame(minHeight: 500)

            Divider()
            HStack {
                Text("\(entries.count) tracks, \(entries.filter { $0.exists }.count) sidecars")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if changedCount > 0 {
                    Text("\(changedCount) changed")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("save sidecars") { saveChangedSidecars() }
                    .garyPrimaryButtonStyle()
                    .disabled(changedCount == 0)
            }
            .padding(14)
        }
        .frame(minWidth: 860, idealWidth: 980, minHeight: 650, idealHeight: 760)
        .onAppear { load() }
    }

    private var trackList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                ForEach(self.entries, id: \.id) { (entry: CareyAceSidecarEntry) in
                    trackRow(entry)
                }
            }
            .padding(10)
        }
    }

    private func trackRow(_ entry: CareyAceSidecarEntry) -> some View {
        Button {
            selectedID = entry.id
            message = nil
            error = nil
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.audioName)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(entry.exists ? "txt" : "none")
                        .font(.caption2)
                        .foregroundStyle(entry.exists ? Color.secondary : Color.orange)
                }
                Spacer()
                if entry.id == selectedID {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.accentColor)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(entry.id == selectedID ? Color.accentColor.opacity(0.12) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    private var editorPane: some View {
        if let index = selectedIndex {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(entries[index].audioName)
                            .font(.headline)
                        Text(entries[index].sidecarPath)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }

                    labeledTextEditor("caption", text: stringBinding(\.caption), minHeight: 76)
                    labeledField("genre", text: stringBinding(\.genre))

                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                        GridRow {
                            labeledField("bpm", text: stringBinding(\.bpm), width: 120)
                            provenance(
                                source: entries[index].fields.bpmSource,
                                lm: entries[index].fields.lmBpm,
                                local: entries[index].fields.localBpm,
                                filename: entries[index].fields.filenameBpm
                            )
                        }
                        GridRow {
                            labeledField("key", text: stringBinding(\.keyscale), width: 160)
                            provenance(
                                source: entries[index].fields.keySource,
                                lm: entries[index].fields.lmKeyscale,
                                local: entries[index].fields.localKeyscale,
                                filename: ""
                            )
                        }
                    }

                    HStack(spacing: 12) {
                        labeledField("time signature", text: stringBinding(\.timesignature), width: 130)
                        labeledField("language", text: stringBinding(\.language), width: 110)
                        labeledField("custom tag", text: stringBinding(\.customTag), width: 150)
                        Toggle("instrumental", isOn: boolBinding(\.isInstrumental))
                            .garyCheckboxStyle()
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("lyrics")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("insert template") {
                                insertLyricsTemplate(at: index)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(entries[index].fields.isInstrumental)
                        }
                        Text("bring your own lyrics in this build. ACE will not auto-fill them.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        labeledTextEditor("", text: stringBinding(\.lyrics), minHeight: 180)
                        Text("Suggested shape: [Intro - optional mood], [Verse - optional timestamp + description], [Chorus]. Keep syllable counts fairly even when you can.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            VStack {
                Spacer()
                Text(entries.isEmpty ? "No audio files found." : "Select a track.")
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .frame(maxWidth: .infinity)
        }
    }

    private func labeledField(
        _ label: String,
        text: Binding<String>,
        width: CGFloat? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    private func labeledTextEditor(
        _ label: String,
        text: Binding<String>,
        minHeight: CGFloat
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: text)
                .font(.body)
                .frame(minHeight: minHeight)
                .padding(4)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    private func provenance(
        source: String,
        lm: String,
        local: String,
        filename: String
    ) -> some View {
        let parts = [
            source.isEmpty ? nil : "selected by \(source)",
            lm.isEmpty ? nil : "lm \(lm)",
            local.isEmpty ? nil : "local \(local)",
            filename.isEmpty ? nil : "filename \(filename)",
        ].compactMap { $0 }
        return Text(parts.joined(separator: " · "))
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(2)
    }

    private func warning(_ text: String, color: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "info.circle.fill")
                .foregroundStyle(color)
            Text(text)
                .font(.callout)
            Spacer()
        }
        .padding(10)
        .background(color.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 7))
    }

    private func stringBinding(_ keyPath: WritableKeyPath<CareyAceSidecarFields, String>) -> Binding<String> {
        Binding(
            get: {
                guard let index = selectedIndex else { return "" }
                return entries[index].fields[keyPath: keyPath]
            },
            set: { value in
                guard let index = selectedIndex else { return }
                entries[index].fields[keyPath: keyPath] = value
            }
        )
    }

    private func boolBinding(_ keyPath: WritableKeyPath<CareyAceSidecarFields, Bool>) -> Binding<Bool> {
        Binding(
            get: {
                guard let index = selectedIndex else { return false }
                return entries[index].fields[keyPath: keyPath]
            },
            set: { value in
                guard let index = selectedIndex else { return }
                entries[index].fields[keyPath: keyPath] = value
            }
        )
    }

    private func load() {
        error = nil
        message = nil
        do {
            let loaded = try Self.loadEntries(datasetPath: datasetPath)
            entries = loaded
            originalEntries = loaded
            selectedID = loaded.first?.id
        } catch {
            entries = []
            originalEntries = []
            selectedID = nil
            self.error = error.localizedDescription
        }
    }

    private func saveChangedSidecars() {
        do {
            var saved = 0
            for index in entries.indices {
                guard !originalEntries.indices.contains(index)
                        || entries[index].fields != originalEntries[index].fields else {
                    continue
                }
                try Self.writeSidecar(entries[index])
                entries[index].exists = true
                saved += 1
            }
            originalEntries = entries
            message = saved == 1 ? "Saved 1 sidecar." : "Saved \(saved) sidecars."
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func insertLyricsTemplate(at index: Int) {
        guard entries.indices.contains(index) else { return }
        let existing = entries[index].fields.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        if !existing.isEmpty {
            return
        }
        entries[index].fields.lyrics = Self.defaultLyricsTemplate
        if entries[index].fields.language.isEmpty {
            entries[index].fields.language = "en"
        }
        message = "Inserted lyric template."
        error = nil
    }

    private static func loadEntries(datasetPath: String) throws -> [CareyAceSidecarEntry] {
        let root = URL(fileURLWithPath: datasetPath, isDirectory: true)
        let manager = FileManager.default
        guard manager.fileExists(atPath: root.path) else {
            throw NSError(domain: "CareyAceSidecars", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Dataset folder does not exist."
            ])
        }
        let audioExtensions: Set<String> = ["wav", "mp3", "flac", "ogg", "opus", "m4a"]
        let urls = (manager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )?.compactMap { $0 as? URL } ?? [])
            .filter { audioExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }

        return urls.map { audioURL in
            let sidecarURL = audioURL.deletingPathExtension().appendingPathExtension("txt")
            let parsed = parseSidecar(at: sidecarURL, audioURL: audioURL)
            let fields = fields(from: parsed)
            return CareyAceSidecarEntry(
                id: audioURL.path,
                audioPath: audioURL.path,
                audioName: audioURL.lastPathComponent,
                sidecarPath: sidecarURL.path,
                exists: manager.fileExists(atPath: sidecarURL.path),
                fields: fields
            )
        }
    }

    private static func parseSidecar(at sidecarURL: URL, audioURL: URL) -> [String: String] {
        let manager = FileManager.default
        if manager.fileExists(atPath: sidecarURL.path),
           let content = try? String(contentsOf: sidecarURL, encoding: .utf8),
           !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return parseKeyValueSidecar(content)
        }

        let stem = audioURL.deletingPathExtension()
        let captionURL = stem.appendingPathExtension("caption.txt")
        let lyricsURL = stem.appendingPathExtension("lyrics.txt")
        var result: [String: String] = [:]
        if let caption = try? String(contentsOf: captionURL, encoding: .utf8) {
            result["caption"] = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let lyrics = try? String(contentsOf: lyricsURL, encoding: .utf8) {
            result["lyrics"] = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if result.isEmpty,
           manager.fileExists(atPath: sidecarURL.path),
           let lyrics = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            result["lyrics"] = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return result
    }

    private static let defaultLyricsTemplate = """
[Intro - optional mood]

[Verse - optional timestamp + description]
lyrics go here
even syllable counts help

[Chorus]
more lyrics here (yeah!)
"""

    private static func parseKeyValueSidecar(_ content: String) -> [String: String] {
        let scalarKeys: Set<String> = [
            "caption", "genre", "bpm", "bpm_source", "lm_bpm", "local_bpm",
            "filename_bpm", "key", "keyscale", "key_source", "lm_keyscale",
            "local_keyscale", "signature", "timesignature", "time_signature",
            "language", "is_instrumental", "custom_tag", "prompt_override",
        ]
        var result: [String: String] = [:]
        var currentKey: String?
        var currentLines: [String] = []

        func commit() {
            guard let currentKey else { return }
            result[currentKey] = currentLines.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        for rawLine in content.components(separatedBy: .newlines) {
            if let colon = rawLine.firstIndex(of: ":") {
                let rawKey = String(rawLine[..<colon])
                let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                if scalarKeys.contains(key) || key == "lyrics" {
                    commit()
                    currentKey = key
                    let valueStart = rawLine.index(after: colon)
                    currentLines = [String(rawLine[valueStart...]).trimmingCharacters(in: .whitespaces)]
                    continue
                }
            }
            if currentKey != nil {
                currentLines.append(rawLine)
            }
        }
        commit()
        return result
    }

    private static func fields(from parsed: [String: String]) -> CareyAceSidecarFields {
        CareyAceSidecarFields(
            caption: parsed["caption"] ?? "",
            genre: parsed["genre"] ?? "",
            bpm: parsed["bpm"] ?? "",
            bpmSource: parsed["bpm_source"] ?? "",
            lmBpm: parsed["lm_bpm"] ?? "",
            localBpm: parsed["local_bpm"] ?? "",
            filenameBpm: parsed["filename_bpm"] ?? "",
            keyscale: parsed["keyscale"] ?? parsed["key"] ?? "",
            keySource: parsed["key_source"] ?? "",
            lmKeyscale: parsed["lm_keyscale"] ?? "",
            localKeyscale: parsed["local_keyscale"] ?? "",
            timesignature: parsed["timesignature"] ?? parsed["time_signature"] ?? parsed["signature"] ?? "",
            language: parsed["language"] ?? "",
            isInstrumental: parseBool(parsed["is_instrumental"]),
            customTag: parsed["custom_tag"] ?? "",
            lyrics: parsed["lyrics"] ?? ""
        )
    }

    private static func parseBool(_ value: String?) -> Bool {
        guard let value else { return false }
        return ["1", "true", "yes", "on"].contains(value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    private static func writeSidecar(_ entry: CareyAceSidecarEntry) throws {
        let fields = entry.fields
        var lines: [String] = []
        append("caption", fields.caption, to: &lines)
        append("genre", fields.genre, to: &lines)
        append("bpm", fields.bpm, to: &lines)
        append("bpm_source", fields.bpmSource, to: &lines)
        append("lm_bpm", fields.lmBpm, to: &lines)
        append("local_bpm", fields.localBpm, to: &lines)
        append("filename_bpm", fields.filenameBpm, to: &lines)
        append("keyscale", fields.keyscale, to: &lines)
        append("key_source", fields.keySource, to: &lines)
        append("lm_keyscale", fields.lmKeyscale, to: &lines)
        append("local_keyscale", fields.localKeyscale, to: &lines)
        append("timesignature", fields.timesignature, to: &lines)
        append("language", fields.language, to: &lines)
        lines.append("is_instrumental: \(fields.isInstrumental ? "true" : "false")")
        append("custom_tag", fields.customTag, to: &lines)

        var lyrics = fields.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        if lyrics.isEmpty && fields.isInstrumental {
            lyrics = "[Instrumental]"
        }
        let lyricLines = lyrics.components(separatedBy: .newlines)
        lines.append("lyrics: \(lyricLines.first ?? "")")
        if lyricLines.count > 1 {
            lines.append(contentsOf: lyricLines.dropFirst())
        }

        let sidecarURL = URL(fileURLWithPath: entry.sidecarPath)
        try FileManager.default.createDirectory(
            at: sidecarURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try (lines.joined(separator: "\n") + "\n").write(to: sidecarURL, atomically: true, encoding: .utf8)
    }

    private static func append(_ key: String, _ value: String, to lines: inout [String]) {
        let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
        if !cleaned.isEmpty {
            lines.append("\(key): \(cleaned)")
        }
    }
}
