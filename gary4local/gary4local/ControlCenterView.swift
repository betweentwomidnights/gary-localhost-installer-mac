import SwiftUI
import AppKit
import Combine
import UniformTypeIdentifiers

struct ControlCenterView: View {
    private enum FirstUseHelperStage: String {
        case rebuild
        case menuBar
    }

    private enum SA3AdvancedTab: String, CaseIterable, Identifiable {
        case level
        case latent
        case continuation
        case experimental

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .level:
                return "level"
            case .latent:
                return "latent"
            case .continuation:
                return "continue"
            case .experimental:
                return "experimental"
            }
        }
    }

    @ObservedObject var viewModel: ControlCenterViewModel
    @AppStorage("firstUseHelperDismissedV1") private var firstUseHelperDismissed = false
    @AppStorage("firstUseHelperStageV1") private var firstUseHelperStageRawValue = FirstUseHelperStage.rebuild.rawValue
    @AppStorage("firstUseMenuBarOpenedV1") private var firstUseMenuBarOpened = false
    @State private var didObserveRebuildStart = false
    @State private var onboardingPulse = false
    @State private var sa3AdvancedPanelExpanded = false
    @State private var sa3AdvancedTab: SA3AdvancedTab = .level
    private let showOnboardingResetButton = false

    private var firstUseHelperStage: FirstUseHelperStage {
        get { FirstUseHelperStage(rawValue: firstUseHelperStageRawValue) ?? .rebuild }
        nonmutating set { firstUseHelperStageRawValue = newValue.rawValue }
    }

    private func effectiveFirstUseHelperStage(manager: ServiceManager) -> FirstUseHelperStage {
        isInitialEnvironmentBuild(manager: manager) ? .rebuild : firstUseHelperStage
    }

    private func showsFirstUseHelper(manager: ServiceManager) -> Bool {
        isInitialEnvironmentBuild(manager: manager) || !firstUseHelperDismissed
    }

    var body: some View {
        Group {
            if let error = viewModel.startupError {
                VStack(alignment: .leading, spacing: 12) {
                    Text("control center failed to load manifest.")
                        .font(.title3.weight(.semibold))
                    Text(error)
                        .font(.body.monospaced())
                    Text("manifest path:")
                        .font(.caption)
                    Text(viewModel.manifestPath)
                        .font(.caption.monospaced())
                    Button("retry") { viewModel.loadManifest() }
                }
                .padding(20)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else if let manager = viewModel.manager {
                HStack(spacing: 0) {
                    serviceList(manager: manager)
                        .frame(minWidth: 520, maxWidth: 700)
                    Divider()
                    logPane(manager: manager)
                        .frame(minWidth: 420)
                }
            } else {
                ProgressView("loading...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            if !onboardingPulse {
                withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
                    onboardingPulse = true
                }
            }
        }
        .sheet(isPresented: $viewModel.isModelDownloadSheetPresented) {
            ModelDownloadSheet(viewModel: viewModel)
        }
        .sheet(isPresented: $viewModel.isCareyLoraSheetPresented) {
            CareyLoraManagerSheet(viewModel: viewModel)
        }
        .sheet(isPresented: $viewModel.isCareyAceTrainingSheetPresented) {
            CareyAceTrainingSheet(
                trainer: viewModel.careyAceTrainingManager,
                serviceIsRunning: viewModel.isCareyServiceRunning,
                environmentReady: viewModel.isCareyEnvironmentReady,
                onStart: viewModel.startCareyAceTraining
            )
        }
        .sheet(isPresented: $viewModel.isSA3LoraSheetPresented) {
            SA3LoraManagerSheet(viewModel: viewModel)
        }
        .sheet(isPresented: $viewModel.isSA3TrainingSheetPresented) {
            SA3LoraTrainingSheet(
                trainer: viewModel.sa3TrainingManager,
                serviceIsRunning: viewModel.isSA3ServiceRunning,
                environmentReady: viewModel.isSA3EnvironmentReady,
                tokenConfigured: viewModel.stableAudioTokenConfigured,
                onStart: viewModel.startSA3LoraTraining
            )
        }
        .sheet(item: $viewModel.rebuildFailureReport) { report in
            RebuildFailureSheet(viewModel: viewModel, report: report)
        }
    }

    @ViewBuilder
    private func serviceList(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            controlCenterHeader

            if showsFirstUseHelper(manager: manager) {
                firstUseHelper(manager: manager)
            }

            Text(viewModel.manifestPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            HStack {
                Button(rebuildAllEnvironmentsButtonTitle(manager: manager)) {
                    manager.rebuildAllEnvironments()
                }
                .disabled(manager.isRebuildingAllEnvironments || manager.services.isEmpty)
                .overlay {
                    if showsFirstUseHelper(manager: manager)
                        && effectiveFirstUseHelperStage(manager: manager) == .rebuild {
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(
                                Color.accentColor.opacity(onboardingPulse ? 0.95 : 0.4),
                                lineWidth: 2
                            )
                            .shadow(
                                color: Color.accentColor.opacity(onboardingPulse ? 0.45 : 0.12),
                                radius: onboardingPulse ? 10 : 5
                            )
                            .padding(-3)
                    }
                }
#if DEBUG
                if showOnboardingResetButton {
                    Button("show onboarding") {
                        resetFirstUseHelperState()
                    }
                    .controlSize(.small)
                }
#endif
                Spacer()
            }

            List(manager.services) { runtime in
                ServiceRow(
                    runtime: runtime,
                    displayName: displayName(for: runtime),
                    isSelected: viewModel.selectedServiceID == runtime.id,
                    onSelect: { viewModel.selectService(runtime.id) },
                    onStart: { manager.start(serviceID: runtime.id) },
                    onStop: { manager.stop(serviceID: runtime.id) },
                    onRebuildEnv: { manager.rebuildEnvironment(serviceID: runtime.id) },
                    onDownloadModels: (runtime.id == "audiocraft_mlx" || runtime.id == "melodyflow" || runtime.id == "sa3" || runtime.id == "stable_audio" || runtime.id == "carey" || runtime.id == "foundation") ? {
                        viewModel.openModelDownloadSheet(for: runtime.id)
                    } : nil,
                    onManageLoras: runtime.id == "carey" ? {
                        viewModel.openCareyLoraSheet()
                    } : (runtime.id == "sa3" ? {
                        viewModel.openSA3LoraSheet()
                    } : nil),
                    onTrainLora: runtime.id == "carey" ? {
                        viewModel.openCareyAceTrainingSheet()
                    } : (runtime.id == "sa3" ? {
                        viewModel.openSA3TrainingSheet()
                    } : nil),
                    manageLorasButtonTitle: "add lora",
                    canManageDownloads: viewModel.canManageModelDownloads(for: runtime.id),
                    downloadModelsExtraDisabled: (runtime.id == "stable_audio" || runtime.id == "sa3") && !viewModel.stableAudioTokenConfigured
                )
            }
            .listStyle(.inset)
        }
        .onChange(of: manager.isRebuildingAllEnvironments, initial: false) { _, isRebuilding in
            guard showsFirstUseHelper(manager: manager),
                  effectiveFirstUseHelperStage(manager: manager) == .rebuild else { return }
            if isRebuilding {
                didObserveRebuildStart = true
                return
            }
            guard didObserveRebuildStart else { return }
            didObserveRebuildStart = false
            if configuredServicesAreReadyOrSucceeded(manager: manager) {
                firstUseMenuBarOpened = false
                firstUseHelperStage = .menuBar
            }
        }
        .onChange(of: activeRebuildAllServiceID(manager: manager), initial: true) { _, activeServiceID in
            guard manager.isRebuildingAllEnvironments else { return }
            guard let activeServiceID else { return }
            viewModel.selectService(activeServiceID)
        }
        .onAppear {
            reconcileFirstUseHelperStateIfNeeded(manager: manager)
        }
        .padding(16)
    }

    private var controlCenterHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("gary4local")
                    .font(.title2.weight(.semibold))
                Text(GaryAppReleaseInfo.releaseLabel)
                    .font(.subheadline.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
            }

            Text("recommended companion: gary4juce \(GaryAppReleaseInfo.recommendedGary4JuceVersion)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func firstUseHelper(manager: ServiceManager) -> some View {
        let stage = effectiveFirstUseHelperStage(manager: manager)
        let isStageTwoAttention = stage == .menuBar
        VStack(alignment: .leading, spacing: 10) {
            Text("first use")
                .font(.headline)

            if stage == .rebuild {
                Text("step 1: run \(buildOrRebuildAllEnvironmentsPhrase(manager: manager)).")
                    .font(.subheadline.weight(.semibold))
                Text("click the highlighted \(buildOrRebuildAllEnvironmentsPhrase(manager: manager)) button below.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                Text("environments look ready.")
                    .font(.subheadline.weight(.semibold))
                Text("now use the red gary icon in the top menu bar for quick service start/stop.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack {
                    Button("got it") { firstUseHelperDismissed = true }
                    Spacer()
                }
            }
        }
        .padding(12)
        .scaleEffect(isStageTwoAttention ? (onboardingPulse ? 1.012 : 1.0) : 1.0)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(
                    Color.accentColor.opacity(
                        isStageTwoAttention
                        ? (onboardingPulse ? 0.16 : 0.09)
                        : 0.09
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    Color.accentColor.opacity(
                        isStageTwoAttention
                        ? (onboardingPulse ? 0.92 : 0.28)
                        : 0.22
                    ),
                    lineWidth: isStageTwoAttention ? 2.0 : 1.0
                )
                .shadow(
                    color: isStageTwoAttention
                        ? Color.accentColor.opacity(onboardingPulse ? 0.4 : 0.08)
                        : .clear,
                    radius: isStageTwoAttention ? (onboardingPulse ? 14 : 4) : 0
                )
                .allowsHitTesting(false)
        )
    }

    private func configuredServicesAreReadyOrSucceeded(manager: ServiceManager) -> Bool {
        let configured = manager.services.filter { $0.service.bootstrap != nil }
        guard !configured.isEmpty else { return false }
        return configured.allSatisfy { runtime in
            runtime.bootstrapState == .ready || runtime.bootstrapState == .succeeded
        }
    }

    private func activeRebuildAllServiceID(manager: ServiceManager) -> String? {
        guard manager.isRebuildingAllEnvironments else { return nil }
        return manager.services.first(where: { $0.bootstrapState == .running })?.id
    }

    private func isInitialEnvironmentBuild(manager: ServiceManager) -> Bool {
        let configured = manager.services.filter { $0.service.bootstrap != nil }
        guard !configured.isEmpty else { return false }

        // If any configured venv directory already exists, this is no longer first-time setup.
        let fileManager = FileManager.default
        let hasAnyExistingVenv = configured.contains { runtime in
            guard let bootstrap = runtime.service.bootstrap else { return false }
            return fileManager.fileExists(atPath: bootstrap.venvDirectory.path)
        }
        if hasAnyExistingVenv {
            return false
        }

        return configured.allSatisfy { runtime in
            runtime.bootstrapState == .ready
        }
    }

    private func buildOrRebuildAllEnvironmentsPhrase(manager: ServiceManager) -> String {
        isInitialEnvironmentBuild(manager: manager)
        ? "build all environments"
        : "rebuild all environments"
    }

    private func rebuildAllEnvironmentsButtonTitle(manager: ServiceManager) -> String {
        if manager.isRebuildingAllEnvironments {
            return "rebuilding environments..."
        }
        return buildOrRebuildAllEnvironmentsPhrase(manager: manager)
    }

    private func resetFirstUseHelperState() {
        firstUseHelperDismissed = false
        firstUseHelperStage = .rebuild
        firstUseMenuBarOpened = false
        didObserveRebuildStart = false
    }

    private func reconcileFirstUseHelperStateIfNeeded(manager: ServiceManager) {
        guard isInitialEnvironmentBuild(manager: manager) else { return }
        if firstUseHelperDismissed {
            firstUseHelperDismissed = false
        }
        if firstUseHelperStage != .rebuild {
            firstUseHelperStage = .rebuild
        }
        if firstUseMenuBarOpened {
            firstUseMenuBarOpened = false
        }
        didObserveRebuildStart = false
    }

    @ViewBuilder
    private func logPane(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let selectedID = viewModel.selectedServiceID,
               let runtime = manager.services.first(where: { $0.id == selectedID }) {
                if runtime.id == "stable_audio" {
                    stableAudioAuthPanel()
                    Divider()
                } else if runtime.id == "sa3" {
                    sa3AuthPanel()
                    Divider()
                } else if runtime.id == "melodyflow" {
                    melodyFlowBackendPanel()
                    Divider()
                } else if runtime.id == "carey" {
                    careyBackendPanel()
                    Divider()
                }

                Text("logs: \(displayName(for: runtime))")
                    .font(.headline)
                Text(runtime.service.logFile.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)

                LogTextView(
                    text: viewModel.selectedLogText,
                    placeholder: "(no log output yet)",
                    isPinnedToBottom: viewModel.isLogViewerPinnedToBottom,
                    onPinnedToBottomChanged: viewModel.updateLogViewerPinnedToBottom
                )
                .background(Color(NSColor.textBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.gray.opacity(0.2))
                )

                HStack {
                    Button("refresh") { viewModel.refreshLog() }
                    Button("open log file") { NSWorkspace.shared.open(runtime.service.logFile) }
                    Spacer()
                    Button("clear selection") { viewModel.selectedServiceID = nil }
                }
            } else {
                Text("select a service to view logs.")
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
    }

    @ViewBuilder
    private func stableAudioAuthPanel() -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("jerry backend")
                .font(.headline)
            Text(
                viewModel.stableAudioTokenConfigured
                ? "hugging face token is configured in keychain."
                : "hugging face token is missing."
            )
            .font(.caption)
            .foregroundStyle(viewModel.stableAudioTokenConfigured ? .green : .orange)
            Text("this keychain token is shared with sa3.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 6) {
                Text("generation backend")
                    .font(.subheadline)
                Picker("generation backend", selection: Binding(
                    get: { viewModel.stableAudioBackendEngine },
                    set: { viewModel.setStableAudioBackendEngine($0) }
                )) {
                    ForEach(StableAudioBackendEngine.allCases) { backend in
                        Text(backend.displayName).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .garyPickerAccent()
                Text("applies to localhost generation default. changing this restarts stable audio if it's running.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.stableAudioTokenConfigured {
                stepRow(
                    number: 1,
                    title: "open model page and click 'agree and access repository'",
                    buttonTitle: "open stable audio page",
                    isEnabled: true
                ) {
                    NSWorkspace.shared.open(StableAudioAuthLinks.stableAudioModelPage)
                }

                stepRow(
                    number: 2,
                    title: "create a read token on hugging face",
                    buttonTitle: "open token settings",
                    isEnabled: true
                ) {
                    NSWorkspace.shared.open(StableAudioAuthLinks.tokenPage)
                }

                if let screenshotPath = viewModel.stableAudioStep2ScreenshotPath {
                    HoverExpandableScreenshot(
                        screenshotPath: screenshotPath,
                        caption: "step 2 reference: check 'read access to contents of all public gated repos you can access' (hover to zoom)"
                    )
                }

                HStack {
                    Text("3. paste token here")
                        .font(.subheadline)
                    Spacer()
                }
                SecureField("hf_...", text: $viewModel.stableAudioTokenInput)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                if !viewModel.stableAudioTokenConfigured {
                    Button("save token") { viewModel.saveStableAudioToken() }
                }
                Button("remove token", role: .destructive) { viewModel.clearStableAudioToken() }
                    .disabled(!viewModel.stableAudioTokenConfigured)
                Button("refresh token status") { viewModel.refreshStableAudioTokenState() }
                Spacer()
            }

            if !viewModel.stableAudioTokenStatus.isEmpty {
                Text(viewModel.stableAudioTokenStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.08))
        )
    }

    @ViewBuilder
    private func sa3AuthPanel() -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("sa3 setup")
                .font(.headline)
            Text(
                viewModel.stableAudioTokenConfigured
                ? "shared hugging face token is configured in keychain."
                : "shared hugging face token is missing."
            )
            .font(.caption)
            .foregroundStyle(viewModel.stableAudioTokenConfigured ? .green : .orange)
            Text("sa3 reuses the same keychain token as jerry. gated access only matters on the main stable-audio-3-medium page; the bundled t5gemma files do not need a separate approval step.")
                .font(.caption)
                .foregroundStyle(.secondary)

            stepRow(
                number: 1,
                title: "accept access for stable-audio-3-medium",
                buttonTitle: "open sa3 model page",
                isEnabled: true
            ) {
                NSWorkspace.shared.open(StableAudioAuthLinks.sa3ModelPage)
            }

            stepRow(
                number: 2,
                title: "create or review your read token on hugging face",
                buttonTitle: "open token settings",
                isEnabled: true
            ) {
                NSWorkspace.shared.open(StableAudioAuthLinks.tokenPage)
            }

            if let screenshotPath = viewModel.stableAudioStep2ScreenshotPath {
                HoverExpandableScreenshot(
                    screenshotPath: screenshotPath,
                    caption: "token scope reference: check 'read access to contents of all public gated repos you can access' (hover to zoom)"
                )
            }

            if !viewModel.stableAudioTokenConfigured {
                HStack {
                    Text("3. paste token here")
                        .font(.subheadline)
                    Spacer()
                }
                SecureField("hf_...", text: $viewModel.stableAudioTokenInput)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                if !viewModel.stableAudioTokenConfigured {
                    Button("save token") { viewModel.saveStableAudioToken() }
                }
                Button("remove token", role: .destructive) { viewModel.clearStableAudioToken() }
                    .disabled(!viewModel.stableAudioTokenConfigured)
                Button("refresh token status") { viewModel.refreshStableAudioTokenState() }
                Spacer()
            }

            if !viewModel.stableAudioTokenStatus.isEmpty {
                Text(viewModel.stableAudioTokenStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            sa3AdvancedSettingsPanel()
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.08))
        )
    }

    @ViewBuilder
    private func melodyFlowBackendPanel() -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("terry backend")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("generation backend")
                    .font(.subheadline)
                Picker("generation backend", selection: Binding(
                    get: { viewModel.melodyFlowBackendEngine },
                    set: { viewModel.setMelodyFlowBackendEngine($0) }
                )) {
                    ForEach(MelodyFlowBackendEngine.allCases) { backend in
                        Text(backend.shortDisplayName).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .garyPickerAccent()
                Text(
                    "mps is the quality baseline. mlx+torch keeps mlx flow with torch codec. "
                    + "mlx end-to-end uses native mlx codec and may differ in output quality."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            if !viewModel.melodyFlowBackendStatus.isEmpty {
                Text(viewModel.melodyFlowBackendStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.08))
        )
    }

    @ViewBuilder
    private func careyBackendPanel() -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("carey backend")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("generation backend")
                    .font(.subheadline)
                Picker("generation backend", selection: Binding(
                    get: { viewModel.careyBackendEngine },
                    set: { viewModel.setCareyBackendEngine($0) }
                )) {
                    ForEach(CareyBackendEngine.allCases) { backend in
                        Text(backend.displayName).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .garyPickerAccent()
                Text(
                    "mlx uses native MLX DiT/VAE for faster generation. "
                    + "mps uses the torch+mps path for compatibility checks. "
                    + "changing this restarts carey if it's running."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("model family")
                    .font(.subheadline)
                Toggle(
                    "use xl models",
                    isOn: Binding(
                        get: { viewModel.careyUseXlModels },
                        set: { viewModel.setCareyUseXlModels($0) }
                    )
                )
                Text(
                    "switches carey between regular acestep-v15-base/sft/turbo and "
                    + "xl-base/xl-sft/xl-turbo. lego uses the active base family and can use matching LoRAs. "
                    + "regular base is still the safer default when you are not loading a LoRA. "
                    + "changing this restarts carey if it is already running."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("decoder")
                    .font(.subheadline)
                Toggle(
                    "use ScragVAE",
                    isOn: Binding(
                        get: { viewModel.careyUseScragVae },
                        set: { viewModel.setCareyUseScragVae($0) }
                    )
                )
                .disabled(!viewModel.canEnableCareyScragVae)
                Text(
                    "ScragVAE is a drop-in alternate decoder by scragnog. it often adds a little more air and detail; "
                    + "stock VAE stays one click away if a render feels strange."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    Button("thank scragnog / view model") {
                        guard let url = URL(string: "https://huggingface.co/scragnog/Ace-Step-1.5-ScragVAE") else { return }
                        NSWorkspace.shared.open(url)
                    }
                    .buttonStyle(.link)

                    if !viewModel.isCareyScragVaeDownloaded {
                        Text("download ScragVAE from carey's model list before enabling it.")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }

            if viewModel.showsExperimentalCareyMlxVaeEncodeToggle {
                VStack(alignment: .leading, spacing: 6) {
                    Text("experimental mlx encode")
                        .font(.subheadline)
                    Toggle(
                        "use sampled MLX VAE encode",
                        isOn: Binding(
                            get: { viewModel.careyUseSampledMlxVaeEncode },
                            set: { viewModel.setCareyUseSampledMlxVaeEncode($0) }
                        )
                    )
                    .disabled(viewModel.careyBackendEngine != .mlx)
                    Text(
                        "experimental. on matches upstream MLX VAE encode_and_sample behavior for "
                        + "carey audio conditioning. off keeps the current deterministic encode_mean path. "
                        + "this only affects the MLX backend and changing it restarts carey if it is running."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }

            if !viewModel.careyBackendStatus.isEmpty {
                Text(viewModel.careyBackendStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.08))
        )
    }

    @ViewBuilder
    private func sa3AdvancedSettingsPanel() -> some View {
        DisclosureGroup(isExpanded: $sa3AdvancedPanelExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                Picker("sa3 advanced tab", selection: $sa3AdvancedTab) {
                    ForEach(SA3AdvancedTab.allCases) { tab in
                        Text(tab.displayName).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .garyPickerAccent()

                if sa3AdvancedTab == .level {
                    HStack(alignment: .top, spacing: 12) {
                        sa3AdvancedField(
                            title: "peak normalize dB",
                            text: $viewModel.sa3PeakNormalizeDb,
                            placeholder: "2.0",
                            helpText: "Pre-scales the decoded waveform before the limiter. Use off to disable.",
                            caption: "Pre-scales the decoded waveform before the limiter. Use off to disable."
                        )
                        sa3AdvancedField(
                            title: "limiter ceiling dB",
                            text: $viewModel.sa3LimiterCeilingDb,
                            placeholder: "-0.3",
                            helpText: "Soft anti-clip ceiling. Keep it at or below 0, or use off.",
                            caption: "Soft anti-clip ceiling. Keep it at or below 0, or use off."
                        )
                    }
                } else if sa3AdvancedTab == .latent {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top, spacing: 12) {
                            sa3AdvancedField(
                                title: "latent rescale",
                                text: $viewModel.sa3LatentRescale,
                                placeholder: "1.0",
                                helpText: "Constant multiply before decode. 1.0 leaves latents unchanged.",
                                caption: "Constant multiply before decode. 1.0 leaves latents unchanged."
                            )
                            sa3AdvancedField(
                                title: "latent shift",
                                text: $viewModel.sa3LatentShift,
                                placeholder: "0.0",
                                helpText: "Constant offset before decode. 0.0 leaves latents unchanged.",
                                caption: "Constant offset before decode. 0.0 leaves latents unchanged."
                            )
                        }
                        sa3AdvancedField(
                            title: "adaptive target std",
                            text: $viewModel.sa3LatentTargetStd,
                            placeholder: "off",
                            helpText: "Optional adaptive attenuation for hot LoRAs. Empty or off disables; try 0.9.",
                            caption: "Optional adaptive attenuation for hot LoRAs. Empty or off disables; try 0.9."
                        )
                    }
                } else if sa3AdvancedTab == .continuation {
                    sa3AdvancedField(
                        title: "continuation tail pad seconds",
                        text: $viewModel.sa3ContinuationTailPad,
                        placeholder: "6",
                        helpText: "For continue mode: 0 ends at the cut, 6 keeps a natural tail, 20+ leans seamless.",
                        caption: "For continue mode: 0 ends at the cut, 6 keeps a natural tail, 20+ leans seamless."
                    )
                } else {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 6) {
                            Text("precision")
                                .font(.subheadline)
                            Image(systemName: "questionmark.circle")
                                .foregroundStyle(.secondary)
                                .help("Experimental. Using float32 for the DiT may help loudness consistency, but it slows generation.")
                        }
                        Toggle(
                            "use fp32 DiT",
                            isOn: $viewModel.sa3UseFP32DiT
                        )
                        Text(
                            "experimental. recent tests suggest fp32 DiT can improve loudness consistency, but it slows generation. turn it off to force fp16 DiT for speed testing."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                HStack {
                    Button("save") { viewModel.saveSA3RuntimeSettings() }
                    Button("defaults") { viewModel.resetSA3RuntimeSettingsToDefaults() }
                    Spacer()
                }

                if !viewModel.sa3RuntimeSettingsStatus.isEmpty {
                    Text(viewModel.sa3RuntimeSettingsStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 8)
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text("sa3 advanced")
                    .font(.subheadline.weight(.semibold))
                Text("output shaping, continuation defaults, and experimental precision.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func sa3AdvancedField(
        title: String,
        text: Binding<String>,
        placeholder: String,
        helpText: String,
        caption: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.subheadline)
                Image(systemName: "questionmark.circle")
                    .foregroundStyle(.secondary)
                    .help(helpText)
            }

            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)

            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func stepRow(
        number: Int,
        title: String,
        buttonTitle: String,
        isEnabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        HStack {
            Text("\(number). \(title)")
                .font(.subheadline)
            Spacer()
            Button(buttonTitle, action: action)
                .disabled(!isEnabled)
        }
    }
}

private struct RebuildFailureSheet: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    let report: RebuildFailureReport
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("environment rebuild failed")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("close") {
                    viewModel.clearRebuildFailureReport()
                    dismiss()
                }
            }

            Text(displayName(forServiceID: report.serviceID, fallback: report.serviceName))
                .font(.headline)
            Text(report.summary)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("repair")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        HStack {
                            Button("repair again") { viewModel.retryRebuildFailure() }
                                .help("rerun dependency repair using the current environment.")
                            Button("repair from scratch") { viewModel.cleanRepairRebuildFailure() }
                                .help("recreate the environment and install dependencies without cache.")
                            Spacer()
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("dependencies")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        HStack {
                            Button("edit requirements") { viewModel.openRebuildFailureRequirementsEditor() }
                                .disabled(report.requirementsFile == nil)
                            Button("open in editor") { viewModel.openRebuildFailureRequirementsFile() }
                                .disabled(report.requirementsFile == nil)
                            Spacer()
                        }
                    }

                    Text("flow: start with support if needed, then edit requirements, save, and run repair again.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("repair from scratch recreates the environment when a normal repair is not enough.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)

                VStack(alignment: .leading, spacing: 8) {
                    Text("support")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        Button("email") { viewModel.openSupportEmail() }
                            .garyPrimaryButtonStyle()
                            .controlSize(.large)
                        Button(action: { viewModel.openSupportDiscord() }) {
                            Image("DiscordIcon")
                                .resizable()
                                .renderingMode(.template)
                                .interpolation(.high)
                                .frame(width: 18, height: 18)
                                .foregroundStyle(.white)
                                .padding(.horizontal, 4)
                                .help("join discord")
                        }
                        .help("join discord")
                        .accessibilityLabel("join discord")
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .tint(
                            Color(
                                red: 88.0 / 255.0,
                                green: 101.0 / 255.0,
                                blue: 242.0 / 255.0
                            )
                        )
                    }
                    Text("support: kev@thecollabagepatch.com")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(12)
                .frame(width: 260, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.gray.opacity(0.09))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.2))
                        .allowsHitTesting(false)
                )
            }

            if !viewModel.rebuildDiagnosticsStatusMessage.isEmpty {
                Text(viewModel.rebuildDiagnosticsStatusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 0) {
                HStack {
                    Text("diagnostics")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(action: { viewModel.copyRebuildFailureDiagnostics() }) {
                        Image(systemName: "doc.on.doc")
                            .frame(width: 16, height: 16)
                            .padding(6)
                            .contentShape(Rectangle())
                            .help("copy diagnostics")
                    }
                    .help("copy diagnostics")
                    .accessibilityLabel("copy diagnostics")
                    .buttonStyle(.plain)
                    Button(action: { viewModel.openRebuildFailureLogFile() }) {
                        Image(systemName: "doc.text")
                            .frame(width: 16, height: 16)
                            .padding(6)
                            .contentShape(Rectangle())
                            .help("open log file")
                    }
                    .help("open log file")
                    .accessibilityLabel("open log file")
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 10)
                .padding(.top, 8)
                .padding(.bottom, 6)

                Divider()

                ScrollView {
                    Text(viewModel.diagnosticsReportText(for: report))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
            }
            .background(Color(NSColor.textBackgroundColor))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.gray.opacity(0.2))
                    .allowsHitTesting(false)
            )
        }
        .padding(16)
        .frame(minWidth: 820, minHeight: 560)
        .sheet(isPresented: $viewModel.isRequirementsEditorPresented) {
            RequirementsEditorSheet(viewModel: viewModel)
        }
    }
}

private struct RequirementsEditorSheet: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("edit requirements")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("close") {
                    viewModel.closeRequirementsEditor()
                    dismiss()
                }
            }

            Text(viewModel.requirementsEditorPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            TextEditor(text: $viewModel.requirementsEditorText)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .background(Color(NSColor.textBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.gray.opacity(0.2))
                )

            HStack {
                if !viewModel.requirementsEditorStatusMessage.isEmpty {
                    Text(viewModel.requirementsEditorStatusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("save") { viewModel.saveRequirementsEditor() }
            }
        }
        .padding(16)
        .frame(minWidth: 820, minHeight: 560)
    }
}

private struct SA3LoraManagerSheet: View {
    private struct SA3PromptPoolRow: Identifiable {
        let name: String
        let count: Int

        var id: String { name }
    }

    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var formName: String = ""
    @State private var checkpointPath: String = ""
    @State private var promptsPath: String = ""

    private var canSave: Bool {
        !viewModel.isSA3LoraSaving
            && !formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !checkpointPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canBuild: Bool {
        viewModel.isSA3EnvironmentReady
            && !viewModel.isSA3LoraBuilding
    }

    private var sortedPools: [SA3PromptPoolRow] {
        (viewModel.sa3LoraState?.pools ?? [:])
            .map { SA3PromptPoolRow(name: $0.key, count: $0.value) }
            .sorted { lhs, rhs in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("sa3 lora manager")
                    .font(.title3.weight(.semibold))
                Spacer()
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
                .help("close")
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("pick the sa3 LoRA checkpoint file. if you still have the training dataset folder with txt sidecars, add that folder too and gary4local can build a prompt dice pool for the plugin.")
                        .font(.subheadline)

                    Text("sa3 loads LoRAs into the model at startup. if the model is already resident, reload or restart sa3 before testing a newly added LoRA.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if !viewModel.isSA3EnvironmentReady {
                        Text("build sa3 first. you can still save LoRA entries now, but prompt generation is unavailable.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else if !viewModel.isSA3ServiceRunning {
                        Text("sa3 is not running. you can save LoRA entries now; start sa3 before generation testing.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 10) {
                        Text("new entry")
                            .font(.headline)

                        TextField("lora name", text: $formName)
                            .textFieldStyle(.roundedBorder)

                        VStack(alignment: .leading, spacing: 6) {
                            Text("checkpoint file")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("path to your .ckpt or .safetensors file", text: $checkpointPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("pick file") {
                                    pickCheckpointFile()
                                }
                            }
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("dataset/prompts source folder (optional)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("path to training captions", text: $promptsPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("pick folder") {
                                    pickPromptsFolder()
                                }
                            }
                        }

                        Text("the prompt builder reads each txt sidecar, strips the bpm/key tail, and writes one JSON per LoRA.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        HStack(spacing: 8) {
                            Button(viewModel.isSA3LoraSaving ? "saving..." : "save lora") {
                                Task {
                                    await viewModel.saveSA3Lora(
                                        name: formName,
                                        checkpointPath: checkpointPath,
                                        promptsPath: promptsPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : promptsPath
                                    )
                                    if viewModel.sa3LoraErrorMessage.isEmpty {
                                        resetForm()
                                    }
                                }
                            }
                            .disabled(!canSave)

                            Button("refresh") {
                                Task { await viewModel.refreshSA3LoraState() }
                            }
                            .disabled(viewModel.isSA3LoraLoading || viewModel.isSA3LoraSaving || viewModel.isSA3LoraBuilding)

                            Button(viewModel.isSA3LoraBuilding ? "building..." : "build prompts") {
                                Task { await viewModel.buildSA3LoraPrompts() }
                            }
                            .disabled(!canBuild)

                            Spacer()
                        }
                    }

                    if !viewModel.sa3LoraStatusMessage.isEmpty {
                        Text(viewModel.sa3LoraStatusMessage)
                            .font(.caption)
                            .foregroundStyle(.green)
                    }

                    if !viewModel.sa3LoraErrorMessage.isEmpty {
                        Text(viewModel.sa3LoraErrorMessage)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    if !viewModel.sa3LoraBuildOutput.isEmpty {
                        ScrollView {
                            Text(viewModel.sa3LoraBuildOutput)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                        }
                        .frame(minHeight: 90, maxHeight: 180)
                        .background(Color(NSColor.textBackgroundColor))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.gray.opacity(0.2))
                        )
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("registered loras")
                            .font(.headline)

                        if viewModel.isSA3LoraLoading && viewModel.sa3LoraState == nil {
                            ProgressView()
                                .controlSize(.small)
                        } else if let state = viewModel.sa3LoraState, !state.entries.isEmpty {
                            List {
                                ForEach(state.entries) { entry in
                                    VStack(alignment: .leading, spacing: 6) {
                                        HStack(alignment: .firstTextBaseline) {
                                            Text(entry.name)
                                                .font(.headline)
                                            Spacer()
                                            Button("remove", role: .destructive) {
                                                Task { await viewModel.removeSA3Lora(named: entry.name) }
                                            }
                                            .controlSize(.small)
                                            .disabled(viewModel.isSA3LoraSaving || viewModel.isSA3LoraBuilding)
                                        }

                                        Text("strength \(entry.strength.formatted())")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)

                                        Text("checkpoint: \(entry.path)")
                                            .font(.caption.monospaced())
                                            .foregroundStyle(.secondary)
                                            .textSelection(.enabled)

                                        if let promptsPath = entry.promptsPath, !promptsPath.isEmpty {
                                            Text("prompt source: \(promptsPath)")
                                                .font(.caption.monospaced())
                                                .foregroundStyle(.secondary)
                                                .textSelection(.enabled)
                                        } else if entry.resolvedPromptsPath != nil {
                                            Text("prompt source: using checkpoint folder sidecars")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        } else {
                                            Text("prompt source: not set")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }

                                        Text("prompt JSON: \(entry.promptFilePath)")
                                            .font(.caption.monospaced())
                                            .foregroundStyle(.secondary)
                                            .textSelection(.enabled)

                                        Text("\(entry.captionCount) txt sidecars detected, \(entry.promptCount) prompts built")
                                            .font(.caption)
                                            .foregroundStyle(entry.registered ? Color.secondary : Color.red)

                                        if entry.captionCount == 0 {
                                            Text("no txt sidecars found. the dice button will use defaults until a prompt JSON exists.")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .padding(.vertical, 4)
                                }
                            }
                            .listStyle(.inset)
                            .frame(minHeight: 180, idealHeight: 220)
                        } else {
                            Text("no loras added yet.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !sortedPools.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("prompt pools")
                                .font(.headline)

                            List(sortedPools) { pool in
                                HStack {
                                    Text(pool.name)
                                    Spacer()
                                    Text("\(pool.count)")
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .listStyle(.inset)
                            .frame(minHeight: 120, idealHeight: 150)
                        }
                    }
                }
            }
        }
        .padding(16)
        .frame(minWidth: 820, minHeight: 620)
    }

    private func pickCheckpointFile() {
        if let path = Self.pickFile() {
            checkpointPath = path
            if formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                formName = suggestedName(from: path)
            }
        }
    }

    private func pickPromptsFolder() {
        if let path = Self.pickDirectory() {
            promptsPath = path
            if formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                formName = suggestedName(from: path)
            }
        }
    }

    private func resetForm() {
        formName = ""
        checkpointPath = ""
        promptsPath = ""
    }

    private func suggestedName(from path: String) -> String {
        let filename = URL(fileURLWithPath: path).lastPathComponent
        let withoutExtension = filename.replacingOccurrences(
            of: #"\.(ckpt|safetensors)$"#,
            with: "",
            options: .regularExpression
        )
        let cleaned = withoutExtension
            .lowercased()
            .replacingOccurrences(
                of: #"[^a-z0-9_-]+"#,
                with: "-",
                options: .regularExpression
            )
        let trimmed = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return String(trimmed.prefix(64))
    }

    private static func pickFile() -> String? {
        let panel = NSOpenPanel()
        panel.prompt = "choose"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [
            UTType(filenameExtension: "ckpt", conformingTo: .data),
            UTType(filenameExtension: "safetensors", conformingTo: .data)
        ].compactMap { $0 }
        return panel.runModal() == .OK ? panel.url?.path : nil
    }

    private static func pickDirectory() -> String? {
        let panel = NSOpenPanel()
        panel.prompt = "choose"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        return panel.runModal() == .OK ? panel.url?.path : nil
    }
}

private struct CareyLoraManagerSheet: View {
    private struct CareyCaptionPoolRow: Identifiable {
        let name: String
        let count: Int

        var id: String { name }
    }

    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var formName: String = ""
    @State private var checkpointPath: String = ""
    @State private var captionsPath: String = ""
    @State private var modelFamily: CareyLoraModelFamily = .standard

    private var canSave: Bool {
        !viewModel.isCareyLoraSaving
            && !formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !checkpointPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canBuild: Bool {
        viewModel.isCareyEnvironmentReady
            && viewModel.isCareyServiceRunning
            && !viewModel.isCareyLoraBuilding
    }

    private var sortedPools: [CareyCaptionPoolRow] {
        (viewModel.careyLoraState?.pools ?? [:])
            .map { CareyCaptionPoolRow(name: $0.key, count: $0.value) }
            .sorted { lhs, rhs in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("carey lora manager")
                    .font(.title3.weight(.semibold))
                Spacer()
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
                .help("close")
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("add local loras, keep the registry in sync, and build caption pools for the gary4juce dice button.")
                        .font(.subheadline)

                    Text("if your `.txt` sidecars live somewhere other than the adapter folder, point the captions/source path at that directory.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("the model family tag controls which loras are exposed when carey is running in standard vs xl mode.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if !viewModel.isCareyEnvironmentReady {
                        Text("build carey first. you can save lora entries now, but captions generation is unavailable until the carey env exists.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else if !viewModel.isCareyServiceRunning {
                        Text("carey is not running. you can save lora entries now; start carey to build captions and hot-reload the registry.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }

                    if viewModel.careyLoraRequiresMpsBackend {
                        Text("carey is currently set to the mlx backend. adapter registration still works, but lora generation should use mps until ace-step supports lora application in the mlx diffusion path.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 10) {
                        Text("new entry")
                            .font(.headline)

                        TextField("lora name", text: $formName)
                            .textFieldStyle(.roundedBorder)

                        Picker("model family", selection: $modelFamily) {
                            ForEach(CareyLoraModelFamily.allCases) { family in
                                Text(family.rawValue).tag(family)
                            }
                        }
                        .pickerStyle(.segmented)
                        .garyPickerAccent()

                        VStack(alignment: .leading, spacing: 6) {
                            Text("checkpoint folder")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("path to adapter folder", text: $checkpointPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("pick folder") {
                                    pickCheckpointFolder()
                                }
                            }
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("captions/source folder (optional)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("path to caption sidecars", text: $captionsPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("pick folder") {
                                    pickCaptionsFolder()
                                }
                            }
                        }

                        HStack(spacing: 8) {
                            Button(viewModel.isCareyLoraSaving ? "saving..." : "save lora") {
                                Task {
                                    await viewModel.saveCareyLora(
                                        name: formName,
                                        checkpointPath: checkpointPath,
                                        captionsPath: captionsPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : captionsPath,
                                        modelFamily: modelFamily
                                    )
                                    if viewModel.careyLoraErrorMessage.isEmpty {
                                        resetForm()
                                    }
                                }
                            }
                            .disabled(!canSave)

                            Button("refresh") {
                                Task { await viewModel.refreshCareyLoraState() }
                            }
                            .disabled(viewModel.isCareyLoraLoading || viewModel.isCareyLoraSaving || viewModel.isCareyLoraBuilding)

                            Button(viewModel.isCareyLoraBuilding ? "building..." : "build captions") {
                                Task { await viewModel.buildCareyLoraCaptions() }
                            }
                            .disabled(!canBuild)

                            Button("open examples repo") {
                                guard let url = URL(string: "https://github.com/betweentwomidnights/gary-lora-examples") else { return }
                                NSWorkspace.shared.open(url)
                            }

                            Spacer()
                        }
                    }

                    if !viewModel.careyLoraStatusMessage.isEmpty {
                        Text(viewModel.careyLoraStatusMessage)
                            .font(.caption)
                            .foregroundStyle(.green)
                    }

                    if !viewModel.careyLoraErrorMessage.isEmpty {
                        Text(viewModel.careyLoraErrorMessage)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    if !viewModel.careyLoraBuildOutput.isEmpty {
                        ScrollView {
                            Text(viewModel.careyLoraBuildOutput)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                        }
                        .frame(minHeight: 90, maxHeight: 180)
                        .background(Color(NSColor.textBackgroundColor))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.gray.opacity(0.2))
                        )
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("registered loras")
                            .font(.headline)

                        if viewModel.isCareyLoraLoading && viewModel.careyLoraState == nil {
                            ProgressView()
                                .controlSize(.small)
                        } else if let state = viewModel.careyLoraState, !state.entries.isEmpty {
                            List {
                                ForEach(state.entries) { entry in
                                    VStack(alignment: .leading, spacing: 6) {
                                        HStack(alignment: .firstTextBaseline) {
                                            Text(entry.name)
                                                .font(.headline)
                                            Spacer()
                                            Button("remove", role: .destructive) {
                                                Task { await viewModel.removeCareyLora(named: entry.name) }
                                            }
                                            .controlSize(.small)
                                            .disabled(viewModel.isCareyLoraSaving || viewModel.isCareyLoraBuilding)
                                        }

                                        Text("\(entry.modelFamily) / backends: \(entry.backends.joined(separator: ", ")) / scale \(entry.scale.formatted())")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)

                                        Text("checkpoint: \(entry.path)")
                                            .font(.caption.monospaced())
                                            .foregroundStyle(.secondary)
                                            .textSelection(.enabled)

                                        if let captionsPath = entry.captionsPath, !captionsPath.isEmpty {
                                            Text("captions source: \(captionsPath)")
                                                .font(.caption.monospaced())
                                                .foregroundStyle(.secondary)
                                                .textSelection(.enabled)
                                        } else if entry.resolvedCaptionsPath != nil {
                                            Text("captions source: using checkpoint folder sidecars")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        } else {
                                            Text("captions source: not set")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }

                                        if entry.registered {
                                            Text("\(entry.captionCount) caption sidecars detected")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        } else {
                                            Text("checkpoint folder missing or invalid")
                                                .font(.caption)
                                                .foregroundStyle(.red)
                                        }

                                        if entry.captionCount == 0 {
                                            Text("no sidecar txts found. the dice button will fall back to the default caption pool.")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .padding(.vertical, 4)
                                }
                            }
                            .listStyle(.inset)
                            .frame(minHeight: 180, idealHeight: 220)
                        } else {
                            Text("no loras added yet.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !sortedPools.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("captions pools")
                                .font(.headline)

                            List(sortedPools) { pool in
                                HStack {
                                    Text(pool.name)
                                    Spacer()
                                    Text("\(pool.count)")
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .listStyle(.inset)
                            .frame(minHeight: 100, idealHeight: 130, maxHeight: 180)
                        }
                    }

                    if let state = viewModel.careyLoraState {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("catalog: \(state.catalogPath)")
                            Text("registry: \(state.registryPath)")
                            Text("captions: \(state.captionsPath)")
                        }
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(16)
        .frame(minWidth: 820, idealWidth: 900, minHeight: 520, idealHeight: 620)
        .onAppear {
            resetFormIfNeeded()
            if viewModel.careyLoraState == nil {
                Task { await viewModel.refreshCareyLoraState() }
            }
        }
    }

    private func resetFormIfNeeded() {
        guard formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              checkpointPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              captionsPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        resetForm()
    }

    private func resetForm() {
        formName = ""
        checkpointPath = ""
        captionsPath = ""
        modelFamily = viewModel.careyUseXlModels ? .xl : .standard
    }

    private func pickCheckpointFolder() {
        guard let path = Self.pickDirectory() else { return }
        checkpointPath = path
        if formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            formName = suggestedName(from: path)
        }
        if path.lowercased().contains("xl") {
            modelFamily = .xl
        }
    }

    private func pickCaptionsFolder() {
        guard let path = Self.pickDirectory() else { return }
        captionsPath = path
        if formName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            formName = suggestedName(from: path)
        }
    }

    private func suggestedName(from path: String) -> String {
        let base = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        let cleaned = base.replacingOccurrences(
            of: #"[^a-z0-9_-]+"#,
            with: "-",
            options: .regularExpression
        )
        let trimmed = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return String(trimmed.prefix(64))
    }

    private static func pickDirectory() -> String? {
        let panel = NSOpenPanel()
        panel.prompt = "choose"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        return panel.runModal() == .OK ? panel.url?.path : nil
    }
}

private struct ModelDownloadSheet: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("download models: \(viewModel.modelDownloadServiceDisplayName)")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("close") { dismiss() }
            }

            if viewModel.modelDownloadServiceID == "stable_audio" {
                stableAudioPredownloadContent
            } else if viewModel.modelDownloadServiceID == "sa3" {
                sa3SetupContent
            } else if viewModel.modelDownloadServiceID == "carey" {
                careyPredownloadContent
            } else {
                standardPredownloadContent
            }
        }
        .padding(16)
        .frame(minWidth: 760, minHeight: 520)
        .onAppear {
            if viewModel.modelDownloadServiceID != "stable_audio",
               viewModel.modelDownloadServiceID != "sa3",
               viewModel.modelDownloadServiceID != "carey",
               viewModel.downloadableModels.isEmpty {
                viewModel.refreshModelCatalogAndStatuses()
            }
        }
    }

    @ViewBuilder
    private var modelDownloadStatusText: some View {
        if !viewModel.modelDownloadStatusMessage.isEmpty {
            let isAlert = isAlertModelDownloadStatus(viewModel.modelDownloadStatusMessage)
            Text(viewModel.modelDownloadStatusMessage)
                .font(.caption.weight(isAlert ? .semibold : .regular))
                .foregroundStyle(isAlert ? Color.red : Color.secondary)
        }
    }

    private func isAlertModelDownloadStatus(_ message: String) -> Bool {
        let normalized = message.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return normalized.contains("rejected")
            || normalized.contains("invalid token")
            || normalized.contains("save your hugging face token")
            || normalized.contains("token is required")
            || (normalized.contains("token") && normalized.contains("failed"))
    }

    @ViewBuilder
    private var sa3SetupContent: some View {
        Text("pre-download the required sa3 model repos for offline use in gary4juce.")
            .font(.subheadline)
        Text("this service reuses the same keychain hugging face token as jerry.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("accept access for both upstream repos before testing the backend.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("if you skip this step, the first /load or generation request will still fetch the gated assets on demand.")
            .font(.caption)
            .foregroundStyle(.secondary)

        if !viewModel.stableAudioTokenConfigured {
            Text("hugging face token is required. save it in sa3 or jerry setup first.")
                .font(.caption)
                .foregroundStyle(.red)
        }

        modelDownloadStatusText

        HStack(spacing: 8) {
            Button("download required models") {
                viewModel.startSA3PredownloadRequiredModels()
            }
            Button("refresh downloaded") {
                viewModel.refreshSA3PredownloadInventory()
            }
            Spacer()
        }
        .disabled(!viewModel.canManageSA3Predownloads || viewModel.isModelDownloadInProgress)

        if viewModel.isModelDownloadInProgress && viewModel.modelDownloadServiceID == "sa3" {
            ProgressView(value: viewModel.sa3PredownloadProgress)
                .controlSize(.small)
        }

        if !viewModel.sa3InventoryModels.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("downloaded models")
                    .font(.caption.weight(.semibold))
                ForEach(viewModel.sa3InventoryModels) { row in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(row.downloaded ? Color.green : Color.gray.opacity(0.5))
                            .frame(width: 8, height: 8)
                        Text(row.label)
                            .font(.caption.monospaced())
                        Spacer()
                        if row.downloaded {
                            Text("downloaded")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else if !row.missing.isEmpty {
                            Text("missing \(row.missing.joined(separator: ", "))")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("not downloaded")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(.vertical, 2)
        }

        VStack(alignment: .leading, spacing: 6) {
            Text("notes")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text("use the loras button on the sa3 service row to register local checkpoints and build prompt dice pools.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !viewModel.canManageModelDownloads {
                Text("start sa3 when you want to validate health and inspect logs.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)

        Spacer(minLength: 0)
    }

    @ViewBuilder
    private var standardPredownloadContent: some View {
        Text("pre-download models for offline use in gary4juce.")
            .font(.subheadline)
        Text("models also download the first time you use them inside gary4juce.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("first-time model downloads can include shared assets, so progress can appear in multiple stages.")
            .font(.caption)
            .foregroundStyle(.secondary)

        modelDownloadStatusText

        HStack {
            Button("refresh statuses") {
                viewModel.refreshModelCatalogAndStatuses()
            }
            .disabled(viewModel.isModelCatalogLoading || viewModel.isModelDownloadInProgress)

            if viewModel.isModelCatalogLoading {
                ProgressView()
                    .controlSize(.small)
            }
            Spacer()
        }

        if viewModel.downloadableModels.isEmpty, !viewModel.isModelCatalogLoading {
            VStack(alignment: .leading, spacing: 6) {
                Text("no models available.")
                    .foregroundStyle(.secondary)
                if !viewModel.canManageModelDownloads {
                    Text("start \(viewModel.modelDownloadServiceDisplayName), then refresh statuses.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else {
            List {
                ForEach(viewModel.modelDownloadSections) { section in
                    Section(section.title) {
                        ForEach(section.models) { model in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(model.displayName)
                                        .font(.system(.body, design: .monospaced))
                                    Spacer()
                                    statusPill(for: model)
                                    Button(
                                        model.isDownloading ? "downloading..." : "download"
                                    ) {
                                        viewModel.startModelDownload(model.path)
                                    }
                                    .disabled(
                                        model.isDownloading
                                        || model.downloaded
                                        || viewModel.isModelDownloadInProgress
                                        || !viewModel.canManageModelDownloads
                                    )
                                }

                                Text(model.path)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)

                                if model.isDownloading {
                                    ProgressView(value: model.progress)
                                        .controlSize(.small)
                                }

                                Text(model.statusMessage)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .listStyle(.inset)
        }
    }

    @ViewBuilder
    private var stableAudioPredownloadContent: some View {
        Text("pre-download stable audio models and finetune checkpoints for offline use in gary4juce.")
            .font(.subheadline)
        Text("this mirrors jerry's repo + checkpoint flow. downloads go to the same hugging face cache used at runtime.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("stable-audio-open-small is already downloaded/loaded when jerry starts.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("use the quick option below for stable-audio-open-1.0 or fetch finetune checkpoints.")
            .font(.caption)
            .foregroundStyle(.secondary)

        if !viewModel.stableAudioTokenConfigured {
            Text("hugging face token is required. save it in jerry setup first.")
                .font(.caption)
                .foregroundStyle(.red)
        }

        modelDownloadStatusText

        HStack(spacing: 8) {
            Button("download open-1.0") {
                viewModel.startStableAudioPredownloadOpenOne()
            }
            Button("refresh downloaded") {
                viewModel.refreshStableAudioPredownloadInventory(
                    checkpointsHint: viewModel.stableAudioPredownloadCheckpoints
                )
            }
            Spacer()
        }
        .disabled(!viewModel.canManageStableAudioPredownloads || viewModel.isModelDownloadInProgress)

        if !viewModel.stableAudioInventoryModels.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("downloaded models")
                    .font(.caption.weight(.semibold))
                ForEach(viewModel.stableAudioInventoryModels) { row in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(row.downloaded ? Color.green : Color.gray.opacity(0.5))
                            .frame(width: 8, height: 8)
                        Text(row.label)
                            .font(.caption.monospaced())
                        Spacer()
                        if row.downloaded {
                            Text("downloaded")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else if !row.missing.isEmpty {
                            Text("missing \(row.missing.joined(separator: ", "))")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("not downloaded")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(.vertical, 2)
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
            Text("finetune repo")
                .font(.caption.weight(.semibold))
            HStack(spacing: 8) {
                TextField("hugging face repo (e.g. thepatch/jerry_grunge)", text: $viewModel.stableAudioPredownloadRepoInput)
                    .textFieldStyle(.roundedBorder)
                Button("fetch checkpoints") {
                    viewModel.fetchStableAudioPredownloadCheckpoints()
                }
                .disabled(
                    !viewModel.canManageStableAudioPredownloads
                    || viewModel.isModelDownloadInProgress
                    || viewModel.isStableAudioCheckpointFetchInProgress
                )
            }

            Text("optional finetune repos: thepatch/jerry_grunge, S3Sound/kickbass, or your own hugging face repo.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if viewModel.isStableAudioCheckpointFetchInProgress {
                ProgressView()
                    .controlSize(.small)
            }

            if !viewModel.stableAudioPredownloadCheckpoints.isEmpty {
                Picker("checkpoint", selection: $viewModel.stableAudioPredownloadSelectedCheckpoint) {
                    ForEach(viewModel.stableAudioPredownloadCheckpoints, id: \.self) { checkpoint in
                        let downloaded = viewModel.stableAudioPredownloadCheckpointDownloaded[checkpoint] ?? false
                        Text(downloaded ? "✓ \(checkpoint)" : checkpoint).tag(checkpoint)
                    }
                }
                .labelsHidden()
                .garyPickerAccent()

                HStack(spacing: 8) {
                    Button("download selected checkpoint") {
                        viewModel.startStableAudioPredownloadSelectedCheckpoint()
                    }
                    .disabled(
                        !viewModel.canManageStableAudioPredownloads
                        || viewModel.isModelDownloadInProgress
                        || viewModel.stableAudioPredownloadSelectedCheckpoint.isEmpty
                    )
                    Button("use model") {
                        viewModel.useStableAudioSelectedCheckpoint()
                    }
                    .disabled(
                        !viewModel.canManageModelDownloads
                        || viewModel.isModelDownloadInProgress
                        || viewModel.isStableAudioModelSwitchInProgress
                        || viewModel.stableAudioPredownloadSelectedCheckpoint.isEmpty
                    )
                    Spacer()
                }
            }

            if !viewModel.stableAudioCachedFinetunes.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("downloaded finetunes for this repo")
                        .font(.caption.weight(.semibold))
                    ForEach(viewModel.stableAudioCachedFinetunes, id: \.self) { checkpoint in
                        HStack(spacing: 8) {
                            Circle()
                                .fill(Color.green)
                                .frame(width: 8, height: 8)
                            Text(checkpoint)
                                .font(.caption.monospaced())
                                .lineLimit(1)
                            Spacer()
                        }
                    }
                }
            }
        }

        if viewModel.isModelDownloadInProgress {
            ProgressView(value: viewModel.stableAudioPredownloadProgress)
                .controlSize(.small)
            if !viewModel.stableAudioPredownloadTargetLabel.isEmpty {
                Text("active: \(viewModel.stableAudioPredownloadTargetLabel)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }

        if viewModel.isStableAudioModelSwitchInProgress {
            ProgressView()
                .controlSize(.small)
            Text("loading selected model into jerry cache...")
                .font(.caption)
                .foregroundStyle(.secondary)
        }

        Spacer()
    }

    @ViewBuilder
    private var careyPredownloadContent: some View {
        Text("pre-download carey lego dependencies for offline use in gary4juce.")
            .font(.subheadline)
        Text("carey uses ace-step lego mode with `acestep_init_llm=false`. lego uses the active base family only, while complete can use base, sft, or turbo from that same family.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("service startup is lazy. use this sheet to pre-download the selected regular or xl model family before generating.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("plain regular base is still the safer lego default without a LoRA. xl-base gets more interesting once you have a matching xl LoRA.")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text("downloading any one of base, sft, or turbo also pulls in the shared qwen + vae assets. ScragVAE is optional.")
            .font(.caption)
            .foregroundStyle(.secondary)

        modelDownloadStatusText

        HStack(spacing: 8) {
            Button("download all") {
                viewModel.startCareyFocusedDownload()
            }
            .disabled(
                !viewModel.canRunCareyFocusedDownload
                || viewModel.isModelDownloadInProgress
                || viewModel.isCareyLifecycleActionInProgress
            )

            Button("refresh downloaded") {
                viewModel.refreshCareyPredownloadInventory()
            }
            .disabled(viewModel.isCareyDownloadInProgress)

            Spacer()
        }

        if !viewModel.canRunCareyFocusedDownload {
            Text("download script missing. expected `runtime/scripts/download_carey_models.sh` or `scripts/download_carey_models.sh` in dev.")
                .font(.caption)
                .foregroundStyle(.red)
        }

        if viewModel.careyRequiredModels.isEmpty {
            Text("no carey model inventory found yet.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            let groupedRows = careyGroupedRequiredModels(
                from: viewModel.careyRequiredModels,
                useXlModels: viewModel.careyUseXlModels
            )
            let downloadedFileCount = viewModel.careyRequiredModels.filter(\.downloaded).count
            let totalFileCount = viewModel.careyRequiredModels.count
            VStack(alignment: .leading, spacing: 6) {
                Text("required components (\(downloadedFileCount)/\(totalFileCount) files)")
                    .font(.caption.weight(.semibold))
                ForEach(groupedRows) { row in
                    HStack(spacing: 8) {
                        let indicatorColor: Color = row.isComplete
                            ? .green
                            : (row.downloadedFileCount > 0 ? .orange : Color.gray.opacity(0.5))
                        Circle()
                            .fill(indicatorColor)
                            .frame(width: 8, height: 8)
                        Text(row.label)
                            .font(.caption.monospaced())
                        Spacer()
                        Text(
                            row.isComplete
                            ? formatByteCount(row.downloadedBytes)
                            : "\(row.downloadedFileCount)/\(row.totalFileCount) files"
                        )
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if let downloadTarget = row.downloadTarget {
                            Button(
                                viewModel.isCareyDownloadTargetActive(downloadTarget)
                                    ? "downloading..."
                                    : (row.isComplete ? "downloaded" : "download")
                            ) {
                                viewModel.startCareyFocusedDownload(for: downloadTarget)
                            }
                            .disabled(
                                row.isComplete
                                || !viewModel.canRunCareyFocusedDownload
                                || viewModel.isModelDownloadInProgress
                                || viewModel.isCareyLifecycleActionInProgress
                            )
                        }
                    }
                    Text(row.relativePrefix)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 2)
        }

        if !viewModel.careyOptionalModels.isEmpty {
            Divider()

            let scragRows = careyGroupedOptionalModels(from: viewModel.careyOptionalModels)
            let scragDownloaded = viewModel.careyOptionalModels.filter(\.downloaded).count
            let scragTotal = viewModel.careyOptionalModels.count
            VStack(alignment: .leading, spacing: 6) {
                Text("optional components (\(scragDownloaded)/\(scragTotal) files)")
                    .font(.caption.weight(.semibold))
                ForEach(scragRows) { row in
                    HStack(spacing: 8) {
                        let indicatorColor: Color = row.isComplete
                            ? .green
                            : (row.downloadedFileCount > 0 ? .orange : Color.gray.opacity(0.5))
                        Circle()
                            .fill(indicatorColor)
                            .frame(width: 8, height: 8)
                        Text(row.label)
                            .font(.caption.monospaced())
                        Spacer()
                        Text(
                            row.isComplete
                            ? formatByteCount(row.downloadedBytes)
                            : "\(row.downloadedFileCount)/\(row.totalFileCount) files"
                        )
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if let downloadTarget = row.downloadTarget {
                            Button(
                                viewModel.isCareyDownloadTargetActive(downloadTarget)
                                    ? "downloading..."
                                    : (row.isComplete ? "downloaded" : "download")
                            ) {
                                viewModel.startCareyFocusedDownload(for: downloadTarget)
                            }
                            .disabled(
                                row.isComplete
                                || !viewModel.canRunCareyFocusedDownload
                                || viewModel.isModelDownloadInProgress
                                || viewModel.isCareyLifecycleActionInProgress
                            )
                        }
                    }
                    Text(row.relativePrefix)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 2)
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
            Text("model lifecycle")
                .font(.caption.weight(.semibold))
            Text("start carey first. use these controls only when you want to manually load/unload the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("stop the service for a full memory release.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button("load model") {
                    viewModel.loadCareyModel()
                }
                .disabled(
                    !viewModel.canManageModelDownloads
                    || viewModel.isCareyLifecycleActionInProgress
                    || viewModel.isModelDownloadInProgress
                )

                Button("unload model") {
                    viewModel.unloadCareyModel()
                }
                .disabled(
                    !viewModel.canManageModelDownloads
                    || viewModel.isCareyLifecycleActionInProgress
                    || viewModel.isModelDownloadInProgress
                )
                Spacer()
            }
        }

        if viewModel.isCareyDownloadInProgress {
            ProgressView(value: viewModel.careyPredownloadProgress)
                .controlSize(.small)
            if !viewModel.careyPredownloadActiveLabel.isEmpty {
                Text("active: \(viewModel.careyPredownloadActiveLabel)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }

        if viewModel.isCareyLifecycleActionInProgress {
            ProgressView()
                .controlSize(.small)
        }

        Spacer()
    }

    private func formatByteCount(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "missing" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private func careyGroupedRequiredModels(
        from rows: [CareyRequiredModelStatus],
        useXlModels: Bool
    ) -> [CareyRequiredGroupStatus] {
        let baseConfig = useXlModels ? "acestep-v15-xl-base" : "acestep-v15-base"
        let sftConfig = useXlModels ? "acestep-v15-xl-sft" : "acestep-v15-sft"
        let turboConfig = useXlModels ? "acestep-v15-xl-turbo" : "acestep-v15-turbo"
        var groups: [(id: String, label: String, prefix: String, downloadTarget: CareyDownloadTarget?)] = [
            ("dit_base", "DiT Base / Lego Base (\(baseConfig))", "checkpoints/\(baseConfig)/", CareyDownloadTarget.base),
            ("dit_sft", "DiT SFT (\(sftConfig))", "checkpoints/\(sftConfig)/", CareyDownloadTarget.sft),
            ("dit_turbo", "DiT Turbo (\(turboConfig))", "checkpoints/\(turboConfig)/", CareyDownloadTarget.turbo),
            ("qwen_encoder", "Qwen Text Encoder", "checkpoints/Qwen3-Embedding-0.6B/", nil),
            ("vae", "VAE", "checkpoints/vae/", nil),
        ]

        return groups.compactMap { group in
            let matched = rows.filter { $0.relativePath.hasPrefix(group.prefix) }
            guard !matched.isEmpty else { return nil }

            let downloadedFileCount = matched.filter(\.downloaded).count
            let downloadedBytes = matched.reduce(into: Int64(0)) { partial, row in
                if row.downloaded {
                    partial += row.sizeBytes
                }
            }

            return CareyRequiredGroupStatus(
                id: group.id,
                label: group.label,
                relativePrefix: group.prefix,
                downloadTarget: group.downloadTarget,
                downloadedFileCount: downloadedFileCount,
                totalFileCount: matched.count,
                downloadedBytes: downloadedBytes
            )
        }
    }

    private func careyGroupedOptionalModels(
        from rows: [CareyRequiredModelStatus]
    ) -> [CareyRequiredGroupStatus] {
        let groups: [(id: String, label: String, prefix: String, downloadTarget: CareyDownloadTarget?)] = [
            ("scrag_vae", "ScragVAE (alternate decoder)", "checkpoints/scrag-vae/", CareyDownloadTarget.scragVae),
        ]

        return groups.compactMap { group in
            let matched = rows.filter { $0.relativePath.hasPrefix(group.prefix) }
            guard !matched.isEmpty else { return nil }

            let downloadedFileCount = matched.filter(\.downloaded).count
            let downloadedBytes = matched.reduce(into: Int64(0)) { partial, row in
                if row.downloaded {
                    partial += row.sizeBytes
                }
            }

            return CareyRequiredGroupStatus(
                id: group.id,
                label: group.label,
                relativePrefix: group.prefix,
                downloadTarget: group.downloadTarget,
                downloadedFileCount: downloadedFileCount,
                totalFileCount: matched.count,
                downloadedBytes: downloadedBytes
            )
        }
    }

    @ViewBuilder
    private func statusPill(for model: DownloadableModel) -> some View {
        if model.downloaded {
            Text("downloaded")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.18))
                .clipShape(Capsule())
        } else if model.isDownloading {
            Text("\(Int((model.progress * 100).rounded()))%")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.orange.opacity(0.18))
                .clipShape(Capsule())
        } else {
            Text("not downloaded")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.gray.opacity(0.16))
                .clipShape(Capsule())
        }
    }
}

private struct CareyRequiredGroupStatus: Identifiable {
    let id: String
    let label: String
    let relativePrefix: String
    let downloadTarget: CareyDownloadTarget?
    let downloadedFileCount: Int
    let totalFileCount: Int
    let downloadedBytes: Int64

    var isComplete: Bool {
        totalFileCount > 0 && downloadedFileCount == totalFileCount
    }
}

private struct HoverExpandableScreenshot: View {
    let screenshotPath: String
    let caption: String

    @StateObject private var previewWindow = DetachedScreenshotPreviewWindow()
    @State private var isHoveringScreenshot = false

    var body: some View {
        if let screenshot = NSImage(contentsOfFile: screenshotPath) {
            VStack(alignment: .leading, spacing: 6) {
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Image(nsImage: screenshot)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 320, height: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2))
                    )
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(_):
                            guard !isHoveringScreenshot else { return }
                            isHoveringScreenshot = true
                            previewWindow.show(
                                image: screenshot,
                                title: "step 2 detailed preview"
                            )
                        case .ended:
                            isHoveringScreenshot = false
                            previewWindow.hide()
                        }
                    }
            }
            .onAppear { previewWindow.prepare() }
            .onDisappear { previewWindow.hide() }
        }
    }
}

@MainActor
private final class DetachedScreenshotPreviewWindow: ObservableObject {
    private var panel: NSPanel?

    func prepare() {
        _ = ensurePanel()
    }

    func show(image: NSImage, title: String) {
        let panel = ensurePanel()
        let imageSize = image.size
        let aspectRatio = imageSize.width / max(imageSize.height, 1)

        let maxPreviewHeight: CGFloat = 700
        let maxPreviewWidth: CGFloat = 520
        let minPreviewWidth: CGFloat = 320

        let previewHeight: CGFloat
        let previewWidth: CGFloat
        if aspectRatio >= 1 {
            previewWidth = maxPreviewWidth
            previewHeight = min(maxPreviewHeight, previewWidth / aspectRatio)
        } else {
            previewHeight = maxPreviewHeight
            previewWidth = max(minPreviewWidth, min(maxPreviewWidth, previewHeight * aspectRatio))
        }

        panel.contentViewController = NSHostingController(
            rootView: DetachedScreenshotContent(
                title: title,
                image: image,
                previewWidth: previewWidth,
                previewHeight: previewHeight
            )
        )
        panel.setContentSize(
            NSSize(
                width: previewWidth + 24,
                height: previewHeight + 72
            )
        )
        position(panel: panel)
        panel.orderFront(nil)
    }

    func hide() {
        panel?.orderOut(nil)
    }

    private func ensurePanel() -> NSPanel {
        if let panel {
            return panel
        }

        let panel = NSPanel(
            contentRect: .init(x: 0, y: 0, width: 420, height: 560),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.hasShadow = true
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.ignoresMouseEvents = true
        self.panel = panel
        return panel
    }

    private func position(panel: NSPanel) {
        let mouse = NSEvent.mouseLocation
        let fallbackScreen = NSScreen.main ?? NSScreen.screens.first
        let screen = NSScreen.screens.first(where: { NSMouseInRect(mouse, $0.frame, false) }) ?? fallbackScreen
        guard let screen else { return }

        let visible = screen.visibleFrame
        let size = panel.frame.size
        let padding: CGFloat = 12
        let xOffset: CGFloat = 24
        let yOffset: CGFloat = 18

        var x = mouse.x + xOffset
        if x + size.width > visible.maxX - padding {
            x = mouse.x - size.width - xOffset
        }
        x = min(max(x, visible.minX + padding), visible.maxX - size.width - padding)

        var y = mouse.y - size.height + yOffset
        if y < visible.minY + padding {
            y = visible.minY + padding
        }
        if y + size.height > visible.maxY - padding {
            y = visible.maxY - size.height - padding
        }

        panel.setFrameOrigin(.init(x: x, y: y))
    }
}

private struct DetachedScreenshotContent: View {
    let title: String
    let image: NSImage
    let previewWidth: CGFloat
    let previewHeight: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text("check: 'read access to contents of all public gated repos you can access'")
                .font(.caption)
                .foregroundStyle(.secondary)
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
                .frame(width: previewWidth, height: previewHeight)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThickMaterial)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.15))
        )
    }
}

private struct LogTextView: NSViewRepresentable {
    let text: String
    let placeholder: String
    let isPinnedToBottom: Bool
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
            forceScrollToBottom: isPinnedToBottom
        )
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.onPinnedToBottomChanged = onPinnedToBottomChanged
        context.coordinator.applyDisplayedText(
            displayedText,
            in: scrollView,
            forceScrollToBottom: isPinnedToBottom
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
            forceScrollToBottom: Bool
        ) {
            guard let textView else { return }
            guard textView.string != nextText else {
                if forceScrollToBottom {
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

            if forceScrollToBottom || wasPinnedToBottom {
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

struct MenuBarContentView: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    let onOpenMainWindow: () -> Void
    let onMenuPresented: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let manager = viewModel.manager {
                Text("services")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(manager.services) { runtime in
                    MenuBarServiceRow(
                        runtime: runtime,
                        displayName: displayName(for: runtime)
                    ) {
                        if runtime.isRunning {
                            manager.stop(serviceID: runtime.id)
                        } else {
                            manager.start(serviceID: runtime.id)
                        }
                    }
                }
            } else {
                Text("manifest load failed")
                if let error = viewModel.startupError {
                    Text(error).font(.caption)
                }
            }

            Divider()
            HStack {
                Button("open control center", action: onOpenMainWindow)
                Spacer()
                Button("quit") { NSApp.terminate(nil) }
            }
        }
        .padding(12)
        .frame(minWidth: 360)
        .onAppear {
            onMenuPresented()
        }
    }
}

private struct MenuBarServiceRow: View {
    let runtime: ServiceRuntime
    let displayName: String
    let onToggleRunning: () -> Void

    private var actionLabel: String {
        runtime.isRunning ? "stop" : "start"
    }

    private var actionRole: ButtonRole? {
        runtime.isRunning ? .destructive : nil
    }

    private var actionDisabled: Bool {
        runtime.isBootstrapping || runtime.processState == .starting || runtime.processState == .stopping
    }

    private var statusMessage: String {
        if runtime.bootstrapState == .running {
            return "rebuilding environment..."
        }
        if runtime.bootstrapState == .failed {
            return "environment rebuild failed"
        }

        switch runtime.processState {
        case .running:
            switch runtime.healthState {
            case .healthy:
                return "running / healthy"
            case .unhealthy:
                return "running / unhealthy"
            case .unknown:
                return "running / checking health"
            }
        case .starting:
            return "starting..."
        case .stopping:
            return "stopping..."
        case .failed:
            return "failed"
        case .stopped:
            return "stopped"
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            StatusDot(processState: runtime.processState, healthState: runtime.healthState)
            VStack(alignment: .leading, spacing: 2) {
                Text(displayName)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Text(statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(actionLabel, role: actionRole, action: onToggleRunning)
                .controlSize(.small)
                .disabled(actionDisabled)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.primary.opacity(0.08))
        )
    }
}

private struct ServiceRow: View {
    let runtime: ServiceRuntime
    let displayName: String
    let isSelected: Bool
    let onSelect: () -> Void
    let onStart: () -> Void
    let onStop: () -> Void
    let onRebuildEnv: () -> Void
    let onDownloadModels: (() -> Void)?
    let onManageLoras: (() -> Void)?
    let onTrainLora: (() -> Void)?
    let manageLorasButtonTitle: String
    let canManageDownloads: Bool
    let downloadModelsExtraDisabled: Bool

    private var downloadModelsDisabled: Bool {
        guard onDownloadModels != nil else { return true }
        return !canManageDownloads || downloadModelsExtraDisabled
    }

    private var startStopTitle: String {
        runtime.isRunning ? "stop" : "start"
    }

    private func startStopAction() {
        if runtime.isRunning {
            onStop()
        } else {
            onStart()
        }
    }

    private var startStopDisabled: Bool {
        runtime.isBootstrapping || runtime.processState == .starting || runtime.processState == .stopping
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(displayName)
                            .font(.headline)
                        if let onDownloadModels {
                            Button("download models") {
                                onDownloadModels()
                            }
                            .controlSize(.small)
                            .garyPrimaryButtonStyle()
                            .disabled(downloadModelsDisabled)
                        }
                    }
                    Text("id: \(runtime.id)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let pid = runtime.pid {
                    Text("pid \(pid)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            if let error = runtime.lastError, !error.isEmpty {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if let bootstrapMessage = runtime.bootstrapMessage, !bootstrapMessage.isEmpty {
                Text(bootstrapMessage)
                    .font(.caption)
                    .foregroundStyle(runtime.bootstrapState == .failed ? .red : .secondary)
            }

            HStack {
                Button(startStopTitle, action: startStopAction)
                    .disabled(startStopDisabled)
                Button(
                    runtime.isBootstrapping ? "rebuilding..." : "rebuild env",
                    action: onRebuildEnv
                )
                .disabled(runtime.isRunning || runtime.isBootstrapping || runtime.bootstrapState == .notConfigured)
                if let onManageLoras {
                    Button(manageLorasButtonTitle) {
                        onManageLoras()
                    }
                    .garyPrimaryButtonStyle()
                }
                if let onTrainLora {
                    Button("train lora") {
                        onTrainLora()
                    }
                    .garyPrimaryButtonStyle()
                }
                Spacer()
                Text(
                    "proc: \(runtime.processState.rawValue) / health: \(runtime.healthState.rawValue) / env: \(runtime.bootstrapState.rawValue)"
                )
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .onTapGesture(perform: onSelect)
    }
}

private func displayName(for runtime: ServiceRuntime) -> String {
    switch runtime.id {
    case "audiocraft_mlx":
        return "gary (musicgen)"
    case "melodyflow":
        return "terry (melodyflow)"
    case "sa3":
        return "sa3 (stable audio 3)"
    case "stable_audio":
        return "jerry (stable audio)"
    case "carey":
        return "carey (ace lego)"
    case "foundation":
        return "foundation-1"
    default:
        return runtime.service.name
    }
}

private func displayName(forServiceID serviceID: String, fallback: String) -> String {
    switch serviceID {
    case "audiocraft_mlx":
        return "gary (musicgen)"
    case "melodyflow":
        return "terry (melodyflow)"
    case "sa3":
        return "sa3 (stable audio 3)"
    case "stable_audio":
        return "jerry (stable audio)"
    case "carey":
        return "carey (ace lego)"
    case "foundation":
        return "foundation-1"
    default:
        return fallback
    }
}

private struct StatusDot: View {
    let processState: ProcessState
    let healthState: HealthState

    private var color: Color {
        switch processState {
        case .running:
            return healthState == .healthy ? .green : .orange
        case .starting, .stopping:
            return .yellow
        case .failed:
            return .red
        case .stopped:
            return .gray
        }
    }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 10, height: 10)
    }
}
