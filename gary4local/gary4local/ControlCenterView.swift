import SwiftUI
import AppKit
import Combine

struct ControlCenterView: View {
    @ObservedObject var viewModel: ControlCenterViewModel

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
        .sheet(isPresented: $viewModel.isModelDownloadSheetPresented) {
            ModelDownloadSheet(viewModel: viewModel)
        }
        .sheet(item: $viewModel.rebuildFailureReport) { report in
            RebuildFailureSheet(viewModel: viewModel, report: report)
        }
    }

    @ViewBuilder
    private func serviceList(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(viewModel.manifestPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            HStack {
                Button(
                    manager.isRebuildingAllEnvironments
                    ? "rebuilding environments..."
                    : "rebuild all environments"
                ) {
                    manager.rebuildAllEnvironments()
                }
                .disabled(manager.isRebuildingAllEnvironments || manager.services.isEmpty)
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
                    onRestart: { manager.restart(serviceID: runtime.id) },
                    onRebuildEnv: { manager.rebuildEnvironment(serviceID: runtime.id) },
                    onDownloadModels: runtime.id == "audiocraft_mlx" ? {
                        viewModel.openModelDownloadSheet()
                    } : nil
                )
            }
            .listStyle(.inset)
        }
        .padding(16)
    }

    @ViewBuilder
    private func logPane(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let selectedID = viewModel.selectedServiceID,
               let runtime = manager.services.first(where: { $0.id == selectedID }) {
                if runtime.id == "stable_audio" {
                    stableAudioAuthPanel()
                    Divider()
                } else if runtime.id == "melodyflow" {
                    melodyFlowBackendPanel()
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
                    NSWorkspace.shared.open(StableAudioAuthLinks.modelPage)
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
                                .help("recreate the environment and install dependencies with --no-cache-dir.")
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
                            .buttonStyle(.borderedProminent)
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

private struct ModelDownloadSheet: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("download models")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("close") { dismiss() }
            }

            Text("pre-download models for offline use in gary4juce.")
                .font(.subheadline)
            Text("models also download the first time you use them inside gary4juce.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if !viewModel.modelDownloadStatusMessage.isEmpty {
                Text(viewModel.modelDownloadStatusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

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
                        Text("start the audiocraft mlx service, then refresh statuses.")
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
        .padding(16)
        .frame(minWidth: 760, minHeight: 520)
        .onAppear {
            if viewModel.downloadableModels.isEmpty {
                viewModel.refreshModelCatalogAndStatuses()
            }
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

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let manager = viewModel.manager {
                ForEach(manager.services) { runtime in
                    HStack {
                        StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                        Text(displayName(for: runtime))
                        Spacer()
                        if runtime.isRunning {
                            Button("stop") { manager.stop(serviceID: runtime.id) }
                        } else {
                            Button("start") { manager.start(serviceID: runtime.id) }
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
            Button("open control center", action: onOpenMainWindow)
            if let manager = viewModel.manager {
                Button(
                    manager.isRebuildingAllEnvironments
                    ? "rebuilding all envs..."
                    : "rebuild all envs"
                ) {
                    manager.rebuildAllEnvironments()
                }
                .disabled(manager.isRebuildingAllEnvironments)
            }
            Button("quit") { NSApp.terminate(nil) }
        }
        .padding(12)
        .frame(minWidth: 360)
    }
}

private struct ServiceRow: View {
    let runtime: ServiceRuntime
    let displayName: String
    let isSelected: Bool
    let onSelect: () -> Void
    let onStart: () -> Void
    let onStop: () -> Void
    let onRestart: () -> Void
    let onRebuildEnv: () -> Void
    let onDownloadModels: (() -> Void)?

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
                            .disabled(!runtime.isRunning || runtime.isBootstrapping)
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
                Button("start", action: onStart)
                    .disabled(runtime.isRunning || runtime.isBootstrapping)
                Button("stop", action: onStop)
                    .disabled(!runtime.isRunning)
                Button("restart", action: onRestart)
                    .disabled(runtime.isBootstrapping)
                Button(
                    runtime.isBootstrapping ? "rebuilding..." : "rebuild env",
                    action: onRebuildEnv
                )
                .disabled(runtime.isRunning || runtime.isBootstrapping || runtime.bootstrapState == .notConfigured)
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
    case "stable_audio":
        return "jerry (stable audio)"
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
    case "stable_audio":
        return "jerry (stable audio)"
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
