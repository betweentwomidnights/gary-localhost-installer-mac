import SwiftUI
import AppKit

struct ControlCenterView: View {
    @ObservedObject var viewModel: ControlCenterViewModel

    var body: some View {
        Group {
            if let error = viewModel.startupError {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Control Center failed to load manifest.")
                        .font(.title3.weight(.semibold))
                    Text(error)
                        .font(.body.monospaced())
                    Text("Manifest path:")
                        .font(.caption)
                    Text(viewModel.manifestPath)
                        .font(.caption.monospaced())
                    Button("Retry") { viewModel.loadManifest() }
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
                ProgressView("Loading...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    @ViewBuilder
    private func serviceList(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Gary Localhost Services")
                .font(.title3.weight(.semibold))
            Text(viewModel.manifestPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            HStack {
                Button(
                    manager.isRebuildingAllEnvironments
                    ? "Rebuilding Environments..."
                    : "Rebuild All Environments"
                ) {
                    manager.rebuildAllEnvironments()
                }
                .disabled(manager.isRebuildingAllEnvironments || manager.services.isEmpty)
                Spacer()
            }

            List(manager.services) { runtime in
                ServiceRow(
                    runtime: runtime,
                    isSelected: viewModel.selectedServiceID == runtime.id,
                    onSelect: { viewModel.selectService(runtime.id) },
                    onStart: { manager.start(serviceID: runtime.id) },
                    onStop: { manager.stop(serviceID: runtime.id) },
                    onRestart: { manager.restart(serviceID: runtime.id) },
                    onRebuildEnv: { manager.rebuildEnvironment(serviceID: runtime.id) }
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
                }

                Text("Logs: \(runtime.service.name)")
                    .font(.headline)
                Text(runtime.service.logFile.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)

                ScrollView {
                    Text(viewModel.selectedLogText.isEmpty ? "(no log output yet)" : viewModel.selectedLogText)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .background(Color(NSColor.textBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.gray.opacity(0.2))
                )

                HStack {
                    Button("Refresh") { viewModel.refreshLog() }
                    Button("Open Log File") { NSWorkspace.shared.open(runtime.service.logFile) }
                    Spacer()
                    Button("Clear Selection") { viewModel.selectedServiceID = nil }
                }
            } else {
                Text("Select a service to view logs.")
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
    }

    @ViewBuilder
    private func stableAudioAuthPanel() -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Stable Audio Access Setup")
                .font(.headline)
            Text(
                viewModel.stableAudioTokenConfigured
                ? "Hugging Face token is configured in Keychain."
                : "Hugging Face token is missing."
            )
            .font(.caption)
            .foregroundStyle(viewModel.stableAudioTokenConfigured ? .green : .orange)

            if !viewModel.stableAudioTokenConfigured {
                stepRow(
                    number: 1,
                    title: "Open model page and click 'Agree and access repository'",
                    buttonTitle: "Open Stable Audio Page",
                    isEnabled: true
                ) {
                    NSWorkspace.shared.open(StableAudioAuthLinks.modelPage)
                }

                stepRow(
                    number: 2,
                    title: "Create a read token on Hugging Face",
                    buttonTitle: "Open Token Settings",
                    isEnabled: true
                ) {
                    NSWorkspace.shared.open(StableAudioAuthLinks.tokenPage)
                }

                if let screenshotPath = viewModel.stableAudioStep2ScreenshotPath {
                    HoverExpandableScreenshot(
                        screenshotPath: screenshotPath,
                        caption: "Step 2 reference: check 'Read access to contents of all public gated repos you can access' (hover to zoom)"
                    )
                }

                HStack {
                    Text("3. Paste token here")
                        .font(.subheadline)
                    Spacer()
                }
                SecureField("hf_...", text: $viewModel.stableAudioTokenInput)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                if !viewModel.stableAudioTokenConfigured {
                    Button("Save Token") { viewModel.saveStableAudioToken() }
                }
                Button("Remove Token", role: .destructive) { viewModel.clearStableAudioToken() }
                    .disabled(!viewModel.stableAudioTokenConfigured)
                Button("Refresh Token Status") { viewModel.refreshStableAudioTokenState() }
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

private struct HoverExpandableScreenshot: View {
    let screenshotPath: String
    let caption: String

    @StateObject private var previewWindow = DetachedScreenshotPreviewWindow()

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
                    .onHover { hovering in
                        if hovering {
                            previewWindow.show(
                                image: screenshot,
                                title: "Step 2 detailed preview"
                            )
                        } else {
                            previewWindow.hide()
                        }
                    }
            }
            .onDisappear { previewWindow.hide() }
        }
    }
}

@MainActor
private final class DetachedScreenshotPreviewWindow: ObservableObject {
    private var panel: NSPanel?

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
            Text("Check: 'Read access to contents of all public gated repos you can access'")
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

struct MenuBarContentView: View {
    @ObservedObject var viewModel: ControlCenterViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let manager = viewModel.manager {
                ForEach(manager.services) { runtime in
                    HStack {
                        StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                        Text(runtime.service.name)
                        Spacer()
                        if runtime.isRunning {
                            Button("Stop") { manager.stop(serviceID: runtime.id) }
                        } else {
                            Button("Start") { manager.start(serviceID: runtime.id) }
                        }
                    }
                }
            } else {
                Text("Manifest load failed")
                if let error = viewModel.startupError {
                    Text(error).font(.caption)
                }
            }

            Divider()
            if let manager = viewModel.manager {
                Button(
                    manager.isRebuildingAllEnvironments
                    ? "Rebuilding All Envs..."
                    : "Rebuild All Envs"
                ) {
                    manager.rebuildAllEnvironments()
                }
                .disabled(manager.isRebuildingAllEnvironments)
            }
            Button("Reload Manifest") { viewModel.loadManifest() }
            Button("Quit") { NSApp.terminate(nil) }
        }
        .padding(12)
        .frame(minWidth: 360)
    }
}

private struct ServiceRow: View {
    let runtime: ServiceRuntime
    let isSelected: Bool
    let onSelect: () -> Void
    let onStart: () -> Void
    let onStop: () -> Void
    let onRestart: () -> Void
    let onRebuildEnv: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                VStack(alignment: .leading, spacing: 2) {
                    Text(runtime.service.name).font(.headline)
                    Text("id: \(runtime.id)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let pid = runtime.pid {
                    Text("PID \(pid)")
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
                Button("Start", action: onStart)
                    .disabled(runtime.isRunning || runtime.isBootstrapping)
                Button("Stop", action: onStop)
                    .disabled(!runtime.isRunning)
                Button("Restart", action: onRestart)
                    .disabled(runtime.isBootstrapping)
                Button(
                    runtime.isBootstrapping ? "Rebuilding..." : "Rebuild Env",
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
