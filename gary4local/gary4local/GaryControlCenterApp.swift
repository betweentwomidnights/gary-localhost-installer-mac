import SwiftUI
import AppKit
import Combine

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var viewModel: ControlCenterViewModel?
    private lazy var dockMenu: NSMenu = {
        let menu = NSMenu(title: "gary4local")
        menu.autoenablesItems = false
        return menu
    }()

    private func presentMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        let opened = NSApp.sendAction(Selector(("newWindow:")), to: nil, from: nil)
        if !opened {
            NSApp.sendAction(#selector(NSApplication.arrangeInFront(_:)), to: nil, from: nil)
        }
    }

    private func pruneTopLevelMenus() {
        guard let mainMenu = NSApp.mainMenu else { return }
        let titlesToRemove = ["Format"]
        for title in titlesToRemove {
            if let index = mainMenu.items.firstIndex(where: { $0.title.caseInsensitiveCompare(title) == .orderedSame }) {
                mainMenu.removeItem(at: index)
            }
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            self.presentMainWindow()
            self.pruneTopLevelMenus()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                self.pruneTopLevelMenus()
            }
        }
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        pruneTopLevelMenus()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            presentMainWindow()
        }
        return true
    }

    @objc(applicationDockMenu:)
    dynamic
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        return buildDockMenu()
    }

    private func buildDockMenu() -> NSMenu {
        dockMenu.removeAllItems()

        let openItem = NSMenuItem(
            title: "open control center",
            action: #selector(openMainWindowFromMenu(_:)),
            keyEquivalent: ""
        )
        openItem.target = self
        dockMenu.addItem(openItem)

        if let services = viewModel?.manager?.services, !services.isEmpty {
            dockMenu.addItem(.separator())

            let servicesHeader = NSMenuItem(title: "services", action: nil, keyEquivalent: "")
            servicesHeader.isEnabled = false
            dockMenu.addItem(servicesHeader)

            for runtime in services {
                dockMenu.addItem(makeDockStatusItem(for: runtime))
                dockMenu.addItem(makeDockActionItem(for: runtime))
            }
        }
        return dockMenu
    }

    @objc
    private func openMainWindowFromMenu(_ sender: Any?) {
        DispatchQueue.main.async {
            self.presentMainWindow()
        }
    }

    @objc
    private func toggleServiceFromDockMenu(_ sender: NSMenuItem) {
        guard let serviceID = sender.representedObject as? String,
              let manager = viewModel?.manager,
              let runtime = manager.services.first(where: { $0.id == serviceID }) else {
            return
        }

        if runtime.isRunning {
            manager.stop(serviceID: serviceID)
        } else {
            manager.start(serviceID: serviceID)
        }
    }

    private func makeDockStatusItem(for runtime: ServiceRuntime) -> NSMenuItem {
        let statusItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        statusItem.attributedTitle = attributedDockStatusTitle(for: runtime)
        return statusItem
    }

    private func makeDockActionItem(for runtime: ServiceRuntime) -> NSMenuItem {
        let name = dockDisplayName(for: runtime)
        let item: NSMenuItem

        if runtime.isBootstrapping {
            item = NSMenuItem(
                title: "rebuilding \(name)...",
                action: nil,
                keyEquivalent: ""
            )
            item.isEnabled = false
            return item
        }

        if runtime.processState == .starting || runtime.processState == .stopping {
            item = NSMenuItem(
                title: "working on \(name)...",
                action: nil,
                keyEquivalent: ""
            )
            item.isEnabled = false
            return item
        }

        item = NSMenuItem(
            title: runtime.isRunning ? "stop \(name)" : "start \(name)",
            action: #selector(toggleServiceFromDockMenu(_:)),
            keyEquivalent: ""
        )
        item.target = self
        item.representedObject = runtime.id
        return item
    }

    private func attributedDockStatusTitle(for runtime: ServiceRuntime) -> NSAttributedString {
        let dotAttributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: dockStatusColor(for: runtime)
        ]
        let textAttributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.labelColor
        ]

        let title = NSMutableAttributedString(
            string: "● ",
            attributes: dotAttributes
        )
        title.append(
            NSAttributedString(
                string: "\(dockDisplayName(for: runtime))  \(dockStatusSummary(for: runtime))",
                attributes: textAttributes
            )
        )
        return title
    }

    private func dockDisplayName(for runtime: ServiceRuntime) -> String {
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

    private func dockStatusSummary(for runtime: ServiceRuntime) -> String {
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

    private func dockStatusColor(for runtime: ServiceRuntime) -> NSColor {
        switch runtime.processState {
        case .running:
            return runtime.healthState == .healthy ? .systemGreen : .systemOrange
        case .starting, .stopping:
            return .systemYellow
        case .failed:
            return .systemRed
        case .stopped:
            return .systemGray
        }
    }
}

private struct GaryHelpView: View {
    private func iconLink(_ title: String, imageName: String, destination: String) -> some View {
        Link(destination: URL(string: destination)!) {
            Label {
                Text(title)
            } icon: {
                Image(imageName)
                    .renderingMode(.original)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 14, height: 14)
            }
        }
        .buttonStyle(.plain)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("gary4local help")
                        .font(.title2.bold())
                    Text("v1.0 guidance for backend selection, support, and model references.")
                        .foregroundStyle(.secondary)
                }

                GroupBox("backend notes") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("mps is usually the safest default on macOS Sequoia.")
                        Text("mlx is expected to become increasingly preferred on Tahoe-era Apple Silicon systems.")
                        Text("on some Tahoe setups, certain torch/mps operations can fall back to cpu. if that happens, try mlx.")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("implementation repositories") {
                    VStack(alignment: .leading, spacing: 8) {
                        Link("stable-audio-mlx", destination: URL(string: "https://github.com/betweentwomidnights/stable-audio-mlx")!)
                        Link("melodyflow-mlx", destination: URL(string: "https://github.com/betweentwomidnights/melodyflow-mlx")!)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("model architecture references") {
                    VStack(alignment: .leading, spacing: 8) {
                        Link("gary (musicgen): facebookresearch/audiocraft", destination: URL(string: "https://github.com/facebookresearch/audiocraft")!)
                        Link("jerry (stable audio): stabilityAI/stable-audio-tools", destination: URL(string: "https://github.com/stabilityAI/stable-audio-tools")!)
                        Link("jerry model repo: stable-audio-open-small", destination: URL(string: "https://huggingface.co/stabilityAI/stable-audio-open-small")!)
                        Link("terry (melodyflow) project page", destination: URL(string: "https://huggingface.co/spaces/facebook/Melodyflow")!)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("support") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("discord is the preferred support channel.")
                        HStack(spacing: 12) {
                            iconLink("join discord", imageName: "DiscordIcon", destination: "https://discord.gg/xUkpsKNvM6")
                            Link(destination: URL(string: "mailto:kev@thecollabagepatch.com")!) {
                                Label("email support", systemImage: "envelope.fill")
                            }
                            .buttonStyle(.plain)
                        }
                        Text("finetunes for audiocraft musicgen and stable-audio-open-small are supported with minor adjustments. ask in discord for tuning guidance.")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(20)
        }
    }
}

private struct GaryAboutView: View {
    private var appIconImage: NSImage {
        if let iconURL = Bundle.main.url(forResource: "icon", withExtension: "icns"),
           let icon = NSImage(contentsOf: iconURL) {
            return icon
        }
        return NSApp.applicationIconImage
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 12) {
                Image(nsImage: appIconImage)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text("about gary4local")
                        .font(.title2.bold())
                    Text("local backend control center")
                        .foregroundStyle(.secondary)
                }
            }

            Text("gary4local runs local backend services for gary4juce.")
                .foregroundStyle(.secondary)

            Text("this v1.0 flow is tested against gary4juce v2.2.")
            Link("gary4juce releases", destination: URL(string: "https://github.com/betweentwomidnights/gary4juce")!)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("support")
                    .font(.headline)
                HStack(spacing: 12) {
                    Link("discord", destination: URL(string: "https://discord.gg/xUkpsKNvM6")!)
                    Link("email", destination: URL(string: "mailto:kev@thecollabagepatch.com")!)
                }
            }

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(20)
    }
}

private struct GaryCoreCommands: Commands {
    let controlCenterWindowID: String
    let aboutWindowID: String
    @Environment(\.openWindow) private var openWindow

    private func activateAndOpen(windowID: String) {
        NSApp.activate(ignoringOtherApps: true)
        openWindow(id: windowID)
    }

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("about gary4local") {
                activateAndOpen(windowID: aboutWindowID)
            }
        }

        CommandGroup(replacing: .newItem) {
            Button("open control center") {
                activateAndOpen(windowID: controlCenterWindowID)
            }
            .keyboardShortcut("o", modifiers: [.command])
        }
    }
}

private struct GaryPrunedDefaultMenusCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .saveItem) {}
        CommandGroup(replacing: .importExport) {}
        CommandGroup(replacing: .printItem) {}
        CommandGroup(replacing: .undoRedo) {}
        CommandGroup(replacing: .pasteboard) {}
        CommandGroup(replacing: .textEditing) {}
        CommandGroup(replacing: .textFormatting) {}
        CommandGroup(replacing: .toolbar) {}
        CommandGroup(replacing: .sidebar) {}
        CommandGroup(replacing: .windowArrangement) {}
    }
}

private struct GaryHelpCommands: Commands {
    let helpWindowID: String
    @Environment(\.openWindow) private var openWindow

    private func openHelp() {
        NSApp.activate(ignoringOtherApps: true)
        openWindow(id: helpWindowID)
    }

    var body: some Commands {
        CommandGroup(replacing: .help) {
            Button("gary4local help") {
                openHelp()
            }
            Divider()
            Link("join discord", destination: URL(string: "https://discord.gg/xUkpsKNvM6")!)
            Link("github", destination: URL(string: "https://github.com/betweentwomidnights/gary-localhost-installer-mac")!)
            Link("email support", destination: URL(string: "mailto:kev@thecollabagepatch.com")!)
        }
    }
}

@main
struct GaryControlCenterApp: App {
    private enum FirstUseHelperStage: String {
        case rebuild
        case menuBar
    }

    private enum MenuBarIconStyle {
        case brandColor
        case monochrome
    }

    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @Environment(\.openWindow) private var openWindow
    @StateObject private var viewModel = ControlCenterViewModel()
    @AppStorage("firstUseHelperDismissedV1") private var firstUseHelperDismissed = false
    @AppStorage("firstUseHelperStageV1") private var firstUseHelperStageRawValue = FirstUseHelperStage.rebuild.rawValue
    @AppStorage("firstUseMenuBarOpenedV1") private var firstUseMenuBarOpened = false
    @State private var menuBarAttentionHighlightVisible = false
    private let menuBarIconName = "MenuBarIcon"
    private let menuBarFallbackSymbolName = "waveform"
    private let menuBarIconStyle: MenuBarIconStyle = .brandColor
    private let controlCenterWindowID = "control-center"
    private let helpWindowID = "help-center"
    private let aboutWindowID = "about-window"
    private let menuBarAttentionTimer = Timer.publish(every: 0.75, on: .main, in: .common).autoconnect()

    private var shouldAnimateMenuBarIconAttention: Bool {
        !firstUseHelperDismissed
        && firstUseHelperStageRawValue == FirstUseHelperStage.menuBar.rawValue
        && !firstUseMenuBarOpened
    }

    private var activeMenuBarIconStyle: MenuBarIconStyle {
        shouldAnimateMenuBarIconAttention && menuBarAttentionHighlightVisible
        ? .monochrome
        : menuBarIconStyle
    }

    private func openMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async {
            openWindow(id: controlCenterWindowID)
        }
    }

    private func handleMenuBarPresented() {
        guard shouldAnimateMenuBarIconAttention else { return }
        firstUseMenuBarOpened = true
        menuBarAttentionHighlightVisible = false
    }

    private func updateMenuBarAttentionState(for shouldAnimate: Bool) {
        if shouldAnimate {
            menuBarAttentionHighlightVisible = true
        } else {
            menuBarAttentionHighlightVisible = false
        }
    }

    private func tickMenuBarAttention() {
        guard shouldAnimateMenuBarIconAttention else {
            if menuBarAttentionHighlightVisible {
                menuBarAttentionHighlightVisible = false
            }
            return
        }
        withAnimation(.easeInOut(duration: 0.2)) {
            menuBarAttentionHighlightVisible.toggle()
        }
    }

    @ViewBuilder
    private func menuBarIconImage(style: MenuBarIconStyle) -> some View {
        if NSImage(named: NSImage.Name(menuBarIconName)) != nil {
            switch style {
            case .brandColor:
                Image(menuBarIconName)
                    .renderingMode(.original)
                    .interpolation(.high)
            case .monochrome:
                Image(menuBarIconName)
                    .renderingMode(.template)
                    .interpolation(.high)
            }
        } else {
            Image(systemName: menuBarFallbackSymbolName)
        }
    }

    @ViewBuilder
    private var menuBarIconLabel: some View {
        ZStack {
            menuBarIconImage(style: activeMenuBarIconStyle)
            if shouldAnimateMenuBarIconAttention && menuBarAttentionHighlightVisible {
                Circle()
                    .stroke(Color.red.opacity(0.9), lineWidth: 1.25)
                    .frame(width: 17, height: 17)
                    .shadow(color: Color.red.opacity(0.45), radius: 2)
            }
        }
        .frame(width: 18, height: 18)
    }

    var body: some Scene {
        WindowGroup("gary4local", id: controlCenterWindowID) {
            ControlCenterView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 620)
                .onAppear {
                    appDelegate.viewModel = viewModel
                    if (NSApp.delegate as AnyObject?) !== appDelegate {
                        NSApp.delegate = appDelegate
                    }
                }
        }
        .defaultSize(width: 1180, height: 720)
        .commands {
            GaryCoreCommands(
                controlCenterWindowID: controlCenterWindowID,
                aboutWindowID: aboutWindowID
            )
            GaryPrunedDefaultMenusCommands()
            GaryHelpCommands(helpWindowID: helpWindowID)
        }

        WindowGroup("gary4local help", id: helpWindowID) {
            GaryHelpView()
                .frame(minWidth: 720, minHeight: 560)
        }
        .defaultSize(width: 780, height: 620)

        WindowGroup("about gary4local", id: aboutWindowID) {
            GaryAboutView()
                .frame(minWidth: 460, minHeight: 320)
        }
        .defaultSize(width: 520, height: 360)

        MenuBarExtra {
            MenuBarContentView(
                viewModel: viewModel,
                onOpenMainWindow: openMainWindow,
                onMenuPresented: handleMenuBarPresented
            )
        } label: {
            menuBarIconLabel
                .simultaneousGesture(
                    TapGesture().onEnded {
                        handleMenuBarPresented()
                    }
                )
                .onReceive(menuBarAttentionTimer) { _ in
                    tickMenuBarAttention()
                }
                .onChange(of: shouldAnimateMenuBarIconAttention, initial: true) { _, shouldAnimate in
                    updateMenuBarAttentionState(for: shouldAnimate)
                }
        }
        .menuBarExtraStyle(.window)
    }
}
