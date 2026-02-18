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

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            self.presentMainWindow()
        }
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
