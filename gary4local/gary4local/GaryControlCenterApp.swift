import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
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

    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        let menu = NSMenu()
        let openItem = NSMenuItem(
            title: "open control center",
            action: #selector(openMainWindowFromMenu(_:)),
            keyEquivalent: ""
        )
        openItem.target = self
        menu.addItem(openItem)
        return menu
    }

    @objc
    private func openMainWindowFromMenu(_ sender: Any?) {
        DispatchQueue.main.async {
            self.presentMainWindow()
        }
    }
}

@main
struct GaryControlCenterApp: App {
    private enum MenuBarIconStyle {
        case brandColor
        case monochrome
    }

    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var viewModel = ControlCenterViewModel()
    private let menuBarIconName = "MenuBarIcon"
    private let menuBarFallbackSymbolName = "waveform"
    private let menuBarIconStyle: MenuBarIconStyle = .brandColor

    private func openMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        let opened = NSApp.sendAction(Selector(("newWindow:")), to: nil, from: nil)
        if !opened {
            NSApp.sendAction(#selector(NSApplication.arrangeInFront(_:)), to: nil, from: nil)
        }
    }

    @ViewBuilder
    private var menuBarIconLabel: some View {
        if NSImage(named: NSImage.Name(menuBarIconName)) != nil {
            switch menuBarIconStyle {
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

    var body: some Scene {
        WindowGroup("gary4local") {
            ControlCenterView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 620)
        }
        .defaultSize(width: 1180, height: 720)

        MenuBarExtra {
            MenuBarContentView(
                viewModel: viewModel,
                onOpenMainWindow: openMainWindow
            )
        } label: {
            menuBarIconLabel
        }
        .menuBarExtraStyle(.window)
    }
}
