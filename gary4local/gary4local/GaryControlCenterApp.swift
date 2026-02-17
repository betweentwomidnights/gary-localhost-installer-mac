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
            title: "Open Control Center",
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
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var viewModel = ControlCenterViewModel()

    private func openMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        let opened = NSApp.sendAction(Selector(("newWindow:")), to: nil, from: nil)
        if !opened {
            NSApp.sendAction(#selector(NSApplication.arrangeInFront(_:)), to: nil, from: nil)
        }
    }

    var body: some Scene {
        WindowGroup("gary4local") {
            ControlCenterView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 620)
        }
        .defaultSize(width: 1180, height: 720)

        MenuBarExtra("gary4local", systemImage: "slider.horizontal.3") {
            MenuBarContentView(
                viewModel: viewModel,
                onOpenMainWindow: openMainWindow
            )
        }
    }
}
