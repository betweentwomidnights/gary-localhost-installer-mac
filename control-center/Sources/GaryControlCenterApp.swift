import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        let menuBarOnly = ProcessInfo.processInfo.environment["GARY_MENU_BAR_ONLY"] == "1"
        NSApp.setActivationPolicy(menuBarOnly ? .accessory : .regular)
        if !menuBarOnly {
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

@main
struct GaryControlCenterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var viewModel = ControlCenterViewModel()

    var body: some Scene {
        WindowGroup("Gary Localhost Control Center") {
            ControlCenterView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 620)
        }
        .defaultSize(width: 1180, height: 720)

        MenuBarExtra("Gary Localhost", systemImage: "slider.horizontal.3") {
            MenuBarContentView(viewModel: viewModel)
        }
    }
}
