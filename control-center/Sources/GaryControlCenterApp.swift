import SwiftUI

@main
struct GaryControlCenterApp: App {
    @StateObject private var viewModel = ControlCenterViewModel()

    var body: some Scene {
        MenuBarExtra("Gary Localhost", systemImage: "slider.horizontal.3") {
            MenuBarContentView(viewModel: viewModel)
        }

        Window("Gary Localhost Control Center", id: "main") {
            ControlCenterView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 620)
        }
        .defaultSize(width: 1180, height: 720)
    }
}
