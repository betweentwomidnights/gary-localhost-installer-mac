import Foundation
import SwiftUI

@MainActor
final class ControlCenterViewModel: ObservableObject {
    @Published var manager: ServiceManager?
    @Published var startupError: String?
    @Published var manifestPath: String = ""
    @Published var selectedServiceID: String?
    @Published var selectedLogText: String = ""

    private var logRefreshTask: Task<Void, Never>?

    init() {
        loadManifest()
    }

    deinit {
        logRefreshTask?.cancel()
    }

    func loadManifest() {
        let defaultURL = ManifestLoader.defaultManifestURL()
        manifestPath = defaultURL.path

        do {
            let manifest = try ManifestLoader.load(from: defaultURL)
            let manager = ServiceManager(manifest: manifest)
            self.manager = manager
            startupError = nil
            selectedServiceID = manager.services.first?.id
            selectedLogText = selectedServiceID.map { manager.readLogTail(serviceID: $0) } ?? ""
            manager.startAutoStartServices()
            startLogRefreshLoop()
        } catch {
            self.manager = nil
            startupError = error.localizedDescription
        }
    }

    func selectService(_ serviceID: String) {
        selectedServiceID = serviceID
        refreshLog()
    }

    func refreshLog() {
        guard let manager, let selectedServiceID else {
            selectedLogText = ""
            return
        }
        selectedLogText = manager.readLogTail(serviceID: selectedServiceID)
    }

    private func startLogRefreshLoop() {
        logRefreshTask?.cancel()
        logRefreshTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await MainActor.run {
                    self?.refreshLog()
                }
            }
        }
    }
}
