import Foundation
import Combine
import AppKit

@MainActor
final class YueyController: ObservableObject {
    @Published var settings = YueySettings.load()
    @Published private(set) var installedTiers: [String] = []
    @Published private(set) var sharedReady = false
    @Published private(set) var recommendedTier = "Q4_K_M"
    @Published private(set) var isBusy = false
    @Published private(set) var progress: Double = 0
    @Published private(set) var message = ""
    @Published private(set) var filesOnDisk: Set<String> = []
    private(set) var service: ResolvedService?
    private var task: Task<Void, Never>?
    private var operationID: UUID?
    var canMutate: () -> Bool = { false }
    var operationChanged: (Bool) -> Void = { _ in }
    deinit { task?.cancel() }

    var directory: URL? { service.map(YueyModelCatalog.modelDirectory) }
    func configure(service: ResolvedService?) {
        cancel()
        self.service = service
        refresh()
    }
    func refresh() {
        guard let service, let directory else { return }
        installedTiers = YueyModelCatalog.installedTiers(in: directory)
        sharedReady = YueyModelCatalog.shared.allSatisfy { YueyModelCatalog.installed($0, in: directory) }
        filesOnDisk = Set((YueyModelCatalog.shared + YueyModelCatalog.generation).filter {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0.name).path)
        }.map(\.name))
        recommendedTier = YueyModelCatalog.defaultTier(
            recommended: NativeRuntimeReceipt.read(for: service)?.recommendedEncoding ?? "Q4_K_M")
    }
    func saveSettings(manager: ServiceManager) {
        settings = settings.normalized
        UserDefaults.standard.set(try? JSONEncoder().encode(settings), forKey: YueySettings.defaultsKey)
        if manager.services.first(where: { $0.id == "yuey" })?.isRunning == true {
            manager.restart(serviceID: "yuey")
            message = "settings saved; restarting Yuey. Requests can lower, not raise, the natural duration cap."
        } else { message = "settings saved. Models unload after every job." }
    }
    func cancel() { task?.cancel() }
    func download(tier: String? = nil, includeShared: Bool = false) {
        guard !isBusy, canMutate(), let directory else {
            message = "stop Yuey and finish any runtime install before changing its models."
            return
        }
        var files = includeShared ? YueyModelCatalog.shared : []
        if let tier, let file = YueyModelCatalog.file(for: tier) { files.append(file) }
        guard !files.isEmpty else { return }
        isBusy = true
        let downloadID = UUID()
        operationID = downloadID
        operationChanged(true)
        progress = 0
        message = "verifying existing files before downloading..."
        task = Task { [weak self] in
            guard let self else { return }
            defer {
                self.isBusy = false
                self.operationChanged(false)
                self.task = nil
                self.operationID = nil
                self.refresh()
            }
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                for (index, file) in files.enumerated() {
                    try Task.checkCancellation()
                    let destination = directory.appendingPathComponent(file.name)
                    self.message = "checking \(file.name)..."
                    if YueyModelCatalog.installed(file, in: directory) {
                        let hash = try await VerifiedNativeDownload.hashFile(destination)
                        try Task.checkCancellation()
                        if hash == file.sha256 {
                            self.progress = Double(index + 1) / Double(files.count)
                            continue
                        }
                    }
                    let staging = directory.appendingPathComponent(".download-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: staging) }
                    let filesystem = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
                    let available = (filesystem[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
                    guard available >= file.bytes + 268_435_456 else {
                        throw NativeServiceError.invalid("Not enough free space for \(file.name). Need at least \(ByteCountFormatter.string(fromByteCount: file.bytes + 268_435_456, countStyle: .file)).")
                    }
                    let url = URL(string: "https://huggingface.co/\(YueyModelCatalog.repository)/resolve/\(YueyModelCatalog.revision)/\(file.name)")!
                    self.message = "downloading \(file.name)..."
                    let count = files.count
                    let owner = self
                    try await VerifiedNativeDownload.fetch(url, to: staging, sha256: file.sha256, expectedBytes: file.bytes) { fraction in
                        Task { @MainActor in
                            guard owner.operationID == downloadID else { return }
                            owner.progress = (Double(index) + fraction) / Double(count)
                        }
                    }
                    try Task.checkCancellation()
                    let downloadedSize = try staging.resourceValues(forKeys: [.fileSizeKey]).fileSize
                    guard downloadedSize == Int(file.bytes) else {
                        throw NativeServiceError.invalid("Unexpected model file size: \(file.name).")
                    }
                    var backup: URL?
                    if FileManager.default.fileExists(atPath: destination.path) {
                        let previous = directory.appendingPathComponent(".replaced-\(UUID().uuidString)-\(file.name)")
                        try FileManager.default.moveItem(at: destination, to: previous)
                        backup = previous
                    }
                    do { try FileManager.default.moveItem(at: staging, to: destination) }
                    catch {
                        if let backup { try? FileManager.default.moveItem(at: backup, to: destination) }
                        throw error
                    }
                    if let backup { try? FileManager.default.trashItem(at: backup, resultingItemURL: nil) }
                    self.progress = Double(index + 1) / Double(count)
                }
                self.message = "model files verified and ready."
            } catch {
                self.message = Task.isCancelled ? "download cancelled; completed model files were kept." : error.localizedDescription
            }
        }
    }
    // Called only after the model sheet's explicit confirmation. Unknown files,
    // other services' models and the parent directory are never removal targets.
    func remove(_ files: [YueyModelFile]) {
        guard !isBusy, canMutate(), let directory else {
            message = "stop Yuey before removing model files."
            return
        }
        do {
            let allowed = Set((YueyModelCatalog.shared + YueyModelCatalog.generation).map(\.name))
            guard files.allSatisfy({ allowed.contains($0.name) }) else {
                throw NativeServiceError.invalid("Unknown Yuey model removal target.")
            }
            for file in files {
                let target = directory.appendingPathComponent(file.name)
                if FileManager.default.fileExists(atPath: target.path) {
                    try FileManager.default.trashItem(at: target, resultingItemURL: nil)
                }
            }
            message = "model files moved to Trash; they can be recovered there."
        } catch { message = error.localizedDescription }
        refresh()
    }
}
