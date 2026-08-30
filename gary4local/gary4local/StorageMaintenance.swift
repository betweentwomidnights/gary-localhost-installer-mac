import Combine
import Foundation

/// Sizes of the things gary4local puts on disk, and the duplicate-space pass.
///
/// The install is mostly weights, and the weights were being stored twice: the
/// downloader fetched into the Hugging Face cache and then copied the file into
/// the checkpoints tree. The downloader clones now, but copies already on disk
/// stay until this runs.
///
/// This is not the same duplication the Windows build reclaims. There the hub
/// cannot symlink without developer mode and writes each file twice inside one
/// repo cache. macOS symlinks fine, so that never happens here.

struct StorageUsage: Equatable {
    var huggingFaceCacheBytes: Int64 = 0
    var careyCheckpointsBytes: Int64 = 0
    var environmentsBytes: Int64 = 0
    var totalBytes: Int64 = 0

    var otherBytes: Int64 {
        max(0, totalBytes - huggingFaceCacheBytes - careyCheckpointsBytes - environmentsBytes)
    }
}

struct StorageReclaimResult: Equatable {
    var matchedBytes: Int64
    var reclaimedBytes: Int64
    var reclaimedFiles: Int
    var skipped: Int
    var freeDeltaBytes: Int64
}

@MainActor
final class StorageMaintenanceManager: ObservableObject {
    @Published private(set) var usage: StorageUsage?
    @Published private(set) var isScanning = false
    @Published private(set) var isReclaiming = false
    @Published private(set) var progressMessage = ""
    @Published private(set) var lastResult: StorageReclaimResult?
    @Published private(set) var errorMessage: String?
    /// Free space on the volume. Instant, and the number that actually moves
    /// when duplicates are reclaimed -- the directory sizes do not, because a
    /// clone still reports its full size on both sides.
    @Published private(set) var volumeFreeBytes: Int64 = 0

    private var process: Process?

    /// Measuring and reclaiming are independent -- an in-progress measurement
    /// used to disable the reclaim buttons, which made them look broken.
    var isBusy: Bool { isScanning || isReclaiming }

    func clearError() { errorMessage = nil }

    // MARK: - Sizing

    /// Directory sizes via `du`, which walks the tree in C and finishes in
    /// about a second on a 200GB install. The Foundation enumerator this
    /// replaced called stat twice per file and took minutes.
    ///
    /// These are logical sizes. A cloned file still reports its full size on
    /// both sides, so reclaiming duplicates does NOT shrink these numbers --
    /// it moves volumeFreeBytes instead. That is why both are shown.
    func scan(appSupportDirectory: URL, checkpointsDirectory: URL?) {
        volumeFreeBytes = Self.freeBytes(at: appSupportDirectory)
        guard !isScanning else { return }
        isScanning = true
        errorMessage = nil

        let hub = appSupportDirectory
            .appendingPathComponent("cache/huggingface", isDirectory: true)
        let envs = appSupportDirectory.appendingPathComponent("venvs", isDirectory: true)
        let checkpoints = checkpointsDirectory
        let root = appSupportDirectory

        Task.detached(priority: .utility) {
            var paths = [hub, envs, root]
            if let checkpoints { paths.append(checkpoints) }
            let sizes = Self.directorySizes(paths)

            let usage = StorageUsage(
                huggingFaceCacheBytes: sizes[hub.path] ?? 0,
                careyCheckpointsBytes: checkpoints.flatMap { sizes[$0.path] } ?? 0,
                environmentsBytes: sizes[envs.path] ?? 0,
                totalBytes: sizes[root.path] ?? 0
            )
            let free = Self.freeBytes(at: root)
            await MainActor.run {
                self.usage = usage
                self.volumeFreeBytes = free
                self.isScanning = false
            }
        }
    }

    nonisolated private static func freeBytes(at url: URL) -> Int64 {
        guard let values = try? url.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ), let available = values.volumeAvailableCapacityForImportantUsage else { return 0 }
        return Int64(available)
    }

    /// One `du -sk` for every path we care about. Returns bytes keyed by path.
    nonisolated private static func directorySizes(_ urls: [URL]) -> [String: Int64] {
        let existing = urls.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !existing.isEmpty else { return [:] }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/du")
        // -s totals, -k in kibibytes. Paths last so du reports one line each.
        task.arguments = ["-sk"] + existing.map(\.path)
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice

        do {
            try task.run()
        } catch {
            return [:]
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8) else { return [:] }

        var sizes: [String: Int64] = [:]
        for line in text.split(separator: "\n") {
            // "<kb>\t<path>" -- the path may contain spaces, so split once.
            guard let tab = line.firstIndex(of: "\t") else { continue }
            let kb = Int64(line[line.startIndex..<tab].trimmingCharacters(in: .whitespaces)) ?? 0
            let path = String(line[line.index(after: tab)...])
            sizes[path] = kb * 1024
        }
        return sizes
    }

    // MARK: - Reclaim

    func reclaim(
        scriptURL: URL,
        checkpointsDirectory: URL,
        huggingFaceHubDirectory: URL
    ) {
        guard !isBusy else { return }
        guard FileManager.default.isReadableFile(atPath: scriptURL.path) else {
            errorMessage = "the reclaim helper is missing from the runtime."
            return
        }
        isReclaiming = true
        errorMessage = nil
        lastResult = nil
        progressMessage = "looking for duplicated files..."

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = [
            "python3",
            scriptURL.path,
            "--checkpoints-dir", checkpointsDirectory.path,
            "--hf-cache", huggingFaceHubDirectory.path,
        ]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()
        process = task

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            for line in text.split(separator: "\n") {
                guard let payload = try? JSONSerialization.jsonObject(
                    with: Data(line.utf8)
                ) as? [String: Any] else { continue }
                Task { @MainActor [weak self] in
                    self?.handle(event: payload)
                }
            }
        }

        task.terminationHandler = { [weak self] finished in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.isReclaiming = false
                self.process = nil
                if finished.terminationStatus != 0, self.errorMessage == nil {
                    self.errorMessage = "the reclaim pass exited with status \(finished.terminationStatus)."
                }
                self.progressMessage = ""
            }
        }

        do {
            try task.run()
        } catch {
            isReclaiming = false
            process = nil
            errorMessage = error.localizedDescription
        }
    }

    private func handle(event: [String: Any]) {
        let name = event["event"] as? String ?? ""
        switch name {
        case "start":
            progressMessage = "looking for duplicated files..."
        case "checking":
            let count = event["files"] as? Int ?? 0
            if count == 0 {
                progressMessage = "nothing to check."
            } else {
                let bytes = Int64(event["bytes"] as? Int ?? 0)
                progressMessage = "checking \(count) file\(count == 1 ? "" : "s") (\(Self.format(bytes)))..."
            }
        case "progress":
            if let index = event["index"] as? Int,
               let total = event["total"] as? Int {
                let reclaimed = Int64(event["reclaimed_bytes"] as? Int ?? 0)
                let file = (event["file"] as? String) ?? ""
                progressMessage = reclaimed > 0
                    ? "\(index)/\(total) — freed \(Self.format(reclaimed)) so far — \(file)"
                    : "\(index)/\(total) — \(file)"
            }
        case "error":
            errorMessage = event["message"] as? String ?? "the reclaim pass failed."
        case "done":
            lastResult = StorageReclaimResult(
                matchedBytes: Int64(event["matched_bytes"] as? Int ?? 0),
                reclaimedBytes: Int64(event["reclaimed_bytes"] as? Int ?? 0),
                reclaimedFiles: event["reclaimed_files"] as? Int ?? 0,
                skipped: event["skipped"] as? Int ?? 0,
                freeDeltaBytes: Int64(event["free_delta_bytes"] as? Int ?? 0)
            )
            progressMessage = ""
        default:
            break
        }
    }

    nonisolated static func format(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        return formatter.string(fromByteCount: bytes)
    }

    func cancel() {
        process?.terminate()
        process = nil
        isReclaiming = false
        progressMessage = ""
    }
}
