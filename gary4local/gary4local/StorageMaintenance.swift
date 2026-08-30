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

nonisolated struct StorageEntry: Equatable {
    /// What the files add up to. What Finder and `du` report.
    var logicalBytes: Int64 = 0
    /// What the disk would give back. Files sharing storage with something
    /// already counted are not counted again.
    var actualBytes: Int64 = 0

    var sharedBytes: Int64 { max(0, logicalBytes - actualBytes) }
    var isMostlyShared: Bool { sharedBytes > actualBytes }
}

nonisolated struct StorageUsage: Equatable {
    var huggingFaceCache = StorageEntry()
    var careyCheckpoints = StorageEntry()
    var environments = StorageEntry()
    var other = StorageEntry()

    var totalLogicalBytes: Int64 {
        huggingFaceCache.logicalBytes + careyCheckpoints.logicalBytes
            + environments.logicalBytes + other.logicalBytes
    }
    var totalActualBytes: Int64 {
        huggingFaceCache.actualBytes + careyCheckpoints.actualBytes
            + environments.actualBytes + other.actualBytes
    }
}

struct StorageReclaimResult: Equatable {
    /// Content that is duplicated. Stays true after a successful pass, so this
    /// is NOT what you can expect to get back.
    var matchedBytes: Int64
    /// What this run actually recovered.
    var reclaimedBytes: Int64
    var reclaimedFiles: Int
    /// Files that were duplicates but already shared storage from an earlier
    /// pass, so replacing them freed nothing.
    var alreadySharedFiles: Int
    var alreadySharedBytes: Int64
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

    /// Real occupancy, via scripts/measure_storage.py.
    ///
    /// `du` counts a cloned file in full on both sides, so after the duplicate
    /// pass it reported a total far bigger than the disk was really using. The
    /// helper asks APFS for each large file's physical extent and counts shared
    /// storage once, which is the number that actually moves when space is
    /// recovered.
    func scan(scriptURL: URL?, appSupportDirectory: URL, checkpointsDirectory: URL?) {
        volumeFreeBytes = Self.freeBytes(at: appSupportDirectory)
        guard !isScanning else { return }
        guard let scriptURL else {
            errorMessage = "the storage measurement helper is missing from the runtime."
            return
        }
        isScanning = true
        errorMessage = nil

        let hub = appSupportDirectory.appendingPathComponent("cache/huggingface", isDirectory: true)
        let envs = appSupportDirectory.appendingPathComponent("venvs", isDirectory: true)
        let checkpoints = checkpointsDirectory
            ?? appSupportDirectory.appendingPathComponent("cache/carey", isDirectory: true)

        // userInitiated, not utility: this runs because someone opened the
        // panel and is watching it. Utility gets throttled disk I/O, which is
        // most of what this does.
        Task.detached(priority: .userInitiated) {
            // Order matters. The cache is charged for storage it shares with
            // the checkpoints tree, because the cache is what the hub needs to
            // keep -- so the checkpoints row shows what removing it would give
            // back, which is the useful way round.
            var arguments = [
                "python3", scriptURL.path,
                "--path", "hub=\(hub.path)",
                "--path", "checkpoints=\(checkpoints.path)",
                "--path", "venvs=\(envs.path)",
                "--path", "other=\(appSupportDirectory.path)",
            ]
            let output = Self.runJSON(arguments: &arguments)
            let free = Self.freeBytes(at: appSupportDirectory)

            await MainActor.run {
                self.isScanning = false
                self.volumeFreeBytes = free
                guard let paths = output?["paths"] as? [String: Any] else {
                    if self.usage == nil {
                        self.errorMessage = "could not measure storage."
                    }
                    return
                }
                func entry(_ key: String) -> StorageEntry {
                    guard let row = paths[key] as? [String: Any] else { return StorageEntry() }
                    return StorageEntry(
                        logicalBytes: Int64(row["logical"] as? Int ?? 0),
                        actualBytes: Int64(row["actual"] as? Int ?? 0)
                    )
                }
                self.usage = StorageUsage(
                    huggingFaceCache: entry("hub"),
                    careyCheckpoints: entry("checkpoints"),
                    environments: entry("venvs"),
                    other: entry("other")
                )
            }
        }
    }

    nonisolated private static func runJSON(arguments: inout [String]) -> [String: Any]? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = arguments
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do { try task.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    nonisolated private static func freeBytes(at url: URL) -> Int64 {
        guard let values = try? url.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ), let available = values.volumeAvailableCapacityForImportantUsage else { return 0 }
        return Int64(available)
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
                alreadySharedFiles: event["already_shared_files"] as? Int ?? 0,
                alreadySharedBytes: Int64(event["already_shared_bytes"] as? Int ?? 0),
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
