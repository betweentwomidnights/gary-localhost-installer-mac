import SwiftUI

struct StorageSettingsSheet: View {
    @ObservedObject var viewModel: ControlCenterViewModel
    @Environment(\.dismiss) private var dismiss

    private var storage: StorageMaintenanceManager { viewModel.storageMaintenanceManager }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("storage")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            usageSection
            Divider()
            reclaimSection

            if let error = storage.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(minWidth: 620, minHeight: 460)
        .onAppear { viewModel.scanStorageUsage() }
    }

    // MARK: - Usage

    @ViewBuilder
    private var usageSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("on disk")
                    .font(.caption.weight(.semibold))
                Spacer()
                Button(storage.isScanning ? "measuring..." : "refresh") {
                    viewModel.scanStorageUsage()
                }
                .disabled(storage.isBusy)
            }

            // Free space is instant and is the number that moves when
            // duplicates are reclaimed, so it is shown even while du runs.
            row("free on this volume", storage.volumeFreeBytes, emphasised: true)

            if let usage = storage.usage {
                Divider()
                row("hugging face cache", usage.huggingFaceCacheBytes)
                row("carey checkpoints", usage.careyCheckpointsBytes)
                row("service environments", usage.environmentsBytes)
                row("everything else", usage.otherBytes)
                row("gary4local total", usage.totalBytes, emphasised: true)
            } else if storage.isScanning {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("measuring...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func row(_ label: String, _ bytes: Int64, emphasised: Bool = false) -> some View {
        HStack {
            Text(label)
                .font(emphasised ? .caption.weight(.semibold) : .caption)
            Spacer()
            Text(formatBytes(bytes))
                .font(emphasised
                      ? .caption.monospaced().weight(.semibold)
                      : .caption.monospaced())
                .foregroundStyle(emphasised ? .primary : .secondary)
        }
    }

    // MARK: - Reclaim

    @ViewBuilder
    private var reclaimSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("duplicated model files")
                .font(.caption.weight(.semibold))
            Text("carey models used to be stored twice — once in the hugging face cache, once in the checkpoints folder. this replaces the second copy with a clone sharing the same storage. nothing is deleted and both copies stay readable, so the folder sizes above will not change — the free space will.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button(storage.isReclaiming ? "reclaiming..." : "find and reclaim duplicates") {
                    viewModel.reclaimDuplicateModels()
                }
                .disabled(!viewModel.canReclaimDuplicateModels || storage.isReclaiming)

                if storage.isReclaiming {
                    ProgressView().controlSize(.small)
                    Button("cancel") { storage.cancel() }
                }
                Spacer()
            }

            if !viewModel.canReclaimDuplicateModels {
                Text(viewModel.reclaimBlockedReason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if storage.isReclaiming {
                Text(storage.progressMessage.isEmpty ? "starting..." : storage.progressMessage)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("this reads every model file that could be a duplicate, so it takes a while on a large install.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let result = storage.lastResult, !storage.isReclaiming {
                resultText(result)
            }
        }
    }

    @ViewBuilder
    private func resultText(_ result: StorageReclaimResult) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if result.reclaimedFiles > 0 {
                Text("reclaimed \(formatBytes(result.reclaimedBytes)) across \(result.reclaimedFiles) file\(result.reclaimedFiles == 1 ? "" : "s").")
                    .font(.caption)
            } else if result.alreadySharedFiles > 0 {
                // The duplicated content is still there and still matches; it
                // just already shares storage, so there is nothing left to get.
                Text("already deduplicated — \(result.alreadySharedFiles) duplicated file\(result.alreadySharedFiles == 1 ? "" : "s") already share storage, so there was nothing to reclaim.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("no duplicated model files found — nothing to reclaim.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if result.reclaimedFiles > 0, result.alreadySharedFiles > 0 {
                Text("\(result.alreadySharedFiles) other file\(result.alreadySharedFiles == 1 ? " was" : "s were") already sharing storage.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if result.skipped > 0 {
                Text("\(result.skipped) file\(result.skipped == 1 ? "" : "s") skipped.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        return formatter.string(fromByteCount: bytes)
    }
}
