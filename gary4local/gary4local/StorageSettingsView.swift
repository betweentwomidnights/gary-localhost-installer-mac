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

            if let usage = storage.usage {
                row("hugging face cache", usage.huggingFaceCacheBytes)
                row("carey checkpoints", usage.careyCheckpointsBytes)
                row("service environments", usage.environmentsBytes)
                row("everything else", usage.otherBytes)
                Divider()
                row("total", usage.totalBytes, emphasised: true)
            } else if storage.isScanning {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("measuring — a full install is a lot of files")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("no measurement yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
            Text("carey models used to be stored twice — once in the hugging face cache, once in the checkpoints folder. this replaces the second copy with a clone that shares the same storage. nothing is deleted, and both copies stay readable.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button("check for duplicates") {
                    viewModel.reclaimDuplicateModels(dryRun: true)
                }
                .disabled(!viewModel.canReclaimDuplicateModels || storage.isBusy)

                if let result = storage.lastResult, result.wasDryRun, result.matchedBytes > 0 {
                    Button("reclaim \(formatBytes(result.matchedBytes))") {
                        viewModel.reclaimDuplicateModels(dryRun: false)
                    }
                    .disabled(!viewModel.canReclaimDuplicateModels || storage.isBusy)
                }

                if storage.isReclaiming {
                    Button("cancel") { storage.cancel() }
                }
                Spacer()
            }

            if !viewModel.canReclaimDuplicateModels {
                Text(viewModel.reclaimBlockedReason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if storage.isReclaiming {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(storage.progressMessage)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            if let result = storage.lastResult, !storage.isReclaiming {
                resultText(result)
            }
        }
    }

    @ViewBuilder
    private func resultText(_ result: StorageReclaimResult) -> some View {
        if result.wasDryRun {
            if result.matchedBytes > 0 {
                Text("found \(formatBytes(result.matchedBytes)) of duplicated model files.")
                    .font(.caption)
            } else {
                Text("no duplicated model files found. nothing to reclaim.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else {
            VStack(alignment: .leading, spacing: 2) {
                Text("reclaimed \(formatBytes(result.freeDeltaBytes)) across \(result.reclaimedFiles) file\(result.reclaimedFiles == 1 ? "" : "s").")
                    .font(.caption)
                if result.skipped > 0 {
                    Text("\(result.skipped) file\(result.skipped == 1 ? "" : "s") skipped.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
