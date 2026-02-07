import SwiftUI
import AppKit

struct ControlCenterView: View {
    @ObservedObject var viewModel: ControlCenterViewModel

    var body: some View {
        Group {
            if let error = viewModel.startupError {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Control Center failed to load manifest.")
                        .font(.title3.weight(.semibold))
                    Text(error)
                        .font(.body.monospaced())
                    Text("Manifest path:")
                        .font(.caption)
                    Text(viewModel.manifestPath)
                        .font(.caption.monospaced())
                    Button("Retry") { viewModel.loadManifest() }
                }
                .padding(20)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else if let manager = viewModel.manager {
                HStack(spacing: 0) {
                    serviceList(manager: manager)
                        .frame(minWidth: 520, maxWidth: 700)
                    Divider()
                    logPane(manager: manager)
                        .frame(minWidth: 420)
                }
            } else {
                ProgressView("Loading...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    @ViewBuilder
    private func serviceList(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Gary Localhost Services")
                .font(.title3.weight(.semibold))
            Text(viewModel.manifestPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            List(manager.services) { runtime in
                ServiceRow(
                    runtime: runtime,
                    isSelected: viewModel.selectedServiceID == runtime.id,
                    onSelect: { viewModel.selectService(runtime.id) },
                    onStart: { manager.start(serviceID: runtime.id) },
                    onStop: { manager.stop(serviceID: runtime.id) },
                    onRestart: { manager.restart(serviceID: runtime.id) }
                )
            }
            .listStyle(.inset)
        }
        .padding(16)
    }

    @ViewBuilder
    private func logPane(manager: ServiceManager) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let selectedID = viewModel.selectedServiceID,
               let runtime = manager.services.first(where: { $0.id == selectedID }) {
                Text("Logs: \(runtime.service.name)")
                    .font(.headline)
                Text(runtime.service.logFile.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)

                ScrollView {
                    Text(viewModel.selectedLogText.isEmpty ? "(no log output yet)" : viewModel.selectedLogText)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .background(Color(NSColor.textBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.gray.opacity(0.2))
                )

                HStack {
                    Button("Refresh") { viewModel.refreshLog() }
                    Button("Open Log File") { NSWorkspace.shared.open(runtime.service.logFile) }
                    Spacer()
                    Button("Clear Selection") { viewModel.selectedServiceID = nil }
                }
            } else {
                Text("Select a service to view logs.")
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
    }
}

struct MenuBarContentView: View {
    @ObservedObject var viewModel: ControlCenterViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let manager = viewModel.manager {
                ForEach(manager.services) { runtime in
                    HStack {
                        StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                        Text(runtime.service.name)
                        Spacer()
                        if runtime.isRunning {
                            Button("Stop") { manager.stop(serviceID: runtime.id) }
                        } else {
                            Button("Start") { manager.start(serviceID: runtime.id) }
                        }
                    }
                }
            } else {
                Text("Manifest load failed")
                if let error = viewModel.startupError {
                    Text(error).font(.caption)
                }
            }

            Divider()
            Button("Reload Manifest") { viewModel.loadManifest() }
            Button("Quit") { NSApp.terminate(nil) }
        }
        .padding(12)
        .frame(minWidth: 360)
    }
}

private struct ServiceRow: View {
    let runtime: ServiceRuntime
    let isSelected: Bool
    let onSelect: () -> Void
    let onStart: () -> Void
    let onStop: () -> Void
    let onRestart: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                StatusDot(processState: runtime.processState, healthState: runtime.healthState)
                VStack(alignment: .leading, spacing: 2) {
                    Text(runtime.service.name).font(.headline)
                    Text("id: \(runtime.id)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let pid = runtime.pid {
                    Text("PID \(pid)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            if let error = runtime.lastError, !error.isEmpty {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Button("Start", action: onStart)
                    .disabled(runtime.isRunning)
                Button("Stop", action: onStop)
                    .disabled(!runtime.isRunning)
                Button("Restart", action: onRestart)
                Spacer()
                Text("\(runtime.processState.rawValue) / \(runtime.healthState.rawValue)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .onTapGesture(perform: onSelect)
    }
}

private struct StatusDot: View {
    let processState: ProcessState
    let healthState: HealthState

    private var color: Color {
        switch processState {
        case .running:
            return healthState == .healthy ? .green : .orange
        case .starting, .stopping:
            return .yellow
        case .failed:
            return .red
        case .stopped:
            return .gray
        }
    }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 10, height: 10)
    }
}
