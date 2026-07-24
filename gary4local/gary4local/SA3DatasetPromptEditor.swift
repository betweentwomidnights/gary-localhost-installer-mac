import AppKit
import SwiftUI

private struct SA3DatasetPromptEntry: Identifiable {
    let audioPath: String
    let relativePath: String
    let sidecarPath: String
    var content: String
    var originalContent: String
    let jsonSidecarExists: Bool

    var id: String { audioPath }
    var isDirty: Bool { content != originalContent }
}

struct SA3DatasetPromptEditorSheet: View {
    let datasetPath: String
    @ObservedObject var autolabeler: SA3AutolabelManager
    let sa3EnvironmentReady: Bool
    let careyServiceIsRunning: Bool
    let careyEnvironmentReady: Bool
    let careyTrainingIsActive: Bool
    let onSuggestMetadata: (String) async throws -> SA3MetadataSuggestion
    let onStartAutolabel: (String, SA3PromptStyle) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var entries: [SA3DatasetPromptEntry] = []
    @State private var selectedIndex = 0
    @State private var templateText = Self.barePrompt
    @State private var promptStyle = SA3PromptStyle.bare
    @State private var showStyleHelp = false
    @State private var isLoading = false
    @State private var isSaving = false
    @State private var isSuggesting = false
    @State private var lastAutolabelDone = 0
    @State private var errorMessage: String?
    @State private var statusMessage: String?

    private static let barePrompt = "technical death metal, 145 bpm, C minor"
    private static let labeledPrompt =
        "TrackType: Music, VocalType: Instrumental, Genre: technical death metal, Mood: absurd, BPM: 145, Key: C minor"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    description
                    templatePanel

                    if let errorMessage {
                        notice(errorMessage, color: .red)
                    }
                    if let statusMessage {
                        notice(statusMessage, color: .green)
                    }
                    if let launchError = autolabeler.launchError {
                        notice(launchError, color: .red)
                    }
                    if let state = relevantAutolabelState,
                       state.isActive || state.isTerminal {
                        autolabelNotice(state)
                    }

                    if isLoading {
                        ProgressView("scanning audio files...")
                            .frame(maxWidth: .infinity, minHeight: 260)
                    } else if entries.isEmpty {
                        Text("no supported audio files found in this folder.")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, minHeight: 260)
                    } else {
                        editor
                    }
                }
                .padding(18)
            }

            Divider()
            footer
        }
        .frame(minWidth: 820, idealWidth: 920, minHeight: 650, idealHeight: 760)
        .onAppear {
            loadEntries()
            autolabeler.refresh()
            lastAutolabelDone = relevantAutolabelState?.done ?? 0
        }
        .onChange(of: autolabeler.state) { _, state in
            handleAutolabelStateChange(state)
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("optional dataset prompts")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("edit sa3 text sidecars")
                    .font(.title2.weight(.semibold))
            }
            Spacer()
            Button("refresh") { loadEntries() }
                .disabled(isLoading || isSaving || autolabelIsActive)
            Button("close") { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(18)
    }

    private var description: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(
                "Give each audio file an optional same-name .txt file, such as song.wav and song.txt. Everything in the text file is used as that track's prompt."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            Text(
                "The shared trigger word is applied separately during training and is never written to these sidecars or exported dice prompts."
            )
            .font(.caption)
            .foregroundStyle(.secondary)

            HStack(spacing: 14) {
                Button("open underfit metadata guide") {
                    NSWorkspace.shared.open(
                        URL(string: "https://github.com/dada-bots/underfit#2-optional-add-metadata-for-prompts")!
                    )
                }
                .buttonStyle(.link)
                Button("open official sa3 prompting guide") {
                    NSWorkspace.shared.open(
                        URL(string: "https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md")!
                    )
                }
                .buttonStyle(.link)
            }
        }
    }

    private var templatePanel: some View {
        GroupBox("fill missing prompts") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Text("prompt style")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("prompt style", selection: $promptStyle) {
                        ForEach(SA3PromptStyle.allCases) { style in
                            Text(style.displayName).tag(style)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(width: 230)
                    .disabled(autolabelIsActive)
                    .onChange(of: promptStyle) { oldStyle, newStyle in
                        updateStarterPrompt(from: oldStyle, to: newStyle)
                    }
                    Button {
                        showStyleHelp.toggle()
                    } label: {
                        Image(systemName: "info.circle")
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("about prompt styles")
                }

                if showStyleHelp {
                    Text(
                        "Barebones matches the prompt tail Gary adds at inference. The official SA3 style uses labeled fields. Either trains correctly; BPM and key are omitted from the dice pool and supplied separately."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Text("editable starter prompt — \(promptStyle.displayName) style")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $templateText)
                    .frame(minHeight: 58)
                    .overlay(
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(Color.gray.opacity(0.25))
                    )
                    .disabled(autolabelIsActive)

                HStack {
                    Button("fill missing", action: fillMissing)
                        .disabled(entries.isEmpty || isLoading || autolabelIsActive)
                    if autolabelIsActive {
                        Button(
                            "cancel auto-label (\(relevantAutolabelState?.done ?? 0)/\(relevantAutolabelState?.total ?? entries.count))"
                        ) {
                            autolabeler.cancel()
                        }
                    } else {
                        Button("auto-label all", action: startAutolabel)
                            .disabled(!canStartAutolabel)
                            .help(autolabelHelp)
                    }
                    Spacer()
                    Text("\(captionedCount) of \(entries.count) tracks have prompt text")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 5)
        }
    }

    private var editor: some View {
        HSplitView {
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(Array(entries.enumerated()), id: \.element.id) { index, entry in
                        Button {
                            selectedIndex = index
                            statusMessage = nil
                        } label: {
                            HStack {
                                if autolabelIsActive,
                                   index == (relevantAutolabelState?.done ?? -1) {
                                    ProgressView()
                                        .controlSize(.mini)
                                }
                                Text(entry.relativePath)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer()
                                Text(entry.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : "txt")
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(
                                        entry.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                            ? Color.secondary
                                            : Color.green
                                    )
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 7)
                            .background(
                                index == selectedIndex
                                    ? Color.accentColor.opacity(0.16)
                                    : Color.clear
                            )
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .frame(minWidth: 230, idealWidth: 280)

            if entries.indices.contains(selectedIndex) {
                trackEditor
                    .frame(minWidth: 470)
            }
        }
        .frame(minHeight: 330)
    }

    private var trackEditor: some View {
        let entry = entries[selectedIndex]
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(entry.relativePath)
                        .font(.headline)
                    Text(entry.sidecarPath)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                Text("\(selectedIndex + 1) / \(entries.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            if entry.jsonSidecarExists {
                notice(
                    "a json sidecar exists for this track and takes precedence over .txt during training.",
                    color: .orange
                )
            }

            Text("literal text-sidecar prompt")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $entries[selectedIndex].content)
                .frame(minHeight: 100)
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .stroke(Color.gray.opacity(0.25))
                )
                .disabled(autolabelIsActive)

            HStack(spacing: 10) {
                Button(isSuggesting ? "analyzing..." : "suggest bpm/key") {
                    suggestBPMKey()
                }
                .disabled(
                    isSuggesting
                        || autolabelIsActive
                        || !sa3EnvironmentReady
                )
                if isSuggesting {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(
                    sa3EnvironmentReady
                        ? "uses the local Carey tempo/key estimators"
                        : "build the SA3 environment to enable suggestions"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 5) {
                Text("dice button result")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(selectedDicePrompt.isEmpty ? "not added to the lora prompt pool" : selectedDicePrompt)
                if entry.content.trimmingCharacters(in: .whitespacesAndNewlines) != selectedDicePrompt {
                    Text("trailing BPM/key tags are omitted because Gary supplies them separately.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            HStack {
                Button("clear") {
                    entries[selectedIndex].content = ""
                    statusMessage = nil
                }
                .disabled(autolabelIsActive)
                Button("restore") {
                    entries[selectedIndex].content = entries[selectedIndex].originalContent
                    statusMessage = nil
                }
                .disabled(!entries[selectedIndex].isDirty || autolabelIsActive)
                Spacer()
                Button("previous") {
                    selectedIndex = max(0, selectedIndex - 1)
                }
                .disabled(selectedIndex == 0)
                Button("next") {
                    selectedIndex = min(entries.count - 1, selectedIndex + 1)
                }
                .disabled(selectedIndex == entries.count - 1)
            }
        }
        .padding(.leading, 14)
    }

    private var footer: some View {
        HStack {
            Text(
                dirtyCount == 0
                    ? "all changes saved"
                    : "\(dirtyCount) unsaved change\(dirtyCount == 1 ? "" : "s")"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            Spacer()
            Button(isSaving ? "saving..." : "save sidecars", action: saveEntries)
                .garyPrimaryButtonStyle()
                .disabled(
                    isSaving
                        || isSuggesting
                        || dirtyCount == 0
                        || autolabelIsActive
                )
        }
        .padding(14)
    }

    private var dirtyCount: Int {
        entries.filter(\.isDirty).count
    }

    private var captionedCount: Int {
        entries.filter {
            !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.count
    }

    private var relevantAutolabelState: SA3AutolabelState? {
        guard let state = autolabeler.state,
              let statePath = state.datasetPath else {
            return nil
        }
        let expected = URL(fileURLWithPath: datasetPath).standardizedFileURL.path
        let actual = URL(fileURLWithPath: statePath).standardizedFileURL.path
        return expected == actual ? state : nil
    }

    private var autolabelIsActive: Bool {
        relevantAutolabelState?.isActive == true
    }

    private var canStartAutolabel: Bool {
        !entries.isEmpty
            && dirtyCount == 0
            && !isLoading
            && !isSaving
            && !isSuggesting
            && careyEnvironmentReady
            && !careyServiceIsRunning
            && !careyTrainingIsActive
            && autolabeler.state?.isActive != true
    }

    private var autolabelHelp: String {
        if dirtyCount > 0 {
            return "Save or discard unsaved sidecar changes before auto-labeling."
        }
        if !careyEnvironmentReady {
            return "Build Carey before auto-labeling."
        }
        if careyServiceIsRunning {
            return "Stop Carey before auto-labeling."
        }
        if careyTrainingIsActive {
            return "Wait for Carey training to finish."
        }
        if autolabeler.state?.isActive == true {
            return "Another SA3 dataset is currently being auto-labeled."
        }
        return "Overwrites every text sidecar with captioned genre plus locally reconciled BPM and key."
    }

    private var selectedDicePrompt: String {
        guard entries.indices.contains(selectedIndex) else { return "" }
        return Self.dicePrompt(from: entries[selectedIndex].content)
    }

    private func updateStarterPrompt(
        from oldStyle: SA3PromptStyle,
        to newStyle: SA3PromptStyle
    ) {
        let current = templateText.trimmingCharacters(in: .whitespacesAndNewlines)
        let known = [Self.barePrompt, Self.labeledPrompt]
        guard known.contains(current) else { return }
        templateText = newStyle == .bare ? Self.barePrompt : Self.labeledPrompt
    }

    private func startAutolabel() {
        guard dirtyCount == 0 else {
            errorMessage =
                "save or discard unsaved sidecar changes before auto-labeling."
            return
        }
        errorMessage = nil
        statusMessage = nil
        autolabeler.clearError()
        lastAutolabelDone = 0
        onStartAutolabel(datasetPath, promptStyle)
    }

    private func suggestBPMKey() {
        guard entries.indices.contains(selectedIndex) else { return }
        let entryID = entries[selectedIndex].id
        let audioPath = entries[selectedIndex].audioPath
        isSuggesting = true
        errorMessage = nil
        statusMessage = nil
        Task {
            do {
                let result = try await onSuggestMetadata(audioPath)
                let tag = metadataTag(
                    bpm: result.bpm,
                    keyscale: result.keyscale ?? ""
                )
                guard !tag.isEmpty else {
                    statusMessage = "no BPM or key could be detected."
                    isSuggesting = false
                    return
                }
                if let index = entries.firstIndex(where: { $0.id == entryID }) {
                    entries[index].content = Self.spliceMetadata(
                        into: entries[index].content,
                        tag: tag
                    )
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isSuggesting = false
        }
    }

    private func metadataTag(bpm: Int?, keyscale: String) -> String {
        var parts: [String] = []
        if let bpm {
            parts.append(promptStyle == .labeled ? "BPM: \(bpm)" : "\(bpm) bpm")
        }
        let keyscale = keyscale.trimmingCharacters(in: .whitespacesAndNewlines)
        if !keyscale.isEmpty {
            parts.append(promptStyle == .labeled ? "Key: \(keyscale)" : keyscale)
        }
        return parts.joined(separator: ", ")
    }

    private func handleAutolabelStateChange(_ state: SA3AutolabelState?) {
        guard let state,
              relevantAutolabelState != nil else {
            return
        }
        let done = state.done ?? 0
        if done > lastAutolabelDone {
            lastAutolabelDone = done
            loadEntries(preservingMessages: true)
            selectedIndex = min(max(0, done - 1), max(0, entries.count - 1))
        } else if state.isTerminal {
            loadEntries(preservingMessages: true)
        }
        if state.status == "failed" {
            errorMessage = state.error ?? state.message ?? "auto-labeling failed."
        } else if state.status == "completed" {
            statusMessage = state.message ?? "auto-labeling complete."
        } else if state.status == "cancelled" {
            statusMessage = state.message ?? "auto-labeling cancelled."
        }
    }

    private func loadEntries(preservingMessages: Bool = false) {
        isLoading = true
        if !preservingMessages {
            errorMessage = nil
            statusMessage = nil
        }
        do {
            entries = try Self.scan(datasetPath: datasetPath)
            selectedIndex = min(selectedIndex, max(0, entries.count - 1))
        } catch {
            entries = []
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    private func fillMissing() {
        let template = templateText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !template.isEmpty else {
            errorMessage = "enter a template first."
            return
        }

        var filled = 0
        for index in entries.indices {
            guard entries[index].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  !entries[index].jsonSidecarExists else {
                continue
            }
            entries[index].content = template
            filled += 1
        }
        errorMessage = nil
        statusMessage = filled == 0
            ? "no empty text sidecars found. tracks with json metadata were skipped."
            : "filled \(filled) missing sidecar draft\(filled == 1 ? "" : "s")."
    }

    private func saveEntries() {
        isSaving = true
        errorMessage = nil
        statusMessage = nil
        var saved = 0
        var removed = 0

        do {
            for index in entries.indices where entries[index].isDirty {
                let sidecarURL = URL(fileURLWithPath: entries[index].sidecarPath)
                let content = entries[index].content
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if content.isEmpty {
                    if FileManager.default.fileExists(atPath: sidecarURL.path) {
                        try FileManager.default.removeItem(at: sidecarURL)
                        removed += 1
                    }
                } else {
                    try "\(content)\n".write(
                        to: sidecarURL,
                        atomically: true,
                        encoding: .utf8
                    )
                    saved += 1
                }
                entries[index].content = content
                entries[index].originalContent = content
            }
            let parts = [
                saved > 0 ? "\(saved) saved" : nil,
                removed > 0 ? "\(removed) removed" : nil,
            ].compactMap { $0 }
            statusMessage = parts.isEmpty
                ? "sidecars are already current."
                : "sidecars updated: \(parts.joined(separator: ", "))."
        } catch {
            errorMessage = "could not save sidecars: \(error.localizedDescription)"
        }
        isSaving = false
    }

    private func notice(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.callout)
            .padding(9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(color.opacity(0.09))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func autolabelNotice(_ state: SA3AutolabelState) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                if state.isActive {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(state.message ?? "SA3 auto-label")
                    .font(.callout.weight(.medium))
                Spacer()
                if let done = state.done, let total = state.total, total > 0 {
                    Text("\(done) / \(total)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if state.isActive, let done = state.done, let total = state.total, total > 0 {
                ProgressView(value: Double(done), total: Double(total))
            }
            if let path = state.currentPath, !path.isEmpty {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            if !autolabeler.logText.isEmpty {
                DisclosureGroup("auto-label log") {
                    ScrollView {
                        Text(autolabeler.logText)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 150)
                }
                .font(.caption)
            }
        }
        .padding(9)
        .background(
            (state.status == "failed" ? Color.red : Color.blue).opacity(0.09)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private static func scan(datasetPath: String) throws -> [SA3DatasetPromptEntry] {
        let fileManager = FileManager.default
        let root = URL(fileURLWithPath: datasetPath, isDirectory: true).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw NSError(
                domain: "SA3DatasetPromptEditor",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "choose a valid audio folder first."]
            )
        }

        let extensions = Set(["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "aif"])
        let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        var audioURLs: [URL] = []
        while let candidate = enumerator?.nextObject() as? URL {
            let values = try? candidate.resourceValues(
                forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
            )
            guard values?.isRegularFile == true,
                  values?.isSymbolicLink != true,
                  extensions.contains(candidate.pathExtension.lowercased()) else {
                continue
            }
            audioURLs.append(candidate.standardizedFileURL)
        }
        audioURLs.sort {
            relativePath(for: $0, root: root).localizedCaseInsensitiveCompare(
                relativePath(for: $1, root: root)
            ) == .orderedAscending
        }

        return audioURLs.map { audioURL in
            let txtURL = audioURL.deletingPathExtension().appendingPathExtension("txt")
            let content = (try? String(contentsOf: txtURL, encoding: .utf8))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                ?? ""
            return SA3DatasetPromptEntry(
                audioPath: audioURL.path,
                relativePath: relativePath(for: audioURL, root: root),
                sidecarPath: txtURL.path,
                content: content,
                originalContent: content,
                jsonSidecarExists: jsonSidecarExists(
                    audioURL: audioURL,
                    root: root
                )
            )
        }
    }

    private static func relativePath(for url: URL, root: URL) -> String {
        let rootPath = root.path.hasSuffix("/") ? root.path : "\(root.path)/"
        guard url.path.hasPrefix(rootPath) else { return url.lastPathComponent }
        return String(url.path.dropFirst(rootPath.count))
    }

    private static func jsonSidecarExists(audioURL: URL, root: URL) -> Bool {
        let direct = audioURL.deletingPathExtension().appendingPathExtension("json")
        if FileManager.default.fileExists(atPath: direct.path) {
            return true
        }
        let parent = audioURL.deletingLastPathComponent()
        guard parent.path != root.path else { return false }
        let sibling = parent.deletingLastPathComponent()
            .appendingPathComponent("json", isDirectory: true)
            .appendingPathComponent(audioURL.deletingPathExtension().lastPathComponent)
            .appendingPathExtension("json")
        return sibling.path.hasPrefix(root.path)
            && FileManager.default.fileExists(atPath: sibling.path)
    }

    private static func dicePrompt(from text: String) -> String {
        let pattern =
            #"(?i)(?:[,;]\s*)?(?:BPM\s*[:=]?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*BPM|(?:Key|Scale)\s*[:=]\s*[A-G][#b♯♭]?\s+(?:maj(?:or)?|min(?:or)?)|[A-G][#b♯♭]?\s+(?:major|minor))\s*$"#
        var prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        while true {
            let next = prompt.replacingOccurrences(
                of: pattern,
                with: "",
                options: .regularExpression
            )
            .trimmingCharacters(in: CharacterSet(charactersIn: " ,;"))
            if next == prompt {
                return prompt
            }
            prompt = next
        }
    }

    private static func spliceMetadata(into content: String, tag: String) -> String {
        let base = dicePrompt(from: content)
        guard !tag.isEmpty else { return base }
        guard !base.isEmpty else { return tag }
        return "\(base), \(tag)"
    }
}
