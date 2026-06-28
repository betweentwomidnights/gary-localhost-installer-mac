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
    let triggerText: String

    @Environment(\.dismiss) private var dismiss
    @State private var entries: [SA3DatasetPromptEntry] = []
    @State private var selectedIndex = 0
    @State private var templateText =
        "TrackType: Music, VocalType: Instrumental, Genre: technical death metal, Mood: absurd, BPM: 145"
    @State private var includeTrigger = false
    @State private var isLoading = false
    @State private var isSaving = false
    @State private var errorMessage: String?
    @State private var statusMessage: String?

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
        .onAppear(perform: loadEntries)
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
            Button("refresh", action: loadEntries)
                .disabled(isLoading || isSaving)
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
                Text("editable starter prompt using sa3 tags and underfit example values")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $templateText)
                    .frame(minHeight: 58)
                    .overlay(
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(Color.gray.opacity(0.25))
                    )

                Toggle(
                    triggerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        ? "prepend custom trigger word"
                        : "prepend custom trigger word: \(triggerText)",
                    isOn: $includeTrigger
                )
                .tint(GaryTheme.red)
                .disabled(triggerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                HStack {
                    Button("fill missing", action: fillMissing)
                        .disabled(entries.isEmpty || isLoading)
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

            VStack(alignment: .leading, spacing: 5) {
                Text("dice button result")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(selectedDicePrompt.isEmpty ? "not added to the lora prompt pool" : selectedDicePrompt)
                if entry.content.trimmingCharacters(in: .whitespacesAndNewlines) != selectedDicePrompt {
                    Text("a trailing bpm tag is omitted because gary supplies tempo separately.")
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
                Button("restore") {
                    entries[selectedIndex].content = entries[selectedIndex].originalContent
                    statusMessage = nil
                }
                .disabled(!entries[selectedIndex].isDirty)
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
                .disabled(isSaving || dirtyCount == 0)
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

    private var selectedDicePrompt: String {
        guard entries.indices.contains(selectedIndex) else { return "" }
        return Self.dicePrompt(from: entries[selectedIndex].content)
    }

    private func loadEntries() {
        isLoading = true
        errorMessage = nil
        statusMessage = nil
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
        let rendered = includeTrigger
            ? Self.compose(trigger: triggerText, prompt: template)
            : template
        guard !rendered.isEmpty else {
            errorMessage = "enter a template or enable the custom trigger first."
            return
        }

        var filled = 0
        for index in entries.indices {
            guard entries[index].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  !entries[index].jsonSidecarExists else {
                continue
            }
            entries[index].content = rendered
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

    private static func compose(trigger: String, prompt: String) -> String {
        let trigger = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trigger.isEmpty else { return prompt }
        guard !prompt.isEmpty else { return trigger }
        let lowerPrompt = prompt.lowercased()
        let lowerTrigger = trigger.lowercased()
        if lowerPrompt == lowerTrigger
            || lowerPrompt.hasPrefix("\(lowerTrigger),")
            || lowerPrompt.hasPrefix("\(lowerTrigger) ") {
            return prompt
        }
        return "\(trigger), \(prompt)"
    }

    private static func dicePrompt(from text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(
                of: #"(?i)(?:[,;]\s*)?(?:BPM\s*:\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*BPM)\s*$"#,
                with: "",
                options: .regularExpression
            )
            .trimmingCharacters(in: CharacterSet(charactersIn: " ,;"))
    }
}
