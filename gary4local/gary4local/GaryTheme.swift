import SwiftUI

enum GaryTheme {
    static let red = Color(red: 0.78, green: 0.17, blue: 0.20)
}

struct GaryCheckboxToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button {
            configuration.isOn.toggle()
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: configuration.isOn ? "checkmark.square.fill" : "square")
                    .foregroundStyle(configuration.isOn ? GaryTheme.red : .secondary)
                configuration.label
            }
        }
        .buttonStyle(.plain)
    }
}

extension View {
    func garyPrimaryButtonStyle() -> some View {
        self
            .buttonStyle(.borderedProminent)
            .tint(GaryTheme.red)
    }

    func garyPickerAccent() -> some View {
        self
            .tint(GaryTheme.red)
    }

    func garyCheckboxStyle() -> some View {
        self
            .toggleStyle(GaryCheckboxToggleStyle())
    }
}
