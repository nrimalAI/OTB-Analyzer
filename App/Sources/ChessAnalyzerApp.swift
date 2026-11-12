import SwiftUI

@main
struct ChessAnalyzerApp: App {
    @StateObject private var model = AppModel()

    init() {
        Theme.applyBoardTypography()
    }

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $model.path) {
                HomeView()
                    .navigationDestination(for: AppRoute.self) { route in
                        switch route {
                        case .scan:
                            ScanView()
                        case .board:
                            BoardWorkspaceView()
                        }
                    }
            }
            .environmentObject(model)
            .environmentObject(model.history)
            .tint(Theme.ink)
            // The app is paper-and-ink everywhere except the scan screen,
            // which is dark like a camera; this also sets the status bar.
            .preferredColorScheme(model.path.last == .scan ? .dark : .light)
            // Layouts are dense (board + tray + controls); follow Dynamic
            // Type up to a point rather than letting rows collide.
            .dynamicTypeSize(...DynamicTypeSize.xxLarge)
            .task {
                if EngineSelfTest.isRequested {
                    await EngineSelfTest.run()
                }
                if let path = RecognitionSelfTest.requestedImagePath {
                    await RecognitionSelfTest.run(imagePath: path, recognizer: model.recognizer)
                }
                // Dev-only deep link for screenshots/automation:
                // SIMCTL_CHILD_AUTOFLOW=scan|review|analysis
                switch ProcessInfo.processInfo.environment["AUTOFLOW"] {
                case "scan":
                    model.path = [.scan]
                case "review":
                    model.startFromScratch(startingPosition: true)
                case "analysis":
                    model.startFromScratch(startingPosition: true)
                    model.tab = .analysis
                default:
                    break
                }
            }
        }
    }
}
