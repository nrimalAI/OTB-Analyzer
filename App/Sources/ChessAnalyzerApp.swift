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
            // How far Dynamic Type goes is set per screen (`TypeLimit`):
            // the lists and sheets of text follow it into the accessibility
            // sizes; the board's screens, dense with controls, stop earlier.
            .task {
                if EngineSelfTest.isRequested {
                    await EngineSelfTest.run()
                }
                if let path = RecognitionSelfTest.requestedImagePath {
                    await RecognitionSelfTest.run(imagePath: path, recognizer: model.recognizer)
                }
                // Dev-only deep link for screenshots/automation:
                // SIMCTL_CHILD_AUTOFLOW=scan|review|analysis|scanned
                switch ProcessInfo.processInfo.environment["AUTOFLOW"] {
                case "scan":
                    model.path = [.scan]
                #if DEBUG
                case "scanned":
                    model.showDemoScan()
                #endif
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
