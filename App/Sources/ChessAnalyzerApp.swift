import SwiftUI

@main
struct ChessAnalyzerApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $model.path) {
                CaptureView()
                    .navigationDestination(for: AppRoute.self) { route in
                        switch route {
                        case .review:
                            ReviewView()
                        case .analysis:
                            AnalysisView()
                        }
                    }
            }
            .environmentObject(model)
            .task {
                if EngineSelfTest.isRequested {
                    await EngineSelfTest.run()
                }
                if let path = RecognitionSelfTest.requestedImagePath {
                    await RecognitionSelfTest.run(imagePath: path, recognizer: model.recognizer)
                }
                // Dev-only deep link for screenshots/automation:
                // SIMCTL_CHILD_AUTOFLOW=review|analysis
                switch ProcessInfo.processInfo.environment["AUTOFLOW"] {
                case "review":
                    model.startFromScratch(startingPosition: true)
                case "analysis":
                    model.startFromScratch(startingPosition: true)
                    model.path.append(.analysis)
                default:
                    break
                }
            }
        }
    }
}
