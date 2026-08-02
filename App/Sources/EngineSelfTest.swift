import ChessAnalysis
import Foundation

/// Headless engine verification, run when the app is launched with
/// `ENGINE_SELFTEST=1` (used by CI / `simctl launch`). Confirms the NNUE
/// networks load and Stockfish finds a forced mate, then writes a
/// machine-readable verdict to Documents/selftest.txt.
enum EngineSelfTest {

    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["ENGINE_SELFTEST"] == "1"
    }

    private static var logURL: URL {
        URL.documentsDirectory.appending(path: "selftest.txt")
    }

    private static func report(_ line: String) {
        print(line)
        let existing = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
        try? (existing + line + "\n").write(to: logURL, atomically: true, encoding: .utf8)
    }

    static func run() async {
        try? FileManager.default.removeItem(at: logURL)
        report("TRACE start; nnuePresent=\(AnalysisEngine.evaluationNetworksArePresent)")
        guard AnalysisEngine.evaluationNetworksArePresent else {
            report("SELFTEST FAIL: NNUE networks missing from bundle")
            return
        }

        let engine = AnalysisEngine()
        await engine.start(multiPV: 1)
        report("TRACE engine started; running=\(await engine.isEngineRunning)")

        // White mates in one with Ra8#.
        let fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
        var final: AnalysisUpdate?
        var updateCount = 0
        let stream = await engine.analyze(fen: fen, depth: 10)
        for await update in stream {
            updateCount += 1
            final = update
        }
        report("TRACE stream finished; updates=\(updateCount)")
        await engine.shutdown()

        guard let final, let best = final.bestLine else {
            report("SELFTEST FAIL: no analysis produced")
            return
        }
        let moveOK = best.lanMoves.first == "a1a8"
        let scoreOK = best.score == .mate(1)
        if moveOK && scoreOK {
            report("SELFTEST PASS: best=\(best.moves.first ?? "?") score=\(best.score.displayString)")
        } else {
            report("SELFTEST FAIL: best=\(best.lanMoves.first ?? "none") score=\(best.score.displayString)")
        }
    }
}
