import ChessKitEngine
import Foundation

/// Owns the Stockfish process and turns its UCI chatter into typed
/// ``AnalysisUpdate`` values.
///
/// Stockfish 17 needs its two NNUE networks in `Bundle.main`
/// (`nn-1111cefa1111.nnue`, `nn-37f18f62d772.nnue`); chesskit-engine finds and
/// configures them automatically at start. Check
/// ``evaluationNetworksArePresent`` before starting and surface a helpful
/// error if the download script hasn't been run.
public actor AnalysisEngine {

    private let engine = Engine(type: .stockfish)
    private var isStarted = false

    public init() {}

    /// Whether the Stockfish NNUE files are bundled. Without them the engine
    /// produces no meaningful evaluation (see `Scripts/download-nnue.sh`).
    public static var evaluationNetworksArePresent: Bool {
        Bundle.main.url(forResource: "nn-1111cefa1111", withExtension: "nnue") != nil
            && Bundle.main.url(forResource: "nn-37f18f62d772", withExtension: "nnue") != nil
    }

    public func start(multiPV: Int = 3) async {
        guard !isStarted else { return }
        await engine.start(multipv: multiPV)
        // `Engine.start` kicks off the UCI handshake but returns before it
        // completes; commands sent before `isRunning` flips are dropped.
        // Wait (bounded) for readiness.
        for _ in 0..<100 {  // ≤ 5s
            if await engine.isRunning { break }
            try? await Task.sleep(for: .milliseconds(50))
        }
        isStarted = true
    }

    public func shutdown() async {
        guard isStarted else { return }
        await engine.stop()
        isStarted = false
    }

    /// Whether the underlying UCI engine has completed its handshake.
    public var isEngineRunning: Bool {
        get async { await engine.isRunning }
    }

    /// Interrupts the current search (the stream then finishes after the
    /// engine's `bestmove`).
    public func stopSearch() async {
        await engine.send(command: .stop)
    }

    /// Analyzes a position, yielding progressively deeper updates until the
    /// engine reaches `depth` (or the stream's consumer cancels).
    ///
    /// Scores are normalized to White's perspective.
    public func analyze(fen: String, depth: Int = 18) async -> AsyncStream<AnalysisUpdate> {
        let engine = self.engine
        let whiteToMove = Self.whiteToMove(fen: fen)

        return AsyncStream { continuation in
            let task = Task {
                await engine.send(command: .stop)
                await engine.send(command: .position(.fen(fen)))
                await engine.send(command: .go(depth: depth))

                guard let responses = await engine.responseStream else {
                    continuation.finish()
                    return
                }

                var lines: [Int: PrincipalVariation] = [:]
                var latestDepth = 0

                for await response in responses {
                    if Task.isCancelled { break }

                    switch response {
                    case .info(let info):
                        guard let pv = info.pv, !pv.isEmpty, let score = info.score else { continue }
                        let rank = info.multipv ?? 1
                        let normalized = Score(
                            cp: score.cp, mate: score.mate, whiteToMove: whiteToMove)
                        lines[rank] = PrincipalVariation(
                            rank: rank,
                            score: normalized,
                            moves: SANConverter.convert(fen: fen, lanMoves: pv),
                            lanMoves: pv,
                            depth: info.depth ?? 0
                        )
                        latestDepth = max(latestDepth, info.depth ?? 0)
                        continuation.yield(
                            AnalysisUpdate(
                                depth: latestDepth,
                                lines: lines.values.sorted { $0.rank < $1.rank },
                                isFinal: false
                            ))

                    case .bestmove:
                        continuation.yield(
                            AnalysisUpdate(
                                depth: latestDepth,
                                lines: lines.values.sorted { $0.rank < $1.rank },
                                isFinal: true
                            ))
                        continuation.finish()
                        return

                    default:
                        break
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func whiteToMove(fen: String) -> Bool {
        let fields = fen.split(separator: " ")
        guard fields.count >= 2 else { return true }
        return fields[1] == "w"
    }
}
